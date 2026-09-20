"""PyTorch model architecture analysis backed by FX graphs.

The analyzer counts operations that participate in ``forward`` rather than every
module merely registered on the model.  It supports ordinary ``torch.fx``
symbolic tracing and graph capture through ``torch.compile`` when example inputs
are supplied.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import fx, nn


_ACTIVATION_MODULE_NAMES = (
    "CELU", "ELU", "GELU", "GLU", "Hardshrink", "Hardsigmoid",
    "Hardswish", "Hardtanh", "LeakyReLU", "LogSigmoid", "LogSoftmax",
    "Mish", "PReLU", "ReLU", "ReLU6", "RReLU", "SELU", "SiLU",
    "Sigmoid", "Softmax", "Softmax2d", "Softmin", "Softplus",
    "Softshrink", "Softsign", "Tanh", "Tanhshrink", "Threshold",
)
ACTIVATION_MODULES = tuple(
    getattr(nn, name) for name in _ACTIVATION_MODULE_NAMES if hasattr(nn, name)
)

_ACTIVATION_FUNCTIONS = {
    name.lower(): name
    for name in _ACTIVATION_MODULE_NAMES
}
_ACTIVATION_FUNCTIONS.update({
    "log_softmax": "LogSoftmax",
    "leaky_relu": "LeakyReLU",
    "relu6": "ReLU6",
    "hardshrink": "Hardshrink",
    "hardsigmoid": "Hardsigmoid",
    "hardswish": "Hardswish",
    "hardtanh": "Hardtanh",
    "logsigmoid": "LogSigmoid",
    "softmax": "Softmax",
    "softmin": "Softmin",
    "softplus": "Softplus",
    "softshrink": "Softshrink",
    "tanhshrink": "Tanhshrink",
})


@dataclass(frozen=True)
class ModelReport:
    model_name: str
    graph_source: str
    graph_count: int
    total_operations: int
    total_layers: int
    module_layers: int
    function_calls: int
    method_calls: int
    linear_layers: int
    activation_functions: dict[str, int]
    layer_types: dict[str, int]
    function_types: dict[str, int]
    total_parameters: int
    trainable_parameters: int

    @property
    def total_activations(self) -> int:
        return sum(self.activation_functions.values())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["total_activations"] = self.total_activations
        return result

    def __str__(self) -> str:
        activations = ", ".join(
            f"{name}: {count}" for name, count in self.activation_functions.items()
        ) or "none detected"
        layers = ", ".join(
            f"{name}: {count}" for name, count in self.layer_types.items()
        ) or "none"
        functions = ", ".join(
            f"{name}: {count}" for name, count in self.function_types.items()
        ) or "none"
        return "\n".join((
            f"Model: {self.model_name}",
            f"Graph source: {self.graph_source} ({self.graph_count} graph(s))",
            f"Total graph operations: {self.total_operations}",
            f"Module layers: {self.total_layers}",
            f"  Function calls: {self.function_calls}",
            f"  Method calls: {self.method_calls}",
            f"Linear layers/operations: {self.linear_layers}",
            f"Activation functions ({self.total_activations}): {activations}",
            f"Layer types: {layers}",
            f"Function types: {functions}",
            f"Parameters: {self.total_parameters:,} total, "
            f"{self.trainable_parameters:,} trainable",
        ))


def _target_name(target: Any) -> str:
    """Return a stable readable name for an FX function or method target."""
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if name:
        return name
    text = str(target)
    return text.rsplit(".", 1)[-1].replace(">", "").strip()


def _canonical_activation(name: str) -> str | None:
    key = name.lower().replace("torch.nn.functional.", "").replace("torch.", "")
    key = key.split(".")[-1]
    return _ACTIVATION_FUNCTIONS.get(key)


def _is_linear_function(name: str) -> bool:
    # Compilers may lower a Linear module to linear or addmm.
    return name.lower().split(".")[-1] in {"linear", "addmm"}


def _capture_compiled_graphs(
    model: nn.Module,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
    fullgraph: bool,
) -> list[fx.GraphModule]:
    graphs: list[fx.GraphModule] = []

    def capture_backend(
        graph_module: fx.GraphModule, _example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(model, backend=capture_backend, fullgraph=fullgraph)
    with torch.no_grad():
        compiled(*example_args, **dict(example_kwargs))
    if not graphs:
        raise RuntimeError("torch.compile completed without producing an FX graph")
    return graphs


def analyze_model(
    model: nn.Module,
    example_args: Sequence[Any] | None = None,
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    use_compile: bool = False,
    fullgraph: bool = False,
) -> ModelReport:
    """Analyze a PyTorch model through one or more FX graphs.

    Args:
        model: Any ``torch.nn.Module``.
        example_args: Positional inputs. Required when ``use_compile=True``.
        example_kwargs: Optional keyword inputs.
        use_compile: Capture the executed graph(s) through ``torch.compile``.
            This is the best choice for models that use functional operations.
        fullgraph: Ask Dynamo to fail rather than split on graph breaks.

    ``total_layers`` is the number of ``call_module`` nodes, while
    ``total_operations`` includes module, function and method calls.
    Placeholders, constants, attributes and output nodes are excluded.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    kwargs = example_kwargs or {}
    if use_compile:
        if example_args is None:
            raise ValueError("example_args are required when use_compile=True")
        graphs = _capture_compiled_graphs(model, tuple(example_args), kwargs, fullgraph)
        graph_source = "torch.compile FX capture"
    else:
        try:
            graphs = [fx.symbolic_trace(model)]
        except Exception as exc:
            raise RuntimeError(
                "FX symbolic tracing failed. Pass representative example_args and "
                "set use_compile=True for input-dependent models."
            ) from exc
        graph_source = "torch.fx.symbolic_trace"

    layer_types: Counter[str] = Counter()
    function_types: Counter[str] = Counter()
    activations: Counter[str] = Counter()
    module_layers = function_calls = method_calls = linear_layers = 0

    for graph_module in graphs:
        modules = dict(graph_module.named_modules())
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module_layers += 1
                module = modules[str(node.target)]
                module_name = type(module).__name__
                layer_types[module_name] += 1
                if isinstance(module, nn.Linear):
                    linear_layers += 1
                if isinstance(module, ACTIVATION_MODULES):
                    activations[module_name] += 1
            elif node.op == "call_function":
                function_calls += 1
                name = _target_name(node.target)
                function_types[name] += 1
                activation = _canonical_activation(name)
                if activation:
                    activations[activation] += 1
                if _is_linear_function(name):
                    linear_layers += 1
            elif node.op == "call_method":
                method_calls += 1
                name = str(node.target)
                function_types[name] += 1
                activation = _canonical_activation(name)
                if activation:
                    activations[activation] += 1

    total_operations = module_layers + function_calls + method_calls
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return ModelReport(
        model_name=type(model).__name__,
        graph_source=graph_source,
        graph_count=len(graphs),
        total_operations=total_operations,
        total_layers=module_layers,
        module_layers=module_layers,
        function_calls=function_calls,
        method_calls=method_calls,
        linear_layers=linear_layers,
        activation_functions=dict(sorted(activations.items())),
        layer_types=dict(sorted(layer_types.items())),
        function_types=dict(sorted(function_types.items())),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )
