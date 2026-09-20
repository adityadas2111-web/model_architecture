from collections import Counter
from typing import Any, Mapping, Sequence

from torch import fx, nn

from activations import (
    ACTIVATION_MODULES,
    identify_activation,
)
from fx_graph import (
    capture_compiled_graphs,
    get_target_name,
    is_linear_operation,
    symbolic_trace_model,
)
from model_report import ModelReport


def analyze_graphs(
    model: nn.Module,
    graphs: list[fx.GraphModule],
    graph_source: str,
) -> ModelReport:
   

    layer_types: Counter[str] = Counter()
    function_types: Counter[str] = Counter()
    activation_counts: Counter[str] = Counter()

    module_layers = 0
    function_calls = 0
    method_calls = 0
    linear_operations = 0

    for graph_module in graphs:
        modules = dict(graph_module.named_modules())

        for node in graph_module.graph.nodes:

        
            if node.op == "call_module":
                module_layers += 1

                module = modules[str(node.target)]
                module_name = type(module).__name__

                layer_types[module_name] += 1

                if isinstance(module, nn.Linear):
                    linear_operations += 1

                if isinstance(module, ACTIVATION_MODULES):
                    activation_counts[module_name] += 1

           
            elif node.op == "call_function":
                function_calls += 1

                function_name = get_target_name(node.target)

                function_types[function_name] += 1

                activation = identify_activation(function_name)

                if activation is not None:
                    activation_counts[activation] += 1

                if is_linear_operation(function_name):
                    linear_operations += 1

           
            elif node.op == "call_method":
                method_calls += 1

                method_name = str(node.target)

                function_types[method_name] += 1

                activation = identify_activation(method_name)

                if activation is not None:
                    activation_counts[activation] += 1

    total_operations = (
        module_layers
        + function_calls
        + method_calls
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return ModelReport(
        model_name=type(model).__name__,
        graph_source=graph_source,
        graph_count=len(graphs),
        total_operations=total_operations,
        module_layers=module_layers,
        function_calls=function_calls,
        method_calls=method_calls,
        linear_operations=linear_operations,
        activation_functions=dict(
            sorted(activation_counts.items())
        ),
        layer_types=dict(
            sorted(layer_types.items())
        ),
        function_types=dict(
            sorted(function_types.items())
        ),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )


def analyze_model(
    model: nn.Module,
    example_args: Sequence[Any] | None = None,
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    use_compile: bool = False,
    fullgraph: bool = False,
) -> ModelReport:


    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module"
        )

    keyword_inputs = example_kwargs or {}

    if use_compile:
        if example_args is None:
            raise ValueError(
                "example_args are required when use_compile=True"
            )

        graphs = capture_compiled_graphs(
            model=model,
            example_args=tuple(example_args),
            example_kwargs=keyword_inputs,
            fullgraph=fullgraph,
        )

        graph_source = "torch.compile FX capture"

    else:
        graphs = symbolic_trace_model(model)

        graph_source = "torch.fx.symbolic_trace"

    return analyze_graphs(
        model=model,
        graphs=graphs,
        graph_source=graph_source,
    )