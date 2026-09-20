from typing import Any, Callable, Mapping, Sequence

import torch
from torch import fx, nn


def get_target_name(target: Any) -> str:
    """
    Convert an FX target into a readable name.
    """

    if isinstance(target, str):
        return target

    name = getattr(target, "__name__", None)

    if name:
        return name

    return str(target)


def is_linear_operation(name: str) -> bool:
    """
    Detect a linear operation.

    torch.compile may convert nn.Linear into lower-level
    operations such as linear or addmm.
    """

    normalized = name.lower()

    return (
        "linear" in normalized
        or "addmm" in normalized
    )


def capture_compiled_graphs(
    model: nn.Module,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
    fullgraph: bool = False,
) -> list[fx.GraphModule]:
    """
    Run a model using torch.compile and capture the generated FX graphs.
    """

    captured_graphs: list[fx.GraphModule] = []

    def capture_backend(
        graph_module: fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Callable[..., Any]:
        # The example inputs are not needed inside the backend.
        del example_inputs

        captured_graphs.append(graph_module)

        # Return the graph's forward function so execution continues.
        return graph_module.forward

    compiled_model = torch.compile(
        model,
        backend=capture_backend,
        fullgraph=fullgraph,
    )

    with torch.no_grad():
        compiled_model(
            *example_args,
            **dict(example_kwargs),
        )

    if not captured_graphs:
        raise RuntimeError(
            "torch.compile did not produce an FX graph."
        )

    return captured_graphs


def symbolic_trace_model(
    model: nn.Module,
) -> list[fx.GraphModule]:
    """
    Trace a model using regular torch.fx symbolic tracing.
    """

    try:
        graph_module = fx.symbolic_trace(model)

    except Exception as error:
        raise RuntimeError(
            "FX symbolic tracing failed. Try providing example "
            "inputs and using compiled graph capture."
        ) from error

    return [graph_module]