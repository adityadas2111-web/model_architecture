from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelReport:
    model_name: str
    graph_source: str
    graph_count: int

    total_operations: int
    module_layers: int
    function_calls: int
    method_calls: int
    linear_operations: int

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
            f"{name}: {count}"
            for name, count in self.activation_functions.items()
        ) or "None detected"

        layers = ", ".join(
            f"{name}: {count}"
            for name, count in self.layer_types.items()
        ) or "None"

        functions = ", ".join(
            f"{name}: {count}"
            for name, count in self.function_types.items()
        ) or "None"

        lines = [
            f"Model: {self.model_name}",
            f"Graph source: {self.graph_source}",
            f"Captured graphs: {self.graph_count}",
            "",
            f"Total graph operations: {self.total_operations}",
            f"Module layers: {self.module_layers}",
            f"Function calls: {self.function_calls}",
            f"Method calls: {self.method_calls}",
            f"Linear operations: {self.linear_operations}",
            "",
            (
                f"Activation functions ({self.total_activations}): "
                f"{activations}"
            ),
            f"Layer types: {layers}",
            f"Function types: {functions}",
            "",
            f"Total parameters: {self.total_parameters:,}",
            f"Trainable parameters: {self.trainable_parameters:,}",
        ]

        return "\n".join(lines)