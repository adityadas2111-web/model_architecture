"""Small demonstration of symbolic and compiled FX analysis."""

import torch
from torch import nn
from torch.nn import functional as F

from fx_model_analyzer import analyze_model


class DemoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(16, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_layer(inputs)
        inputs = self.relu(inputs)       # module activation
        inputs = F.gelu(inputs)          # functional activation
        return self.output_layer(inputs)


model = DemoModel().eval()
sample = torch.randn(4, 8)

print("SYMBOLIC FX REPORT")
print(analyze_model(model))

print("\nCOMPILED FX REPORT")
print(analyze_model(model, (sample,), use_compile=True))
