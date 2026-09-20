import unittest

import torch
from torch import nn
from torch.nn import functional as F

from fx_model_analyzer import analyze_model


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(F.gelu(self.relu(self.fc1(x))))


class AnalyzerTests(unittest.TestCase):
    def test_symbolic_graph_counts_modules_and_functions(self):
        report = analyze_model(MixedModel())
        self.assertEqual(report.linear_layers, 2)
        self.assertEqual(report.activation_functions, {"GELU": 1, "ReLU": 1})
        self.assertEqual(report.module_layers, 3)
        self.assertEqual(report.function_calls, 1)
        self.assertEqual(report.total_layers, 3)
        self.assertEqual(report.total_operations, 4)

    def test_dictionary_output_contains_activation_total(self):
        result = analyze_model(MixedModel()).to_dict()
        self.assertEqual(result["total_activations"], 2)

    def test_compile_requires_inputs(self):
        with self.assertRaises(ValueError):
            analyze_model(MixedModel(), use_compile=True)


if __name__ == "__main__":
    unittest.main()
