import torch.nn as nn
from rnn_model import model

activation_types = (
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.GELU,
    nn.Softmax,
    nn.LeakyReLU,
    nn.ELU,
    nn.SiLU,
    nn.Mish
)

activation_counts = {}

for name, layer in model.named_modules():
    if isinstance(layer, activation_types):
        layer_name = layer.__class__.__name__
        activation_counts[layer_name] = activation_counts.get(layer_name, 0) + 1

print("Model Architecture:")
print(model)

print("\nActivation Functions:")

for name, count in activation_counts.items():
    print(f"{name}: {count}")

print("\nTotal:", sum(activation_counts.values())) #compile graph of the model and then from fx graph try to trace,input and output of each function number of layers metadata it consume the trial and a model should be give to script as input  