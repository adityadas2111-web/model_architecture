import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.relu(x[:, -1, :])
        x = self.fc(x)
        return x

model = SimpleRNN()

print(model)