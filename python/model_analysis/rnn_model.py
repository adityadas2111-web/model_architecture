import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(20, 8)

        self.rnn = nn.RNN(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )

        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(16, 16)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)

        self.gelu = nn.GELU()
        self.fc3 = nn.Linear(16, 16)

        self.leaky_relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(16, 16)

        self.silu = nn.SiLU()
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.embedding(x)

        x, h = self.rnn(x)

        x = x[:, -1, :]

        x = self.tanh(x)
        x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)

        x = self.gelu(x)
        x = self.fc3(x)

        x = self.leaky_relu(x)
        x = self.fc4(x)

        x = self.silu(x)
        x = self.fc5(x)

        return x


model = RNNModel()