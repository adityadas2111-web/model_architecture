import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(10)

sentences = [
    "i love this movie",
    "this movie is amazing",
    "i hate this movie",
    "this movie is terrible"
]

y = torch.tensor([1, 1, 0, 0])

vocab = {}

for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

x = []

for sentence in sentences:
    words = sentence.split()
    numbers = []

    for word in words:
        numbers.append(vocab[word])

    x.append(numbers)

x = torch.tensor(x)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), 8)
        self.rnn = nn.RNN(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, h = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x


model1 = RNN()

model2 = RNN()

model2.load_state_dict(model1.state_dict())

torch._dynamo.config.allow_rnn = True

compiled_model = torch.compile(model2, backend="eager")

graph_model, guards = torch._dynamo.export(model2)(x)

print("Graph:")
print(graph_model.graph)

print("Graph nodes:")

for node in graph_model.graph.nodes:
    print(node.op, "->", node.target)


loss_function = nn.CrossEntropyLoss()

optimizer1 = optim.Adam(model1.parameters(), lr=0.01)

optimizer2 = optim.Adam(compiled_model.parameters(), lr=0.01)

loss1 = []

loss2 = []


for i in range(100):

    optimizer1.zero_grad()

    output1 = model1(x)

    l1 = loss_function(output1, y)

    l1.backward()

    optimizer1.step()

    loss1.append(l1.item())


    optimizer2.zero_grad()

    output2 = compiled_model(x)

    l2 = loss_function(output2, y)

    l2.backward()

    optimizer2.step()

    loss2.append(l2.item())


print("Vocabulary:")
print(vocab)

print("Normal RNN output:")
print(model1(x))

print("Compiled RNN output:")
print(compiled_model(x))

print("Normal final loss:", loss1[-1])

print("Compiled final loss:", loss2[-1])

print("Outputs are close:")
print(torch.allclose(model1(x), compiled_model(x), atol=1e-5))


plt.plot(loss1, label="Normal RNN")

plt.plot(loss2, label="Compiled RNN")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Normal RNN vs PyTorch Compile")

plt.legend()

plt.show()