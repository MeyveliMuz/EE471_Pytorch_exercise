import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import itertools
import time

# Download datasets
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return 100 * correct, test_loss


# Hyperparameter search space (32 combinations)
param_grid = {
    "learning_rate": [1e-2, 5e-3, 1e-3, 1e-4],
    "batch_size": [32, 64, 128, 256],
    "hidden_size": [512],
    "optimizer": ["Adam", "SGD"],
    "epochs": [10],
}

keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))
total_combos = len(combinations)

print(f"Total combinations: {total_combos}\n")

best_accuracy = 0
best_params = None
results = []

for i, combo in enumerate(tqdm(combinations, desc="Hyperparameter Search", unit="combo")):
    params = dict(zip(keys, combo))
    lr = params["learning_rate"]
    bs = params["batch_size"]
    hs = params["hidden_size"]
    opt_name = params["optimizer"]
    epochs = params["epochs"]

    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)

    model = NeuralNetwork(hs).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()
    for epoch in tqdm(range(epochs), desc=f"  [{i+1}/{total_combos}] LR={lr} BS={bs} {opt_name}", leave=False):
        train(train_dataloader, model, loss_fn, optimizer)
    elapsed = time.time() - start

    accuracy, loss = test(test_dataloader, model, loss_fn)
    results.append((params, accuracy, loss))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        torch.save(model.state_dict(), "model_optimized.pth")
        tqdm.write(f"  ★ NEW BEST: Acc={accuracy:.1f}% | LR={lr} BS={bs} {opt_name} ({elapsed:.1f}s)")
    else:
        tqdm.write(f"    Acc={accuracy:.1f}% | LR={lr} BS={bs} {opt_name} ({elapsed:.1f}s)")

print(f"\n{'='*60}")
print(f"Best accuracy: {best_accuracy:.1f}%")
print(f"Best parameters: {best_params}")
print(f"Optimized model saved to model_optimized.pth")
