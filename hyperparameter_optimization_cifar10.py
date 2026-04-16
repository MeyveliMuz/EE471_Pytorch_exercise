import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools
import time

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

training_data = datasets.CIFAR10(root="data", train=True,  download=True, transform=transform)
test_data     = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden_size),
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
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()
    return 100 * correct / size, test_loss / num_batches


# 32 combinations
param_grid = {
    "learning_rate": [1e-2, 5e-3, 1e-3, 1e-4],
    "batch_size":    [32, 64, 128, 256],
    "hidden_size":   [512],
    "optimizer":     ["Adam", "SGD"],
    "epochs":        [10],
}

keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
print(f"Total combinations: {len(combos)}\n")

best_accuracy = 0
best_params   = None

for i, combo in enumerate(tqdm(combos, desc="Hyperparameter Search", unit="combo")):
    params   = dict(zip(keys, combo))
    lr, bs   = params["learning_rate"], params["batch_size"]
    hs       = params["hidden_size"]
    opt_name = params["optimizer"]
    epochs   = params["epochs"]

    train_dl = DataLoader(training_data, batch_size=bs, shuffle=True)
    test_dl  = DataLoader(test_data,     batch_size=bs)

    model    = NeuralNetwork(hs).to(device)
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt_name == "Adam" \
                else torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()
    for epoch in tqdm(range(epochs), desc=f"  [{i+1}/{len(combos)}] LR={lr} BS={bs} {opt_name}", leave=False):
        train(train_dl, model, loss_fn, optimizer)
    elapsed = time.time() - start

    accuracy, loss = test(test_dl, model, loss_fn)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params   = params
        torch.save(model.state_dict(), "model_cifar10_optimized.pth")
        tqdm.write(f"  ★ NEW BEST: Acc={accuracy:.1f}% | LR={lr} BS={bs} {opt_name} ({elapsed:.1f}s)")
    else:
        tqdm.write(f"    Acc={accuracy:.1f}% | LR={lr} BS={bs} {opt_name} ({elapsed:.1f}s)")

print(f"\n{'='*60}")
print(f"Best accuracy: {best_accuracy:.1f}%")
print(f"Best parameters: {best_params}")
print(f"Saved to model_cifar10_optimized.pth")
