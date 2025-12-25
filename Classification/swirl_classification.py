import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy

from Helpers.helpers import plot_decision_boundary

# Setup
device = "mps"
torch.set_default_device(device)

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)

SAVE_DIRECTORY = "models/multi_classification_model"

# Make the dataset
N = 100  # number of points per class
D = 2  # dimensionality
NUM_ClASSES = 5  # number of classes
X = np.zeros((N * NUM_ClASSES, D))  # data matrix (each row = single example)
y = np.zeros(N * NUM_ClASSES, dtype="uint8")  # class labels)
for j in range(NUM_ClASSES):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    # np.c_[] = np.column_stack() but with slice syntax
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

c_map = "copper"
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=mpl.colormaps[c_map])
if not os.path.exists(SAVE_DIRECTORY):
    os.mkdir(SAVE_DIRECTORY)
plt.savefig(f"{SAVE_DIRECTORY}/swirls.png")

# Shuffling the data points
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

train_split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = (
    X[:train_split],
    X[train_split:],
    y[:train_split],
    y[train_split:],
)

model = nn.Sequential(
    nn.Linear(2, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, NUM_ClASSES),
)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
accuracy_function = Accuracy(task="multiclass", num_classes=NUM_ClASSES)

epochs = 500
epoch_count = []
losses = {"train": [], "test": []}
accuracies = {"train": [], "test": []}

for epoch in range(epochs):
    model.train()
    y_logits = model(X_train)

    loss = loss_function(y_logits, y_train)
    accuracy = accuracy_function(y_logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test)

        test_loss = loss_function(y_test_logits, y_test)
        test_accuracy = accuracy_function(y_test_logits, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        losses["train"].append(loss.detach().cpu().numpy())
        losses["test"].append(test_loss.detach().cpu().numpy())
        accuracies["train"].append(accuracy.detach().cpu().numpy())
        accuracies["test"].append(test_accuracy.detach().cpu().numpy())

    print(f"""Epoch: {epoch + 1} | Train Loss: {round(loss.item(), 4)} | Test Loss: {round(test_loss.item(), 4)}
             Train Accuracy: {round(accuracy.item(), 4)} | Test Accuracy: {round(test_accuracy.item(), 4)}""")

# Clear figure
plt.clf()
plt.plot(epoch_count, losses["train"], label="Training Loss")
plt.plot(epoch_count, losses["test"], label="Training Loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(f"{SAVE_DIRECTORY}/loss.png")

plt.clf()
plt.plot(epoch_count, accuracies["train"], label="Model Training Accuracy")
plt.plot(epoch_count, accuracies["test"], label="Model Testing Accuracy")
plt.title("Model Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(f"{SAVE_DIRECTORY}/accuracy.png")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(
    model,
    X_train,  # pyright: ignore[reportArgumentType]
    y_train,  # pyright: ignore[reportArgumentType]
    f"{SAVE_DIRECTORY}/decision_boundary.png",
    c_map=c_map,
)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(
    model, X_test, y_test, f"{SAVE_DIRECTORY}/decision_boundary.png", c_map=c_map
)  # pyright: ignore[reportArgumentType]
