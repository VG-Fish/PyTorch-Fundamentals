import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

# Setup
device = torch.device("mps")
torch.set_default_device(device)

torch.manual_seed(0)
torch.mps.manual_seed(0)

DIRECTORY = "models/pytorch_workflow_model"
if not os.path.exists(DIRECTORY):
    os.mkdir(DIRECTORY)

# Linear Regression Model (y = Ax + B)
weight = A = 0.5
bias = B = 0.1
start, end, step = 0, 1, 0.01

# The input
X = torch.arange(start, end, step).unsqueeze(dim=1)
# The ideal output that the model will learn
y = weight * X + bias

# Splitting data into train and testing sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
) -> None:
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Model Predictions")

    plt.legend(prop={"size": 14})

    plt.savefig(f"{DIRECTORY}/model_predictions.png")


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


model = LinearRegressionModel()
print(model.state_dict())

print(X_test)
print(y_test)

loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Inference mode turns off gradient tracking and makes model predictions faster.
# Same result is also possible with .no_grad(), but .inference_mode() is faster.
# with torch.inference_mode():
#     y_predictions = model(X_test)
#     print(y_predictions)

# Training loop
epochs = 400

epoch_count = []
train_loss_values = []
test_loss_values = []
test_predictions: torch.Tensor = torch.empty(0)

for epoch in range(epochs):
    # Set the model to train mode, which sets all parameters that require gradients to officially require gradients.
    model.train()

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate loss
    loss = loss_function(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Perform gradient descent
    # Optimizer changes will accumulate throughout the loop, so we zero them for the next iteration (step 3)
    optimizer.step()

    # Model testing
    # Turns off gradient tracking and other settings that are not needed
    model.eval()

    with torch.inference_mode():
        # 1. Forward pass
        test_predictions = model(X_test)

        # 2. Calculate loss
        test_loss = loss_function(test_predictions, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().cpu().numpy())
        test_loss_values.append(test_loss.detach().cpu().numpy())

        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

plt.plot(epoch_count, train_loss_values, label="Training Loss")
plt.plot(epoch_count, test_loss_values, label="Testing Loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(f"{DIRECTORY}/loss.png")

print(model.state_dict())
MODEL_PATH = Path(DIRECTORY)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "pytorch_workflow_model.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

print(f"Saving model to: {MODEL_SAVE_PATH}")

# Model visualization
plot_predictions(
    X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), test_predictions.cpu()
)
