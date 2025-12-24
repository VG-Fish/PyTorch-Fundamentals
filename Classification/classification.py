import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from Helpers.helpers import plot_decision_boundary

# Set up
torch.set_default_device("mps")

SAMPLE_COUNT = 1000
RANDOM_SEED = 1290
SAVE_DIRECTORY = "models/classification_model"

# Dataset creation and visualization
X, y = make_moons(SAMPLE_COUNT, noise=0.05, random_state=RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

# print(plt.colormaps())
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=mpl.colormaps["RdGy_r"])  # pyright: ignore[reportArgumentType, reportCallIssue]
if not os.path.exists(SAVE_DIRECTORY):
    os.mkdir(SAVE_DIRECTORY)
plt.savefig(f"{SAVE_DIRECTORY}/moons.png")

# Type of torch.float32
X = torch.from_numpy(X).type(torch.float).to("mps")
y = torch.from_numpy(y).type(torch.float).to("mps")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=RANDOM_SEED
)


# Model creation
class MoonClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We have 5 hidden neurons; this number is chosen by us.
        # Since the dataset is small, five neurons should be enough.
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.activation_function = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.activation_function(self.layer_1(X)))


model = nn.Sequential(
    nn.Linear(in_features=2, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1),
)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


# Accuracy Evaluation:
# While a loss function measures how wrong a model, an evaluation metric how right a model is.
# Evaluation metrics offer different perspectives about model performance.
def accuracy_function(y_predictions, y_true):
    correct = torch.eq(y_predictions, y_true).sum().item()
    return correct / len(y_predictions) * 100


epoch_count = []
accuracies = []
testing_accuracies = []
training_losses = []
testing_losses = []
epochs = 500

for epoch in range(epochs):
    model.train()
    # y_predictions are called logits (raw outputs of our model)
    y_logits = model(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))

    loss = loss_function(y_logits, y_train)
    accuracy = accuracy_function(y_predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_function(test_logits, y_test)
        test_accuracy = accuracy_function(test_predictions, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        accuracies.append(accuracy)
        testing_accuracies.append(test_accuracy)
        training_losses.append(loss.detach().cpu().numpy())
        testing_losses.append(test_loss.detach().cpu().numpy())

    print(
        f"""Epoch: {epoch}: Training Loss: {round(loss.item(), 5)} | Testing Loss: {round(test_loss.item(), 5)}
            Training Accuracy: {accuracy} | Testing Accuracy: {test_accuracy}"""
    )

# Clear figure
plt.clf()
plt.plot(epoch_count, training_losses, label="Training Loss")
plt.plot(epoch_count, testing_losses, label="Training Loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(f"{SAVE_DIRECTORY}/loss.png")

plt.clf()
plt.plot(epoch_count, accuracies, label="Model Training Accuracy")
plt.plot(epoch_count, testing_accuracies, label="Model Testing Accuracy")
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
)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)  # pyright: ignore[reportArgumentType]
