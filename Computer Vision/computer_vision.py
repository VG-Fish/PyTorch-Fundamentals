from typing import Dict

import mlxtend
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics.metric import Metric
from torchvision import datasets
from tqdm.auto import tqdm

device = "mps" if torch.mps.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
g = torch.Generator().manual_seed(RANDOM_SEED)
LABELS = LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def eval_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    metric_fns: Dict[str, Metric],
):
    total_loss: float = 0.0
    for metric_fn in metric_fns.values():
        metric_fn.reset()
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            total_loss += loss_fn(y_pred, y).item()
            for metric_fn in metric_fns.values():
                metric_fn.update(y_pred, y)

    metrics: Dict[str, float] = {}
    for name, metric_fn in metric_fns.items():
        metrics[name] = metric_fn.compute().item()

    return total_loss / len(loader), metrics


def train_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    train_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(loader)


def main():
    transform = T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root="Computer Vision/data/fer2013_images/train",
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        root="Computer Vision/data/fer2013_images/test",
        transform=transform,
    )

    BATCH_SIZE = 128
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        generator=g,
    )

    # Only one channel as our input is in grayscale
    input_shape = 1
    output_shape = len(LABELS)
    hidden_units = 128
    # TinyVGG CNN Architecture
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(
            input_shape,
            # hidden_units = # of kernels to learn
            hidden_units,
            # kernel_size can also be a tuple
            3,
            stride=1,
            # padding = kernel_size // 2 for odd numbered kernel sizes
            # This choice keeps input and output sizes equal
            padding="same",
        ),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        # This layers normalizes the distribution
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            # Usually stride = kernel_size, so we can omit stride
            stride=2,
        ),
        # Block 2
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 4
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        nn.AvgPool2d(2),
        # Block 5
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding="same"),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.ReLU(),
        # Classifier
        nn.Flatten(),
        # Randomly zeros out some neurons to prevent overfittig
        nn.Dropout(p=0.25),
        # In X ** 2, X = 48 * 2^(-num_max_pool_2d)
        nn.Linear(hidden_units * 3**2, hidden_units),
        nn.Linear(hidden_units, output_shape),
    ).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    # lr is usually 1e-3 or 1e-4 for CNN image models
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    f1_score_fn = F1Score(
        task="multiclass", num_classes=len(LABELS), average="weighted"
    ).to(device)
    accuracy_fn = Accuracy(task="multiclass", num_classes=len(LABELS)).to(device)

    epochs = 100
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.3f}\n")

        print("Getting loss and metrics...")
        loss, metrics = eval_model(
            model,
            test_loader,
            loss_fn,
            {"F1Score": f1_score_fn, "Accuracy": accuracy_fn},
        )
        print(f"Loss: {loss:.03f}", end="  ")
        for name, metric in metrics.items():
            print(f"{name}: {metric * 100:.03f}%", end="  ")
        print()

    print("Getting test loss and metrics...")
    loss, metrics = eval_model(
        model, test_loader, loss_fn, {"F1Score": f1_score_fn, "Accuracy": accuracy_fn}
    )
    print(f"Test Loss: {loss:.03f}", end="  ")
    for name, metric in metrics.items():
        print(f"{name}: {metric * 100:.03f}%", end="  ")
    print()


if __name__ == "__main__":
    main()
