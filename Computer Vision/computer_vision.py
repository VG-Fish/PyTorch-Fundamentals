import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics.classification import Accuracy
from torchmetrics.metric import Metric
from torchvision import datasets

device = "mps"

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
g = torch.Generator().manual_seed(RANDOM_SEED)
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def eval_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Metric,
):
    loss: torch.Tensor = torch.tensor(0.0).to(device)
    accuracy_fn.reset()
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)

            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            accuracy_fn.update(y_pred, y)

        loss /= len(loader)
    return loss.item(), accuracy_fn.compute().item()


def train_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    train_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(loader):
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)

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
            T.PILToTensor(),
            T.Grayscale(num_output_channels=1),
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

    BATCH_SIZE = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        generator=g,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        generator=g,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(48 * 48, 100),
        nn.ReLU(),
        nn.Linear(100, 7),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    accuracy_fn = Accuracy(task="multiclass", num_classes=len(LABELS)).to(device)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.3f}\n")

    print("Getting test loss and accuracy...")
    loss, accuracy = eval_model(model, test_loader, loss_fn, accuracy_fn)
    print(f"Test Loss: {loss} | Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
