import mlxtend
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics.classification import F1Score
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
    metric_fn: Metric,
):
    total_loss: float = 0.0
    metric_fn.reset()
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            total_loss += loss_fn(y_pred, y).item()
            metric_fn.update(y_pred, y)
    return total_loss / len(loader), metric_fn.compute().item()


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

    BATCH_SIZE = 4096
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
    hidden_units = 32
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
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            hidden_units,
            hidden_units,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=2,
            # Usually stride = kernel_size so we can omit stride
            stride=2,
        ),
        # Block 2
        nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3
        nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Classifier
        nn.Flatten(),
        # 48 -> 24 -> 12 -> 6 after MaxPool2d
        nn.Linear(hidden_units * 6**2, output_shape),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # lr is usually 1e-3 or 1e-4 for CNN image models
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    f1_score_fn = F1Score(task="multiclass", num_classes=len(LABELS)).to(device)

    epochs = 100
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.3f}\n")

    print("Getting test loss and F1Score...")
    loss, f1_score = eval_model(model, test_loader, loss_fn, f1_score_fn)
    print(f"Test Loss: {loss} | F1Score: {f1_score}")


if __name__ == "__main__":
    main()
