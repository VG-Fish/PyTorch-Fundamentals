import os

from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

SAVE_DIRECTORY = "Computer Vision/data/fer2013_images"
LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]
transform = transforms.ToTensor()

for label in LABELS:
    os.makedirs(f"{SAVE_DIRECTORY}/train/{label}", exist_ok=True)
    os.makedirs(f"{SAVE_DIRECTORY}/test/{label}", exist_ok=True)

train_data = datasets.FER2013(root="Computer Vision/data", split="train")
test_data = datasets.FER2013(root="Computer Vision/data", split="test")

for i in tqdm(range(len(train_data)), desc="Converting FER2013_train to ImageFolder"):
    image, label = train_data[i]
    image = transform(image)  # pyright: ignore[reportCallIssue]
    image_path = f"{SAVE_DIRECTORY}/train/{LABELS[label]}/{i:05d}.png"
    save_image(image, image_path)  # pyright: ignore[reportArgumentType]

for i in tqdm(range(len(test_data)), desc="Converting FER2013_test to ImageFolder"):
    image, label = test_data[i]
    image = transform(image)  # pyright: ignore[reportCallIssue]
    image_path = f"{SAVE_DIRECTORY}/test/{LABELS[label]}/{i:05d}.png"
    save_image(image, image_path)  # pyright: ignore[reportArgumentType]
