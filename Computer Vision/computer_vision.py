import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets

train_dataset = datasets.ImageFolder(root="Computer Vision/data/fer2013_images/train")
test_dataset = datasets.ImageFolder(root="Computer Vision/data/fer2013_images/test")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=True, num_workers=4
)
