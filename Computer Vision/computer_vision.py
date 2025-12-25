import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FER2013(root="Computer Vision/data", split="train")
test_data = datasets.FER2013(root="Computer Vision/data", split="test")
