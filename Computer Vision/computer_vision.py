import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FER2013(root="data/fer2013")
