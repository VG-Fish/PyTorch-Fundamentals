import torch

# Initialization
device = torch.device("mps")
torch.set_default_device(device)

# Creating tensors
scalar = torch.tensor(7)
print(scalar.ndim)
print(scalar.item())

print()

vector = torch.tensor([6, 7])
print(vector.ndim)
print(vector.tolist())

print()

MATRIX = torch.tensor([[6, 7], [6, 9]])
print(MATRIX.ndim)
print(MATRIX.tolist())

print()

TENSOR = torch.tensor([[[1, 2], [3, 6], [6, 12]]])
print(TENSOR.ndim)
print(TENSOR.tolist())

print()

# Creating random tensors
random_tensor = torch.rand(3, 4)
# or do: random_tensor = torch.rand(size=(3, 4))
print(random_tensor.tolist())

# Images are usually represented with tensors of shape: (3, X, Y)

print()

# Creating tensors of all zeros or ones
zeros = torch.zeros(3, 4)
print(zeros.tolist())

ones = torch.ones(3, 4)
print(ones.tolist())

print()

# Creating a range of tensors and tensors-like
range_tensor = torch.arange(0, 2)
print(range_tensor.tolist())
