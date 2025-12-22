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
range_tensor = torch.arange(0, 21, step=2)
print(range_tensor.tolist())

# Will make a tensor that has the same shape as range_tensor but will only be filled with zeros.
zeros_like = torch.zeros_like(range_tensor)
print(zeros_like.tolist())

print()

# Tensor data types
example_tensor = torch.tensor(
    [1],
    dtype=torch.int8,
    requires_grad=False,  # whether gradients operations should be tracked
)
print(example_tensor)

example_float_16_tensor = example_tensor.type(torch.float16)

print()

# Tensor operations
tensor = torch.tensor([1, 2])
print(f"{tensor = }")
print(tensor + 10)
print(tensor * tensor)
print(torch.matmul(tensor, tensor))
print(torch.mm(torch.rand(3, 4), torch.rand(4, 3)))
print(tensor @ tensor)
print()

# Tensor aggregations
a_range = torch.arange(0, 20, dtype=torch.float32)
print(torch.min(a_range))
# Finding the index of where the min value is
print(torch.argmin(a_range))
print(torch.max(a_range), a_range.max())
print(torch.argmax(a_range))
print(torch.mean(a_range))
print(torch.sum(a_range))
print()

# Advanced tensor operations
# Reshaping: reshapes an input tensor to a desired shape (if possible)
# View: returns a view of a tensor of a specific shape but shares the same memory as the input tensor
# Stacking: combining multiple tensors into a stack along a dimension
# Squeezing: removes all dimensions with one element from a tensor
# Unsqueezing: adds a '1' dimension to the input tensor
# Permuting: returns a view of the input tensor but with the dimensions swapped in a specific way

X = torch.arange(1, 21)
print(X)
print(X.reshape(2, 2, 5))

Y = X.view(2, 2, 5)
print(Y)

X[0] = 11
print(f"{X = }\n{Y = }")

Y = Y.reshape_as(X)
print(Y)

print()
print(torch.stack([X, Y], dim=0))
print(torch.vstack([X, Y]))
print(torch.stack([X, Y], dim=1))
print(torch.hstack([X, Y]))
print()

A = torch.rand(1, 2, 1, 2)
print(A)
print(torch.squeeze(A))
print(torch.unsqueeze(torch.rand(2, 2), dim=0))
print()

B = torch.rand(2, 3)
print(B.shape)
print(B.permute(1, 0).shape)
print()

# Tensor indexing
X = torch.rand(3, 1, 3)
print(X)
print(X[:, 0, 2])

# PyTorch & Numpy
# # Default numpy datatype is float64
X = torch.from_numpy(torch.rand(1).cpu().numpy())
print(X)
print(X.numpy())
print()

# Reproducibility
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)

# Setting up scripts
if torch.cuda.is_available():
    device = "cuda"  # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps"  # Use Apple Silicon GPU (if available)
else:
    device = "cpu"  # Default to CPU if no GPU is available
print(torch.mps.device_count())
print()

# Moving tensors between devices
T = torch.rand(1)
print(T)
T = T.to("cpu")
print(T)
print()
