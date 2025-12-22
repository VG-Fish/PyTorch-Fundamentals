import torch


def matrix_multiplication(
    A: torch.Tensor, B: torch.Tensor, dtype=torch.float32
) -> torch.Tensor:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Either A.ndim or B.ndim != 2")

    if A.shape[1] != B.shape[0]:
        raise ValueError("A.shape[1] != B.shape[0]")

    out = torch.zeros(A.shape[0], B.shape[1])
    for i, row in enumerate(A):
        for j, col in enumerate(B.T):
            out[i][j] = torch.dot(row, col).item()
    return out.type(dtype)


A = torch.tensor([[1, 2], [3, 4]])
print(A)
B = torch.tensor([[5, 6], [7, 8]])
print(B)

C = matrix_multiplication(A, B, dtype=torch.int8)
print(C)
