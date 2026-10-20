import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if arr.ndim not in (1, 2):
        raise ValueError("Input tensor must be 1D or 2D.")

    if norm_type == 'l1':
        result = torch.sum(torch.abs(arr))

    elif norm_type == 'l2':
        result = torch.linalg.vector_norm(arr.reshape(-1), ord=2)

    elif norm_type == 'linf':
        if arr.numel() == 0:
            raise ValueError("Cannot compute the infinity norm of an empty tensor.")
        result = torch.max(torch.abs(arr))

    elif norm_type == 'frobenius':
        if arr.ndim != 2:
            raise ValueError("Frobenius norm requires a 2D tensor.")
        result = torch.linalg.matrix_norm(arr, ord='fro')

    else:
        raise ValueError(
            "Invalid norm_type. Choose 'l1', 'l2', 'linf', or 'frobenius'."
        )

    return result.item()

