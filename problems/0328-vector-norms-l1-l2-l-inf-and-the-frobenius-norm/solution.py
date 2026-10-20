import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    flat_arr = arr.flatten()
    if norm_type == 'l1':
        result = np.sum(np.abs(flat_arr))
    elif norm_type == 'l2':
        result = np.linalg.norm(arr.ravel())
    elif norm_type == 'linf':
        if arr.size == 0:
            raise ValueError("Cannot compute the infinity norm of an empty array.")
        result = np.max(np.abs(arr))
    elif norm_type == 'frobenius':
        if arr.ndim != 2:
            raise ValueError("Frobenius norm requires a 2D array.")
        result = np.linalg.norm(arr, ord='fro')
    return float(result)
