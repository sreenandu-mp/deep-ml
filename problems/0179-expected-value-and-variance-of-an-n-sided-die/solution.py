import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    if n<1:
        raise ValueError("n must be a positive integer")
    
    outcomes = torch.arange(1, n + 1, dtype=torch.float64)

    probabilities = torch.full((n,), 1.0 / n, dtype=torch.float64)

    expected_value = torch.sum(probabilities * outcomes)

    variance = torch.sum(probabilities * (outcomes - expected_value) ** 2)

    return expected_value.item(), variance.item()