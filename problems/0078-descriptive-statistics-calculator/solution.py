import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    x = torch.as_tensor(data)

    if x.numel == 0:
        raise ValueError("Dataset cannot be empty")
    
    mean = torch.mean(x)

    median = torch.median(x) if x.numel() % 2 else torch.quantile(x, 0.5)

    values, counts = torch.unique(x, return_counts=True)
    mode = values[torch.argmax(counts)]

    variance = torch.var(x, correction=0)
    standard_deviation = torch.sqrt(variance)

    percentiles = torch.quantile(
        x,
        torch.tensor([0.25, 0.50, 0.75], dtype = x.dtype)
    )
    q25, q50, q75 = percentiles
    iqr = q75 - q25
    
    return {
        "mean": mean.item(),
        "median": median.item(),
        "mode": mode.item(),
        "variance": variance.item(),
        "standard_deviation": standard_deviation.item(),
        "25th_percentile": q25.item(),
        "50th_percentile": q50.item(),
        "75th_percentile": q75.item(),
        "interquartile_range": iqr.item()
    }