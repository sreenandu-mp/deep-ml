import torch

def empirical_pmf(samples: torch.Tensor) -> list:
    """
    Given a 1D tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    if samples.numel() == 0:
        return []
    
    values, counts = torch.unique(samples, return_counts = True, sorted = True)

    probabilities = counts/samples.numel()

    return list(zip(values.tolist(), probabilities.tolist()))