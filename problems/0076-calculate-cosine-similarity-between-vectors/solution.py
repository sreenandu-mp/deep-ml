import torch
import torch.nn.functional as F

def l2_norm(vec: torch.Tensor):
    return torch.sqrt(torch.sum(vec ** 2))

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    dot_product = torch.dot(v1, v2)
    l2_v1 = l2_norm(v1)
    l2_v2 = l2_norm(v2)

    denominator = l2_v1 * l2_v2
    if denominator.item() == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return (dot_product / denominator).item()

