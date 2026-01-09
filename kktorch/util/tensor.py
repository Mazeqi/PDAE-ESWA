import torch


__all__ = [
    "create_allpattern",
]


def create_allpattern(dim: int, dtype=torch.float32) -> torch.Tensor:
    """
    Usage::
    >>> create_allpattern(2)
    tensor([[[0., 1.],
            [1., 0.],
            [0., 1.],
            [1., 0.]],

            [[0., 1.],
            [0., 1.],
            [1., 0.],
            [1., 0.]]])
    >>> create_allpattern(2).shape
    torch.Size([2, 4, 2])
    """
    indices = torch.arange(2 ** dim)
    offsets = 2 ** torch.arange(dim)
    bin_codes = (torch.div(indices.view(1, -1), offsets.view(-1, 1), rounding_mode='trunc') % 2).to(dtype)
    #bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
    bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
    return bin_codes_1hot
