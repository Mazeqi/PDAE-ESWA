import torch
from torch import nn
import entmax


__all__ = [
    "Entmax15",
    "Sparsemax",
    "Entmoid15",

]
Entmax15  = entmax.Entmax15
Sparsemax = entmax.Sparsemax


class Entmoid15(Entmax15):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.einsum   = ""
        self.is_check = True
        self.param    = None
        self.dtype    = dtype
        self.forward_output = lambda x: x
    def forward(self, input: torch.Tensor):
        if self.is_check:
            base_str    = list("abcdefghijk")
            n_dim       = len(input.shape) + 1
            assert n_dim >= 2 and n_dim < 6
            if   n_dim == 2: self.forward_output = lambda x: x[:, 0]
            elif n_dim == 3: self.forward_output = lambda x: x[:, :, 0]
            elif n_dim == 4: self.forward_output = lambda x: x[:, :, :, 0]
            elif n_dim == 5: self.forward_output = lambda x: x[:, :, :, :, 0]
            self.einsum = "".join(base_str[:n_dim]) + "," + "".join(base_str[n_dim-1:n_dim+1]) + "->" + "".join(base_str[:n_dim-1]) + base_str[n_dim]
            self.param  = torch.Tensor([[1,0]]).to(input.device).to(self.dtype)
            self.is_check = False
        output = input.reshape(-1, *input.shape[1:], 1)
        output = torch.einsum(self.einsum, output, self.param)
        output = super().forward(output)
        return self.forward_output(output)