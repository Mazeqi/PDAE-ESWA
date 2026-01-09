import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


__all__ = [
    "DropBatch",
    "PatchEmbed",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
]


class DropBatch(_DropoutNd):
    """
    Usage::
    >>> DropBatch(p=0.5)(torch.rand(4,2,2,5))
    tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],

            [[[1.6440, 0.7861, 1.2762, 1.5844, 0.7000],
            [1.6535, 0.9399, 1.5625, 1.0104, 0.9968]],
            [[1.7697, 1.1133, 1.1082, 0.4414, 0.2374],
            [0.9488, 1.8923, 0.1295, 0.7643, 1.9394]]],

            [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],

            [[[1.3391, 1.9098, 0.9932, 0.4776, 1.2583],
            [1.9585, 1.9327, 1.8663, 0.5800, 1.2611]],
            [[1.2127, 0.1276, 0.2146, 0.2466, 0.5820],
            [0.1971, 1.6241, 0.7061, 1.2834, 1.7624]]]])
    """
    def forward(self, input: torch.Tensor):
        if not self.training or self.p == 0.:
            return input
        keep_prob = 1 - self.p
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
        random_tensor = torch.floor(random_tensor)  # binarize
        output = input.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """
    Usage::
        >>> PatchEmbed()(torch.rand(2,3,224,224)).shape
        torch.Size([2, 196, 768])
    Comment::
        The following figure shows the sequence in the order of the patch.
        -------
        |1|2|3|
        -------
        |4|5|6|
        -------
    """
    def __init__(self, patch_size=16, in_channels=3, out_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, input: torch.Tensor):
        B, C, H, W = input.shape
        output = self.proj(input).flatten(2).transpose(1, 2)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, *dim, init_eval: str=None, requires_grad: bool=True, add_type: str=""):
        assert isinstance(init_eval, str)
        assert isinstance(add_type, str) and add_type in ["", "drop", "interpolate"]
        super().__init__()
        param = torch.empty(*dim)
        param = eval(init_eval, {"torch": torch, "param": param})
        self.pe = nn.Parameter(param, requires_grad=requires_grad)
        self.add_type = add_type
        self.is_check = True
        self.pe_proc  = lambda batch, shape: self.pe
        if   add_type == "drop":        self.pe_proc = self.pe_drop
        elif add_type == "interpolate": self.pe_proc = self.pe_interpolate
    def forward(self, input: torch.Tensor):
        """
        Params::
            input: [Batch, Sequence, Dim]
        """
        batch, *shape = input.shape
        if self.is_check:
            assert len(shape) == len(self.pe.shape)
            if   self.add_type == "drop":
                assert sum([shape[i] > x for i, x in enumerate(self.pe.shape)]) == 0
            elif self.add_type == "interpolate":
                assert len(input.shape) == 4
            self.is_check = False
        if shape == list(self.pe.shape):
            return input + self.pe.expand(batch, *self.pe.shape)
        pe = self.pe_proc(batch, shape)
        return input + pe.expand(batch, *pe.shape)
    def pe_drop(self, batch, shape):
        pe = self.pe[[slice(0,x) for x in shape]]
        return pe
    def pe_interpolate(self, batch, shape):
        n_dim,    patch_w,    patch_h    = shape
        n_dim_pe, patch_pe_w, patch_pe_h = self.pe.shape
        pe = torch.nn.functional.interpolate(
            self.pe.unsqueeze(0), scale_factor=(patch_w / patch_pe_w, patch_h / patch_pe_h),
            mode='bicubic', align_corners=False, recompute_scale_factor=False
        )
        return pe[0]


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int=8, qkv_bias: bool=False, qk_scale: float=None,
        p_drop_attn: float=0., p_drop_proj: float=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p_drop_attn)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p_drop_proj)
    def forward(self, input: torch.Tensor):
        B, S, D = input.shape
        qkv = self.qkv(input).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B, S, D)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output, attn
