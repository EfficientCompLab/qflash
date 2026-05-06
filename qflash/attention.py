from typing import Dict, Optional

import torch
from torch import nn

from timm.layers import to_2tuple, trunc_normal_
from timm.models.swin_transformer import get_relative_position_index

from .kernel import qflash


class QAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = qflash.forward(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

    @classmethod
    def from_orig(cls, src: nn.Module) -> "QAttention":
        new = cls(
            dim=src.qkv.in_features,
            num_heads=src.num_heads,
            qkv_bias=src.qkv.bias is not None,
        ).to(device=src.qkv.weight.device, dtype=src.qkv.weight.dtype)
        new.load_state_dict(src.state_dict())
        return new


class QWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.relative_position_bias_table = nn.Parameter(
            torch.empty((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(win_h, win_w),
            persistent=False,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_area, self.window_area, -1).permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_mask = self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
            attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)

        x = qflash.forward(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B_, N, -1))

    @classmethod
    def from_orig(cls, src: nn.Module) -> "QWindowAttention":
        new = cls(
            dim=src.dim,
            num_heads=src.num_heads,
            window_size=src.window_size,
            qkv_bias=src.qkv.bias is not None,
        ).to(device=src.qkv.weight.device, dtype=src.qkv.weight.dtype)
        new.load_state_dict(src.state_dict())
        return new


def patch_attention(model: nn.Module) -> Dict[str, int]:
    import timm.layers.attention as _ta
    import timm.models.swin_transformer as _ts

    targets = []
    for name, module in model.named_modules():
        if isinstance(module, _ta.Attention):
            targets.append((name, "attn"))
        elif isinstance(module, _ts.WindowAttention):
            targets.append((name, "wattn"))

    counts = {"attn": 0, "wattn": 0}
    for name, kind in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        src = getattr(parent, child_name)
        new = (QAttention if kind == "attn" else QWindowAttention).from_orig(src)
        setattr(parent, child_name, new)
        counts[kind] += 1
    return counts
