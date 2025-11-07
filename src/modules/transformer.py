import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from src.modules.attention import MultiHeadAttention
from src.modules.ffn import FFN
from src.modules.norm import RMSNorm


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)

        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "b seq_len d_model"]):
        out = x + self.mha(self.rms_norm1(x))
        out = out + self.ffn(self.rms_norm2(out))

        return out
