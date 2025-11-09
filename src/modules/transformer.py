from typing import OrderedDict, Self

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from src.modules.attention import MultiHeadAttention
from src.modules.embedding import Embedding
from src.modules.ffn import FFN
from src.modules.linear import Linear
from src.modules.norm import RMSNorm
from src.modules.rope import RotaryPositionEmbedding


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RotaryPositionEmbedding | None = None,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            rope=rope,
        )

        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)

        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "b seq_len d_model"],
        token_positions: Int[Tensor, "b seq_len"] | None = None,
    ):
        out = x + self.mha(self.rms_norm1(x), token_positions)
        out = out + self.ffn(self.rms_norm2(out))

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        theta: float = 10000,
        eps: float = 0.00001,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Use the same rope layer for each block
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_heads)
        if theta is not None:
            self.rope = RotaryPositionEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=context_length,
                device=device,
            )

        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embedding=vocab_size, embedding_dim=d_model
        )
        nn.Sequential()
        self.blocks = nn.Sequential(
            *[
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    rope=self.rope,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, eps, device, dtype)

        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        # Rope to be used across all layers
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) should be divisible by num_heads ({num_heads}): {d_model / num_heads:.2f}"
            )

    def forward(
        self, token_ids: Int[Tensor, "b seq_len"]
    ) -> Float[Tensor, "b seq_len vocab_size"]:
        x: Float[Tensor, "b seq_len d_model"] = self.token_embeddings(
            token_ids
        )

        x = self.blocks(x)

        x = self.ln_final(x)

        return self.lm_head(x)


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    max_seq_len = 1024
    theta = 10000.0
    d_ff = 2048

    block = Block(d_model, num_heads, max_seq_len, theta, d_ff)

    x = torch.randn(32, 100, 512)
    out = block(x)

    print(out.shape)
