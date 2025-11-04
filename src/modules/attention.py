import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

from src.modules.linear import Linear
from src.modules.rope import RotaryPositionEmbedding


def softmax(
    x: Float[torch.Tensor, "..."], dim: int
) -> Float[torch.Tensor, "..."]:
    """Compute the softmax operation across dim.

    Args:
        x (Float[torch.Tensor, "..."]): Tensor to be transformed.
        dim (int): Dimension over which to calculate the softmax function.

    Returns:
        Float[torch.Tensor, "..."]: Softmax normalized tensor.
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values.expand_as(x)
    exp = torch.exp(x)
    return exp / (torch.sum(exp, dim=dim, keepdim=True))


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "b ... seq_len d_k"],
    k: Float[torch.Tensor, "b ... seq_len d_k"],
    v: Float[torch.Tensor, "b ... seq_len d_v"],
    mask: Bool[torch.Tensor, "seq_len seq_len"] | None = None,
) -> Float[torch.Tensor, "b ... seq_len d_v"]:
    """Function to compute the scaled dot product attention.

    Args:
        q (Float[torch.Tensor, &quot;b ... seq_len d_k&quot;]): Query matrix.
        k (Float[torch.Tensor, &quot;b ... seq_len d_k&quot;]): Key matrix.
        v (Float[torch.Tensor, &quot;b ... seq_len d_v&quot;]): Value matrix.
        mask (Bool[torch.Tensor, &quot;seq_len seq_len&quot;] | None, optional): Causal attention mask. Defaults to None.

    Returns:
        Float[torch.Tensor, "b ... seq_len d_v"]: Attention-weigthed values.
    """
    *rest, d_k = q.shape
    presoftmax = einsum(
        q,
        k,
        "b ... seq_len_i d_k, b ... seq_len_j d_k -> b ... seq_len_i seq_len_j",
    )

    presoftmax = presoftmax / pow(d_k, 1 / 2)

    if mask is not None:
        presoftmax = torch.where(mask, presoftmax, -torch.inf)

    att_weights = softmax(presoftmax, dim=-1)

    return einsum(
        att_weights,
        v,
        "b ... seq_len_i seq_len_j, b ... seq_len_j d_v -> b ... seq_len_i d_v",
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) should be divisible by num_heads ({num_heads}): {d_model / num_heads:.2f}"
            )
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Q = Linear(d_model, d_model, device, dtype)
        self.K = Linear(d_model, d_model, device, dtype)
        self.V = Linear(d_model, d_model, device, dtype)
        self.O = Linear(d_model, d_model, device, dtype)
        self.rope = None

        if theta is not None:
            self.rope = RotaryPositionEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ):
        q, k, v = self.Q(x), self.K(x), self.V(x)

        _, seq_len, _ = x.shape

        mask = torch.tril(torch.full(size=(seq_len, seq_len), fill_value=True))

        q = rearrange(
            q,
            "b seq_len (num_heads d_k) -> b num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "b seq_len (num_heads d_k) -> b num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "b seq_len (num_heads d_k) -> b num_heads seq_len d_k",
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        att_output = scaled_dot_product_attention(q, k, v, mask)

        att_output = rearrange(
            att_output, "b num_heads seq_len d_k -> b seq_len (num_heads d_k)"
        )

        return self.O(att_output)


if __name__ == "__main__":
    torch.manual_seed(1234)

    b, seq_len, d_model = 4, 12, 64
    q = torch.rand(size=(b, seq_len, d_model))
    k = torch.rand(size=(b, seq_len, d_model))
    v = torch.rand(size=(b, seq_len, d_model))

    scaled_dot_product_attention(q, k, v)

    num_heads = 4
    x = torch.rand(size=(b, seq_len, d_model))
    token_positions = torch.arange(seq_len).unsqueeze(0)

    mh_layer = MultiHeadAttention(d_model, num_heads)
    mh_out = mh_layer(x)
    print(mh_out.shape)

    mh_layer = MultiHeadAttention(d_model, num_heads, 10000, seq_len)
    mh_out = mh_layer(x, token_positions)
    print(mh_out.shape)
