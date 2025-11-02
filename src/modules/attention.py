import torch
from einops import einsum
from jaxtyping import Float


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
    mask: Float[torch.Tensor, "seq_len seq_len"] | None = None,
) -> Float[torch.Tensor, "b ... seq_len d_v"]:
    """Function to compute the scaled dot product attention.

    Args:
        q (Float[torch.Tensor, &quot;b ... seq_len d_k&quot;]): Query matrix.
        k (Float[torch.Tensor, &quot;b ... seq_len d_k&quot;]): Key matrix.
        v (Float[torch.Tensor, &quot;b ... seq_len d_v&quot;]): Value matrix.
        mask (Float[torch.Tensor, &quot;seq_len seq_len&quot;] | None, optional): Causal attention mask. Defaults to None.

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


if __name__ == "__main__":
    torch.manual_seed(1234)
    q = torch.rand(size=(4, 10, 64))
    k = torch.rand(size=(4, 10, 64))
    v = torch.rand(size=(4, 10, 32))

    scaled_dot_product_attention(q, k, v)
