import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int


def get_rotation_matrices_old(
    max_seq_len: int, d_k: int, theta: int
) -> Float[torch.Tensor, "max_seq_len d_k_2 2 2"]:
    positions = torch.arange(1, max_seq_len + 1).unsqueeze(-1)
    k_s = torch.arange(1, (d_k / 2) + 1)
    exponent = ((2 * k_s) - 1) / d_k
    denominator = torch.pow(theta, exponent)
    angles = (positions / denominator).unsqueeze(-1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    concat = torch.concat(
        (cos, -sin, sin, cos),
        dim=-1,
    )
    return rearrange(concat, "... (b1 b2) -> ... b1 b2", b1=2)


def get_rotation_matrices(
    max_seq_len: int, d_k: int, theta: int
) -> Float[torch.Tensor, "max_seq_len d_k_2 2 2"]:
    positions = torch.arange(0, max_seq_len).unsqueeze(-1)
    k_s = torch.arange(0, (d_k / 2))
    exponent = (2 * k_s) / d_k
    denominator = torch.pow(theta, exponent)
    angles = (positions / denominator).unsqueeze(-1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    concat = torch.concat(
        (cos, -sin, sin, cos),
        dim=-1,
    )
    return rearrange(concat, "... (b1 b2) -> ... b1 b2", b1=2)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        rot_matrices = get_rotation_matrices(max_seq_len, d_k, theta).to(
            device
        )
        self.register_buffer("rot_matrices", rot_matrices, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ):
        x = rearrange(
            x, "... seq_len (d_k_2 dim) -> ... seq_len d_k_2 dim ", dim=2
        )

        *rest, seq_len, d_k = x.shape

        idx = (
            token_positions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*rest, seq_len, d_k)
        )
        reordered = torch.gather(x, dim=1, index=idx)

        rotated = einsum(
            self.rot_matrices,
            reordered,
            "seq_len d_k2 dim1 dim2, ... seq_len d_k2 dim2 -> ... seq_len d_k2 dim1",
        )

        return rearrange(
            rotated, "b seq_len d_k2 dim1 -> b seq_len (d_k2 dim1)"
        )


if __name__ == "__main__":
    b, seq_len, d_k = 4, 12, 64
    x = torch.rand(b, seq_len, d_k)
    rope_layer = RotaryPositionEmbedding(
        theta=0.1, d_k=d_k, max_seq_len=seq_len
    )

    output = rope_layer.forward_2(
        x,
        torch.randint(0, seq_len, size=(seq_len,)),
    )

    print(output.shape)
