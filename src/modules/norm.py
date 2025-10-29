import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self, x: Float[torch.Tensor, "b seq_len d_out"]
    ) -> Float[torch.Tensor, "b seq_len d_out"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms: Float[torch.Tensor, "b seq_len"] = torch.sqrt(
            (einsum(torch.pow(x, 2), "... d_model -> ... ") / self.d_model)
            + self.eps
        )
        norm: Float[torch.Tensor, "b seq_len d_out"] = x / rms.unsqueeze(-1)

        return einsum(norm, self.g, "... d_model, d_model -> ... d_model").to(
            in_dtype
        )


if __name__ == "__main__":
    d_model = 512

    layer_norm = RMSNorm(d_model)

    x = torch.Tensor(32, 100, 512)

    out = layer_norm(x)

    print(out.shape)
