import rootutils
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float

if __name__ == "__main__":
    import rootutils

    rootutils.setup_root(__file__, pythonpath=True)


from src.modules import Linear


def silu(
    x: Float[torch.Tensor, "b seq_len d_model"],
) -> Float[torch.Tensor, "b seq_len d_model"]:
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            self.d_ff = round(((8 / 3) * self.d_model) / 64) * 64
        else:
            self.d_ff = d_ff

        if d_ff % 64 != 0:
            print(f"WARNING: d_ff ({self.d_ff}) is not divisible by 64")

        self.l_1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.l_3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.l_2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "b seq_len d_model"],
    ) -> Float[torch.Tensor, "b seq_len d_model"]:
        activation = silu(self.l_1(x))
        inner_gated = einsum(
            activation,
            self.l_3(x),
            "b seq_len d_model, b seq_len d_model -> b seq_len d_model",
        )

        return self.l_2(inner_gated)


if __name__ == "__main__":
    d_model = 512
    x = torch.rand(2, 10, 512)
    ffn_layer = FFN(d_model)

    print(ffn_layer(x).shape)
