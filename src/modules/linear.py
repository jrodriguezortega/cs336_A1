import math
from turtle import forward

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """Class representing a Linear transformation layer or module.

    Args:
        nn.Module: Pytorch nn.Module class.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Initializer parameter W as nn.Parameter object
        W = torch.empty(size=(d_out, d_in), device=device, dtype=dtype)

        linear_std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(
            W, mean=0, std=linear_std, a=-3 * linear_std, b=3 * linear_std
        )

        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method to apply the linear transformation to input "x".

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.
        """

        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


if __name__ == "__main__":
    layer = Linear(32, 64)
    tensor = torch.rand(32)

    out = layer(tensor)

    print(tensor.shape)
    print(out.shape)

    # Try load_state_dict

    weights = torch.rand((64, 32), dtype=torch.float32)

    layer = Linear(32, 64)
    layer.load_state_dict({"W": weights})

    layer(tensor)
