import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
    """Class representing the an embedding layer or module.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initilization function of the embedding layer.

        Args:
            num_embedding (int): Number of embeddings. Vocabulary size in the context of LLMs.
            embedding_dim (int): Dimension of each embedding.
            device (torch.device | None, optional): _description_. Defaults to None.
            dtype (torch.dtype | None, optional): _description_. Defaults to None.
        """
        super().__init__()

        W = torch.empty(
            (num_embedding, embedding_dim), device=device, dtype=dtype
        )
        nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(W)

    def forward(
        self, token_ids: Int[Tensor, "b seq_len"]
    ) -> Float[Tensor, "b seq_len embedding_dim"]:
        return self.W[token_ids]


if __name__ == "__main__":
    num_embedding = 1000
    emb_dim = 512
    token_ids = torch.randint(0, num_embedding, (32, 100))

    emb_layer = Embedding(num_embedding, emb_dim)

    print(emb_layer(token_ids).shape)
