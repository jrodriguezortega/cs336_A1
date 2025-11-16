import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

from src.modules.attention import softmax


def perplexity(
    ce_losses: Float[Tensor, "... seq_len"],
) -> Float[Tensor, "..."]:
    return torch.exp(ce_losses.mean(dim=-1))


def cross_entropy_loss(
    logits: Float[Tensor, "... seq_len vocab_size"],
    targets: Float[Tensor, "... seq_len"],
) -> Float[Tensor, "... seq_len"]:
    """Compute the cross entropy loss with respect to
    the logits outputed by the network.

    Returns:
        Float[Tensor, '... seq_len']: CEL averaged along the batch.
    """
    max_logits = logits.max(-1, keepdim=True)[0]
    norm_logits = logits - max_logits
    predicted = torch.gather(norm_logits, -1, targets.unsqueeze(-1)).squeeze(
        -1
    )

    sum_logits = torch.exp(norm_logits).sum(-1)

    ce_loss = -(predicted - torch.log(sum_logits))

    return ce_loss.mean(0)
