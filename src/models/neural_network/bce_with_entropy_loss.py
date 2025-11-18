"""Custom loss functions for neural network training with entropy regularization."""

import torch
import torch.nn as nn


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p).

    Args:
        probs: Predicted probabilities in [0, 1]
    """
    epsilon = 1e-15
    return -(probs * torch.log(probs + epsilon) + (1 - probs) * torch.log(1 - probs + epsilon))


class BCEWithEntropyLoss(nn.Module):
    """
    Binary cross-entropy loss with entropy penalty to push models to make overconfident predictions.

    Loss: L = BCE(y_pred, y_true) + entropy_weight * mean(H(y_pred))
    where H(p) = -p*log(p) - (1-p)*log(1-p) is the binary entropy.

    The entropy is higher for uncertain predictions (max at 0.5) and lower for confident predictions.
    Hence adding the entropy to the cross entropy makes uncertain predictions have a higher loss.

    Args:
        entropy_weight (float): Weight for entropy penalty. Higher values encourage
            less confident predictions. Default: 1.0
    """

    def __init__(self, entropy_weight: float = 3.5, reduction: str = "mean"):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction)
        self.entropy_weight = entropy_weight
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with entropy penalty.

        Args:
            inputs: Predicted probabilities in [0, 1]
            targets: Ground truth labels (0 or 1)

        Returns:
            Scalar loss value
        """
        if self._reduction == "mean":
            entropy_term = entropy(inputs).mean()
        elif self._reduction == "sum":
            entropy_term = entropy(inputs).sum()
        else:
            entropy_term = entropy(inputs)

        return self.bce(inputs, targets) + self.entropy_weight * entropy_term
