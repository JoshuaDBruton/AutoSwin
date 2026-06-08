import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss to address class imbalance.

        Parameters:
            - alpha: Class weights (list, tensor, or None). 
                     If a list, it will be converted to a tensor.
            - gamma: Focusing parameter (default: 2).
            - reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Parameters:
            - inputs: Logits from the model (B, C).
            - targets: Ground truth labels (B).

        Returns:
            - Focal loss (scalar or tensor based on reduction).
        """
        # Convert logits to log probabilities
        inputs = F.log_softmax(inputs, dim=-1)

        # Gather the log probabilities of the target class
        log_pt = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt = log_pt.exp()  # Convert log probabilities to probabilities

        # Apply alpha (class weights)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]  # (B,)
            log_pt = log_pt * at

        # Compute the focal loss
        loss = -((1 - pt) ** self.gamma) * log_pt

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

