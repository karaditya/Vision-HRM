import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class VisionClassificationLossHead(nn.Module):
    """Loss head for vision classification tasks."""
    
    def __init__(self, model: nn.Module, loss_type: str = "cross_entropy", **kwargs):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        
    def forward(self, carry: Any, batch: Dict[str, torch.Tensor], return_keys: Any = None) -> tuple:
        """Forward pass with loss computation."""
        # Forward through model
        carry, metrics, preds, _, all_finish = self.model(carry, batch, return_keys)

        # Training loop expects (carry, loss, metrics, preds, all_finish)
        loss = metrics.get("loss")
        # Remove loss from metrics dict when returning metrics separately
        if isinstance(metrics, dict) and "loss" in metrics:
            metrics = {k: v for k, v in metrics.items() if k != "loss"}

        return carry, loss, metrics, preds, all_finish

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Proxy to the underlying model's initial_carry."""
        return getattr(self.model, "initial_carry")(batch)  # type: ignore[attr-defined]
