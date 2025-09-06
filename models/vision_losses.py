from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


class VisionClassificationLossHead(nn.Module):
    """Loss head for vision classification tasks with HRM."""
    
    def __init__(self, model: nn.Module, loss_type: str = "cross_entropy"):
        super().__init__()
        self.model = model
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Forward pass for vision classification."""
        
        # Model forward pass
        new_carry, outputs = self.model(**model_kwargs)
        
        # Get labels - single label per example
        labels = new_carry.current_data["labels"]  # Shape: (batch_size, 1)
        labels = labels.squeeze(-1)  # Shape: (batch_size,)
        
        # Classification logits from the model
        classification_logits = outputs["logits"]  # Shape: (batch_size, num_classes)

        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(classification_logits, dim=-1)
            is_correct = (predictions == labels)
            
            # Only consider halted sequences for metrics
            valid_metrics = new_carry.halted
            
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, is_correct.to(torch.float32), 0).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
            
            # Add predictions to outputs
            outputs["predictions"] = predictions

        # Compute losses - only for halted sequences
        classification_mask = new_carry.halted
        if classification_mask.any():
            masked_logits = classification_logits[classification_mask]
            masked_labels = labels[classification_mask]
            classification_loss = F.cross_entropy(masked_logits, masked_labels, reduction="sum")
        else:
            classification_loss = torch.tensor(0.0, device=classification_logits.device)
        
        # Q-learning loss for halting decision
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )

        metrics.update({
            "classification_loss": classification_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue loss (for bootstrapping)
        q_continue_loss = torch.tensor(0.0, device=classification_logits.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                outputs["target_q_continue"], 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Total loss
        total_loss = classification_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()