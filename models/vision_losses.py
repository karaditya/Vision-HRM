import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


IGNORE_LABEL_ID = -100


class VisionClassificationLossHead(nn.Module):
    """Loss head for vision classification tasks with deep supervision."""
    
    def __init__(self, model: nn.Module, loss_type: str = "cross_entropy", **kwargs):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        
    def forward(self, carry: Any, batch: Dict[str, torch.Tensor], return_keys: Any = None) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool]:
        new_carry, outputs = self.model(carry, batch, return_keys)
        labels = new_carry.current_data["labels"]
        if labels.dim() > 1:
            labels = labels[:, 0]  # Assume scalar labels
        
        lm_loss = F.cross_entropy(outputs["logits"], labels.long())
        with torch.no_grad():
            is_correct = (torch.argmax(outputs["logits"], dim=-1) == labels.long())
            valid_metrics = new_carry.halted
            total_valid = valid_metrics.sum().float().item()  # Compute total_valid as a scalar
            accuracy = torch.where(valid_metrics, is_correct.float(), 0).sum() / total_valid if total_valid > 0 else torch.tensor(0.0, device=lm_loss.device)
            exact_accuracy = (valid_metrics & is_correct).sum().float() / total_valid if total_valid > 0 else torch.tensor(0.0, device=lm_loss.device)
            metrics = {
                "count": valid_metrics.sum().float(),
                "accuracy": accuracy,
                "exact_accuracy": exact_accuracy,
                "steps": torch.where(valid_metrics, new_carry.steps.float(), 0).sum(),
            }
            q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], is_correct.float(), reduction="sum")
            q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
            if "target_q_continue" in outputs:
                q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics.update({
                "lm_loss": lm_loss.detach(),  # Keep as tensor
                "q_halt_loss": q_halt_loss.detach(),  # Keep as tensor
                "q_continue_loss": q_continue_loss.detach(),  # Keep as tensor
            })
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs} if return_keys else {}
        all_finish = new_carry.halted.all()
        return new_carry, total_loss, metrics, detached_outputs, all_finish

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Proxy to the underlying model's initial_carry."""
        return self.model.initial_carry(batch)  # type: ignore
