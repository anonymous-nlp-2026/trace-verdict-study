"""
GradientMaskTrainer: SFTTrainer subclass for gradient dilution ablation (MF-2).

Randomly masks a fraction of valid token losses to simulate full_trace's
gradient sparsity when training on exception_only data.
Verdict tokens (last N valid tokens covering the assistant turn) are always preserved.

Usage: --gradient_mask_ratio 0.998 keeps ~0.2% of token gradients.
"""

import torch
from trl import SFTTrainer

# Chat template: ...<|im_start|>assistant\n{verdict}<|im_end|>\n
# Protect last 3 valid shifted-label positions to cover verdict + <|im_end|> + \n
VERDICT_PROTECT_N = 3


class GradientMaskTrainer(SFTTrainer):
    """SFTTrainer with random per-token loss masking for gradient dilution ablation."""

    def __init__(self, gradient_mask_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gradient_mask_ratio = gradient_mask_ratio

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override loss computation to apply random token-level gradient masking.

        For each batch:
        1. Forward pass to get logits
        2. Compute per-token cross-entropy (unreduced)
        3. Among valid tokens (labels != -100), randomly keep (1 - mask_ratio) fraction
        4. Always preserve the verdict tokens (last VERDICT_PROTECT_N valid tokens)
        5. Return mean loss over kept tokens
        """
        if self.gradient_mask_ratio <= 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM: predict next token from current
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        valid_mask = (shift_labels != -100).float()

        # Stochastic keep mask
        keep_ratio = 1.0 - self.gradient_mask_ratio
        random_keep = torch.rand_like(valid_mask) < keep_ratio

        # Protect verdict: last VERDICT_PROTECT_N valid tokens per sequence
        for i in range(shift_labels.size(0)):
            valid_pos = (shift_labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid_pos) > 0:
                protect = valid_pos[-VERDICT_PROTECT_N:]
                random_keep[i, protect] = True

        final_mask = valid_mask * random_keep.float()
        masked_loss = (per_token_loss * final_mask).sum() / final_mask.sum().clamp(min=1)

        # Log stats every 10 steps
        if self.state.global_step % 10 == 0:
            total_valid = valid_mask.sum().item()
            kept = final_mask.sum().item()
            mask_ratio = 1.0 - kept / max(total_valid, 1)
            print(
                f"[GradMask] step={self.state.global_step} "
                f"valid={int(total_valid)} kept={int(kept)} "
                f"mask_ratio={mask_ratio:.4f}"
            )

        return (masked_loss, outputs) if return_outputs else masked_loss
