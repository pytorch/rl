# LLM Objectives

Specialized loss functions for LLM training.

## GRPO

| [`GRPOLoss`](generated/torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss)(*args, **kwargs) | GRPO loss. |
| --- | --- |
| [`GRPOLossOutput`](generated/torchrl.objectives.llm.GRPOLossOutput.html#torchrl.objectives.llm.GRPOLossOutput)(loss_objective, ...[, ...]) | |
| [`MCAdvantage`](generated/torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage)(grpo_size[, prompt_key, ...]) | Monte-Carlo advantage computation engine. |
| [`MCAdvantageSelector`](generated/torchrl.objectives.llm.MCAdvantageSelector.html#torchrl.objectives.llm.MCAdvantageSelector)([strategy, ...]) | Select trajectories from an oversampled Monte-Carlo advantage group. |
| [`RayMCAdvantage`](generated/torchrl.objectives.llm.RayMCAdvantage.html#torchrl.objectives.llm.RayMCAdvantage)(grpo_size[, prompt_key, ...]) | Ray actor-backed [`MCAdvantage`](generated/torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage). |

## SFT

| [`SFTLoss`](generated/torchrl.objectives.llm.SFTLoss.html#torchrl.objectives.llm.SFTLoss)(*args, **kwargs) | Supervised fine-tuning loss. |
| --- | --- |
| [`SFTLossOutput`](generated/torchrl.objectives.llm.SFTLossOutput.html#torchrl.objectives.llm.SFTLossOutput)(loss_sft[, loss_kl_to_ref, ...]) | |

## Distillation

| [`DistillationLoss`](generated/torchrl.objectives.llm.DistillationLoss.html#torchrl.objectives.llm.DistillationLoss)(*args, **kwargs) | Token-level knowledge-distillation loss for LLM policies. |
| --- | --- |
| [`DistillationLossOutput`](generated/torchrl.objectives.llm.DistillationLossOutput.html#torchrl.objectives.llm.DistillationLossOutput)(loss_distill, ...[, ...]) | |
| [`k3_kl_token_estimate`](generated/torchrl.objectives.llm.k3_kl_token_estimate.html#torchrl.objectives.llm.k3_kl_token_estimate)(target_log_prob, log_prob) | Per-token k3 estimate of the KL divergence to a target distribution. |