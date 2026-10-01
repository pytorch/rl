# LLM Objectives

Specialized loss functions for LLM training.

## GRPO

| [`GRPOLoss`](generated/torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss)(*args, **kwargs) | GRPO loss. |
| --- | --- |
| [`GRPOLossOutput`](generated/torchrl.objectives.llm.GRPOLossOutput.html#torchrl.objectives.llm.GRPOLossOutput)(loss_objective, ...[, ...]) | |
| [`MCAdvantage`](generated/torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage)(grpo_size[, prompt_key, ...]) | Monte-Carlo advantage computation engine. |

## SFT

| [`SFTLoss`](generated/torchrl.objectives.llm.SFTLoss.html#torchrl.objectives.llm.SFTLoss)(*args, **kwargs) | Supervised fine-tuning loss. |
| --- | --- |
| [`SFTLossOutput`](generated/torchrl.objectives.llm.SFTLossOutput.html#torchrl.objectives.llm.SFTLossOutput)(loss_sft[, loss_kl_to_ref, ...]) | |