# reward_model_loss

*class*torchrl.objectives.llm.reward_model_loss(*chosen_scores: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *rejected_scores: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *reduction: Literal['mean', 'sum', 'none']*)[[source]](../../_modules/torchrl/objectives/llm/reward.html#reward_model_loss)

Compute the Bradley-Terry pairwise reward-model loss.

The loss is computed as `-log_sigmoid(chosen_scores - rejected_scores)`. It is
small when the reward model assigns a higher score to the chosen response than to
the rejected one, and large otherwise.

\[\text{loss} = -\log\sigma(r_\theta(x, y_c) - r_\theta(x, y_r))\]

Parameters:

- **chosen_scores** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the scalar scores assigned to the chosen
responses. Must have shape `[B]`.
- **rejected_scores** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the scalar scores assigned to the rejected
responses. Must have shape `[B]`.
- **reduction** (*Literal**[**"mean"**,**"sum"**,**"none"**]*) - the reduction to apply to the loss.

Returns:

The Bradley-Terry loss.

References

- Ralph Allan Bradley, Milton E. Terry, 1952. "Rank Analysis of Incomplete Block
Designs: I. The Method of Paired Comparisons".