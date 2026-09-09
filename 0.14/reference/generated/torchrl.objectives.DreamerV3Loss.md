# DreamerV3Loss

*class*torchrl.objectives.DreamerV3Loss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3Loss)

Compose DreamerV3 world-model, imagination and replay-value objectives.

See also [`DreamerV3LossConfig`](torchrl.trainers.algorithms.configs.DreamerV3LossConfig.html#torchrl.trainers.algorithms.configs.DreamerV3LossConfig).

Posterior states start imagination with detached features. The replay-value
term instead retains its path to the world model. The returned TensorDict
contains scalar loss entries and detached posterior features under
`replay_context` for generation-checked replay updates. Sum the entries
whose names start with `loss_` to obtain the training objective.

Reference: Hafner et al., "Mastering Diverse Domains through World Models"
(2023), [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

Parameters:

- **model_loss** ([*DreamerV3ModelLoss*](torchrl.objectives.DreamerV3ModelLoss.html#torchrl.objectives.DreamerV3ModelLoss)) - World-model objective. Must use
`detach_output=False` when the replay-value weight is nonzero.
- **actor_loss** ([*DreamerV3ActorLoss*](torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss)) - Imagination objective, sharing the
world-model dynamics and the online value network.
- **value_loss** ([*DreamerV3ValueLoss*](torchrl.objectives.DreamerV3ValueLoss.html#torchrl.objectives.DreamerV3ValueLoss)) - Critic objective for imagined and real
sequences.

Keyword Arguments:

- **replay_value_loss_weight** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Replay-value contribution
to the total objective; zero disables this term. Default: `0.3`.
- **continuation_horizon** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Horizon used for replay-value
targets. Default: `333.0`.
- **lmbda** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Lambda-return coefficient for replay-value
targets. Default: `0.95`.

Examples

This small learner shares dynamics and reward parameters between real
sequences and imagined rollouts. All components are public imports.

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... NormalParamExtractor,
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... TensorDictModule,
... TensorDictSequential,
... )
>>> from torchrl.data import Bounded, Composite, Unbounded
>>> from torchrl.envs.model_based import DreamerEnv
>>> from torchrl.modules import (
... MLP,
... RSSMPriorV3,
... RSSMPosteriorV3,
... RSSMRolloutV3,
... TanhNormal,
... WorldModelWrapper,
... )
>>> from torchrl.objectives import (
... DreamerV3ActorLoss,
... DreamerV3Loss,
... DreamerV3ModelLoss,
... DreamerV3ValueLoss,
... SoftUpdate,
... )
>>> prior_net = RSSMPriorV3(
... action_shape=(1,),
... hidden_dim=8,
... rnn_hidden_dim=8,
... num_categoricals=2,
... num_classes=2,
... action_dim=1,
... )
>>> posterior_net = RSSMPosteriorV3(
... hidden_dim=8,
... rnn_hidden_dim=8,
... num_categoricals=2,
... num_classes=2,
... obs_embed_dim=8,
... )
>>> prior = TensorDictModule(
... prior_net,
... in_keys=["state", "belief", "action"],
... out_keys=[
... ("next", "prior_logits"),
... ("next", "state"),
... ("next", "belief"),
... ],
... )
>>> posterior = TensorDictModule(
... posterior_net,
... in_keys=[("next", "belief"), ("next", "encoded")],
... out_keys=[("next", "posterior_logits"), ("next", "state")],
... )
>>> reward_net = MLP(in_features=12, out_features=1, num_cells=8, depth=1)
>>> world_model = TensorDictSequential(
... TensorDictModule(
... torch.nn.Linear(3, 8),
... in_keys=[("next", "observation")],
... out_keys=[("next", "encoded")],
... ),
... RSSMRolloutV3(prior, posterior, reset_key="is_init"),
... TensorDictModule(
... MLP(in_features=12, out_features=3, num_cells=8, depth=1),
... in_keys=[("next", "state"), ("next", "belief")],
... out_keys=[("next", "reco_pixels")],
... ),
... TensorDictModule(
... reward_net,
... in_keys=[("next", "state"), ("next", "belief")],
... out_keys=[("next", "reward")],
... ),
... )
>>> imagination = WorldModelWrapper(
... TensorDictModule(
... prior_net,
... in_keys=["state", "belief", "action"],
... out_keys=["_", "state", "belief"],
... ),
... TensorDictModule(
... reward_net, in_keys=["state", "belief"], out_keys=["reward"]
... ),
... )
>>> env = DreamerEnv(imagination, prior_shape=(4,), belief_shape=(8,))
>>> env.observation_spec = Composite(
... state=Unbounded(4), belief=Unbounded(8)
... )
>>> env.state_spec = env.observation_spec.clone()
>>> env.action_spec = Bounded(-1, 1, (1,))
>>> env.reward_spec = Unbounded((1,))
>>> actor = ProbabilisticTensorDictSequential(
... TensorDictModule(
... MLP(in_features=12, out_features=2, num_cells=8, depth=1),
... in_keys=["state", "belief"],
... out_keys=["params"],
... ),
... TensorDictModule(
... NormalParamExtractor(),
... in_keys=["params"],
... out_keys=["loc", "scale"],
... ),
... ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=TanhNormal,
... return_log_prob=True,
... ),
... )
>>> value = TensorDictModule(
... MLP(in_features=12, out_features=1, num_cells=8, depth=1),
... in_keys=["state", "belief"],
... out_keys=["state_value"],
... )
>>> model_loss = DreamerV3ModelLoss(
... world_model, reward_two_hot=False, detach_output=False
... )
>>> model_loss.set_keys(pixels="observation")
>>> actor_loss = DreamerV3ActorLoss(
... actor, value, env, imagination_horizon=3
... )
>>> value_loss = DreamerV3ValueLoss(
... value, actor_loss=actor_loss, slow_critic_regularization=1.0
... )
>>> target_updater = SoftUpdate(value_loss, tau=0.02)
>>> loss_module = DreamerV3Loss(model_loss, actor_loss, value_loss)
>>> sample = TensorDict(
... {
... "state": torch.zeros(2, 3, 4),
... "belief": torch.zeros(2, 3, 8),
... "action": torch.zeros(2, 3, 1),
... "is_init": torch.zeros(2, 3, 1, dtype=torch.bool),
... "next": {
... "observation": torch.randn(2, 3, 3),
... "reward": torch.randn(2, 3, 1),
... "done": torch.zeros(2, 3, 1, dtype=torch.bool),
... "terminated": torch.zeros(2, 3, 1, dtype=torch.bool),
... },
... },
... [2, 3],
... )
>>> losses = loss_module(sample)
>>> sum(
... value for key, value in losses.items() if key.startswith("loss_")
... ).backward()
>>> assert losses["replay_context"].batch_size == sample.batch_size
>>> assert not losses["replay_context", "state"].requires_grad
```

default_keys

alias of `_AcceptedKeys`

forward(*sample: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3Loss.forward)

Compute scalar objectives and detached posterior replay features.

Parameters:

**sample** (*TensorDictBase*) - Real transition sequences with batch
dimensions `[batch, time]`.

Returns:

Scalar loss and metric entries, plus a posterior TensorDict under
the configured `replay_context` key.

*property*in_keys*: list[NestedKey]*

World-model inputs, reset marker and real transition targets.