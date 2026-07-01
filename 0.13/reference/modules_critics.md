# Value Networks and Critics

Value networks estimate the value of states or state-action pairs.

| [`ValueOperator`](generated/torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)(*args, **kwargs) | General class for value functions in RL. |
| --- | --- |
| [`ValueNorm`](generated/torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm)(*[, shape, epsilon, device]) | Abstract base class for value normalisers. |
| [`PopArtValueNorm`](generated/torchrl.modules.PopArtValueNorm.html#torchrl.modules.PopArtValueNorm)(*[, shape, beta, epsilon, ...]) | PopArt-style EMA value normaliser. |
| [`RunningValueNorm`](generated/torchrl.modules.RunningValueNorm.html#torchrl.modules.RunningValueNorm)(*[, shape, epsilon, device]) | Exact running mean / variance (Welford's online algorithm). |
| [`DuelingCnnDQNet`](generated/torchrl.modules.DuelingCnnDQNet.html#torchrl.modules.DuelingCnnDQNet)(out_features[, ...]) | Dueling CNN Q-network. |
| [`DistributionalDQNnet`](generated/torchrl.modules.DistributionalDQNnet.html#torchrl.modules.DistributionalDQNnet)(*args, **kwargs) | Distributional Deep Q-Network softmax layer. |
| [`ConvNet`](generated/torchrl.modules.ConvNet.html#torchrl.modules.ConvNet)(in_features, depth, num_cells, ...) | A convolutional neural network. |
| [`CrossCriticGroupSpec`](generated/torchrl.modules.CrossCriticGroupSpec.html#torchrl.modules.CrossCriticGroupSpec)(obs_dim, n_agents, ...) | Specification for one agent group used by [`CrossGroupCritic`](generated/torchrl.modules.CrossGroupCritic.html#torchrl.modules.CrossGroupCritic). |
| [`CrossGroupCritic`](generated/torchrl.modules.CrossGroupCritic.html#torchrl.modules.CrossGroupCritic)(*args, **kwargs) | Centralised critic that conditions on observations from multiple agent groups. |
| [`MLP`](generated/torchrl.modules.MLP.html#torchrl.modules.MLP)(in_features, out_features, depth, ...) | A multi-layer perceptron. |
| [`DdpgCnnActor`](generated/torchrl.modules.DdpgCnnActor.html#torchrl.modules.DdpgCnnActor)(action_dim[, conv_net_kwargs, ...]) | DDPG Convolutional Actor class. |
| [`DdpgCnnQNet`](generated/torchrl.modules.DdpgCnnQNet.html#torchrl.modules.DdpgCnnQNet)([conv_net_kwargs, ...]) | DDPG Convolutional Q-value class. |
| [`DdpgMlpActor`](generated/torchrl.modules.DdpgMlpActor.html#torchrl.modules.DdpgMlpActor)(action_dim[, mlp_net_kwargs, ...]) | DDPG Actor class. |
| [`DdpgMlpQNet`](generated/torchrl.modules.DdpgMlpQNet.html#torchrl.modules.DdpgMlpQNet)([mlp_net_kwargs_net1, ...]) | DDPG Q-value MLP class. |
| [`LSTMModule`](generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule)(*args, **kwargs) | An embedder for an LSTM module. |
| [`GRUModule`](generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule)(*args, **kwargs) | An embedder for an GRU module. |
| [`canonicalize_rnn_subset`](generated/torchrl.modules.canonicalize_rnn_subset.html#torchrl.modules.canonicalize_rnn_subset)(data, modules, *[, ...]) | Canonicalize only the union of RNN keys used by `modules`. |
| [`set_recurrent_mode`](generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode)([mode]) | Context manager for setting RNNs recurrent mode. |
| [`OnlineDTActor`](generated/torchrl.modules.OnlineDTActor.html#torchrl.modules.OnlineDTActor)(state_dim, action_dim[, ...]) | Online Decision Transformer Actor class. |
| [`DTActor`](generated/torchrl.modules.DTActor.html#torchrl.modules.DTActor)(state_dim, action_dim[, ...]) | Decision Transformer Actor class. |
| [`DecisionTransformer`](generated/torchrl.modules.DecisionTransformer.html#torchrl.modules.DecisionTransformer)(state_dim, action_dim[, ...]) | Online Decision Transformer. |