# torchrl.trainers.algorithms.configs.modules.TdMpc2WorldModelConfig

*class*torchrl.trainers.algorithms.configs.modules.TdMpc2WorldModelConfig(*observation_dim: int = '???'*, *action_dim: int = '???'*, *latent_dim: int = '???'*, *encoder_dim: int = 256*, *encoder_depth: int = 1*, *mlp_dim: int = 512*, *simnorm_dim: int = 8*, *num_bins: int = 101*, *observation_key: Any = 'observation'*, *action_key: Any = 'action'*, *latent_key: Any = 'latent'*, *reward_logits_key: Any = 'reward_logits'*, *device: Any = None*, *shared: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tdmpc2_world_model'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TdMpc2WorldModelConfig)

Configuration for a TD-MPC2 world model.

The resulting [`WorldModel`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel) encodes observations,
predicts the next latent state from the current latent state and action,
and predicts distributional rewards from the current latent state and
action. TensorDict routing is configured with the specialized key fields
below. Set `shared=True` to place the constructed module in shared
memory.

See also

[`WorldModel`](torchrl.modules.WorldModel.html#torchrl.modules.WorldModel)