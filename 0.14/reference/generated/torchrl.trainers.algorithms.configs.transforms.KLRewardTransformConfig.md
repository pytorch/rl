# torchrl.trainers.algorithms.configs.transforms.KLRewardTransformConfig

*class*torchrl.trainers.algorithms.configs.transforms.KLRewardTransformConfig(*ref_model: Any = None*, *ref_model_factory: Any = None*, *coef: Any = 1.0*, *in_keys: list[str] | None = None*, *out_keys: list[str] | None = None*, *log_prob_key: Any = ('log_probs', 'full')*, *device: Any = None*, *add_to_reward: bool = True*, *tokenizer: Any = None*, *assistant_only: bool = True*, *padding_side: str = 'left'*, *use_ray_service: bool = False*, *_target_: str = 'torchrl.envs.llm.transforms.kl.KLRewardTransform'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/transforms.html#KLRewardTransformConfig)

Hydra configuration for [`KLRewardTransform`](torchrl.envs.llm.transforms.KLRewardTransform.html#torchrl.envs.llm.transforms.KLRewardTransform).

Every kwarg accepted by `KLRewardTransform.__init__` is exposed as a field
here. Instantiating without `ref_model` or `ref_model_factory` still
fails at runtime because the class requires one of them.