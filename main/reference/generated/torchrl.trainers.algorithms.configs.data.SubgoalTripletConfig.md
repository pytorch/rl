# torchrl.trainers.algorithms.configs.data.SubgoalTripletConfig

*class*torchrl.trainers.algorithms.configs.data.SubgoalTripletConfig(*_target_: str = 'torchrl.data.replay_buffers.SubgoalTriplet'*, *goal_discount: float | None = 0.99*, *policy_goal_discount: float | None = None*, *done_key: Any = ('next', 'done')*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#SubgoalTripletConfig)

Hydra configuration for `SubgoalTriplet`.

Every kwarg accepted by `SubgoalTriplet.__init__` is exposed as a field
here with the same default.