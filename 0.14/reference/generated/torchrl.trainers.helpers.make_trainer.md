# make_trainer

torchrl.trainers.helpers.make_trainer(*collector: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)*, *loss_module: [LossModule](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*, *recorder: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | None = None*, *target_net_updater: TargetNetUpdater | None = None*, *policy_exploration: None | TensorDictModuleWrapper | TensorDictModule = None*, *replay_buffer: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) | None = None*, *logger: Logger | None = None*, *cfg: DictConfig = None*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)[[source]](../../_modules/torchrl/trainers/helpers/trainers.html#make_trainer)

Creates a Trainer instance given its constituents.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - A data collector to be used to collect data.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - A TorchRL loss module
- **recorder** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*,**optional*) - a recorder environment. If None, the trainer will train the policy without
testing it.
- **target_net_updater** (*TargetNetUpdater**,**optional*) - A target network update object.
- **policy_exploration** (*TDModule**or**TensorDictModuleWrapper**,**optional*) - a policy to be used for recording and exploration
updates (should be synced with the learnt policy).
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - a replay buffer to be used to collect data.
- **logger** (*Logger**,**optional*) - a Logger to be used for logging.
- **cfg** (*DictConfig**,**optional*) - a DictConfig containing the arguments of the script. If None, the default
arguments are used.

Returns:

A trainer built with the input objects. The optimizer is built by this helper function using the cfg provided.

Examples

```
>>> import torch
>>> import tempfile
>>> from torchrl.trainers.loggers import TensorboardLogger
>>> from torchrl.trainers import Trainer
>>> from torchrl.envs import EnvCreator
>>> from torchrl.collectors import Collector
>>> from torchrl.data import TensorDictReplayBuffer
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.modules import TensorDictModuleWrapper, SafeModule, ValueOperator, EGreedyWrapper
>>> from torchrl.objectives.common import LossModule
>>> from torchrl.objectives.utils import TargetNetUpdater
>>> from torchrl.objectives import DDPGLoss
>>> env_maker = EnvCreator(lambda: GymEnv("Pendulum-v0"))
>>> env_proof = env_maker()
>>> obs_spec = env_proof.observation_spec
>>> action_spec = env_proof.action_spec
>>> net = torch.nn.Linear(env_proof.observation_spec.shape[-1], action_spec.shape[-1])
>>> net_value = torch.nn.Linear(env_proof.observation_spec.shape[-1], 1) # for the purpose of testing
>>> policy = SafeModule(action_spec, net, in_keys=["observation"], out_keys=["action"])
>>> value = ValueOperator(net_value, in_keys=["observation"], out_keys=["state_action_value"])
>>> collector = Collector(env_maker, policy, total_frames=100)
>>> loss_module = DDPGLoss(policy, value, gamma=0.99)
>>> recorder = env_proof
>>> target_net_updater = None
>>> policy_exploration = EGreedyWrapper(policy)
>>> replay_buffer = TensorDictReplayBuffer()
>>> dir = tempfile.gettempdir()
>>> logger = TensorboardLogger(exp_name=dir)
>>> trainer = make_trainer(collector, loss_module, recorder, target_net_updater, policy_exploration,
... replay_buffer, logger)
>>> print(trainer)
```