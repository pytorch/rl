Note

Go to the end
to download the full example code.

# Get started with logging

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

The final chapter of this series before we orchestrate everything in a
training script is to learn about logging.

## Loggers

Logging is crucial for reporting your results to the outside world and for
you to check that your algorithm is learning properly. TorchRL has several
loggers that interface with custom backends such as
wandb ([`WandbLogger`](../reference/generated/torchrl.record.loggers.wandb.WandbLogger.html#torchrl.record.loggers.wandb.WandbLogger)),
tensorboard (`TensorBoardLogger`) or a lightweight and
portable CSV logger ([`CSVLogger`](../reference/generated/torchrl.record.loggers.csv.CSVLogger.html#torchrl.record.loggers.csv.CSVLogger)) that you can use
pretty much everywhere.

Loggers are located in the `torchrl.record` module and the various classes
can be found in the [API reference](../reference/trainers_loggers.html#ref-loggers).

We tried to keep the loggers APIs as similar as we could, given the
differences in the underlying backends. While execution of the loggers will
mostly be interchangeable, their instantiation can differ.

Usually, building a logger requires
at least an experiment name and possibly a logging directory and other
hyperparameters.

```
from torchrl.record import CSVLogger

logger = CSVLogger(exp_name="my_exp")
```

Once the logger is instantiated, the only thing left to do is call the
logging methods! For example, `log_scalar()`
is used in several places across the training examples to log values such as
reward, loss value or time elapsed for executing a piece of code.

```
logger.log_scalar("my_scalar", 0.4)
```

## Recording videos

Finally, it can come in handy to record videos of a simulator. Some
environments (e.g., Atari games) are already rendered as images whereas
others require you to create them as such. Fortunately, in most common cases,
rendering and recording videos isn't too difficult.

Let's first see how we can create a Gym environment that outputs images
alongside its observations. [`GymEnv`](../reference/generated/torchrl.envs.GymEnv.html#torchrl.envs.GymEnv) accept two keywords
for this purpose:
- `from_pixels=True` will make the env `step` function
write a `"pixels"` entry containing the images corresponding to your
observations, and

- `pixels_only=False` will indicate that you want the

observations to be returned as well.

```
from torchrl.envs import GymEnv

env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)

print(env.rollout(max_steps=3))

from torchrl.envs import TransformedEnv
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 pixels: Tensor(shape=torch.Size([3, 400, 600, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 pixels: Tensor(shape=torch.Size([3, 400, 600, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

We now have built an environment that renders images with its observations.
To record videos, we will need to combine that environment with a recorder
and the logger (the logger providing the backend to save the video).
This will happen within a transformed environment, like the one we saw in
the [first tutorial](getting-started-0.html#gs-env-ted).

```
from torchrl.record import VideoRecorder

recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)
```

When running this environment, all the `"pixels"` entries will be saved in
a local buffer (i.e. RAM) and dumped in a video on demand (to prevent excessive
RAM usage, you are advised to call this method whenever appropriate!):

```
rollout = record_env.rollout(max_steps=3)
# Uncomment this line to save the video on disk:
# recorder.dump()
```

In this specific case, the video format can be chosen when instantiating
the CSVLogger.

(If you want to customise how your video is recorded, have a look at [our knowledge base](../reference/knowledge_base.html#ref-knowledge-base).)

This is all we wanted to cover in the getting started tutorial.
You should now be ready to code your
[first training loop with TorchRL](getting-started-5.html#gs-first-training)!

**Total running time of the script:** (0 minutes 0.111 seconds)

[`Download Jupyter notebook: getting-started-4.ipynb`](../_downloads/0edaef9fb710ac597eeeb399f4a3bd6e/getting-started-4.ipynb)

[`Download Python source code: getting-started-4.py`](../_downloads/2dadb08a5dab3a561b1c797300baf13d/getting-started-4.py)

[`Download zipped: getting-started-4.zip`](../_downloads/8f12eae910cb18a3a4c44ea4196bc20d/getting-started-4.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)