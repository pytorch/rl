# Common PyTorch errors and solutions

## Gradient-related errors \[Newcomers\]

Newcomers often face gradient-related issues when coding up an RL algorithm from scratch.
The typical training loop can usually be sketched as follows:
```python

obs = env.reset()

for _ in range(n_training_steps):
    # STEP 1: data collection
    # Get a new datapoint "online"
    observations = []
    actions = []
    others = []
    for _ in range(n_data_per_training):
        with torch.no_grad():
            action = policy(obs)
        obs, *other = env.step(action)
        observations.append(obs)
        actions.append(action)
        others.append(other)
    replay_buffer.extend(observations, actions, others)

    # STEP 2: loss and optimization
    # => compute loss "offline"
    loss = loss_fn(replay_buffer.sample(batch_size))
    
    loss.backward()
    optim.step()
    optim.zero_grad()

```

A series of errors come from wanting to backpropagate through the policy operation
that is decorated by the `no_grad()` context manager. In fact, this operation should
(in most cases) not be part of any computational graph. Instead, all the differentiable
operations should be executed in the `loss_fn(...)` abstraction.
In general, RL is a domain where one should pay attention to understanding well
what should be considered as non-differentiable "data" (e.g. environment 
interactions, advantage and return computation, 'denominator' log-probability in PPO)
and what should be considered as  differentiable loss artifacts
(e.g. value error, 'numerator' log-probability in PPO).

Errors to look for that may be related to this misconception are the following:
- `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).`
  This error usually appears after a datapoint that is part of a compuational graph is used twice
  in the loss function. Some users try to fix this by calling `loss.backward(retain_graph=True)`, but this will lead
  to the next error of this list.
  **Related discussed PyTorch errors**:
  - [here](https://discuss.pytorch.org/t/how-to-properly-create-a-batch-with-torch-tensor/169217)
  - [here](https://discuss.pytorch.org/t/i-am-training-my-multi-agents-reinforcement-learning-project-and-i-got-an-error-trying-to-backward-through-the-graph-a-second-time/152352)

- `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`
  This typically occurs after one fixes the first error with a `retain_graph=True` flag. Instead, the operation
  that is to be differentiated through should be re-computed in the `loss_fn`.
  Another common reason is that two modules are updated using a shared compuational graph (e.g. the policy and the critic).
  In that case the `retain_graph=True` flag should be used, although one should be careful as this
  may accumulate gradients of one loss onto the other. In general, it's better practice to
  re-compute each intermediate value for each loss separately while excluding the parameters
  that are not necessary from the specific graph, even if the forward call of some submodules match.
  **Related discussed PyTorch errors**:
  - [here](https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-3-1-which-is-output-0-of-tanhbackward-is-at-version-1-expected-version-0-instead/87630)
  - [here](https://discuss.pytorch.org/t/in-place-operation-error-while-training-maddpg/151622)

- Algorithm is not learning / `param.grad` is 0 or None.
  An algorithm not learning can have multiple causes. The first thing to look at
  is the value of the parameter gradients, whose norm should be strictly non-negative.
  **Related PyTorch discussed errors**:
  - [here](https://discuss.pytorch.org/t/multi-threaded-backprop-failing-in-a3c-implementation/157132/5)

## My Training is too slow \[Newcomers / intermediate\]
- RL is known to be CPU-intensive in some instances. Even when running a few
  environments in parallel, you can see a great speed-up by asking for more cores on your cluster
  than the number of environments you're working with (twice as much for instance). This
  is also and especially true for environments that are rendered (even if they are rendered on GPU). 
- The speed of training depends upon several factors and there is not a one-fits-all
  solution to every problem. The common bottlnecks are:
  - **data collection**: the simulator speed may affect performance, as can the data
    transformation that follows. Speeding up environment interactions is usually
    done via vectorization (if the simulators enables it, e.g. Brax and other Jax-based
    simulators) or parallelization (which is improperly called vectorized envs in gym
    and other libraries). In TorchRL, transformations can usually be executed on device.
  - **Replay buffer storage and sampling**: storing items in a replay buffer can
    take time if the underlying operation requires some heavy memory manipulation
    or tedeious indexing (e.g. with prioritized replay buffers). Sampling can
    also take a considerable amount of time if the data isn't stored contiguously
    and/or if costly stacking of concatenation operations are performed.
    TorchRL provides efficient contiguous storage solutions and efficient writing
    and sampling solutions in these cases.
  - **Advantage computation**: computing advantage functions can also constitute
    a computational bottleneck as these are usually coded using plain for loops.
    If profiling indicates that this operation is taking a considerable amount
    of time, consider using our fully vectorized solutions instead.
  - **Loss compuation**: The loss computation and the optimization
    steps are frequently responsible of a significant share of the compute time.
    Some techniques can speed things up. For instance, if multiple target networks
    are being used, using vectorized maps and functional programming (through 
    functorch) instead of looping over the model configurations can provide a
    significant speedup.

## Common bugs
- For bugs related to mujoco (incl. DeepMind Control suite and other libraries),
  refer to the [MUJOCO_INSTALLATION](MUJOCO_INSTALLATION.md) file.
- `ValueError: bad value(s) in fds_to_keep`: this can have multiple reasons. One that is common in torchrl
  is that you are trying to send a tensor across processes that is a view of another tensor.
  For instance, when sending the tensor `b = tensor.expand(new_shape)` across processes, the reference to the original
  content will be lost (as the `expand` operation keeps the reference to the original tensor).
  To debug this, look for such operations (`view`, `permute`, `expand`, etc.) and call `clone()` or `contiguous()` after
  the call to the function.
