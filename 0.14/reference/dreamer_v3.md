# DreamerV3 in a nutshell

[DreamerV3](https://arxiv.org/abs/2301.04104) is a model-based reinforcement
learning algorithm. It learns a compact model of the environment from replayed
experience, then trains an actor and a critic on trajectories generated inside
that model. The real environment supplies data for the world model; most policy
improvement happens in latent-space imagination.

## Paper and maintained implementation

This page treats the [DreamerV3 paper](https://arxiv.org/abs/2301.04104) as
the source of truth for the algorithm. The author-maintained
[JAX implementation](https://github.com/danijar/dreamerv3) continues to
evolve and its named experiment presets can differ from the protocol reported
in the paper. TorchRL documents those presets as separate reproduction targets
rather than redefining the paper algorithm around the latest JAX configuration.

Some constructor defaults predate full paper parity and remain for backward
compatibility. The runnable DreamerV3 recipes pass the paper-compatible loss
settings explicitly; changes to public defaults require the normal deprecation
cycle.

The high-level data flow is:

```
real transition sequences
 |
 v
encoder -> posterior RSSM state -> reconstruction, reward, continuation
 ^ |
 | v
 prior dynamics + action
 |
 v
 imagined trajectories
 |
 v
 actor + online critic
 |
 v
 slow (target) critic
```

## Nomenclature

Dreamer papers and implementations use several names for closely related
objects. In the TorchRL API:

| Term | Meaning |
| --- | --- |
| **World model** | The observation encoder, recurrent state-space model (RSSM), observation decoder, reward predictor, and optional continuation predictor. |
| **Belief** or **deterministic state** (`h_t`) | The recurrent hidden state that summarizes history. TorchRL stores it under the `"belief"` key. |
| **State** or **stochastic state** (`z_t`) | A sample from the RSSM's categorical latent variables. TorchRL stores the flattened straight-through one-hot sample under `"state"`. |
| **Prior**, **dynamics**, or **transition model** | Predicts the next categorical state from the previous state, belief, and action, without seeing the next observation. See `RSSMPriorV3`. |
| **Posterior** or **representation model** | Corrects the prior using the encoded next observation. It is used while learning from real sequences. See `RSSMPosteriorV3`. |
| **Imagination** | A latent rollout that uses the prior, reward model, continuation model, and actor, but no real observations. |
| **Critic** or **value model** | The online network that predicts lambda returns from the RSSM state and belief. |
| **Slow critic**, **target critic**, or **EMA critic** | A lagged copy of the online critic. It is updated by Polyak averaging and provides a stable auxiliary target for critic regularization. "Slow" refers to its parameter updates, not its optimizer or runtime. |
| **Continuation** | The learned probability that an imagined trajectory continues. It replaces a fixed survival assumption when weighting returns and losses. |

## How the RSSM works

The RSSM splits its latent representation into a deterministic recurrent state
and a stochastic categorical state. At each real-data time step:

1. `RSSMPriorV3` updates the belief from the previous
stochastic state and action, then predicts prior categorical logits.
2. `RSSMPosteriorV3` combines that belief with the
encoded observation and predicts posterior categorical logits.
3. Both networks sample hard categorical states with a straight-through
gradient estimator. `unimix` can mix a small uniform component into the
probabilities to prevent overconfident categories.
4. `RSSMRolloutV3` carries the posterior state and
belief through a sequence and resets them at episode boundaries.

During imagination there is no observation, so only the prior advances the
latent state. The actor and prediction heads consume both `state` and
`belief`. `RSSMPriorV3` supports a conventional GRU and the grouped
`"block_gru"` core used by the full DreamerV3 example. The accompanying
[`DreamerV3MLP`](generated/torchrl.modules.DreamerV3MLP.html#torchrl.modules.DreamerV3MLP) provides the RMS-normalized SiLU MLP
blocks used by the example's encoder, decoder, actor, critic, and prediction
heads.

For recurrent features outside an RSSM, [`DreamerV3BlockGRUCell`](generated/torchrl.modules.DreamerV3BlockGRUCell.html#torchrl.modules.DreamerV3BlockGRUCell)
exposes the same block-diagonal update as a single-step module, while
[`DreamerV3BlockGRU`](generated/torchrl.modules.DreamerV3BlockGRU.html#torchrl.modules.DreamerV3BlockGRU) executes batch-major sequences with
mixed episode resets.

## Selecting the sequence backend

The sequence backend is selected directly on the high-level module:

```
from torchrl.modules import DreamerV3BlockGRU

gru = DreamerV3BlockGRU(
 input_size=512,
 hidden_size=512,
 recurrent_backend="triton",
).cuda()
```

The three backends trade portability for speed:

- `"reference"` (default) runs the time loop with ordinary autograd. It
works on every supported device, floating dtype, and elementwise activation,
and it is the only backend that supports double backward
(`create_graph=True`). It is the slowest option on long sequences.
- `"scan"` fuses the time loop through `torch._higher_order_ops.scan` and
carries only the hidden cotangent in a specialized reverse scan. It runs on
CPU and CUDA, requires a recent PyTorch with the `hoptorch` package, and
supports the same activations as the reference backend. Mixed input/hidden
dtypes are promoted like the reference backend. Its backward consumes saved
gate states, so double backward raises instead of silently returning wrong
second-order gradients.
- `"triton"` fuses the complete forward and reverse-time recurrences into
one CUDA kernel each, keeping the carry on-chip across the whole horizon.
It requires an NVIDIA GPU and Triton 3.3 or newer, supports `nn.SiLU`,
`nn.Tanh`, and `nn.ReLU` dynamics, and runs in `float32` or
`bfloat16` (mixed input and hidden dtypes are promoted like the reference
backend; other dtypes raise an error). Parameters stay in `float32` and
accumulation is performed in `float32` in both directions. Like the scan
backend, double backward raises. Kernels are autotuned, so the first calls
for a new sequence-length/width configuration pay a tuning warmup. On
DreamerV3-sized workloads it is roughly an order of magnitude faster than
the scan backend in both directions.

Select `"scan"` or `"triton"` explicitly so missing dependencies or
unsupported devices are reported instead of silently changing execution; the
optimized backends never fall back to another implementation.

To compare the backends on your own shapes and hardware (synchronized forward
and backward timings, peak memory, and 95% confidence intervals), run the
developer benchmark from a source checkout:

```
python benchmarks/bench_rnn_backward.py --rnn block_gru \
 --backends reference,scan,triton --batches 16 --seq-lens 64,512 \
 --hiddens 512 --input-size 512 --projection-size 512 --blocks 8 \
 --dtype bfloat16 --warmup 10 --iters 30
```

Use the batch size, sequence length, widths, block count, dtype, and compile
modes from the intended workload: backend performance is hardware- and
shape-dependent.

## The three objectives

### World model

[`DreamerV3ModelLoss`](generated/torchrl.objectives.DreamerV3ModelLoss.html#torchrl.objectives.DreamerV3ModelLoss) trains the model on real
transition sequences. Its components are:

- a dynamics KL that trains the prior toward a stopped-gradient posterior;
- a representation KL that trains the posterior toward a stopped-gradient
prior;
- free nats and optional uniform mixing for the categorical distributions;
- an L1 or L2 reconstruction loss in symlog space;
- a reward loss using symlog-spaced two-hot bins, or symlog MSE; and
- an optional binary continuation loss.

`kl_mode="separate"` exposes the dynamics and representation KL terms
separately, as used by the full example. `kl_mode="balanced"` provides the
combined balanced-KL form.

### Actor

[`DreamerV3ActorLoss`](generated/torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss) starts from posterior states
produced by the world model and rolls the actor through a
[`DreamerEnv`](generated/torchrl.envs.model_based.dreamer.DreamerEnv.html#torchrl.envs.model_based.dreamer.DreamerEnv). It computes lambda
returns from predicted rewards, values, and optional continuation
probabilities. It supports:

- TD(0), TD(1), and TD(lambda) return estimators;
- REINFORCE with a stopped-gradient advantage, or reparameterization gradients
for suitable continuous policies;
- an entropy bonus;
- cumulative discount/continuation weighting; and
- EMA percentile-range normalization of REINFORCE returns.

### Critic and slow critic

[`DreamerV3ValueLoss`](generated/torchrl.objectives.DreamerV3ValueLoss.html#torchrl.objectives.DreamerV3ValueLoss) fits the online critic to the
lambda returns produced by the actor loss. The critic can use symlog MSE or a
distributional two-hot cross-entropy loss.

Setting `slow_critic_regularization` to a positive value creates target
critic parameters inside the value loss. The slow critic is a soft-updated
copy of the online critic:

\[\theta_{\mathrm{slow}} \leftarrow
(1 - \tau)\,\theta_{\mathrm{slow}} +
\tau\,\theta_{\mathrm{online}}.\]

The slow critic's stopped-gradient prediction is an additional target for the
online critic. In the current TorchRL objective, the online critic still
provides the bootstrap values used to form imagined lambda returns; the slow
critic regularizes critic learning rather than replacing that bootstrap.

Target updates are deliberately external to the loss. Associate a
`SoftUpdate` with the value loss and call it after
each critic optimizer step:

```
from torchrl.objectives import DreamerV3ValueLoss
from torchrl.objectives.utils import SoftUpdate

value_loss = DreamerV3ValueLoss(
 value_model,
 value_loss="two_hot",
 actor_loss=actor_loss,
 slow_critic_regularization=1.0,
)
slow_critic_updater = SoftUpdate(value_loss, tau=0.02)

# After loss.backward() and optimizer.step():
slow_critic_updater.step()
```

### Replay critic loss

The author-maintained JAX implementation also fits the critic on the real
replay sequences, not only on imagined trajectories.
[`replay_value_loss()`](generated/torchrl.objectives.DreamerV3ValueLoss.html#torchrl.objectives.DreamerV3ValueLoss.replay_value_loss) computes that
term. Its return at each replay state uses the following replay reward and
bootstraps from the first imagined lambda return of the next state, so the
critic is fitted on real replay states as well as imagined states. The method
reads its
`reward`, `done`, `terminated` and `bootstrap` entries through
`tensor_keys`, so
[`set_keys()`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.set_keys) can redirect them:

```
value_loss.set_keys(bootstrap="first_imagined_return")
replay_td = value_loss.replay_value_loss(replay_features)
loss = replay_td["loss_replay_value"]
```

Because the input features stay attached, this term also trains the RSSM
representation when the world-model loss returns live features.

## Optimization and training loop

The loss modules do not create optimizers. This keeps optimizer ownership and
the update schedule explicit. A typical update cycle is:

1. Sample contiguous real transition sequences from replay.
2. Update the world model on KL, reconstruction, reward, and continuation
losses.
3. Detach posterior states from the real sequence and use them as imagination
starting points.
4. Update the actor on imagined lambda returns.
5. Update the online critic on those same detached returns.
6. Soft-update the slow critic.

The runnable `sota-implementations/dreamer_v3` example uses a single optimizer
over the world model, actor and critic parameters, reproducing the current JAX
implementation's adaptive gradient clipping, LaProp-style RMS scaling followed
by momentum, and warmup chain.
Those choices belong to the training recipe rather than the loss API, so users
can substitute another optimizer or schedule without changing the objectives.

## API map

| Component | Purpose |
| --- | --- |
| `RSSMPriorV3` | Categorical latent dynamics and deterministic recurrent update. |
| `RSSMPosteriorV3` | Observation-conditioned categorical representation model. |
| `RSSMRolloutV3` | Sequential prior/posterior filtering over replayed trajectories. |
| [`DreamerV3MLP`](generated/torchrl.modules.DreamerV3MLP.html#torchrl.modules.DreamerV3MLP) | RMS-normalized MLP building block. |
| [`SymExpTwoHot`](generated/torchrl.modules.SymExpTwoHot.html#torchrl.modules.SymExpTwoHot) | Symlog-spaced categorical scalar encoder, decoder, and loss helper. |
| [`DreamerV3ModelLoss`](generated/torchrl.objectives.DreamerV3ModelLoss.html#torchrl.objectives.DreamerV3ModelLoss) | World-model objective. |
| [`DreamerV3ActorLoss`](generated/torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss) | Latent-imagination actor objective and lambda-return construction. |
| [`DreamerV3ValueLoss`](generated/torchrl.objectives.DreamerV3ValueLoss.html#torchrl.objectives.DreamerV3ValueLoss) | Online and slow-critic objective. |
| `SoftUpdate` | External Polyak update for the slow critic. |
| [`symlog()`](generated/torchrl.objectives.symlog.html#torchrl.objectives.symlog), [`symexp()`](generated/torchrl.objectives.symexp.html#torchrl.objectives.symexp), and two-hot helpers | Scale-robust scalar transformations for custom heads and losses. |

For a complete training setup, see the
[DreamerV3 example](https://github.com/pytorch/rl/tree/main/sota-implementations/dreamer_v3).