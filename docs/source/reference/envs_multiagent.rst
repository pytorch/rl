.. currentmodule:: torchrl.envs

.. _MARL-environment-API:

Multi-agent Environments
========================

TorchRL supports multi-agent learning out-of-the-box.
*The same classes used in a single-agent learning pipeline can be seamlessly used in multi-agent contexts,
without any modification or dedicated multi-agent infrastructure.*

In this view, environments play a core role for multi-agent. In multi-agent environments,
many decision-making agents act in a shared world.
Agents can observe different things, act in different ways and also be rewarded differently.
Therefore, many paradigms exist to model multi-agent environments (DecPODPs, Markov Games).
Some of the main differences between these paradigms include:

- **observation** can be per-agent and also have some shared components
- **reward** can be per-agent or shared
- **done** (and ``"truncated"`` or ``"terminated"``) can be per-agent or shared.

TorchRL accommodates all these possible paradigms thanks to its :class:`tensordict.TensorDict` data carrier.
Per-agent keys live in nested **group** tensordicts. Each group has an extra agent dimension
so that data that differs across agents can be stacked. Shared keys stay at the root,
as in single-agent cases.

The simplest layout uses a single group named ``"agents"`` (the VMAS default shown
below). When agents belong to several groups -- for example two competing teams --
each group is its own nested tensordict. See :ref:`MARL-multiple-groups` for the
general contract a native :class:`~torchrl.envs.EnvBase` ``_step()`` must implement.

Let's look at the single-group case first. For this example we are going to use
`VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`_, a multi-robot task simulator also
based on PyTorch, which runs parallel batched simulation on device.

We can create a VMAS environment and look at what the output from a random step looks like:

.. code-block::
   :caption: Example of multi-agent step tensordict

        >>> from torchrl.envs.libs.vmas import VmasEnv
        >>> env = VmasEnv("balance", num_envs=3, n_agents=5)
        >>> td = env.rand_step()
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([3, 5, 2]))},
                    batch_size=torch.Size([3, 5])),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                info: TensorDict(
                                    fields={
                                        ground_rew: Tensor(shape=torch.Size([3, 5, 1])),
                                        pos_rew: Tensor(shape=torch.Size([3, 5, 1]))},
                                    batch_size=torch.Size([3, 5])),
                                observation: Tensor(shape=torch.Size([3, 5, 16])),
                                reward: Tensor(shape=torch.Size([3, 5, 1]))},
                            batch_size=torch.Size([3, 5])),
                        done: Tensor(shape=torch.Size([3, 1]))},
                    batch_size=torch.Size([3]))},
            batch_size=torch.Size([3]))

We can observe that *keys that are shared by all agents*, such as **done** are present in the root tensordict with
batch size `(num_envs,)`, which represents the number of environments simulated.

On the other hand, *keys that are different between agents*, such as **action**, **reward**, **observation**,
and **info** are present in the nested "agents" tensordict with batch size `(num_envs, n_agents)`,
which represents the additional agent dimension.

Multi-agent tensor specs will follow the same style as in tensordicts.
Specs relating to values that vary between agents will need to be nested in the
group entry (here, ``"agents"``).

Here is an example of how specs can be created in a multi-agent environment where
only the done flag is shared across agents (as in VMAS):

.. code-block::
   :caption: Example of multi-agent spec creation

        >>> action_specs = []
        >>> observation_specs = []
        >>> reward_specs = []
        >>> info_specs = []
        >>> for i in range(env.n_agents):
        ...    action_specs.append(agent_i_action_spec)
        ...    reward_specs.append(agent_i_reward_spec)
        ...    observation_specs.append(agent_i_observation_spec)
        >>> env.action_spec = Composite(
        ...    {
        ...        "agents": Composite(
        ...            {"action": torch.stack(action_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.reward_spec = Composite(
        ...    {
        ...        "agents": Composite(
        ...            {"reward": torch.stack(reward_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.observation_spec = Composite(
        ...    {
        ...        "agents": Composite(
        ...            {"observation": torch.stack(observation_specs)}, shape=(env.n_agents,)
        ...        )
        ...    }
        ...)
        >>> env.done_spec = Categorical(
        ...    n=2,
        ...    shape=torch.Size((1,)),
        ...    dtype=torch.bool,
        ... )

As you can see, it is very simple! Per-agent keys will have the nested composite spec and shared keys will follow
single agent standards.

.. note::
  Since reward, done and action keys may have the additional group prefix (e.g., ``("agents", "action")``),
  the default keys used in the arguments of other TorchRL components (e.g. ``"action"``) will not match exactly.
  Therefore, TorchRL provides the ``env.action_key``, ``env.reward_key``, and ``env.done_key`` attributes,
  which will automatically point to the right key to use. Make sure you pass these attributes to the various
  components in TorchRL to inform them of the right key (e.g., the ``loss.set_keys()`` function).
  When there is more than one action, reward or done key (as with multiple groups),
  use the plural ``env.action_keys``, ``env.reward_keys`` and ``env.done_keys`` instead --
  the singular attributes raise ``KeyError``.

.. note::
  TorchRL abstracts these nested specs away for ease of use.
  This means that accessing `env.reward_spec` will always return the leaf
  spec if the accessed spec is Composite. Therefore, if in the example above
  we run `env.reward_spec` after env creation, we would get the same output as `torch.stack(reward_specs)}`.
  To get the full composite spec with the "agents" key, you can run
  `env.output_spec["full_reward_spec"]`. The same is valid for action and done specs.
  Note that `env.reward_spec == env.output_spec["full_reward_spec"][env.reward_key]`.


.. _MARL-multiple-groups:

Multiple agent groups
---------------------

``"agents"`` is just a group name, not a required key. A native
:class:`~torchrl.envs.EnvBase` can expose any number of groups -- ``"red"`` /
``"blue"``, or the ``"agents"`` / ``"adversaries"`` teams of *simple_tag* --
as long as each group is a nested tensordict whose last batch dimension
indexes the agents in that group.

Agents that share a policy (and typically a spec) belong in the same group so
their tensors can be stacked. Heterogeneous or competing teams go in separate
groups and are processed by separate modules. The grouping is the
``group_map`` dict ``{group_name: [agent_name, ...]}``; see
:class:`~torchrl.envs.MarlGroupMapType` and
:func:`~torchrl.envs.check_marl_grouping`.

The canonical trained example of this layout is the
:doc:`competitive MADDPG tutorial <../tutorials/multiagent_competitive_ddpg>`
(in particular the *Rollout* section), which consumes the ``"agents"`` /
``"adversaries"`` groups of *simple_tag*.

What ``_step()`` must return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~torchrl.envs.EnvBase.step` is a thin wrapper: it calls the private
``_step()``, then stores that result under the ``"next"`` key of the input
tensordict. Consequently a custom ``_step()`` must:

- **read** actions from the *input* tensordict at ``(group, "action")``;
- **write** next observations, rewards and done flags into a **new**
  tensordict (out-of-place);
- **not** wrap that tensordict in ``"next"`` -- the public
  :meth:`~torchrl.envs.EnvBase.step` does that;
- **not** write actions onto the output.

``_reset()`` uses the same key tree minus rewards (there is no reward at
reset time). After a public ``step()`` the input (with actions) stays at the
root and the ``_step()`` output sits under ``"next"``.

For two groups ``"red"`` (2 agents) and ``"blue"`` (3 agents) and an
environment batch ``B``, the tensordict ``_step()`` returns looks like this:

.. code-block::
   :caption: Multi-group key tree returned by a native ``_step()``

        TensorDict(
            fields={
                red: TensorDict(
                    fields={
                        observation: Tensor(shape=torch.Size([*B, 2, obs_red])),
                        reward: Tensor(shape=torch.Size([*B, 2, 1]))},
                    batch_size=torch.Size([*B, 2])),
                blue: TensorDict(
                    fields={
                        observation: Tensor(shape=torch.Size([*B, 3, obs_blue])),
                        reward: Tensor(shape=torch.Size([*B, 3, 1]))},
                    batch_size=torch.Size([*B, 3])),
                done: Tensor(shape=torch.Size([*B, 1])),
                terminated: Tensor(shape=torch.Size([*B, 1])),
                truncated: Tensor(shape=torch.Size([*B, 1]))},
            batch_size=torch.Size([*B,]))

The corresponding public ``step()`` / ``rand_step()`` output is the input
tensordict (root ``red`` / ``blue`` entries hold ``action`` only) plus a
``next`` tensordict that is exactly the tree above.

Where each field lives
~~~~~~~~~~~~~~~~~~~~~~

- **Action.** Input of ``_step()`` only, at ``(group, "action")``, shape
  ``(*batch, n_agents_in_group, *action_shape)``. Never written by
  ``_step()``.
- **Observation.** Output of ``_step()`` and ``_reset()``, typically at
  ``(group, "observation")`` with shape
  ``(*batch, n_agents_in_group, *obs_shape)``. A shared / global observation
  (for example a global ``"state"``) is a root-level key with no extra agent
  dimension, exactly as in the single-agent case.
- **Reward.** Output of ``_step()`` only (not ``_reset()``). Per-agent or
  per-group rewards live at ``(group, "reward")`` with shape
  ``(*batch, n_agents_in_group, 1)``. A fully shared reward is a root
  ``"reward"`` with shape ``(*batch, 1)``.
- **Done / terminated / truncated** can sit at three levels, which can be
  combined:

  * **Shared (root).** ``"done"``, ``"terminated"`` and (if used)
    ``"truncated"`` at the root, shape ``(*batch, 1)``. This is the signal
    TorchRL uses to reset the environment. A native env must write at least
    this shared flag (VMAS writes only this).
  * **Per-group / per-agent, stacked.** ``(group, "done")`` (and the
    ``terminated`` / ``truncated`` siblings) with shape
    ``(*batch, n_agents_in_group, 1)``. PettingZoo writes these *and*
    aggregates them into the root flag (``any`` or ``all``, controlled by
    ``done_on_any``).
  * **One group per agent.** With
    :attr:`~torchrl.envs.MarlGroupMapType.ONE_GROUP_PER_AGENT` each group
    has a single agent, so ``(agent_name, "done")`` has shape
    ``(*batch, 1)`` and there is no extra agent dimension.

Specs must mirror the nesting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every leaf that ``_step()`` (or ``_reset()``) writes must have a spec at the
same nested key, with the same leading shape. The group-level
:class:`~torchrl.data.Composite` carries shape ``(*batch, n_agents_in_group)``;
root-level (shared) specs carry shape ``(*batch,)`` or ``(*batch, 1)``.

.. code-block::
   :caption: Specs for a two-group env (shared done, per-group obs/reward/action)

        >>> from torchrl.data import Bounded, Categorical, Composite, Unbounded
        >>> n_red, n_blue = 2, 3
        >>> obs_dim, act_dim = 8, 2
        >>> bs = env.batch_size
        >>> env.action_spec = Composite(
        ...     {
        ...         "red": Composite(
        ...             {"action": Bounded(-1, 1, shape=(*bs, n_red, act_dim))},
        ...             shape=(*bs, n_red),
        ...         ),
        ...         "blue": Composite(
        ...             {"action": Bounded(-1, 1, shape=(*bs, n_blue, act_dim))},
        ...             shape=(*bs, n_blue),
        ...         ),
        ...     },
        ...     shape=bs,
        ... )
        >>> env.observation_spec = Composite(
        ...     {
        ...         "red": Composite(
        ...             {"observation": Unbounded(shape=(*bs, n_red, obs_dim))},
        ...             shape=(*bs, n_red),
        ...         ),
        ...         "blue": Composite(
        ...             {"observation": Unbounded(shape=(*bs, n_blue, obs_dim))},
        ...             shape=(*bs, n_blue),
        ...         ),
        ...     },
        ...     shape=bs,
        ... )
        >>> env.reward_spec = Composite(
        ...     {
        ...         "red": Composite(
        ...             {"reward": Unbounded(shape=(*bs, n_red, 1))},
        ...             shape=(*bs, n_red),
        ...         ),
        ...         "blue": Composite(
        ...             {"reward": Unbounded(shape=(*bs, n_blue, 1))},
        ...             shape=(*bs, n_blue),
        ...         ),
        ...     },
        ...     shape=bs,
        ... )
        >>> env.done_spec = Composite(
        ...     {
        ...         "done": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
        ...         "terminated": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
        ...         "truncated": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
        ...     },
        ...     shape=bs,
        ... )

To also expose per-agent done flags, nest them in each group the same way
as ``reward``, and keep the root flags as the reset signal:

.. code-block::

        >>> env.done_spec["red"] = Composite(
        ...     {
        ...         "done": Categorical(n=2, shape=(*bs, n_red, 1), dtype=torch.bool),
        ...         "terminated": Categorical(n=2, shape=(*bs, n_red, 1), dtype=torch.bool),
        ...         "truncated": Categorical(n=2, shape=(*bs, n_red, 1), dtype=torch.bool),
        ...     },
        ...     shape=(*bs, n_red),
        ... )

Call :func:`~torchrl.envs.check_env_specs` after construction: it runs a
short rollout and checks that every key ``_step()`` / ``_reset()`` writes
matches the spec tree.

Native ``_step()`` sketch
~~~~~~~~~~~~~~~~~~~~~~~~~

The following is a copy-paste starting point for a two-team
:class:`~torchrl.envs.EnvBase`. It is not a PettingZoo or VMAS wrapper:
actions are read from the group tensordicts and the returned tensordict
follows the tree above. ``_initial_obs`` / ``_apply_dynamics`` are the
only placeholders -- replace them with a real reset distribution and
transition. In ``_step``, take devices from the action tensors.

.. code-block:: python
   :caption: Native multi-group ``EnvBase._step()``

    import torch
    from tensordict import TensorDict, TensorDictBase
    from torchrl.data import Bounded, Categorical, Composite, Unbounded
    from torchrl.envs import EnvBase, check_env_specs


    class TwoTeamEnv(EnvBase):
        """Minimal two-group env. Replace the two helpers with real physics."""

        def __init__(self, n_red=2, n_blue=3, obs_dim=8, act_dim=2, **kwargs):
            super().__init__(**kwargs)
            self.n_red = n_red
            self.n_blue = n_blue
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.group_map = {
                "red": [f"red_{i}" for i in range(n_red)],
                "blue": [f"blue_{i}" for i in range(n_blue)],
            }
            bs = self.batch_size
            self.action_spec = Composite(
                {
                    "red": Composite(
                        {"action": Bounded(-1, 1, shape=(*bs, n_red, act_dim))},
                        shape=(*bs, n_red),
                    ),
                    "blue": Composite(
                        {"action": Bounded(-1, 1, shape=(*bs, n_blue, act_dim))},
                        shape=(*bs, n_blue),
                    ),
                },
                shape=bs,
            )
            self.observation_spec = Composite(
                {
                    "red": Composite(
                        {"observation": Unbounded(shape=(*bs, n_red, obs_dim))},
                        shape=(*bs, n_red),
                    ),
                    "blue": Composite(
                        {"observation": Unbounded(shape=(*bs, n_blue, obs_dim))},
                        shape=(*bs, n_blue),
                    ),
                },
                shape=bs,
            )
            self.reward_spec = Composite(
                {
                    "red": Composite(
                        {"reward": Unbounded(shape=(*bs, n_red, 1))},
                        shape=(*bs, n_red),
                    ),
                    "blue": Composite(
                        {"reward": Unbounded(shape=(*bs, n_blue, 1))},
                        shape=(*bs, n_blue),
                    ),
                },
                shape=bs,
            )
            self.done_spec = Composite(
                {
                    "done": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
                    "terminated": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
                    "truncated": Categorical(n=2, shape=(*bs, 1), dtype=torch.bool),
                },
                shape=bs,
            )

        def _set_seed(self, seed):
            if seed is not None:
                torch.manual_seed(seed)

        def _initial_obs(self):
            # Replace with a real reset distribution.
            red_obs = torch.zeros(
                *self.batch_size, self.n_red, self.obs_dim, device=self.device
            )
            blue_obs = torch.zeros(
                *self.batch_size, self.n_blue, self.obs_dim, device=self.device
            )
            return red_obs, blue_obs

        def _apply_dynamics(self, red_action, blue_action):
            # Replace with a real transition. Devices come from the actions.
            red_obs = red_action.new_zeros(*self.batch_size, self.n_red, self.obs_dim)
            blue_obs = blue_action.new_zeros(*self.batch_size, self.n_blue, self.obs_dim)
            red_rew = red_action.new_zeros(*self.batch_size, self.n_red, 1)
            blue_rew = blue_action.new_zeros(*self.batch_size, self.n_blue, 1)
            done = red_action.new_zeros(*self.batch_size, 1, dtype=torch.bool)
            return red_obs, blue_obs, red_rew, blue_rew, done

        def _reset(self, tensordict):
            red_obs, blue_obs = self._initial_obs()
            done = red_obs.new_zeros(*self.batch_size, 1, dtype=torch.bool)
            return TensorDict(
                {
                    "red": TensorDict(
                        {"observation": red_obs},
                        batch_size=(*self.batch_size, self.n_red),
                    ),
                    "blue": TensorDict(
                        {"observation": blue_obs},
                        batch_size=(*self.batch_size, self.n_blue),
                    ),
                    "done": done,
                    "terminated": done.clone(),
                    "truncated": done.clone(),
                },
                batch_size=self.batch_size,
            )

        def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
            # Actions live on the *input* tensordict, never on the output.
            red_action = tensordict["red", "action"]    # (*batch, n_red, act_dim)
            blue_action = tensordict["blue", "action"]  # (*batch, n_blue, act_dim)

            red_obs, blue_obs, red_rew, blue_rew, done = self._apply_dynamics(
                red_action, blue_action
            )
            # Optional per-agent flags go on this returned tensordict,
            # e.g. out["red", "done"] with shape (*batch, n_red, 1).

            return TensorDict(
                {
                    "red": TensorDict(
                        {"observation": red_obs, "reward": red_rew},
                        batch_size=(*self.batch_size, self.n_red),
                    ),
                    "blue": TensorDict(
                        {"observation": blue_obs, "reward": blue_rew},
                        batch_size=(*self.batch_size, self.n_blue),
                    ),
                    "done": done,
                    "terminated": done.clone(),
                    "truncated": done.new_zeros(done.shape, dtype=torch.bool),
                },
                batch_size=self.batch_size,
            )


    env = TwoTeamEnv()
    check_env_specs(env)
    # env.action_keys == [("blue", "action"), ("red", "action")]
    # env.reward_keys == [("blue", "reward"), ("red", "reward")]
    # env.done_keys   == ["done", "terminated", "truncated"]

Collectors and replay buffers transport these nested keys automatically; they
do not take ``env.action_keys``, ``env.reward_keys`` or ``env.done_keys`` as
configuration arguments. Compose the group policies so that each reads and
writes its group's nested keys, then use one loss per group. Configure each loss
with the individual :class:`~tensordict.NestedKey` values accepted by its
``set_keys()`` method. For example, a loss for the red group in the environment
above can use ``reward=("red", "reward")`` together with the root
``done="done"`` and ``terminated="terminated"`` keys. The
:doc:`competitive MADDPG tutorial <../tutorials/multiagent_competitive_ddpg>`
shows a full two-group training loop.


.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    MarlGroupMapType
    check_marl_grouping
