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
In particular, in multi-agent environments, per-agent keys will be carried in a nested "agents" TensorDict.
This TensorDict will have the additional agent dimension and thus group data that is different for each agent.
The shared keys, on the other hand, will be kept in the first level, as in single-agent cases.

Let's look at an example to understand this better. For this example we are going to use
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
Specs relating to values that vary between agents will need to be nested in the "agents" entry.

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
  Since reward, done and action keys may have the additional "agent" prefix (e.g., `("agents","action")`),
  the default keys used in the arguments of other TorchRL components (e.g. "action") will not match exactly.
  Therefore, TorchRL provides the `env.action_key`, `env.reward_key`, and `env.done_key` attributes,
  which will automatically point to the right key to use. Make sure you pass these attributes to the various
  components in TorchRL to inform them of the right key (e.g., the `loss.set_keys()` function).

.. note::
  TorchRL abstracts these nested specs away for ease of use.
  This means that accessing `env.reward_spec` will always return the leaf
  spec if the accessed spec is Composite. Therefore, if in the example above
  we run `env.reward_spec` after env creation, we would get the same output as `torch.stack(reward_specs)}`.
  To get the full composite spec with the "agents" key, you can run
  `env.output_spec["full_reward_spec"]`. The same is valid for action and done specs.
  Note that `env.reward_spec == env.output_spec["full_reward_spec"][env.reward_key]`.


.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    MarlGroupMapType
    check_marl_grouping
