# Multi-Agent Trainer Examples

These examples run the trainer-based versions of the VMAS multi-agent baselines.

```bash
python sota-implementations/multiagent_trainer/train.py --config-name maddpg
python sota-implementations/multiagent_trainer/train.py --config-name iddpg
python sota-implementations/multiagent_trainer/train.py --config-name masac
python sota-implementations/multiagent_trainer/train.py --config-name isac
python sota-implementations/multiagent_trainer/train.py --config-name mappo
python sota-implementations/multiagent_trainer/train.py --config-name ippo
python sota-implementations/multiagent_trainer/train.py --config-name qmix
python sota-implementations/multiagent_trainer/train.py --config-name vdn
python sota-implementations/multiagent_trainer/train.py --config-name iql
```

The centralized/decentralized critic variants are split into separate config files for clarity.
For value-decomposition methods, ``qmix`` and ``vdn`` use mean reward aggregation across agents
