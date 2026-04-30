# Multi-Agent Trainer Examples

These examples run the trainer-based versions of the VMAS multi-agent baselines.

```bash
python sota-implementations/multiagent_trainer/train.py --config-name maddpg
python sota-implementations/multiagent_trainer/train.py --config-name iddpg
python sota-implementations/multiagent_trainer/train.py --config-name masac
python sota-implementations/multiagent_trainer/train.py --config-name isac
python sota-implementations/multiagent_trainer/train.py --config-name mappo
python sota-implementations/multiagent_trainer/train.py --config-name ippo
```

The centralized/decentralized critic variants are split into separate config files for clarity.
