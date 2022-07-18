# Examples

We provide examples to train the following algorithms:
- [DQN](dqn/dqn.py)
- [DDPG](ddpg/ddpg.py)
- [SAC](sac/sac.py)
- [REDQ](redq/redq.py)
- [PPO](ppo/ppo.py)

To run these examples, make sure you have installed hydra:
```
pip install hydra-code
```

Then, go to the directory that interests you and run
```
python sac.py
```
or similar. Hyperparameters can be easily changed by providing the arguments to hydra:
```
python sac frames_per_batch=63
```
