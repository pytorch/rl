# HIL-SERL in simulation

SAC with demos, human corrections and a learned success reward, following
Luo et al., "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
Reinforcement Learning" ([arXiv:2410.21845](https://arxiv.org/abs/2410.21845)).

Robosuite Lift stands in for the robot.

- Corrections: a scripted controller plays the human. It takes over when the
  policy gets stuck or wanders off, for up to 30 steps, with some noise.
- Replay: one demo buffer and one online buffer. Each batch is half from each,
  and corrected steps are stored in both.
- Reward: the simulator's success, or a success classifier. Whichever is used
  also decides when an episode ends. The simulator's success is always stored
  for evaluation, and the loss never reads it.
- SAC: settings from the authors' code (critic updated twice per actor update,
  discount 0.97, batch 256), with 10 critics, which keeps the critic stable
  when the human takes over often.

## Setup

```bash
pip install robosuite h5py "mujoco<3.3.6"
export MUJOCO_GL=disabled OMP_NUM_THREADS=1
```

The robomimic Lift demos (low_dim_v15.hdf5) were recorded with robosuite 1.5
at 20 Hz. The script checks that the simulator reproduces them before
training. Without a demo file, demos are recorded with the scripted
controller instead.

## Run

```bash
python hil_serl.py demos.path=/path/to/lift/ph/low_dim_v15.hdf5

python train_reward_classifier.py classifier.out=classifier.pt
python hil_serl.py demos.path=... reward.source=classifier reward.classifier_path=classifier.pt
```

The classifier script labels Lift states with the simulator, keeps 200
successes and 1000 failures, and saves the held-out error rates next to the
model.

The logs include the true evaluation success, how often the human takes over,
and the total time the human spent correcting.
