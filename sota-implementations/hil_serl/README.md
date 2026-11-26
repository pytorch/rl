# HIL-SERL in simulation

SAC with demos, human corrections and a learned success reward, following
Luo et al., "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
Reinforcement Learning" ([arXiv:2410.21845](https://arxiv.org/abs/2410.21845)).

Robosuite Lift stands in for the robot.

- Corrections: a scripted controller plays the human. It takes over when the
  policy gets stuck or wanders off, for up to 30 steps, with some noise.
  `intervention.reaction_delay` makes it wait a few steps before taking over.
- Replay: one demo buffer and one online buffer. Each batch is half from each,
  and corrected steps are stored in both.
- Reward: the simulator's success, or a success classifier. Whichever is used
  also decides when an episode ends, unless `reward.end_episode_on_success`
  is false: then episodes run to `env.max_steps` and successes are only
  marked terminal in the stored data, so a false success from the classifier
  does not cut the episode short. The simulator's success is always stored
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

The policy soon finds states the classifier wrongly calls successes. As in
HIL-SERL, save them, add them to the classifier's failures and retrain it,
then train the policy again; repeat for a few rounds if needed:

```bash
python hil_serl.py ... reward.classifier_path=classifier.pt reward.false_positives_out=fp_0
python train_reward_classifier.py classifier.out=classifier_1.pt "classifier.extra_negatives=[fp_0]"
python hil_serl.py ... reward.classifier_path=classifier_1.pt reward.false_positives_out=fp_1
python train_reward_classifier.py classifier.out=classifier_2.pt "classifier.extra_negatives=[fp_0,fp_1]"
```

On a robot a person reviews the claimed successes; here the simulator's
success flag does. The logs count the claims and the false ones.

The logs include the true evaluation success, how often the human takes over,
and the total time the human spent correcting.
