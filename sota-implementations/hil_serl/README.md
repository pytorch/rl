# HIL-SERL in simulation

SAC with demos, human corrections and a learned success reward, following
Luo et al., "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
Reinforcement Learning" ([arXiv:2410.21845](https://arxiv.org/abs/2410.21845)).
Robosuite Lift stands in for the robot and a scripted controller for the
person.

## How it works

- Corrections: the scripted controller takes over when the policy stalls or
  leaves the workspace, for up to `intervention.max_takeover` steps, with
  some noise. `intervention.reaction_delay` makes it wait a few steps first.
- Replay: one demo buffer and one online buffer. Each batch takes
  `replay.demo_fraction` of its samples (half by default) from the demo
  buffer, and corrected steps are stored in both.
- Reward: the simulator's success (`reward.source=oracle`) or a success
  classifier (`reward.source=classifier`). A success also ends the episode,
  unless `reward.end_episode_on_success=false`: then episodes run to
  `env.max_steps` and successes are only marked terminal in the stored data.
  The simulator's success is always logged for evaluation, and the loss never
  reads it in classifier mode.
- SAC: settings from the authors' code (discount 0.97, batch 256, critic
  updated twice per actor update, no entropy term in the critic's target),
  with 10 critics. With the entropy term (`optim.backup_entropy=true`) or with
  fewer critics, training often collapses on this task.

## Setup

```bash
pip install robosuite h5py "mujoco<3.3.6"
export MUJOCO_GL=disabled OMP_NUM_THREADS=1
```

The robomimic Lift demos (low_dim_v15.hdf5) were recorded with robosuite 1.5
at 20 Hz. The script checks that the simulator reproduces them before
training. Without a demo file, demos are recorded with the scripted
controller instead.

On a Mac, pass `network.device=cpu`: training on the MPS device learns much
worse, for reasons not yet known.

## Run

```bash
python hil_serl.py demos.path=/path/to/lift/ph/low_dim_v15.hdf5
```

The update is compiled with CUDA graphs by default, which about halves the
run time; `compile.compile=false` turns it off. The logs include the true
evaluation success, how often the human takes over and the total time the
human spent correcting.

## Learned reward

```bash
python train_reward_classifier.py classifier.out=classifier.pt
python hil_serl.py demos.path=... reward.source=classifier reward.classifier_path=classifier.pt
```

The classifier script labels Lift states with the simulator, keeps 200
successes and 1000 failures, and saves the held-out error rates next to the
model.

The policy soon finds states the classifier wrongly calls successes. As in
HIL-SERL, save them, add them to the classifier's failures, retrain it and
train the policy again, for a few rounds:

```bash
python hil_serl.py ... reward.classifier_path=classifier.pt reward.false_positives_out=fp_0
python train_reward_classifier.py classifier.out=classifier_1.pt "classifier.extra_negatives=[fp_0]"
python hil_serl.py ... reward.classifier_path=classifier_1.pt reward.false_positives_out=fp_1
python train_reward_classifier.py classifier.out=classifier_2.pt "classifier.extra_negatives=[fp_0,fp_1]"
```

On a robot a person reviews the claimed successes; here the simulator's
success flag does. The logs count the claims and the false ones.

## Expected results

Lift, 40k frames, final success over 30 evaluation episodes, mean of 10
seeds:

| Reward | Success |
| --- | --- |
| Simulator | 77% |
| Classifier, no retraining | 1% |
| Classifier, after 3 rounds of retraining | 34% |

The first classifier is wrong on under 1% of held-out states, yet the policy
soon learns to close the gripper near the cube without lifting it, which the
classifier calls a success. Three rounds of retraining add about 3,000
labelled states per seed.
