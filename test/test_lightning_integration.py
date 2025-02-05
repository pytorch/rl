"""Tests for Lightning integration."""

import lightning.pytorch as pl
import pytest
from lightning.pytorch.loggers import CSVLogger

from torchrl.trainers.ppo import PPOPendulum


def test_example_ppo_pl() -> None:
    """Tray to run the example from here,
    to make sure it is tested."""
    import os, sys

    sys.path.append(os.path.join("examples", "lightning"))
    from train_ppo_on_pendulum_with_lightning import main

    main()


def test_ppo() -> None:
    """Test PPO on InvertedDoublePendulum."""
    frame_skip = 1
    frames_per_batch = frame_skip * 5
    total_frames = 100
    model = PPOPendulum(
        frame_skip=frame_skip,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        n_mlp_layers=4,
        use_checkpoint_callback=True,
    )
    # Rollout
    rollout = model.env.rollout(3)
    print(f"Rollout of three steps: {rollout}")
    print(f"Shape of the rollout TensorDict: {rollout.batch_size}")
    print(f"Env reset: {model.env.reset()}")
    print(f"Running policy: {model.policy_module(model.env.reset())}")
    # Collector
    model.setup()
    collector = model.train_dataloader()
    for _, tensordict_data in enumerate(collector):
        print(f"Tensordict data:\n{tensordict_data}")
        batch_size = int(tensordict_data.batch_size[0])
        rollout_size = int(tensordict_data.batch_size[1])
        assert rollout_size == int(frames_per_batch // frame_skip)
        assert batch_size == model.num_envs
        break
    # Training
    max_steps = 2
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=max_steps,
        val_check_interval=2,
        log_every_n_steps=1,
        logger=CSVLogger(
            save_dir="pytest_artifacts",
            name=model.__class__.__name__,
        ),
    )
    trainer.fit(model)
    # Test we stopped quite because the max number of steps was reached
    assert max_steps >= trainer.global_step


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
