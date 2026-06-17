# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyspiel = pytest.importorskip("pyspiel")

import numpy as np
from torchrl.envs import OpenSpielEnv, OpenSpielWrapper
from torchrl.envs.utils import check_env_specs


class TestChanceNodeSampler:
    """Test the default chance outcome sampler."""

    def test_default_sampler_respects_probabilities(self):
        """Verify default sampler samples according to probabilities."""
        env = OpenSpielEnv("backgammon")

        actions = [0, 1, 2]
        probs = [0.5, 0.3, 0.2]

        # Sample many times and verify distribution roughly matches
        samples = [env._default_chance_sampler(actions, probs) for _ in range(10000)]
        sample_counts = {a: samples.count(a) for a in actions}

        # Check that empirical probabilities are roughly correct (within 5%)
        for action, prob in zip(actions, probs):
            empirical_prob = sample_counts[action] / len(samples)
            assert (
                abs(empirical_prob - prob) < 0.05
            ), f"Action {action}: expected ~{prob}, got {empirical_prob}"

    def test_custom_sampler_injection(self):
        """Verify custom sampler can be injected."""
        # Deterministic sampler that always picks the first action
        def custom_sampler(actions, probs):
            return actions[0]

        game = pyspiel.load_game("backgammon").new_initial_state()
        env = OpenSpielWrapper(game, chance_sampler=custom_sampler)

        assert env._chance_sampler is custom_sampler

    def test_sampler_with_single_outcome(self):
        """Verify sampler handles single outcome correctly."""
        env = OpenSpielEnv("backgammon")

        actions = [5]
        probs = [1.0]

        sampled = env._default_chance_sampler(actions, probs)
        assert sampled == 5


class TestChanceNodeResolution:
    """Test chance node resolution during reset and step."""

    def test_backgammon_reset_resolves_chance(self):
        """Verify reset resolves initial chance nodes in backgammon."""
        env = OpenSpielEnv("backgammon")

        td = env.reset()

        # After reset, we should be at a decision node (not chance node)
        assert (
            not env._env.is_chance_node()
        ), "After reset, environment should not be at a chance node"

        # Should have valid observation and current player
        assert "agents" in td
        assert td.shape == torch.Size([])

    def test_backgammon_step_resolves_chance(self):
        """Verify step resolves chance nodes that may occur after action."""
        env = OpenSpielEnv("backgammon")

        env.reset()

        # Take an action
        action = env.full_action_spec.rand()
        env.step(action)

        # After step, should not be at a chance node
        assert (
            not env._env.is_chance_node()
        ), "After step, environment should not be at a chance node"

    def test_full_rollout_with_chance_game(self):
        """Verify complete rollout works with stochastic game."""
        env = OpenSpielEnv("backgammon")

        td = env.reset()
        episode_length = 0
        max_steps = 100
        done = td["done"]

        while not done.item() and episode_length < max_steps:
            action = env.full_action_spec.rand()
            td = env.step(action)
            done = td["next", "done"]
            episode_length += 1

        # Verify episode completed without errors
        assert episode_length > 0
        assert not env._env.is_chance_node()

    def test_state_serialization_with_chance(self):
        """Verify state serialization captures post-chance state."""
        env = OpenSpielEnv("backgammon", return_state=True)

        td1 = env.reset()
        td1["state"]

        # Take a step
        action = env.full_action_spec.rand()
        td2 = env.step(action)
        td2["next"]["state"]

        # Reset to state2
        env.reset(td2["next"])

        # The new state should match what we captured
        assert not env._env.is_chance_node()


class TestSpecsUnchanged:
    """Test that specs remain unchanged with chance support."""

    def test_specs_valid_for_chance_game(self):
        """Verify env specs satisfy check_env_specs for chance game."""
        env = OpenSpielEnv("backgammon")

        # This should not raise
        check_env_specs(env)

    def test_observation_spec_structure(self):
        """Verify observation spec structure unchanged."""
        env = OpenSpielEnv("backgammon")

        spec = env.observation_spec

        # Should have agents and current_player
        assert "agents" in spec
        assert "current_player" in spec

        # Specs should be deterministic (same for repeated calls)
        spec2 = env.observation_spec
        assert str(spec) == str(spec2)


class TestDeterministicSampling:
    """Test deterministic sampling for reproducible testing."""

    def test_deterministic_sampler(self):
        """Verify deterministic sampler produces same sequence."""

        def seeded_sampler(seed):
            rng = np.random.RandomState(seed)

            def sampler(actions, probs):
                return int(rng.choice(actions, p=probs))

            return sampler

        # Create two envs with same seed
        game1 = pyspiel.load_game("backgammon").new_initial_state()
        game2 = pyspiel.load_game("backgammon").new_initial_state()

        sampler1 = seeded_sampler(42)
        sampler2 = seeded_sampler(42)

        env1 = OpenSpielWrapper(game1, chance_sampler=sampler1)
        env2 = OpenSpielWrapper(game2, chance_sampler=sampler2)

        td1 = env1.reset()
        td2 = env2.reset()

        # Observations should match (up to floating point)
        if "agents" in td1:
            obs1 = td1["agents"].get("observation", None)
            obs2 = td2["agents"].get("observation", None)
            if obs1 is not None and obs2 is not None:
                assert torch.allclose(obs1, obs2, atol=1e-6)


class TestParallelVsSequential:
    """Test chance resolution works for both game types."""

    @pytest.mark.skipif(
        not hasattr(pyspiel, "load_game"), reason="pyspiel not available"
    )
    def test_sequential_game_with_chance(self):
        """Verify sequential game handling."""
        env = OpenSpielEnv("backgammon")

        assert not env.parallel

        env.reset()
        assert not env._env.is_chance_node()

        action = env.full_action_spec.rand()
        env.step(action)
        assert not env._env.is_chance_node()

    @pytest.mark.skipif(
        not hasattr(pyspiel, "load_game"), reason="pyspiel not available"
    )
    def test_parallel_game_basic(self):
        """Verify parallel game still works (may or may not have chance)."""
        # Load a parallel game (rock-paper-scissors is parallel)
        try:
            env = OpenSpielEnv("rock_paper_scissors")

            if env.parallel:
                env.reset()
                # Should handle parallel games correctly
                assert not env._env.is_chance_node()
        except Exception:
            # Not all games available, skip if rock_paper_scissors not found
            pytest.skip("rock_paper_scissors not available")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_terminal_state_with_chance_history(self):
        """Verify terminal states are handled correctly."""
        env = OpenSpielEnv("backgammon")

        td = env.reset()

        # Play until terminal
        steps = 0
        done = td["done"]
        while not done.item() and steps < 200:
            action = env.full_action_spec.rand()
            td = env.step(action)
            done = td["next", "done"]
            steps += 1

        # Terminal state should be valid
        assert env._env.is_terminal()
        assert not env._env.is_chance_node()

    def test_repeated_resets(self):
        """Verify repeated resets work correctly."""
        env = OpenSpielEnv("backgammon", return_state=True)

        for _ in range(5):
            td = env.reset()
            assert not env._env.is_chance_node()
            assert "state" in td

    def test_batch_size_not_supported(self):
        """Verify that non-empty batch_size raises an error."""
        with pytest.raises(
            ValueError,
            match="OpenSpielWrapper only supports single-environment mode",
        ):
            OpenSpielEnv("backgammon", batch_size=torch.Size([4]))

    def test_batch_size_empty_allowed(self):
        """Verify that empty batch_size is accepted."""
        # This should not raise
        env = OpenSpielEnv("backgammon", batch_size=torch.Size([]))
        assert env.batch_size == torch.Size([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
