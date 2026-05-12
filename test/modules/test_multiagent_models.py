# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import re

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import MultiAgentConvNet, MultiAgentMLP, QMixer, VDNMixer
from torchrl.modules.models.multiagent import MultiAgentNetBase

from torchrl.testing import retry


class TestMultiAgent:
    def _get_mock_input_td(
        self, n_agents, n_agents_inputs, state_shape=(64, 64, 3), T=None, batch=(2,)
    ):
        if T is not None:
            batch = batch + (T,)
        obs = torch.randn(*batch, n_agents, n_agents_inputs)
        state = torch.randn(*batch, *state_shape)

        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs},
                    [*batch, n_agents],
                ),
                "state": state,
            },
            batch_size=batch,
        )
        return td

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("n_agent_inputs", [6, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_mlp(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
        )
        if n_agent_inputs is None:
            n_agent_inputs = 6
        td = self._get_mock_input_td(n_agents, n_agent_inputs, batch=batch)
        obs = td.get(("agents", "observation"))

        out = mlp(obs)
        assert out.shape == (*batch, n_agents, n_agent_outputs)
        for i in range(n_agents):
            if centralized and share_params:
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[..., i, :], out[..., j, :])

        obs[..., 0, 0] += 1
        out2 = mlp(obs)
        for i in range(n_agents):
            if centralized:
                # a modification to the input of agent 0 will impact all agents
                assert not torch.allclose(out[..., i, :], out2[..., i, :])
            elif i > 0:
                assert torch.allclose(out[..., i, :], out2[..., i, :])

        obs = (
            torch.randn(*batch, 1, n_agent_inputs)
            .expand(*batch, n_agents, n_agent_inputs)
            .clone()
        )
        out = mlp(obs)
        for i in range(n_agents):
            if share_params:
                # same input same output
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    # same input different output
                    assert not torch.allclose(out[..., i, :], out[..., j, :])
        pattern = rf"""MultiAgentMLP\(
    MLP\(
      \(0\): Linear\(in_features=\d+, out_features=32, bias=True\)
      \(1\): Tanh\(\)
      \(2\): Linear\(in_features=32, out_features=32, bias=True\)
      \(3\): Tanh\(\)
      \(4\): Linear\(in_features=32, out_features=2, bias=True\)
    \),
    n_agents={n_agents},
    share_params={share_params},
    centralized={centralized},
    agent_dim={-2}\)"""
        assert re.match(pattern, str(mlp), re.DOTALL)

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("n_agent_inputs", [6, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_mlp_init(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
        )
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                assert not isinstance(m.weight, nn.Parameter)
                assert m.weight.device == torch.device("meta")
                break
        else:
            raise RuntimeError("could not find a Linear module")
        if n_agent_inputs is None:
            n_agent_inputs = 6
        td = self._get_mock_input_td(n_agents, n_agent_inputs, batch=batch)
        obs = td.get(("agents", "observation"))
        mlp(obs)
        snet = mlp.get_stateful_net()
        assert snet is not mlp._empty_net

        def zero_inplace(mod):
            if hasattr(mod, "weight"):
                mod.weight.data *= 0
            if hasattr(mod, "bias"):
                mod.bias.data *= 0

        snet.apply(zero_inplace)
        assert (mlp.params == 0).all()

        def one_outofplace(mod):
            if hasattr(mod, "weight"):
                mod.weight = nn.Parameter(torch.ones_like(mod.weight.data))
            if hasattr(mod, "bias"):
                mod.bias = nn.Parameter(torch.ones_like(mod.bias.data))

        snet.apply(one_outofplace)
        assert (mlp.params == 0).all()
        mlp.from_stateful_net(snet)
        assert (mlp.params == 1).all()

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [3])
    @pytest.mark.parametrize("share_params", [True])
    @pytest.mark.parametrize("centralized", [True])
    @pytest.mark.parametrize("n_agent_inputs", [6])
    @pytest.mark.parametrize("batch", [(4,)])
    @pytest.mark.parametrize("tdparams", [True, False])
    def test_multiagent_mlp_tdparams(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        tdparams,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
            use_td_params=tdparams,
        )
        if tdparams:
            assert list(mlp._empty_net.parameters()) == []
            assert list(mlp.params.parameters()) == list(mlp.parameters())
        else:
            assert list(mlp._empty_net.parameters()) == list(mlp.parameters())
            assert not hasattr(mlp.params, "parameters")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return
        mlp = nn.Sequential(mlp)
        mlp.to(device)
        param_set = set(mlp.parameters())
        for p in mlp[0].params.values(True, True):
            assert p in param_set

    def test_multiagent_mlp_lazy(self):
        torch.manual_seed(0)
        mlp = MultiAgentMLP(
            n_agent_inputs=None,
            n_agent_outputs=6,
            n_agents=3,
            centralized=True,
            share_params=False,
            depth=2,
        )
        optim = torch.optim.SGD(mlp.parameters(), lr=1e-3)
        for p in mlp.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for _ in range(2):
            td = self._get_mock_input_td(3, 4, batch=(10,))
            obs = td.get(("agents", "observation"))
            out = mlp(obs)
            assert (
                not mlp.params[0]
                .apply(lambda x, y: torch.isclose(x, y), mlp.params[1])
                .any()
            )
            out.mean().backward()
            optim.step()
        for p in mlp.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    def test_multiagent_reset_mlp(
        self,
        n_agents,
        centralized,
        share_params,
    ):
        actor_net = MultiAgentMLP(
            n_agent_inputs=4,
            n_agent_outputs=6,
            num_cells=(4, 4),
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
        )
        params_before = actor_net.params.clone()
        actor_net.reset_parameters()
        params_after = actor_net.params
        assert not params_before.apply(
            lambda x, y: torch.isclose(x, y), params_after, batch_size=[]
        ).any()
        if params_after.numel() > 1:
            assert (
                not params_after[0]
                .apply(lambda x, y: torch.isclose(x, y), params_after[1], batch_size=[])
                .any()
            )

    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("agent_dim", [1, -3])
    def test_multiagent_custom_agent_dim(self, share_params, agent_dim):
        """Test that custom agent_dim values work correctly.

        Regression test for https://github.com/pytorch/rl/issues/3288
        """
        n_agents = 3
        obs_dim = 5
        seq_len = 6
        output_dim = 4

        class SingleAgentMLP(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 32),
                    nn.Tanh(),
                    nn.Linear(32, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        class MultiAgentPolicyNet(MultiAgentNetBase):
            def __init__(
                self,
                obs_dim,
                output_dim,
                n_agents,
                share_params,
                agent_dim,
                device=None,
            ):
                self.obs_dim = obs_dim
                self.output_dim = output_dim
                self._agent_dim = agent_dim

                super().__init__(
                    n_agents=n_agents,
                    centralized=False,
                    share_params=share_params,
                    agent_dim=agent_dim,
                    device=device,
                )

            def _build_single_net(self, *, device, **kwargs):
                net = SingleAgentMLP(self.obs_dim, self.output_dim)
                return net.to(device) if device is not None else net

            def _pre_forward_check(self, inputs):
                if inputs.shape[self._agent_dim] != self.n_agents:
                    raise ValueError(
                        f"Multi-agent network expected input with shape[{self._agent_dim}]={self.n_agents},"
                        f" but got {inputs.shape}"
                    )
                return inputs

        policy_net = MultiAgentPolicyNet(
            obs_dim=obs_dim,
            output_dim=output_dim,
            n_agents=n_agents,
            share_params=share_params,
            agent_dim=agent_dim,
        )

        # Input shape: (batch, n_agents, seq_len, obs_dim) with agents at dim 1
        batch_size = 4
        obs = torch.randn(batch_size, n_agents, seq_len, obs_dim)
        out = policy_net(obs)

        # Output should preserve agent dimension position
        expected_shape = (batch_size, n_agents, seq_len, output_dim)
        assert (
            out.shape == expected_shape
        ), f"Expected {expected_shape}, got {out.shape}"

        # Verify different agents produce different outputs (unless share_params with same input)
        if not share_params:
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[:, i], out[:, j])

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("channels", [3, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_cnn(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        channels,
        x=15,
        y=15,
    ):
        torch.manual_seed(0)
        cnn = MultiAgentConvNet(
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            in_features=channels,
            kernel_sizes=3,
        )
        if channels is None:
            channels = 3
        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.randn(*batch, n_agents, channels, x, y)},
                    [*batch, n_agents],
                )
            },
            batch_size=batch,
        )
        obs = td[("agents", "observation")]
        out = cnn(obs)
        assert out.shape[:-1] == (*batch, n_agents)
        if centralized and share_params:
            torch.testing.assert_close(out, out[..., :1, :].expand_as(out))
        else:
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[..., i, :], out[..., j, :])
        obs[..., 0, 0, 0, 0] += 1
        out2 = cnn(obs)
        if centralized:
            # a modification to the input of agent 0 will impact all agents
            assert not torch.isclose(out, out2).all()
        elif n_agents > 1:
            assert not torch.isclose(out[..., 0, :], out2[..., 0, :]).all()
            torch.testing.assert_close(out[..., 1:, :], out2[..., 1:, :])

        obs = torch.randn(*batch, 1, channels, x, y).expand(
            *batch, n_agents, channels, x, y
        )
        out = cnn(obs)
        for i in range(n_agents):
            if share_params:
                # same input same output
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    # same input different output
                    assert not torch.allclose(out[..., i, :], out[..., j, :])

    def test_multiagent_cnn_lazy(self):
        torch.manual_seed(42)
        n_agents = 5
        n_channels = 3
        cnn = MultiAgentConvNet(
            n_agents=n_agents,
            centralized=False,
            share_params=False,
            in_features=None,
            kernel_sizes=3,
        )
        optim = torch.optim.SGD(cnn.parameters(), lr=1e-3)
        for p in cnn.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for _ in range(2):
            td = TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": torch.randn(4, n_agents, n_channels, 15, 15)},
                        [4, 5],
                    )
                },
                batch_size=[4],
            )
            obs = td[("agents", "observation")]
            out = cnn(obs)
            assert (
                not cnn.params[0]
                .apply(lambda x, y: torch.isclose(x, y), cnn.params[1])
                .any()
            )
            out.mean().backward()
            optim.step()
        for p in cnn.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    def test_multiagent_reset_cnn(
        self,
        n_agents,
        centralized,
        share_params,
    ):
        torch.manual_seed(42)
        actor_net = MultiAgentConvNet(
            in_features=4,
            num_cells=[5, 5],
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
        )
        params_before = actor_net.params.clone()
        actor_net.reset_parameters()
        params_after = actor_net.params
        assert not params_before.apply(
            lambda x, y: torch.isclose(x, y), params_after, batch_size=[]
        ).any()
        if params_after.numel() > 1:
            assert (
                not params_after[0]
                .apply(lambda x, y: torch.isclose(x, y), params_after[1], batch_size=[])
                .any()
            )

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("batch", [(10,), (10, 3), ()])
    def test_vdn(self, n_agents, batch):
        torch.manual_seed(0)
        mixer = VDNMixer(n_agents=n_agents, device="cpu")

        td = self._get_mock_input_td(n_agents, batch=batch, n_agents_inputs=1)
        obs = td.get(("agents", "observation"))
        assert obs.shape == (*batch, n_agents, 1)
        out = mixer(obs)
        assert out.shape == (*batch, 1)
        assert torch.equal(obs.sum(-2), out)

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("batch", [(10,), (10, 3), ()])
    @pytest.mark.parametrize("state_shape", [(64, 64, 3), (10,)])
    def test_qmix(self, n_agents, batch, state_shape):
        torch.manual_seed(0)
        mixer = QMixer(
            n_agents=n_agents,
            state_shape=state_shape,
            mixing_embed_dim=32,
            device="cpu",
        )

        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=1, state_shape=state_shape
        )
        obs = td.get(("agents", "observation"))
        state = td.get("state")
        assert obs.shape == (*batch, n_agents, 1)
        assert state.shape == (*batch, *state_shape)
        out = mixer(obs, state)
        assert out.shape == (*batch, 1)

    @pytest.mark.parametrize("mixer", ["qmix", "vdn"])
    def test_mixer_malformed_input(
        self, mixer, n_agents=3, batch=(32,), state_shape=(64, 64, 3)
    ):
        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=3, state_shape=state_shape
        )
        if mixer == "qmix":
            mixer = QMixer(
                n_agents=n_agents,
                state_shape=state_shape,
                mixing_embed_dim=32,
                device="cpu",
            )
        else:
            mixer = VDNMixer(n_agents=n_agents, device="cpu")
        obs = td.get(("agents", "observation"))
        state = td.get("state")

        if mixer.needs_state:
            with pytest.raises(
                ValueError,
                match="Mixer that needs state was passed more than 2 inputs",
            ):
                mixer(obs)
        else:
            with pytest.raises(
                ValueError,
                match="Mixer that doesn't need state was passed more than 1 input",
            ):
                mixer(obs, state)

        in_put = [obs, state] if mixer.needs_state else [obs]
        with pytest.raises(
            ValueError,
            match="Mixer network expected chosen_action_value with last 2 dimensions",
        ):
            mixer(*in_put)
        if mixer.needs_state:
            state_diff = state.unsqueeze(-1)
            with pytest.raises(
                ValueError,
                match="Mixer network expected state with ending shape",
            ):
                mixer(obs, state_diff)

        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=1, state_shape=state_shape
        )
        obs = td.get(("agents", "observation"))
        state = td.get("state")
        obs = obs.sum(-2)
        in_put = [obs, state] if mixer.needs_state else [obs]
        with pytest.raises(
            ValueError,
            match="Mixer network expected chosen_action_value with last 2 dimensions",
        ):
            mixer(*in_put)

        obs = td.get(("agents", "observation"))
        state = td.get("state")
        in_put = [obs, state] if mixer.needs_state else [obs]
        mixer(*in_put)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
