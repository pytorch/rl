# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import sys

import pytest
import torch
import torchrl.modules
from _modules_common import (
    _has_functorch,
    _has_triton,
    _triton_skip_reason,
    TORCH_VERSION,
)
from packaging import version
from tensordict import pad, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.utils import assert_close
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    Compose,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    SerialEnv,
    TensorDictPrimer,
    TransformedEnv,
)
from torchrl.envs.utils import step_mdp
from torchrl.modules import (
    ConsistentDropoutModule,
    GRU,
    GRUCell,
    GRUModule,
    LSTM,
    LSTMCell,
    LSTMModule,
    MLP,
    ProbabilisticActor,
    set_recurrent_mode,
)
from torchrl.modules.utils import (
    get_env_transforms_from_module,
    get_primers_from_module,
)
from torchrl.modules.utils.utils import _compute_missing_env_transforms

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import CountingEnv, DiscreteActionVecMockEnv

if _has_functorch:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
def test_python_lstm_cell(device, bias):
    lstm_cell1 = LSTMCell(10, 20, device=device, bias=bias)
    lstm_cell2 = nn.LSTMCell(10, 20, device=device, bias=bias)

    lstm_cell1.load_state_dict(lstm_cell2.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(
        lstm_cell1.named_parameters(), lstm_cell2.named_parameters()
    ):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    # Run loop
    input = torch.randn(2, 3, 10, device=device)
    h0 = torch.randn(3, 20, device=device)
    c0 = torch.randn(3, 20, device=device)
    with torch.no_grad():
        for i in range(input.size()[0]):
            h1, c1 = lstm_cell1(input[i], (h0, c0))
            h2, c2 = lstm_cell2(input[i], (h0, c0))

            # Make sure the final hidden states have the same shape
            assert h1.shape == h2.shape
            assert c1.shape == c2.shape
            torch.testing.assert_close(h1, h2)
            torch.testing.assert_close(c1, c2)
            h0 = h1
            c0 = c1


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
def test_python_gru_cell(device, bias):
    gru_cell1 = GRUCell(10, 20, device=device, bias=bias)
    gru_cell2 = nn.GRUCell(10, 20, device=device, bias=bias)

    gru_cell2.load_state_dict(gru_cell1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(
        gru_cell1.named_parameters(), gru_cell2.named_parameters()
    ):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (v1 == v2).all()
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    # Run loop
    input = torch.randn(2, 3, 10, device=device)
    h0 = torch.zeros(3, 20, device=device)
    with torch.no_grad():
        for i in range(input.size()[0]):
            h1 = gru_cell1(input[i], h0)
            h2 = gru_cell2(input[i], h0)

            # Make sure the final hidden states have the same shape
            assert h1.shape == h2.shape
            torch.testing.assert_close(h1, h2)
            h0 = h1


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_python_lstm(device, bias, dropout, batch_first, num_layers):
    B = 5
    T = 3
    lstm1 = LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    lstm2 = nn.LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )

    lstm2.load_state_dict(lstm1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(lstm1.named_parameters(), lstm2.named_parameters()):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    if batch_first:
        input = torch.randn(B, T, 10, device=device)
    else:
        input = torch.randn(T, B, 10, device=device)

    h0 = torch.randn(num_layers, 5, 20, device=device)
    c0 = torch.randn(num_layers, 5, 20, device=device)

    # Test without hidden states
    with torch.no_grad():
        output1, (h1, c1) = lstm1(input)
        output2, (h2, c2) = lstm2(input)

    assert h1.shape == h2.shape
    assert c1.shape == c2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(c1, c2)

    # Test with hidden states
    with torch.no_grad():
        output1, (h1, c1) = lstm1(input, (h0, c0))
        output2, (h2, c2) = lstm1(input, (h0, c0))

    assert h1.shape == h2.shape
    assert c1.shape == c2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(c1, c2)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_python_gru(device, bias, dropout, batch_first, num_layers):
    B = 5
    T = 3
    gru1 = GRU(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    gru2 = nn.GRU(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    gru2.load_state_dict(gru1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(gru1.named_parameters(), gru2.named_parameters()):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        torch.testing.assert_close(v1, v2)
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    if batch_first:
        input = torch.randn(B, T, 10, device=device)
    else:
        input = torch.randn(T, B, 10, device=device)

    h0 = torch.randn(num_layers, 5, 20, device=device)

    # Test without hidden states
    with torch.no_grad():
        output1, h1 = gru1(input)
        output2, h2 = gru2(input)

    assert h1.shape == h2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)

    # Test with hidden states
    with torch.no_grad():
        output1, h1 = gru1(input, h0)
        output2, h2 = gru2(input, h0)

    assert h1.shape == h2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)


class TestLSTMModule:
    def test_errs(self):
        with pytest.raises(ValueError, match="batch_first"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=False,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys="abc",
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_key="smth",
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys="abc",
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_key="smth",
                out_keys=["intermediate", ("next", "hidden0")],
            )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        td = TensorDict({"observation": torch.randn(3)}, [])
        with pytest.raises(KeyError, match="is_init"):
            lstm_module(td)
        with pytest.raises(ValueError, match="recurrent_backend"):
            LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="other",
            )

    @pytest.mark.parametrize("default_val", [False, True, None])
    def test_set_recurrent_mode(self, default_val):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=default_val,
        )
        assert lstm_module.recurrent_mode is bool(default_val)
        with set_recurrent_mode(True):
            assert lstm_module.recurrent_mode
            with set_recurrent_mode(False):
                assert not lstm_module.recurrent_mode
                with set_recurrent_mode("recurrent"):
                    assert lstm_module.recurrent_mode
                    with set_recurrent_mode("sequential"):
                        assert not lstm_module.recurrent_mode
                    assert lstm_module.recurrent_mode
                assert not lstm_module.recurrent_mode
            assert lstm_module.recurrent_mode
        assert lstm_module.recurrent_mode is bool(default_val)

    def test_python_cudnn(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            dropout=0,
            num_layers=2,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=True,
        )
        obs = torch.rand(10, 20, 3)

        hidden0 = torch.rand(10, 20, 2, 12)
        hidden1 = torch.rand(10, 20, 2, 12)

        is_init = torch.zeros(10, 20, dtype=torch.bool)
        assert isinstance(lstm_module.lstm, nn.LSTM)
        outs_ref = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )

        lstm_module.make_python_based()
        assert isinstance(lstm_module.lstm, torchrl.modules.LSTM)
        outs_rl = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )
        torch.testing.assert_close(outs_ref, outs_rl)

        lstm_module.make_cudnn_based()
        assert isinstance(lstm_module.lstm, nn.LSTM)
        outs_cudnn = lstm_module(
            observation=obs, hidden0=hidden0, hidden1=hidden1, is_init=is_init
        )
        torch.testing.assert_close(outs_ref, outs_cudnn)

    def test_noncontiguous(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["bork", "h0", "h1"],
            out_keys=["dork", ("next", "h0"), ("next", "h1")],
        )
        td = TensorDict(
            {
                "bork": torch.randn(3, 3),
                "is_init": torch.zeros(3, 1, dtype=torch.bool),
            },
            [3],
        )
        padded = pad(td, [0, 5])
        lstm_module(padded)

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step(self, shape, python_based):
        td = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        td = lstm_module(td)
        td_next = step_mdp(td, keep_other=True)
        td_next = lstm_module(td_next)

        assert not torch.isclose(
            td_next["next", "hidden0"], td["next", "hidden0"]
        ).any()

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [False, True])
    def test_multi_consecutive(self, shape, python_based):
        t = 20
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        if shape:
            td["is_init"][0, ..., 13, :] = True
        else:
            td["is_init"][13, :] = True

        lstm_module_ss = LSTMModule(
            input_size=3,
            hidden_size=12,
            num_layers=4,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        with set_recurrent_mode(True):
            lstm_module_ss(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            td_ss["is_init"][:] = td["is_init"][..., _t, :]
            lstm_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        # import ipdb; ipdb.set_trace()  # assert fails when python_based is True, why?
        torch.testing.assert_close(
            td_ss["intermediate"], td["intermediate"][..., -1, :]
        )

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    @pytest.mark.parametrize("within", [False, True])
    def test_lstm_parallel_env(
        self, python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
    ):
        self._test_lstm_parallel_env(
            python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
        )

    def _test_lstm_parallel_env(
        self, python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
    ):

        torch.manual_seed(0)
        num_envs = 3
        device = "cuda" if torch.cuda.device_count() else "cpu"
        # tests that hidden states are carried over with parallel envs
        lstm_module = LSTMModule(
            input_size=7,
            hidden_size=12,
            num_layers=2,
            in_key="observation",
            out_key="features",
            device=device,
            python_based=python_based,
        )
        if parallel:
            cls = maybe_fork_ParallelEnv
        else:
            cls = SerialEnv

        if within:

            def create_transformed_env():
                primer = lstm_module.make_tensordict_primer()
                env = DiscreteActionVecMockEnv(
                    categorical_action_encoding=True, device=device
                )
                env = TransformedEnv(env)
                env.append_transform(InitTracker())
                env.append_transform(primer)
                return env

        else:
            create_transformed_env = functools.partial(
                DiscreteActionVecMockEnv,
                categorical_action_encoding=True,
                device=device,
            )

        if heterogeneous:
            create_transformed_env = [
                EnvCreator(create_transformed_env) for _ in range(num_envs)
            ]
        env = cls(
            create_env_fn=create_transformed_env,
            num_workers=num_envs,
        )
        if not within:
            env = env.append_transform(InitTracker())
            env.append_transform(lstm_module.make_tensordict_primer())

        mlp = TensorDictModule(
            MLP(
                in_features=12,
                out_features=7,
                num_cells=[],
                device=device,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )

        actor_model = TensorDictSequential(lstm_module, mlp)

        actor = ProbabilisticActor(
            module=actor_model,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        for break_when_any_done in [False, True]:
            data = env.rollout(10, actor, break_when_any_done=break_when_any_done)
            assert (data.get(("next", "recurrent_state_c")) != 0.0).all()
            assert (data.get("recurrent_state_c") != 0.0).any()
        return data  # noqa

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    def test_lstm_parallel_within(
        self, python_based, parallel, heterogeneous, maybe_fork_ParallelEnv
    ):
        out_within = self._test_lstm_parallel_env(
            python_based,
            parallel,
            heterogeneous,
            within=True,
            maybe_fork_ParallelEnv=maybe_fork_ParallelEnv,
        )
        out_not_within = self._test_lstm_parallel_env(
            python_based,
            parallel,
            heterogeneous,
            within=False,
            maybe_fork_ParallelEnv=maybe_fork_ParallelEnv,
        )
        assert_close(out_within, out_not_within)

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
    def test_lstm_vmap_complex_model(self):
        # Tests that all ops in GRU are compatible with VMAP (when build using
        # the PT backend).
        # This used to fail when splitting the input based on the is_init mask.
        # This test is intended not only as a non-regression test but also
        # to make sure that any change provided to RNNs is compliant with vmap
        torch.manual_seed(0)
        input_size = 4
        hidden_size = 5
        num_layers = 1
        output_size = 3
        out_key = "out"

        embedding_module = TensorDictModule(
            in_keys=["observation"],
            out_keys=["embed"],
            module=torch.nn.Linear(input_size, input_size),
        )

        lstm_module = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_key="embed",
            out_key="features",
            python_based=True,
        )
        mlp = TensorDictModule(
            MLP(
                in_features=hidden_size,
                out_features=output_size,
                num_cells=[],
            ),
            in_keys=["features"],
            out_keys=[out_key],
        )
        training_model = TensorDictSequential(embedding_module, lstm_module, mlp)
        is_init = torch.zeros(50, 11, 1, dtype=torch.bool).bernoulli_(0.1)
        data = TensorDict(
            {"observation": torch.randn(50, 11, input_size), "is_init": is_init},
            [50, 11],
        )
        with set_recurrent_mode(True):
            training_model(data)
        params = TensorDict.from_module(training_model)
        params = params.expand(2)

        def call(data, params):
            with set_recurrent_mode(True), params.to_module(training_model):
                return training_model(data)

        assert vmap(call, (None, 0))(data, params).shape == torch.Size((2, 50, 11))

    @pytest.mark.parametrize("python_based", [False, True])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_recurrent_state_at_traj_end(self, python_based, num_layers):
        # Regression test for https://github.com/pytorch/rl/issues/3711:
        # in recurrent mode, when a batch contains trajectories of different
        # lengths, the recurrent_state stored at the end of each trajectory
        # must be the LSTM hidden state after consuming that trajectory's
        # last real observation -- not the hidden state after consuming the
        # padded tail.
        torch.manual_seed(0)
        B, T, F, H = 1, 5, 2, 4
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True  # traj 1: steps 0..2, traj 2: steps 3..4
        obs = torch.ones(B, T, F)
        obs[0, 3:] = 2.0
        data = TensorDict({"obs": obs, "is_init": is_init}, [B, T])
        lstm_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=num_layers,
            in_key="obs",
            out_keys=[
                "feat",
                ("next", "recurrent_state_h"),
                ("next", "recurrent_state_c"),
            ],
            python_based=python_based,
        )
        with set_recurrent_mode(True), torch.no_grad():
            out = lstm_module(data)
        with torch.no_grad():
            _, (h1, c1) = lstm_module.lstm(obs[:, :3])
            _, (h2, c2) = lstm_module.lstm(obs[:, 3:])
        # Stored states have shape [B, T, num_layers, H]; expected per-layer
        # final states have shape [num_layers, 1, H] (batch dim collapsed).
        torch.testing.assert_close(
            out["next", "recurrent_state_h"][0, 2], h1.squeeze(1)
        )
        torch.testing.assert_close(
            out["next", "recurrent_state_c"][0, 2], c1.squeeze(1)
        )
        torch.testing.assert_close(
            out["next", "recurrent_state_h"][0, 4], h2.squeeze(1)
        )
        torch.testing.assert_close(
            out["next", "recurrent_state_c"][0, 4], c2.squeeze(1)
        )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="torch.compile scan tests need a C compiler"
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_scan_prototype(self):
        # Opt-in prototype: torch._higher_order_ops.scan-based time loop.
        # Must be exercised under torch.compile -- scan is unusable in eager.
        from torchrl.modules.tensordict_module.rnn import LSTM

        torch.manual_seed(0)
        B, T, F_in, H, L = 2, 5, 3, 4, 2
        ref = LSTM(input_size=F_in, hidden_size=H, num_layers=L, batch_first=True)
        scn = LSTM(
            input_size=F_in,
            hidden_size=H,
            num_layers=L,
            batch_first=True,
            use_scan=True,
        )
        scn.load_state_dict(ref.state_dict())

        x = torch.randn(B, T, F_in)
        h0 = torch.zeros(L, B, H)
        c0 = torch.zeros(L, B, H)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, 3:] = False  # batch 1 ends at t=3

        with torch.no_grad():
            y_ref, (hn_ref, cn_ref) = ref(x, (h0, c0), mask=mask)

        @torch.compile(fullgraph=True)
        def call(x, h0, c0, mask):
            return scn(x, (h0, c0), mask=mask)

        with torch.no_grad():
            y_s, (hn_s, cn_s) = call(x, h0, c0, mask)
        torch.testing.assert_close(y_ref, y_s)
        torch.testing.assert_close(hn_ref, hn_s)
        torch.testing.assert_close(cn_ref, cn_s)

    @pytest.mark.parametrize("python_based", [False, True])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_scan_backend_matches_pad(self, python_based, monkeypatch):
        torch.manual_seed(0)
        B, T, F, H, L = 4, 7, 3, 5, 2
        pad_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
        )
        scan_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
            recurrent_backend="scan",
        )
        auto_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            python_based=python_based,
            recurrent_backend="auto",
        )
        scan_module.load_state_dict(pad_module.state_dict())
        auto_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        hidden0 = torch.zeros(B, T, L, H)
        hidden1 = torch.zeros(B, T, L, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        reset_shape = (int(is_init.sum()), L, H)
        hidden0[is_init.squeeze(-1)] = torch.randn(reset_shape)
        hidden1[is_init.squeeze(-1)] = torch.randn(reset_shape)
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            scan_out = scan_module(data.clone())
            auto_pad_out = auto_module(data.clone())

        torch.testing.assert_close(pad_out["feat"], scan_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden0"], scan_out["next", "hidden0"]
        )
        torch.testing.assert_close(
            pad_out["next", "hidden1"], scan_out["next", "hidden1"]
        )
        torch.testing.assert_close(pad_out["feat"], auto_pad_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden0"], auto_pad_out["next", "hidden0"]
        )
        torch.testing.assert_close(
            pad_out["next", "hidden1"], auto_pad_out["next", "hidden1"]
        )

        from torchrl.modules.tensordict_module import rnn as rnn_module

        monkeypatch.setattr(rnn_module, "is_compiling", lambda: True)
        with set_recurrent_mode(True), torch.no_grad():
            auto_scan_out = auto_module(data.clone())
        torch.testing.assert_close(scan_out["feat"], auto_scan_out["feat"])
        torch.testing.assert_close(
            scan_out["next", "hidden0"], auto_scan_out["next", "hidden0"]
        )
        torch.testing.assert_close(
            scan_out["next", "hidden1"], auto_scan_out["next", "hidden1"]
        )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [16, 64])
    def test_lstm_module_triton_backend_matches_pad(self, H, compute_dtype):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F = 4, 7, 3
        pad_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            device=device,
        )
        triton_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="triton",
            recurrent_compute_dtype=compute_dtype,
            device=device,
        )
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden0 = torch.randn(B, T, 1, H, device=device)
        hidden1 = torch.randn(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            triton_out = triton_module(data.clone())

        atol = 5e-2 if compute_dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(
            pad_out["feat"], triton_out["feat"], atol=atol, rtol=atol
        )
        torch.testing.assert_close(
            pad_out["next", "hidden0"],
            triton_out["next", "hidden0"],
            atol=atol,
            rtol=atol,
        )
        torch.testing.assert_close(
            pad_out["next", "hidden1"],
            triton_out["next", "hidden1"],
            atol=atol,
            rtol=atol,
        )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize(
        "module_kwargs,training",
        [
            ({"num_layers": 2}, False),
            ({"num_layers": 2, "dropout": 0.3}, True),
            ({"num_layers": 2, "dropout": 0.3}, False),
            ({"bidirectional": True}, False),
            ({"proj_size": 8}, False),
            ({"num_layers": 2, "bidirectional": True, "proj_size": 8}, False),
        ],
    )
    def test_lstm_module_triton_extended_forward_matches_pad(
        self, module_kwargs, training
    ):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        num_layers = module_kwargs.get("num_layers", 1)
        num_directions = 2 if module_kwargs.get("bidirectional", False) else 1
        proj_size = module_kwargs.get("proj_size", 0)
        hidden_out = proj_size if proj_size > 0 else H
        state_shape_h = (B, T, num_layers * num_directions, hidden_out)
        state_shape_c = (B, T, num_layers * num_directions, H)
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            "device": device,
            **module_kwargs,
        }
        pad_module = LSTMModule(**kwargs)
        triton_module = LSTMModule(**kwargs, recurrent_backend="triton")
        triton_module.load_state_dict(pad_module.state_dict())
        pad_module.train(training)
        triton_module.train(training)

        obs = torch.randn(B, T, F, device=device)
        hidden0 = torch.randn(*state_shape_h, device=device)
        hidden1 = torch.randn(*state_shape_c, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            torch.manual_seed(1)
            pad_out = pad_module(data.clone())
            torch.manual_seed(1)
            triton_out = triton_module(data.clone())

        # Bit-exact equivalence isn't achievable in training mode with dropout:
        # cuDNN's nn.LSTM stores its dropout mask state in a cuDNN dropout
        # descriptor that advances independently of torch's global RNG, while
        # the triton backend's between-layer ``F.dropout`` consumes torch's
        # RNG directly. Verify both produce sane, same-shape outputs instead.
        dropout_active = training and module_kwargs.get("dropout", 0.0) > 0
        if dropout_active:
            for key in ["feat", ("next", "hidden0"), ("next", "hidden1")]:
                assert pad_out[key].shape == triton_out[key].shape
                assert pad_out[key].dtype == triton_out[key].dtype
                assert torch.isfinite(pad_out[key]).all()
                assert torch.isfinite(triton_out[key]).all()
        else:
            torch.testing.assert_close(
                pad_out["feat"], triton_out["feat"], atol=5e-3, rtol=5e-3
            )
            torch.testing.assert_close(
                pad_out["next", "hidden0"],
                triton_out["next", "hidden0"],
                atol=5e-3,
                rtol=5e-3,
            )
            torch.testing.assert_close(
                pad_out["next", "hidden1"],
                triton_out["next", "hidden1"],
                atol=5e-3,
                rtol=5e-3,
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_lstm_module_triton_backward(self):
        """Backward path: gradients match pad backend within tolerance."""
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 64
        pad_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            device=device,
        )
        triton_module = LSTMModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="triton",
            device=device,
        )
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden0 = torch.zeros(B, T, 1, H, device=device)
        hidden1 = torch.zeros(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        def loss_for(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            out = mod(data.clone())
            return out["feat"].pow(2).sum()

        with set_recurrent_mode(True):
            loss_pad = loss_for(pad_module)
            loss_pad.backward()
            grads_pad = {
                k: p.grad.detach().clone() for k, p in pad_module.named_parameters()
            }
            loss_triton = loss_for(triton_module)
            loss_triton.backward()
            grads_triton = {
                k: p.grad.detach().clone() for k, p in triton_module.named_parameters()
            }

        for k in grads_pad:
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=5e-3, rtol=5e-3
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize(
        "module_kwargs",
        [
            {"num_layers": 2},
            {"bidirectional": True},
            {"proj_size": 8},
        ],
    )
    def test_lstm_module_triton_extended_backward_matches_pad(self, module_kwargs):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        num_layers = module_kwargs.get("num_layers", 1)
        num_directions = 2 if module_kwargs.get("bidirectional", False) else 1
        proj_size = module_kwargs.get("proj_size", 0)
        hidden_out = proj_size if proj_size > 0 else H
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            "device": device,
            **module_kwargs,
        }
        pad_module = LSTMModule(**kwargs)
        triton_module = LSTMModule(**kwargs, recurrent_backend="triton")
        triton_module.load_state_dict(pad_module.state_dict())

        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        is_init[2, 2] = True

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            obs = torch.randn(B, T, F, device=device, requires_grad=True)
            hidden0 = torch.zeros(
                B,
                T,
                num_layers * num_directions,
                hidden_out,
                device=device,
                requires_grad=True,
            )
            hidden1 = torch.zeros(
                B,
                T,
                num_layers * num_directions,
                H,
                device=device,
                requires_grad=True,
            )
            data = TensorDict(
                {
                    "obs": obs,
                    "hidden0": hidden0,
                    "hidden1": hidden1,
                    "is_init": is_init,
                },
                [B, T],
            )
            out = mod(data)
            loss = (
                out["feat"].pow(2).sum()
                + out["next", "hidden0"].pow(2).sum()
                + out["next", "hidden1"].pow(2).sum()
            )
            loss.backward()
            grads = {k: p.grad.detach().clone() for k, p in mod.named_parameters()}
            return grads, obs.grad, hidden0.grad, hidden1.grad

        with set_recurrent_mode(True):
            torch.manual_seed(1)
            grads_pad, obs_grad_pad, h0_grad_pad, h1_grad_pad = loss_and_grads(
                pad_module
            )
            torch.manual_seed(1)
            (
                grads_triton,
                obs_grad_triton,
                h0_grad_triton,
                h1_grad_triton,
            ) = loss_and_grads(triton_module)

        for k in grads_pad:
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=1e-2, rtol=1e-2
            )
        torch.testing.assert_close(obs_grad_pad, obs_grad_triton, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(h0_grad_pad, h0_grad_triton, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(h1_grad_pad, h1_grad_triton, atol=1e-2, rtol=1e-2)

    def test_lstm_module_triton_requires_triton(self, monkeypatch):
        from torchrl.modules.tensordict_module import rnn as rnn_module

        monkeypatch.setattr(rnn_module, "_has_triton", False)
        with pytest.raises(RuntimeError, match="triton"):
            LSTMModule(
                input_size=3,
                hidden_size=12,
                num_layers=1,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="triton",
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_three_backends_equivalent(self, num_layers):
        """pad / scan / triton agree at the intersection of supported configs.

        Scan does not support dropout, so this test fixes ``dropout=0``; the
        pad-vs-triton dropout case is covered separately by
        ``test_lstm_module_triton_extended_forward_matches_pad``.
        """
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": num_layers,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            "device": device,
        }
        pad_module = LSTMModule(**kwargs)
        scan_module = LSTMModule(**kwargs, recurrent_backend="scan")
        triton_module = LSTMModule(**kwargs, recurrent_backend="triton")
        scan_module.load_state_dict(pad_module.state_dict())
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden0 = torch.randn(B, T, num_layers, H, device=device)
        hidden1 = torch.randn(B, T, num_layers, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            scan_out = scan_module(data.clone())
            triton_out = triton_module(data.clone())

        for key in ["feat", ("next", "hidden0"), ("next", "hidden1")]:
            torch.testing.assert_close(
                pad_out[key], scan_out[key], atol=5e-3, rtol=5e-3
            )
            torch.testing.assert_close(
                pad_out[key], triton_out[key], atol=5e-3, rtol=5e-3
            )


class TestGRUModule:
    def test_errs(self):
        with pytest.raises(ValueError, match="batch_first"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=False,
                in_keys=["observation", "hidden"],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=[
                    "observation",
                    "hidden0",
                    "hidden1",
                ],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys="abc",
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_key="smth",
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_keys=["intermediate", ("next", "hidden"), "other"],
            )
        with pytest.raises(TypeError, match="incompatible function arguments"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_keys="abc",
            )
        with pytest.raises(ValueError, match="out_keys"):
            gru_module = GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_key="smth",
                out_keys=["intermediate", ("next", "hidden"), "other"],
            )
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
        )
        td = TensorDict({"observation": torch.randn(3)}, [])
        with pytest.raises(KeyError, match="is_init"):
            gru_module(td)
        with pytest.raises(ValueError, match="recurrent_backend"):
            GRUModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden"],
                out_keys=["intermediate", ("next", "hidden")],
                recurrent_backend="other",
            )

    @pytest.mark.parametrize("default_val", [False, True, None])
    def test_set_recurrent_mode(self, default_val):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            default_recurrent_mode=default_val,
        )
        assert gru_module.recurrent_mode is bool(default_val)
        with set_recurrent_mode(True):
            assert gru_module.recurrent_mode
            with set_recurrent_mode(False):
                assert not gru_module.recurrent_mode
                with set_recurrent_mode("recurrent"):
                    assert gru_module.recurrent_mode
                    with set_recurrent_mode("sequential"):
                        assert not gru_module.recurrent_mode
                    assert gru_module.recurrent_mode
                assert not gru_module.recurrent_mode
            assert gru_module.recurrent_mode
        assert gru_module.recurrent_mode is bool(default_val)

    @set_recurrent_mode(True)
    def test_python_cudnn(self):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            dropout=0,
            num_layers=2,
            in_keys=["observation", "hidden0"],
            out_keys=["intermediate", ("next", "hidden0")],
        )
        obs = torch.rand(10, 20, 3)

        hidden0 = torch.rand(10, 20, 2, 12)

        is_init = torch.zeros(10, 20, dtype=torch.bool)
        assert isinstance(gru_module.gru, nn.GRU)
        outs_ref = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)

        gru_module.make_python_based()
        assert isinstance(gru_module.gru, torchrl.modules.GRU)
        outs_rl = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)
        torch.testing.assert_close(outs_ref, outs_rl)

        gru_module.make_cudnn_based()
        assert isinstance(gru_module.gru, nn.GRU)
        outs_cudnn = gru_module(observation=obs, hidden0=hidden0, is_init=is_init)
        torch.testing.assert_close(outs_ref, outs_cudnn)

    def test_noncontiguous(self):
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["bork", "h"],
            out_keys=["dork", ("next", "h")],
        )
        td = TensorDict(
            {
                "bork": torch.randn(3, 3),
                "is_init": torch.zeros(3, 1, dtype=torch.bool),
            },
            [3],
        )
        padded = pad(td, [0, 5])
        gru_module(padded)

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step(self, shape, python_based):
        td = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        gru_module = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        td = gru_module(td)
        td_next = step_mdp(td, keep_other=True)
        td_next = gru_module(td_next)

        assert not torch.isclose(td_next["next", "hidden"], td["next", "hidden"]).any()

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("t", [1, 10])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_single_step_vs_multi(self, shape, t, python_based):
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        gru_module_ss = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        with set_recurrent_mode(True):
            gru_module_ss(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            gru_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(td_ss["hidden"], td["next", "hidden"][..., -1, :, :])

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("python_based", [True, False])
    def test_multi_consecutive(self, shape, python_based):
        t = 20
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        if shape:
            td["is_init"][0, ..., 13, :] = True
        else:
            td["is_init"][13, :] = True

        gru_module_ss = GRUModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden"],
            out_keys=["intermediate", ("next", "hidden")],
            python_based=python_based,
        )
        with set_recurrent_mode(True):
            gru_module_ss(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            td_ss["is_init"][:] = td["is_init"][..., _t, :]
            gru_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(
            td_ss["intermediate"], td["intermediate"][..., -1, :]
        )

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    @pytest.mark.parametrize("within", [False, True])
    def test_gru_parallel_env(
        self, python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
    ):
        self._test_gru_parallel_env(
            python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
        )

    def _test_gru_parallel_env(
        self, python_based, parallel, heterogeneous, within, maybe_fork_ParallelEnv
    ):
        torch.manual_seed(0)
        num_workers = 3

        device = "cuda" if torch.cuda.device_count() else "cpu"
        # tests that hidden states are carried over with parallel envs
        gru_module = GRUModule(
            input_size=7,
            hidden_size=12,
            num_layers=2,
            in_key="observation",
            out_key="features",
            device=device,
            python_based=python_based,
        )

        if within:

            def create_transformed_env():
                primer = gru_module.make_tensordict_primer()
                env = DiscreteActionVecMockEnv(
                    categorical_action_encoding=True, device=device
                )
                env = TransformedEnv(env)
                env.append_transform(InitTracker())
                env.append_transform(primer)
                return env

        else:
            create_transformed_env = functools.partial(
                DiscreteActionVecMockEnv,
                categorical_action_encoding=True,
                device=device,
            )

        if parallel:
            cls = maybe_fork_ParallelEnv
        else:
            cls = SerialEnv
        if heterogeneous:
            create_transformed_env = [
                EnvCreator(create_transformed_env) for _ in range(num_workers)
            ]

        env: ParallelEnv | SerialEnv = cls(
            create_env_fn=create_transformed_env,
            num_workers=num_workers,
        )
        if not within:
            primer = gru_module.make_tensordict_primer()
            env = env.append_transform(InitTracker())
            env.append_transform(primer)

        mlp = TensorDictModule(
            MLP(
                in_features=12,
                out_features=7,
                num_cells=[],
                device=device,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )

        actor_model = TensorDictSequential(gru_module, mlp)

        actor = ProbabilisticActor(
            module=actor_model,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        for break_when_any_done in [False, True]:
            data = env.rollout(10, actor, break_when_any_done=break_when_any_done)
            assert (data.get("recurrent_state") != 0.0).any()
            assert (data.get(("next", "recurrent_state")) != 0.0).all()
        return data  # noqa

    @pytest.mark.parametrize("python_based", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("heterogeneous", [True, False])
    def test_gru_parallel_within(
        self, python_based, parallel, heterogeneous, maybe_fork_ParallelEnv
    ):
        out_within = self._test_gru_parallel_env(
            python_based,
            parallel,
            heterogeneous,
            within=True,
            maybe_fork_ParallelEnv=maybe_fork_ParallelEnv,
        )
        out_not_within = self._test_gru_parallel_env(
            python_based,
            parallel,
            heterogeneous,
            within=False,
            maybe_fork_ParallelEnv=maybe_fork_ParallelEnv,
        )
        assert_close(out_within, out_not_within)

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
    def test_gru_vmap_complex_model(self):
        # Tests that all ops in GRU are compatible with VMAP (when build using
        # the PT backend).
        # This used to fail when splitting the input based on the is_init mask.
        # This test is intended not only as a non-regression test but also
        # to make sure that any change provided to RNNs is compliant with vmap
        torch.manual_seed(0)
        input_size = 4
        hidden_size = 5
        num_layers = 1
        output_size = 3
        out_key = "out"

        embedding_module = TensorDictModule(
            in_keys=["observation"],
            out_keys=["embed"],
            module=torch.nn.Linear(input_size, input_size),
        )

        lstm_module = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_key="embed",
            out_key="features",
            python_based=True,
        )
        mlp = TensorDictModule(
            MLP(
                in_features=hidden_size,
                out_features=output_size,
                num_cells=[],
            ),
            in_keys=["features"],
            out_keys=[out_key],
        )
        training_model = TensorDictSequential(embedding_module, lstm_module, mlp)
        is_init = torch.zeros(50, 11, 1, dtype=torch.bool).bernoulli_(0.1)
        data = TensorDict(
            {"observation": torch.randn(50, 11, input_size), "is_init": is_init},
            [50, 11],
        )
        with set_recurrent_mode(True):
            training_model(data)
        params = TensorDict.from_module(training_model)
        params = params.expand(2)

        def call(data, params):
            with set_recurrent_mode(True), params.to_module(training_model):
                return training_model(data)

        assert vmap(call, (None, 0))(data, params).shape == torch.Size((2, 50, 11))

    @pytest.mark.parametrize("python_based", [False, True])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_recurrent_state_at_traj_end(self, python_based, num_layers):
        # Regression test for https://github.com/pytorch/rl/issues/3711 (GRU
        # twin): same fix as LSTM -- the hidden state at the end of each
        # trajectory must reflect the trajectory's last real observation, not
        # the padded tail.
        torch.manual_seed(0)
        B, T, F, H = 1, 5, 2, 4
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True
        obs = torch.ones(B, T, F)
        obs[0, 3:] = 2.0
        data = TensorDict({"obs": obs, "is_init": is_init}, [B, T])
        gru_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=num_layers,
            in_key="obs",
            out_keys=["feat", ("next", "recurrent_state")],
            python_based=python_based,
        )
        with set_recurrent_mode(True), torch.no_grad():
            out = gru_module(data)
        with torch.no_grad():
            _, h1 = gru_module.gru(obs[:, :3])
            _, h2 = gru_module.gru(obs[:, 3:])
        torch.testing.assert_close(out["next", "recurrent_state"][0, 2], h1.squeeze(1))
        torch.testing.assert_close(out["next", "recurrent_state"][0, 4], h2.squeeze(1))

    @pytest.mark.skipif(
        sys.platform == "win32", reason="torch.compile scan tests need a C compiler"
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_scan_prototype(self):
        # Opt-in prototype: see TestLSTMModule.test_lstm_scan_prototype.
        from torchrl.modules.tensordict_module.rnn import GRU

        torch.manual_seed(0)
        B, T, F_in, H, L = 2, 5, 3, 4, 2
        ref = GRU(input_size=F_in, hidden_size=H, num_layers=L, batch_first=True)
        scn = GRU(
            input_size=F_in,
            hidden_size=H,
            num_layers=L,
            batch_first=True,
            use_scan=True,
        )
        scn.load_state_dict(ref.state_dict())

        x = torch.randn(B, T, F_in)
        h0 = torch.zeros(L, B, H)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, 3:] = False

        with torch.no_grad():
            y_ref, hn_ref = ref(x, h0, mask=mask)

        @torch.compile(fullgraph=True)
        def call(x, h0, mask):
            return scn(x, h0, mask=mask)

        with torch.no_grad():
            y_s, hn_s = call(x, h0, mask)
        torch.testing.assert_close(y_ref, y_s)
        torch.testing.assert_close(hn_ref, hn_s)

    @pytest.mark.parametrize("python_based", [False, True])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_module_scan_backend_matches_pad(self, python_based, monkeypatch):
        torch.manual_seed(0)
        B, T, F, H, L = 4, 7, 3, 5, 2
        pad_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            python_based=python_based,
        )
        scan_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            python_based=python_based,
            recurrent_backend="scan",
        )
        auto_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            python_based=python_based,
            recurrent_backend="auto",
        )
        scan_module.load_state_dict(pad_module.state_dict())
        auto_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        hidden = torch.zeros(B, T, L, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        hidden[is_init.squeeze(-1)] = torch.randn(is_init.sum(), L, H)
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            scan_out = scan_module(data.clone())
            auto_pad_out = auto_module(data.clone())

        torch.testing.assert_close(pad_out["feat"], scan_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden"], scan_out["next", "hidden"]
        )
        torch.testing.assert_close(pad_out["feat"], auto_pad_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden"], auto_pad_out["next", "hidden"]
        )

        from torchrl.modules.tensordict_module import rnn as rnn_module

        monkeypatch.setattr(rnn_module, "is_compiling", lambda: True)
        with set_recurrent_mode(True), torch.no_grad():
            auto_scan_out = auto_module(data.clone())
        torch.testing.assert_close(scan_out["feat"], auto_scan_out["feat"])
        torch.testing.assert_close(
            scan_out["next", "hidden"], auto_scan_out["next", "hidden"]
        )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [16, 64])
    def test_gru_module_triton_backend_matches_pad(self, H, compute_dtype):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F = 4, 7, 3
        pad_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            device=device,
        )
        triton_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="triton",
            recurrent_compute_dtype=compute_dtype,
            device=device,
        )
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.randn(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            triton_out = triton_module(data.clone())

        atol = 5e-2 if compute_dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(
            pad_out["feat"], triton_out["feat"], atol=atol, rtol=atol
        )
        torch.testing.assert_close(
            pad_out["next", "hidden"],
            triton_out["next", "hidden"],
            atol=atol,
            rtol=atol,
        )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize(
        "module_kwargs,training",
        [
            ({"num_layers": 2}, False),
            ({"num_layers": 2, "dropout": 0.3}, True),
            ({"num_layers": 2, "dropout": 0.3}, False),
            ({"bidirectional": True}, False),
        ],
    )
    def test_gru_module_triton_extended_forward_matches_pad(
        self, module_kwargs, training
    ):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        num_layers = module_kwargs.get("num_layers", 1)
        num_directions = 2 if module_kwargs.get("bidirectional", False) else 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
            "device": device,
            **module_kwargs,
        }
        pad_module = GRUModule(**kwargs)
        triton_module = GRUModule(**kwargs, recurrent_backend="triton")
        triton_module.load_state_dict(pad_module.state_dict())
        pad_module.train(training)
        triton_module.train(training)

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.randn(B, T, num_layers * num_directions, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            torch.manual_seed(1)
            pad_out = pad_module(data.clone())
            torch.manual_seed(1)
            triton_out = triton_module(data.clone())

        # See comment on test_lstm_module_triton_extended_forward_matches_pad:
        # under dropout + training=True the two backends draw their masks from
        # different RNG state and bit-exact comparison is not meaningful.
        dropout_active = training and module_kwargs.get("dropout", 0.0) > 0
        if dropout_active:
            for key in ["feat", ("next", "hidden")]:
                assert pad_out[key].shape == triton_out[key].shape
                assert pad_out[key].dtype == triton_out[key].dtype
                assert torch.isfinite(pad_out[key]).all()
                assert torch.isfinite(triton_out[key]).all()
        else:
            torch.testing.assert_close(
                pad_out["feat"], triton_out["feat"], atol=5e-3, rtol=5e-3
            )
            torch.testing.assert_close(
                pad_out["next", "hidden"],
                triton_out["next", "hidden"],
                atol=5e-3,
                rtol=5e-3,
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_gru_module_triton_backward(self):
        """Backward path: gradients match pad backend within tolerance."""
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 64
        pad_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            device=device,
        )
        triton_module = GRUModule(
            input_size=F,
            hidden_size=H,
            num_layers=1,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="triton",
            device=device,
        )
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.zeros(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        data = TensorDict({"obs": obs, "hidden": hidden, "is_init": is_init}, [B, T])

        def loss_for(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            out = mod(data.clone())
            return out["feat"].pow(2).sum()

        with set_recurrent_mode(True):
            loss_pad = loss_for(pad_module)
            loss_pad.backward()
            grads_pad = {
                k: p.grad.detach().clone() for k, p in pad_module.named_parameters()
            }
            loss_triton = loss_for(triton_module)
            loss_triton.backward()
            grads_triton = {
                k: p.grad.detach().clone() for k, p in triton_module.named_parameters()
            }

        for k in grads_pad:
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=5e-3, rtol=5e-3
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize(
        "module_kwargs",
        [
            {"num_layers": 2},
            {"bidirectional": True},
        ],
    )
    def test_gru_module_triton_extended_backward_matches_pad(self, module_kwargs):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        num_layers = module_kwargs.get("num_layers", 1)
        num_directions = 2 if module_kwargs.get("bidirectional", False) else 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
            "device": device,
            **module_kwargs,
        }
        pad_module = GRUModule(**kwargs)
        triton_module = GRUModule(**kwargs, recurrent_backend="triton")
        triton_module.load_state_dict(pad_module.state_dict())
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        is_init[2, 2] = True

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            obs = torch.randn(B, T, F, device=device, requires_grad=True)
            hidden = torch.zeros(
                B,
                T,
                num_layers * num_directions,
                H,
                device=device,
                requires_grad=True,
            )
            data = TensorDict(
                {"obs": obs, "hidden": hidden, "is_init": is_init},
                [B, T],
            )
            out = mod(data)
            loss = out["feat"].pow(2).sum() + out["next", "hidden"].pow(2).sum()
            loss.backward()
            grads = {k: p.grad.detach().clone() for k, p in mod.named_parameters()}
            return grads, obs.grad, hidden.grad

        with set_recurrent_mode(True):
            torch.manual_seed(1)
            grads_pad, obs_grad_pad, hidden_grad_pad = loss_and_grads(pad_module)
            torch.manual_seed(1)
            grads_triton, obs_grad_triton, hidden_grad_triton = loss_and_grads(
                triton_module
            )

        for k in grads_pad:
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=1e-2, rtol=1e-2
            )
        torch.testing.assert_close(obs_grad_pad, obs_grad_triton, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            hidden_grad_pad, hidden_grad_triton, atol=1e-2, rtol=1e-2
        )

    def test_gru_module_triton_requires_triton(self, monkeypatch):
        from torchrl.modules.tensordict_module import rnn as rnn_module

        monkeypatch.setattr(rnn_module, "_has_triton", False)
        with pytest.raises(RuntimeError, match="triton"):
            GRUModule(
                input_size=3,
                hidden_size=12,
                num_layers=1,
                in_keys=["obs", "hidden"],
                out_keys=["feat", ("next", "hidden")],
                recurrent_backend="triton",
            )

    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_module_three_backends_equivalent(self, num_layers):
        """pad / scan / triton agree at the intersection of supported configs.

        Scan does not support dropout, so this test fixes ``dropout=0``; the
        pad-vs-triton dropout case is covered separately by
        ``test_gru_module_triton_extended_forward_matches_pad``.
        """
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": num_layers,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
            "device": device,
        }
        pad_module = GRUModule(**kwargs)
        scan_module = GRUModule(**kwargs, recurrent_backend="scan")
        triton_module = GRUModule(**kwargs, recurrent_backend="triton")
        scan_module.load_state_dict(pad_module.state_dict())
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.randn(B, T, num_layers, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[1, 5] = True
        is_init[2, 1] = True
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
            [B, T],
        )

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(data.clone())
            scan_out = scan_module(data.clone())
            triton_out = triton_module(data.clone())

        for key in ["feat", ("next", "hidden")]:
            torch.testing.assert_close(
                pad_out[key], scan_out[key], atol=5e-3, rtol=5e-3
            )
            torch.testing.assert_close(
                pad_out[key], triton_out[key], atol=5e-3, rtol=5e-3
            )


def test_get_primers_from_module():

    # No primers in the model
    module = MLP(in_features=10, out_features=10, num_cells=[])
    transform = get_primers_from_module(module)
    assert transform is None

    # 1 primer in the model
    gru_module = GRUModule(
        input_size=10,
        hidden_size=10,
        num_layers=1,
        in_keys=["input", "gru_recurrent_state", "is_init"],
        out_keys=["features", ("next", "gru_recurrent_state")],
    )
    transform = get_primers_from_module(gru_module)
    assert isinstance(transform, TensorDictPrimer)
    assert "gru_recurrent_state" in transform.primers

    # 2 primers in the model
    composed_model = TensorDictSequential(
        gru_module,
        LSTMModule(
            input_size=10,
            hidden_size=10,
            num_layers=1,
            in_keys=[
                "input",
                "lstm_recurrent_state_c",
                "lstm_recurrent_state_h",
                "is_init",
            ],
            out_keys=[
                "features",
                ("next", "lstm_recurrent_state_c"),
                ("next", "lstm_recurrent_state_h"),
            ],
        ),
    )
    transform = get_primers_from_module(composed_model)
    assert isinstance(transform, Compose)
    assert len(transform) == 2
    assert "gru_recurrent_state" in transform[0].primers
    assert "lstm_recurrent_state_c" in transform[1].primers
    assert "lstm_recurrent_state_h" in transform[1].primers


def test_get_primers_from_module_partial_failure():
    """One submodule's failed primer must not poison primers from siblings.

    ``ConsistentDropoutModule`` without ``input_shape`` raises ``RuntimeError``
    from ``make_tensordict_primer()``. Pre-fix, ``module.apply``'s walk
    propagated that exception and the GRU's well-formed primer was lost too.
    The dry-run path now passes ``strict=False`` to isolate failures
    per-submodule.
    """
    dropout = ConsistentDropoutModule(p=0.1, in_keys="features")
    gru = GRUModule(
        input_size=10,
        hidden_size=10,
        num_layers=1,
        in_keys=["input", "recurrent_state", "is_init"],
        out_keys=["features", ("next", "recurrent_state")],
    )
    policy = TensorDictSequential(gru, dropout)

    # Public default (strict=True): the bad primer still raises.
    with pytest.raises(RuntimeError, match="input_shape"):
        get_primers_from_module(policy)

    # strict=False: GRU primer survives, warning names the failing module.
    with pytest.warns(UserWarning, match="ConsistentDropoutModule"):
        primer = get_primers_from_module(policy, warn=False, strict=False)
    assert primer is not None
    assert "recurrent_state" in primer.primers

    # Collector dry-run end-to-end: bare env gets InitTracker AND the GRU
    # primer. Pre-fix it would have gotten only InitTracker.
    env = CountingEnv()
    transforms = _compute_missing_env_transforms(env, policy)
    type_names = [type(t).__name__ for t in transforms]
    assert "InitTracker" in type_names
    assert "TensorDictPrimer" in type_names


def test_get_env_transforms_from_module_no_rnn():
    """Non-recurrent module returns a bare InitTracker."""
    module = MLP(in_features=10, out_features=10, num_cells=[])
    transforms = get_env_transforms_from_module(module)
    assert isinstance(transforms, InitTracker)


def test_get_env_transforms_from_module_gru():
    """GRUModule returns Compose([InitTracker, TensorDictPrimer])."""
    gru = GRUModule(
        input_size=10,
        hidden_size=10,
        num_layers=1,
        in_keys=["input", "recurrent_state", "is_init"],
        out_keys=["features", ("next", "recurrent_state")],
    )
    transforms = get_env_transforms_from_module(gru)
    assert isinstance(transforms, Compose)
    assert any(isinstance(t, InitTracker) for t in transforms.transforms)
    assert any(isinstance(t, TensorDictPrimer) for t in transforms.transforms)
    # InitTracker comes before TensorDictPrimer
    types = [type(t) for t in transforms.transforms]
    assert types.index(InitTracker) < types.index(TensorDictPrimer)


def test_get_env_transforms_from_module_lstm():
    """LSTMModule returns Compose([InitTracker, TensorDictPrimer])."""
    lstm = LSTMModule(
        input_size=10,
        hidden_size=10,
        num_layers=1,
        in_keys=["input", "h", "c", "is_init"],
        out_keys=["features", ("next", "h"), ("next", "c")],
    )
    transforms = get_env_transforms_from_module(lstm)
    assert isinstance(transforms, Compose)
    assert any(isinstance(t, InitTracker) for t in transforms.transforms)
    assert any(isinstance(t, TensorDictPrimer) for t in transforms.transforms)


def test_get_env_transforms_from_module_custom_init_key():
    """custom init_key is forwarded to InitTracker."""
    # Use a plain MLP so we don't need to thread init_key through GRU validation
    module = MLP(in_features=4, out_features=4, num_cells=[])
    transforms = get_env_transforms_from_module(module, init_key="my_init")
    # No RNN → bare InitTracker
    assert isinstance(transforms, InitTracker)
    assert transforms.init_key == "my_init"


def _make_recurrent_counting_policy():
    obs_to_float = TensorDictModule(
        lambda observation: observation.to(torch.get_default_dtype()),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    gru = GRUModule(
        input_size=1,
        hidden_size=1,
        num_layers=1,
        in_keys=["embed", "recurrent_state", "is_init"],
        out_keys=["embed", ("next", "recurrent_state")],
    )
    action = TensorDictModule(
        lambda embed: torch.ones_like(embed, dtype=torch.bool),
        in_keys=["embed"],
        out_keys=["action"],
    )
    return TensorDictSequential(obs_to_float, gru, action)


def test_env_policy_argument_adds_recurrent_transforms():
    """EnvBase subclasses accept policy=... and return a prepared TransformedEnv."""
    policy = _make_recurrent_counting_policy()
    env = CountingEnv(max_steps=3, policy=policy)
    assert isinstance(env, TransformedEnv)
    assert any(isinstance(t, InitTracker) for t in env.transform)
    assert any(isinstance(t, TensorDictPrimer) for t in env.transform)

    tensordict = env.reset()
    assert "is_init" in tensordict.keys()
    assert "recurrent_state" in tensordict.keys()


def test_collector_adds_recurrent_env_transforms():
    """Collectors prepare bare envs from recurrent policies before rollout."""
    policy = _make_recurrent_counting_policy()
    collector = SyncDataCollector(
        CountingEnv(max_steps=3),
        policy,
        frames_per_batch=3,
        total_frames=3,
        auto_register_policy_transforms=True,
    )
    try:
        assert isinstance(collector.env, TransformedEnv)
        assert any(isinstance(t, InitTracker) for t in collector.env.transform)
        assert any(isinstance(t, TensorDictPrimer) for t in collector.env.transform)
        batch = next(iter(collector))
        assert "is_init" in batch.keys()
        assert "recurrent_state" in batch.keys()
    finally:
        collector.shutdown()


def test_collector_does_not_duplicate_recurrent_env_transforms():
    """Collector auto-setup is idempotent with env policy=... setup."""
    policy = _make_recurrent_counting_policy()
    collector = SyncDataCollector(
        CountingEnv(max_steps=3, policy=policy),
        policy,
        frames_per_batch=3,
        total_frames=3,
        auto_register_policy_transforms=True,
    )
    try:
        transforms = list(collector.env.transform)
        assert sum(isinstance(t, InitTracker) for t in transforms) == 1
        assert sum(isinstance(t, TensorDictPrimer) for t in transforms) == 1
    finally:
        collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
