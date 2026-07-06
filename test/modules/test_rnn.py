# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import importlib.util
import sys
import threading

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

from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
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
    canonicalize_rnn_subset,
    ConsistentDropoutModule,
    get_recurrent_matmul_precision,
    GRU,
    GRUCell,
    GRUModule,
    LSTM,
    LSTMCell,
    LSTMModule,
    MLP,
    ProbabilisticActor,
    set_recurrent_matmul_precision,
    set_recurrent_mode,
    ValueOperator,
)
from torchrl.modules.tensordict_module._rnn_triton import _resolve_save_gates
from torchrl.modules.tensordict_module.rnn import (
    _canonical_contiguous,
    _canonical_stride,
)
from torchrl.modules.utils import (
    get_env_transforms_from_module,
    get_primers_from_module,
)
from torchrl.modules.utils.utils import _compute_missing_env_transforms
from torchrl.objectives.value.advantages import GAE

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import (
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionVecMockEnv,
)

_has_hoptorch = importlib.util.find_spec("hoptorch") is not None
_vmap = None


def _get_vmap():
    global _vmap
    if _vmap is None:
        if hasattr(torch, "vmap"):
            _vmap = torch.vmap
        else:
            from functorch import vmap

            _vmap = vmap
    return _vmap


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

    def test_set_recurrent_mode_is_thread_local(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        values = []

        def worker():
            values.append(lstm_module.recurrent_mode)

        with set_recurrent_mode(True):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert lstm_module.recurrent_mode
        assert values == [False]

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
        vmap = _get_vmap()

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
            # vmap cannot trace through ``torch._higher_order_ops.scan``; the
            # 'pad' backend keeps the time loop as a plain Python call into
            # the Python-based LSTM, which is fully vmap-compatible.
            recurrent_backend="pad",
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

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [16, 64, 512])
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

    @pytest.mark.gpu
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

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_lstm_module_triton_backward(self):
        """Backward path: gradients match pad backend within tolerance.

        Uses the fused path (H below ``_FWD_TILED_H_MIN``). The autotuned fused
        backward at large H exceeds the unit-test timeout on CI GPUs, and the
        tiled backward is only reachable via recompute (see #3752), so large-H
        backward parity is left to the benchmark/recompute paths.
        """
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

    @pytest.mark.gpu
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

    @pytest.mark.gpu
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

    @pytest.mark.parametrize("nested_in_key", [False, True])
    def test_lstm_canonicalize_subset(self, nested_in_key):
        in_key = ("obs", "value") if nested_in_key else "obs"
        module = LSTMModule(
            input_size=3,
            hidden_size=4,
            in_key=in_key,
            out_key="out",
        )
        # obs is canonical, reward is not. canonicalize should touch obs only.
        obs = torch.randn(2, 5, 3)
        reward = torch.randn(2, 5, 1).transpose(0, 1).transpose(0, 1)
        td = TensorDict({in_key: obs, "reward": reward}, batch_size=[2, 5])
        reward_ptr = td["reward"].data_ptr()
        out = module.canonicalize(td)
        assert out["obs" if not nested_in_key else ("obs", "value")].is_contiguous()
        # Unrelated key untouched (same storage).
        assert out["reward"].data_ptr() == reward_ptr

    def test_lstm_canonicalize_inplace(self):
        module = LSTMModule(input_size=3, hidden_size=4, in_key="obs", out_key="out")
        obs = torch.randn(2, 5, 3)
        td = TensorDict({"obs": obs}, batch_size=[2, 5])
        out = module.canonicalize(td, inplace=True)
        assert out is td

    def test_canonicalize_rnn_subset_free_fn(self):
        lstm = LSTMModule(input_size=3, hidden_size=4, in_key="obs", out_key="lstm_out")
        gru = GRUModule(input_size=3, hidden_size=4, in_key="obs", out_key="gru_out")
        td = TensorDict(
            {"obs": torch.randn(2, 5, 3), "reward": torch.randn(2, 5, 1)},
            batch_size=[2, 5],
        )
        reward_ptr = td["reward"].data_ptr()
        out = canonicalize_rnn_subset(td, [lstm, gru])
        assert out["obs"].is_contiguous()
        assert out["reward"].data_ptr() == reward_ptr

    @pytest.mark.parametrize("backend", ["auto", "pad"])
    def test_lstm_recompute_rejected_for_non_recompute_backend(self, backend):
        with pytest.raises(ValueError, match="recurrent_recompute"):
            LSTMModule(
                input_size=3,
                hidden_size=4,
                in_key="obs",
                out_key="out",
                recurrent_backend=backend,
                recurrent_recompute="full",
            )

    def test_lstm_recompute_invalid_value(self):
        with pytest.raises(ValueError, match="recurrent_recompute"):
            LSTMModule(
                input_size=3,
                hidden_size=4,
                in_key="obs",
                out_key="out",
                recurrent_backend="scan",
                recurrent_recompute="partial",
            )

    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_scan_recompute_matches_pad(self, num_layers):
        """Scan + recompute uses a python time-loop, which matches cuDNN to fp precision."""
        torch.manual_seed(0)
        B, T, F, H = 4, 7, 3, 8
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": num_layers,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
        }
        pad_module = LSTMModule(**kwargs)
        rc_module = LSTMModule(
            **kwargs, recurrent_backend="scan", recurrent_recompute="full"
        )
        rc_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        hidden0 = torch.zeros(B, T, num_layers, H)
        hidden1 = torch.zeros(B, T, num_layers, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True
        is_init[1, 2] = True
        is_init[2, 5] = True

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            obs_leaf = obs.detach().clone().requires_grad_(True)
            d = TensorDict(
                {
                    "obs": obs_leaf,
                    "hidden0": hidden0,
                    "hidden1": hidden1,
                    "is_init": is_init,
                },
                [B, T],
            )
            out = mod(d)
            loss = (
                out["feat"].pow(2).sum()
                + out["next", "hidden0"].pow(2).sum()
                + out["next", "hidden1"].pow(2).sum()
            )
            loss.backward()
            return {
                "feat": out["feat"].detach(),
                "h0": out["next", "hidden0"].detach(),
                "h1": out["next", "hidden1"].detach(),
                "obs_grad": obs_leaf.grad.clone(),
                "params": {k: v.grad.clone() for k, v in mod.named_parameters()},
            }

        with set_recurrent_mode(True):
            torch.manual_seed(1)
            pad_res = loss_and_grads(pad_module)
            torch.manual_seed(1)
            rc_res = loss_and_grads(rc_module)
        torch.testing.assert_close(
            pad_res["feat"], rc_res["feat"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            pad_res["obs_grad"], rc_res["obs_grad"], atol=1e-5, rtol=1e-5
        )
        for k in pad_res["params"]:
            torch.testing.assert_close(
                pad_res["params"][k], rc_res["params"][k], atol=1e-5, rtol=1e-5
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_lstm_triton_recompute_parity(self):
        """Triton recompute matches non-recompute on outputs and grads."""
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": 1,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            "device": device,
            "recurrent_backend": "triton",
        }
        m_full = LSTMModule(**kwargs)
        m_rc = LSTMModule(**kwargs, recurrent_recompute="full")
        m_rc.load_state_dict(m_full.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden0 = torch.zeros(B, T, 1, H, device=device)
        hidden1 = torch.zeros(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            out = mod(data.clone())
            (out["feat"].pow(2).sum()).backward()
            return out["feat"].detach().clone(), {
                k: p.grad.detach().clone() for k, p in mod.named_parameters()
            }

        with set_recurrent_mode(True):
            out_full, grads_full = loss_and_grads(m_full)
            out_rc, grads_rc = loss_and_grads(m_rc)
        torch.testing.assert_close(out_full, out_rc, atol=1e-6, rtol=1e-6)
        for k in grads_full:
            torch.testing.assert_close(grads_full[k], grads_rc[k], atol=1e-5, rtol=1e-5)

    def test_resolve_save_gates(self):
        """Save_gates must be False under recompute / no_grad / no-track inputs.

        Avoids the dead allocation of save_i/f/g/o/save_tanhc buffers that
        would otherwise be captured by the CUDA caching allocator / cudagraph
        private pool even when backward will never read them.
        """
        x = torch.randn(2, 3, 4, requires_grad=True)
        x_nograd = x.detach()
        assert _resolve_save_gates(False, x) is True
        assert _resolve_save_gates(True, x) is False
        assert _resolve_save_gates(False, x_nograd) is False
        with torch.no_grad():
            assert _resolve_save_gates(False, x) is False
        # ``None`` entries (optional biases) must not break the check.
        assert _resolve_save_gates(False, x, None) is True

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    def test_scan_backend_backward_matches_pad(self, rnn_type):
        torch.manual_seed(0)
        B, T, F, H, L = 3, 5, 4, 8, 1
        if rnn_type == "lstm":
            kwargs = {
                "input_size": F,
                "hidden_size": H,
                "num_layers": L,
                "in_keys": ["obs", "hidden0", "hidden1"],
                "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            }
            pad_module = LSTMModule(**kwargs)
            scan_module = LSTMModule(**kwargs, recurrent_backend="scan")
        else:
            kwargs = {
                "input_size": F,
                "hidden_size": H,
                "num_layers": L,
                "in_keys": ["obs", "hidden"],
                "out_keys": ["feat", ("next", "hidden")],
            }
            pad_module = GRUModule(**kwargs)
            scan_module = GRUModule(**kwargs, recurrent_backend="scan")
        scan_module.load_state_dict(pad_module.state_dict())

        obs_source = torch.randn(B, T, F)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        is_init[1, 3] = True
        is_init[2, 2] = True

        def make_data():
            obs = obs_source.detach().clone().requires_grad_()
            if rnn_type == "lstm":
                hidden0 = torch.zeros(B, T, L, H)
                hidden1 = torch.zeros(B, T, L, H)
                data = TensorDict(
                    {
                        "obs": obs,
                        "hidden0": hidden0,
                        "hidden1": hidden1,
                        "is_init": is_init.clone(),
                    },
                    [B, T],
                )
            else:
                hidden = torch.zeros(B, T, L, H)
                data = TensorDict(
                    {"obs": obs, "hidden": hidden, "is_init": is_init.clone()},
                    [B, T],
                )
            return data, obs

        def grads(module):
            data, obs = make_data()
            with set_recurrent_mode(True):
                out = module(data)
                loss = out["feat"].square().mean()
            loss.backward()
            param_grads = [
                (name, param.grad.detach().clone())
                for name, param in module.named_parameters()
            ]
            return obs.grad.detach().clone(), param_grads

        pad_obs_grad, pad_param_grads = grads(pad_module)
        scan_obs_grad, scan_param_grads = grads(scan_module)

        torch.testing.assert_close(pad_obs_grad, scan_obs_grad, atol=5e-3, rtol=5e-3)
        for (pad_name, pad_grad), (scan_name, scan_grad) in zip(
            pad_param_grads, scan_param_grads
        ):
            assert pad_name == scan_name
            torch.testing.assert_close(pad_grad, scan_grad, atol=5e-3, rtol=5e-3)

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("backend", ["scan", "triton"])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_backends_non_contiguous_recurrent_inputs(self, backend):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H, L = 4, 6, 3, 8, 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": L,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
            "device": device,
        }
        pad_module = LSTMModule(**kwargs)
        module = LSTMModule(**kwargs, recurrent_backend=backend)
        module.load_state_dict(pad_module.state_dict())

        obs_source = torch.randn(T, B, F, device=device)
        hidden0_source = torch.randn(T, B, L, H, device=device)
        hidden1_source = torch.randn(T, B, L, H, device=device)
        is_init_source = torch.zeros(T, B, 1, dtype=torch.bool, device=device)
        is_init_source[0] = True
        is_init_source[2, 1] = True
        is_init_source[4, 3] = True

        def make_data():
            obs_base = obs_source.detach().clone().requires_grad_()
            hidden0_base = hidden0_source.detach().clone().requires_grad_()
            hidden1_base = hidden1_source.detach().clone().requires_grad_()
            obs = obs_base.transpose(0, 1)
            hidden0 = hidden0_base.transpose(0, 1)
            hidden1 = hidden1_base.transpose(0, 1)
            is_init = is_init_source.transpose(0, 1)
            assert not obs.is_contiguous()
            assert not is_init.is_contiguous()
            return (
                TensorDict(
                    {
                        "obs": obs,
                        "hidden0": hidden0,
                        "hidden1": hidden1,
                        "is_init": is_init,
                    },
                    [B, T],
                ),
                obs_base,
                hidden0_base,
                hidden1_base,
            )

        pad_data, pad_obs, pad_hidden0, pad_hidden1 = make_data()
        data, obs, hidden0, hidden1 = make_data()
        with set_recurrent_mode(True):
            pad_out = pad_module(pad_data)
            out = module(data)

        tol = 5e-3
        for key in ["feat", ("next", "hidden0"), ("next", "hidden1")]:
            torch.testing.assert_close(pad_out[key], out[key], atol=tol, rtol=tol)

        pad_out["feat"].square().mean().backward()
        out["feat"].square().mean().backward()
        torch.testing.assert_close(pad_obs.grad, obs.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(pad_hidden0.grad, hidden0.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(pad_hidden1.grad, hidden1.grad, atol=tol, rtol=tol)
        for (_, pad_param), (_, param) in zip(
            pad_module.named_parameters(), module.named_parameters()
        ):
            torch.testing.assert_close(pad_param.grad, param.grad, atol=tol, rtol=tol)

    def test_canonical_contiguous_helper(self):
        # _canonical_contiguous must be a no-op when the input strides match
        # the C-canonical layout, and must materialize a fresh canonical
        # tensor when they don't (the size-1-dim quirk).
        canonical = torch.randn(3, 4, 5)
        assert _canonical_stride(canonical.shape) == (20, 5, 1)
        out = _canonical_contiguous(canonical)
        assert out.data_ptr() == canonical.data_ptr()

        # A [1, 4, 5] tensor with strides (5, 5, 1) passes is_contiguous()
        # (size-1 leading dim makes stride[0] irrelevant for that check)
        # but is not canonical — canonical would be (20, 5, 1).
        weird = torch.empty_strided((1, 4, 5), (5, 5, 1))
        weird.copy_(torch.randn(1, 4, 5))
        assert weird.is_contiguous()
        assert tuple(weird.stride()) != _canonical_stride(weird.shape)
        out = _canonical_contiguous(weird)
        assert out.data_ptr() != weird.data_ptr()
        assert tuple(out.stride()) == _canonical_stride(out.shape)
        torch.testing.assert_close(out, weird)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="torch.compile scan tests need a C compiler"
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_scan_compile_no_aliasing(self):
        # Under torch.compile, scan's HOP tracer walks the FakeTensor graph
        # and rejects shared storage on inputs. nn.LSTM with cuDNN flattens
        # its parameters into a single storage, so the per-layer weight
        # views alias each other — `_lstm_scan_with_resets` must clone
        # them before closing the scan body over them. This test pins
        # that contract so the clones are not silently removed again.
        torch.manual_seed(0)
        B, T, F_in, H, L = 4, 6, 3, 8, 1
        scan_module = LSTMModule(
            input_size=F_in,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="scan",
        )

        obs = torch.randn(B, T, F_in)
        hidden0 = torch.zeros(B, T, L, H)
        hidden1 = torch.zeros(B, T, L, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        data = TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [B, T],
        )

        prev = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        try:

            @torch.compile(fullgraph=False)
            def call(td):
                with set_recurrent_mode(True):
                    return scan_module(td)

            with torch.no_grad():
                out = call(data.clone())
        finally:
            torch._dynamo.config.capture_scalar_outputs = prev
        assert "feat" in out.keys(True, True)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="torch.compile scan tests need a C compiler"
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_module_scan_compile_no_aliasing(self):
        torch.manual_seed(0)
        B, T, F_in, H, L = 4, 6, 3, 8, 1
        scan_module = GRUModule(
            input_size=F_in,
            hidden_size=H,
            num_layers=L,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="scan",
        )

        obs = torch.randn(B, T, F_in)
        hidden = torch.zeros(B, T, L, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
            [B, T],
        )

        prev = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        try:

            @torch.compile(fullgraph=False)
            def call(td):
                with set_recurrent_mode(True):
                    return scan_module(td)

            with torch.no_grad():
                out = call(data.clone())
        finally:
            torch._dynamo.config.capture_scalar_outputs = prev
        assert "feat" in out.keys(True, True)

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch.compile recurrent path requires Torch >= 2.6.0",
    )
    @pytest.mark.skipif(
        sys.platform == "win32", reason="torch.compile tests need a C compiler"
    )
    def test_module_pad_backend_compile_with_resets(self, rnn_type):
        # Regression: the pad (cuDNN) backend cuts multi-trajectory rollouts via
        # the data-dependent _split_and_pad_sequence / _inv_pad_sequence. Under
        # torch.compile the boolean-mask reconstruction (tensor[mask]) produced
        # a data-dependent shape and crashed with "torch.Size() takes an
        # iterable of 'int' (item 0 is 'FakeTensor')". _split_and_pad_for_reset /
        # _inv_pad_for_reset now run those as eager islands so the pad backend
        # compiles; the compiled output must still match eager. Note there is no
        # capture_scalar_outputs toggle here: the pad path must compile under the
        # default config.
        torch.manual_seed(0)
        B, T, F_in, H, L = 4, 8, 3, 8, 1
        obs = torch.randn(B, T, F_in)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        is_init[1, 4] = True  # mid-row reset -> exercises split/pad/unpad
        if rnn_type == "gru":
            module = GRUModule(
                input_size=F_in,
                hidden_size=H,
                num_layers=L,
                in_keys=["obs", "hidden"],
                out_keys=["feat", ("next", "hidden")],
                recurrent_backend="pad",
            )
            hidden = {"hidden": torch.zeros(B, T, L, H)}
        else:
            module = LSTMModule(
                input_size=F_in,
                hidden_size=H,
                num_layers=L,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="pad",
            )
            hidden = {
                "hidden0": torch.zeros(B, T, L, H),
                "hidden1": torch.zeros(B, T, L, H),
            }
        data = TensorDict({"obs": obs, "is_init": is_init, **hidden}, [B, T])

        with set_recurrent_mode(True), torch.no_grad():
            ref = module(data.clone())

        @torch.compile
        def call(td):
            with set_recurrent_mode(True):
                return module(td)

        with torch.no_grad():
            out = call(data.clone())
        torch.testing.assert_close(out["feat"], ref["feat"])

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_three_backends_ppo_advantage_parity(self):
        # End-to-end pin: feeding the LSTM output through a value head and
        # GAE should produce identical advantages across all backends. The
        # existing three_backends_equivalent test already pins feat and the
        # per-step hidden tensors; this test pins the downstream PPO
        # contract so a regression in either step is caught.
        torch.manual_seed(0)
        B, T, F_in, H, L = 4, 7, 3, 16, 1
        kwargs = {
            "input_size": F_in,
            "hidden_size": H,
            "num_layers": L,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
        }
        pad_module = LSTMModule(**kwargs)
        scan_module = LSTMModule(**kwargs, recurrent_backend="scan")
        scan_module.load_state_dict(pad_module.state_dict())
        critic = ValueOperator(nn.Linear(H, 1), in_keys=["feat"])

        obs = torch.randn(B, T, F_in)
        hidden0 = torch.zeros(B, T, L, H)
        hidden1 = torch.zeros(B, T, L, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        is_init[1, 3] = True
        is_init[2, 5] = True
        reward = torch.randn(B, T, 1)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[1, 2] = True
        done[2, 4] = True
        terminated = done.clone()
        next_obs = torch.randn(B, T, F_in)
        data = TensorDict(
            {
                "obs": obs,
                "hidden0": hidden0,
                "hidden1": hidden1,
                "is_init": is_init,
                "next": TensorDict(
                    {
                        "obs": next_obs,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                    },
                    [B, T],
                ),
            },
            [B, T],
        )

        def pipeline(lstm_module, td):
            with set_recurrent_mode(True), torch.no_grad():
                td = lstm_module(td)
                td = critic(td)
                # Mirror critic onto ("next", "state_value") so GAE has the
                # bootstrap term it expects.
                next_td = td["next"].clone()
                next_td["feat"] = td["feat"]  # placeholder; critic just needs feat
                # use the actual next_feat via a single re-call on next obs
                next_lstm_in = TensorDict(
                    {
                        "obs": td["next", "obs"],
                        "hidden0": td["next", "hidden0"],
                        "hidden1": td["next", "hidden1"],
                        "is_init": td["is_init"],
                    },
                    td.batch_size,
                )
                next_lstm_out = lstm_module(next_lstm_in)
                next_state_value = critic(
                    TensorDict({"feat": next_lstm_out["feat"]}, td.batch_size)
                )["state_value"]
                td["next", "state_value"] = next_state_value
                gae = GAE(
                    gamma=0.99,
                    lmbda=0.95,
                    value_network=None,
                    average_gae=False,
                    shifted=False,
                )
                gae(td)
            return td["advantage"], td["value_target"], td["state_value"]

        adv_pad, vt_pad, val_pad = pipeline(pad_module, data.clone())
        adv_scan, vt_scan, val_scan = pipeline(scan_module, data.clone())
        torch.testing.assert_close(val_pad, val_scan, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(adv_pad, adv_scan, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(vt_pad, vt_scan, atol=5e-3, rtol=5e-3)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_lstm_module_scan_non_canonical_hidden_strides(self):
        # Hidden buffers whose strides pass is_contiguous() but disagree with
        # the canonical row-major layout (the size-1-dim quirk that bit the
        # Isaac PPO run) must not break the scan backend.
        torch.manual_seed(0)
        B, T, F, H, L = 4, 6, 3, 8, 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": L,
            "in_keys": ["obs", "hidden0", "hidden1"],
            "out_keys": ["feat", ("next", "hidden0"), ("next", "hidden1")],
        }
        pad_module = LSTMModule(**kwargs)
        scan_module = LSTMModule(**kwargs, recurrent_backend="scan")
        scan_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        # Build a [B, T, L, H] hidden buffer whose internal [..., 0, :, :]
        # slice then transpose(-3, -2) yields a [L, B, H] view that
        # is_contiguous() but has non-canonical strides. The same buffer
        # shape PPO uses; the stride layout matters more than the values.
        canonical_h = torch.zeros(B, T, L, H)
        h_storage = torch.randn(B * L * H)
        # Lay out so that hidden_buf[:, 0] -> [B, L, H] indexing then
        # transposing yields the size-1-dim non-canonical stride pattern.
        hidden_buf = torch.empty_strided((B, T, L, H), (L * H, L * H, H, 1))
        hidden_buf.zero_()
        hidden_buf[:, 0] = h_storage.view(B, L, H)
        # sanity check that the runtime path does see a non-canonical view
        probe = hidden_buf[:, 0].transpose(-3, -2)
        assert probe.is_contiguous()
        assert tuple(probe.stride()) != _canonical_stride(probe.shape)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        data = TensorDict(
            {
                "obs": obs,
                "hidden0": hidden_buf,
                "hidden1": hidden_buf.clone().contiguous(),
                "is_init": is_init,
            },
            [B, T],
        )
        canonical_data = TensorDict(
            {
                "obs": obs,
                "hidden0": canonical_h.clone(),
                "hidden1": canonical_h.clone(),
                "is_init": is_init.clone(),
            },
            [B, T],
        )
        canonical_data["hidden0"][:, 0] = h_storage.view(B, L, H)
        canonical_data["hidden1"][:, 0] = canonical_data["hidden0"][:, 0]

        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(canonical_data.clone())
            # The weird-stride buffer must not crash scan, and must produce
            # the same outputs as the canonical-stride buffer with matching
            # values.
            scan_out = scan_module(data.clone())
        torch.testing.assert_close(pad_out["feat"], scan_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden0"], scan_out["next", "hidden0"]
        )
        torch.testing.assert_close(
            pad_out["next", "hidden1"], scan_out["next", "hidden1"]
        )


LSTM_LIFECYCLE_MAX_STEPS = 2  # episode runs 3 transitions before done -> very short
LSTM_LIFECYCLE_HIDDEN_SIZE = 8
LSTM_LIFECYCLE_FRAMES_PER_BATCH = 16  # >> episode length, so mid-batch dones occur


def _build_lstm_lifecycle_env_and_policy():
    """Set up the deterministic CountingEnv + LSTM policy used by the test.

    The LSTM rides along the rollout so its hidden state propagates and
    resets at trajectory boundaries; the actual action is driven by
    :class:`CountingEnvCountPolicy` so the env terminates on a known
    schedule and we get reliable mid-batch ``done`` events.
    """
    base_env = CountingEnv(max_steps=LSTM_LIFECYCLE_MAX_STEPS)
    obs_size = base_env.observation_spec["observation"].shape[-1]

    lstm_module = LSTMModule(
        input_size=obs_size,
        hidden_size=LSTM_LIFECYCLE_HIDDEN_SIZE,
        in_keys=["obs_float", "rs_h", "rs_c"],
        out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
        python_based=True,
        dropout=0,
    )
    lstm_module.eval()  # deterministic: no dropout or train-only behavior

    env = TransformedEnv(base_env, InitTracker())
    env = env.append_transform(lstm_module.make_tensordict_primer())
    env.set_seed(0)

    # Cast int32 observation to float so the LSTM can consume it. Kept as a
    # tiny TensorDictModule rather than an env transform so we don't conflate
    # the recurrent-state lifecycle test with transform plumbing.
    def _to_float(obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float32)

    cast_obs = TensorDictModule(
        _to_float, in_keys=["observation"], out_keys=["obs_float"]
    )
    action_module = CountingEnvCountPolicy(action_spec=base_env.action_spec)

    policy = TensorDictSequential(cast_obs, lstm_module, action_module)
    return env, policy, lstm_module


class TestLSTMRecurrentStateLifecycle:
    """End-to-end integration test for the recurrent-state lifecycle."""

    def test_lstm_collector_replay_mid_batch_done_resets_hidden_state(self):
        torch.manual_seed(0)
        env, policy, lstm_module = _build_lstm_lifecycle_env_and_policy()

        # --- Phase 1: collect a batch through Collector -----------------
        collector = Collector(
            env,
            policy=policy,
            frames_per_batch=LSTM_LIFECYCLE_FRAMES_PER_BATCH,
            total_frames=LSTM_LIFECYCLE_FRAMES_PER_BATCH,
            reset_at_each_iter=False,
        )
        try:
            data = next(iter(collector))
        finally:
            collector.shutdown()

        # Structural assertions: lifecycle keys are present, mid-batch
        # trajectory boundaries actually occurred.
        assert "is_init" in data.keys(), "InitTracker did not emit is_init"
        assert "rs_h" in data.keys(), "primer did not surface recurrent state h"
        assert "rs_c" in data.keys(), "primer did not surface recurrent state c"

        is_init = data["is_init"].squeeze(-1)
        assert bool(is_init[0].item()), "is_init must be True at the first step"
        n_resets = int(is_init.sum().item())
        assert n_resets >= 2, (
            f"expected at least 2 trajectory boundaries in "
            f"{LSTM_LIFECYCLE_FRAMES_PER_BATCH} frames with "
            f"max_steps={LSTM_LIFECYCLE_MAX_STEPS}, got {n_resets}"
        )

        # Recurrent state at every is_init=True position must be the primer zero:
        # this is the per-step reset invariant. Without it the LSTM's
        # sequential-mode reset block is broken.
        rs_h_at_inits = data["rs_h"][is_init]
        rs_c_at_inits = data["rs_c"][is_init]
        assert torch.equal(rs_h_at_inits, torch.zeros_like(rs_h_at_inits)), (
            "incoming recurrent_state_h at is_init=True positions should be "
            "the primer zero; found non-zero values, suggesting hidden state "
            "leaked across a trajectory boundary in the collector"
        )
        assert torch.equal(rs_c_at_inits, torch.zeros_like(rs_c_at_inits)), (
            "incoming recurrent_state_c at is_init=True positions should be "
            "the primer zero; see above"
        )

        # --- Phase 2: round-trip through ReplayBuffer + SliceSampler ----
        # slice_len matches one full episode (max_steps + 1 transitions), so
        # each sampled slice should be exactly one trajectory.
        slice_len = LSTM_LIFECYCLE_MAX_STEPS + 1
        num_slices = 2
        buffer = ReplayBuffer(
            storage=LazyTensorStorage(LSTM_LIFECYCLE_FRAMES_PER_BATCH),
            sampler=SliceSampler(
                slice_len=slice_len,
                end_key=("next", "done"),
                strict_length=True,
            ),
        )
        buffer.extend(data)
        sampled = buffer.sample(num_slices * slice_len)
        sampled = sampled.reshape(num_slices, slice_len)

        sampled_is_init = sampled["is_init"].squeeze(-1)
        assert sampled_is_init[:, 0].all(), (
            "every SliceSampler-returned slice should begin at a trajectory "
            "boundary (is_init=True at slice index 0)"
        )
        assert not sampled_is_init[:, 1:].any(), (
            "sliced trajectories should contain no interior is_init=True; "
            "the sampler must respect end_key=('next', 'done')"
        )

        # --- Phase 3: no-leakage check ---------------------------------
        # Build two adjacent trajectories from the flat rollout. Lengths are
        # computed from is_init: a new trajectory starts at every is_init=True,
        # so lengths are the gaps between consecutive trues.
        init_positions = is_init.nonzero(as_tuple=False).squeeze(-1).tolist()
        assert len(init_positions) >= 3, (
            f"need at least 3 is_init boundaries to extract 2 complete "
            f"trajectories; got positions={init_positions}"
        )
        len_a = init_positions[1] - init_positions[0]
        len_b = init_positions[2] - init_positions[1]
        packed_t = len_a + len_b
        packed = data[:packed_t].reshape(1, packed_t).clone()
        b_alone = data[len_a : len_a + len_b].reshape(1, len_b).clone()

        # Seed packed's incoming hidden with non-zero noise to make any leakage
        # detectable: if recurrent-mode forward fails to zero B's hidden at its
        # is_init=True boundary, B's outputs will pick up this noise. b_alone
        # gets the same noise; the split-and-pad path inside LSTMModule.forward
        # must override both.
        noise_h = torch.randn_like(packed["rs_h"])
        noise_c = torch.randn_like(packed["rs_c"])
        packed["rs_h"] = noise_h
        packed["rs_c"] = noise_c
        b_alone["rs_h"] = noise_h[:, len_a:].clone()
        b_alone["rs_c"] = noise_c[:, len_a:].clone()

        with set_recurrent_mode(True):
            packed_out = lstm_module(packed)
            b_alone_out = lstm_module(b_alone)

        # Trajectory B's LSTM outputs inside the packed batch must match the
        # standalone run. Hidden-state leakage from A through the is_init
        # boundary would make these diverge.
        torch.testing.assert_close(
            packed_out["intermediate"][:, len_a:],
            b_alone_out["intermediate"],
            rtol=1e-5,
            atol=1e-6,
            msg="hidden state leaked across is_init trajectory boundary",
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
        vmap = _get_vmap()

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
            # vmap cannot trace through ``torch._higher_order_ops.scan``; the
            # 'pad' backend keeps the time loop as a plain Python call into
            # the Python-based GRU, which is fully vmap-compatible.
            recurrent_backend="pad",
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

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [16, 64, 512])
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

    @pytest.mark.gpu
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

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_gru_module_triton_backward(self):
        """Backward path: gradients match pad backend within tolerance.

        Uses the fused path (H below ``_FWD_TILED_H_MIN``). The autotuned fused
        backward at large H exceeds the unit-test timeout on CI GPUs, and the
        tiled backward is only reachable via recompute (see #3752), so large-H
        backward parity is left to the benchmark/recompute paths.
        """
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

    @pytest.mark.gpu
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

    @pytest.mark.gpu
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

    @pytest.mark.parametrize("nested_in_key", [False, True])
    def test_gru_canonicalize_subset(self, nested_in_key):
        in_key = ("obs", "value") if nested_in_key else "obs"
        module = GRUModule(
            input_size=3,
            hidden_size=4,
            in_key=in_key,
            out_key="out",
        )
        obs = torch.randn(2, 5, 3)
        reward = torch.randn(2, 5, 1)
        td = TensorDict({in_key: obs, "reward": reward}, batch_size=[2, 5])
        reward_ptr = td["reward"].data_ptr()
        out = module.canonicalize(td)
        assert out["obs" if not nested_in_key else ("obs", "value")].is_contiguous()
        assert out["reward"].data_ptr() == reward_ptr

    @pytest.mark.parametrize("backend", ["auto", "pad"])
    def test_gru_recompute_rejected_for_non_recompute_backend(self, backend):
        with pytest.raises(ValueError, match="recurrent_recompute"):
            GRUModule(
                input_size=3,
                hidden_size=4,
                in_key="obs",
                out_key="out",
                recurrent_backend=backend,
                recurrent_recompute="full",
            )

    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_scan_recompute_matches_pad(self, num_layers):
        torch.manual_seed(0)
        B, T, F, H = 4, 7, 3, 8
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": num_layers,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
        }
        pad_module = GRUModule(**kwargs)
        rc_module = GRUModule(
            **kwargs, recurrent_backend="scan", recurrent_recompute="full"
        )
        rc_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        hidden = torch.zeros(B, T, num_layers, H)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[0, 3] = True
        is_init[1, 2] = True

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            obs_leaf = obs.detach().clone().requires_grad_(True)
            d = TensorDict(
                {"obs": obs_leaf, "hidden": hidden, "is_init": is_init}, [B, T]
            )
            out = mod(d)
            (out["feat"].pow(2).sum() + out["next", "hidden"].pow(2).sum()).backward()
            return {
                "feat": out["feat"].detach(),
                "obs_grad": obs_leaf.grad.clone(),
                "params": {k: v.grad.clone() for k, v in mod.named_parameters()},
            }

        with set_recurrent_mode(True):
            torch.manual_seed(1)
            pad_res = loss_and_grads(pad_module)
            torch.manual_seed(1)
            rc_res = loss_and_grads(rc_module)
        torch.testing.assert_close(
            pad_res["feat"], rc_res["feat"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            pad_res["obs_grad"], rc_res["obs_grad"], atol=1e-5, rtol=1e-5
        )
        for k in pad_res["params"]:
            torch.testing.assert_close(
                pad_res["params"][k], rc_res["params"][k], atol=1e-5, rtol=1e-5
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_gru_triton_recompute_parity(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H = 4, 7, 3, 16
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": 1,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
            "device": device,
            "recurrent_backend": "triton",
        }
        m_full = GRUModule(**kwargs)
        m_rc = GRUModule(**kwargs, recurrent_recompute="full")
        m_rc.load_state_dict(m_full.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.zeros(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        data = TensorDict({"obs": obs, "hidden": hidden, "is_init": is_init}, [B, T])

        def loss_and_grads(mod):
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad = None
            out = mod(data.clone())
            out["feat"].pow(2).sum().backward()
            return out["feat"].detach().clone(), {
                k: p.grad.detach().clone() for k, p in mod.named_parameters()
            }

        with set_recurrent_mode(True):
            out_full, grads_full = loss_and_grads(m_full)
            out_rc, grads_rc = loss_and_grads(m_rc)
        torch.testing.assert_close(out_full, out_rc, atol=1e-6, rtol=1e-6)
        for k in grads_full:
            torch.testing.assert_close(grads_full[k], grads_rc[k], atol=1e-5, rtol=1e-5)

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("backend", ["scan", "triton"])
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_module_backends_non_contiguous_recurrent_inputs(self, backend):
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, T, F, H, L = 4, 6, 3, 8, 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": L,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
            "device": device,
        }
        pad_module = GRUModule(**kwargs)
        module = GRUModule(**kwargs, recurrent_backend=backend)
        module.load_state_dict(pad_module.state_dict())

        obs_source = torch.randn(T, B, F, device=device)
        hidden_source = torch.randn(T, B, L, H, device=device)
        is_init_source = torch.zeros(T, B, 1, dtype=torch.bool, device=device)
        is_init_source[0] = True
        is_init_source[2, 1] = True
        is_init_source[4, 3] = True

        def make_data():
            obs_base = obs_source.detach().clone().requires_grad_()
            hidden_base = hidden_source.detach().clone().requires_grad_()
            obs = obs_base.transpose(0, 1)
            hidden = hidden_base.transpose(0, 1)
            is_init = is_init_source.transpose(0, 1)
            assert not obs.is_contiguous()
            assert not is_init.is_contiguous()
            return (
                TensorDict(
                    {"obs": obs, "hidden": hidden, "is_init": is_init},
                    [B, T],
                ),
                obs_base,
                hidden_base,
            )

        pad_data, pad_obs, pad_hidden = make_data()
        data, obs, hidden = make_data()
        with set_recurrent_mode(True):
            pad_out = pad_module(pad_data)
            out = module(data)

        tol = 5e-3
        for key in ["feat", ("next", "hidden")]:
            torch.testing.assert_close(pad_out[key], out[key], atol=tol, rtol=tol)

        pad_out["feat"].square().mean().backward()
        out["feat"].square().mean().backward()
        torch.testing.assert_close(pad_obs.grad, obs.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(pad_hidden.grad, hidden.grad, atol=tol, rtol=tol)
        for (_, pad_param), (_, param) in zip(
            pad_module.named_parameters(), module.named_parameters()
        ):
            torch.testing.assert_close(pad_param.grad, param.grad, atol=tol, rtol=tol)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="torch._higher_order_ops.scan requires Torch >= 2.6.0",
    )
    def test_gru_module_scan_non_canonical_hidden_strides(self):
        # GRU counterpart of test_lstm_module_scan_non_canonical_hidden_strides.
        torch.manual_seed(0)
        B, T, F, H, L = 4, 6, 3, 8, 1
        kwargs = {
            "input_size": F,
            "hidden_size": H,
            "num_layers": L,
            "in_keys": ["obs", "hidden"],
            "out_keys": ["feat", ("next", "hidden")],
        }
        pad_module = GRUModule(**kwargs)
        scan_module = GRUModule(**kwargs, recurrent_backend="scan")
        scan_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F)
        h_storage = torch.randn(B * L * H)
        hidden_buf = torch.empty_strided((B, T, L, H), (L * H, L * H, H, 1))
        hidden_buf.zero_()
        hidden_buf[:, 0] = h_storage.view(B, L, H)
        probe = hidden_buf[:, 0].transpose(-3, -2)
        assert probe.is_contiguous()
        assert tuple(probe.stride()) != _canonical_stride(probe.shape)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0] = True
        canonical_h = torch.zeros(B, T, L, H)
        canonical_h[:, 0] = h_storage.view(B, L, H)
        canonical_data = TensorDict(
            {"obs": obs, "hidden": canonical_h, "is_init": is_init.clone()},
            [B, T],
        )
        data = TensorDict(
            {"obs": obs, "hidden": hidden_buf, "is_init": is_init},
            [B, T],
        )
        with set_recurrent_mode(True), torch.no_grad():
            pad_out = pad_module(canonical_data.clone())
            scan_out = scan_module(data.clone())
        torch.testing.assert_close(pad_out["feat"], scan_out["feat"])
        torch.testing.assert_close(
            pad_out["next", "hidden"], scan_out["next", "hidden"]
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
    collector = Collector(
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
    collector = Collector(
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


class TestRecurrentMatmulPrecision:
    """Tests for the global precision control plumbed into the triton RNN backend.

    Covers (a) the global setter/getter, (b) the env var, (c) per-module
    override, (d) the cuBLAS flip helpers, (e) the GPU-aware presets
    (``"fast"`` / ``"high-prec"``) with monkey-patched device capability,
    and (f) grad-parity for each precision against the pad backend. CPU-only
    tests live first; GPU tests are guarded with the standard ``_has_triton``
    skip.
    """

    def _restore_precision(self):
        # Tests mutate the process-global override; restore to "auto" at exit.
        from torchrl.modules.tensordict_module import _rnn_precision

        _rnn_precision._GLOBAL_OVERRIDE = _rnn_precision._read_env_default()
        # Clear the lru_cache so per-test monkeypatching takes effect.
        _rnn_precision._is_tensor_core_capable.cache_clear()

    def _patch_device_capability(
        self, monkeypatch, *, has_cuda, major, minor=0, hip=False
    ):
        """Pretend we're on a specific GPU for preset resolution.

        Clears the ``_is_tensor_core_capable`` lru_cache so the patch takes
        effect even after a previous resolution.
        """
        from torchrl.modules.tensordict_module import _rnn_precision

        monkeypatch.setattr(torch.cuda, "is_available", lambda: has_cuda)
        if has_cuda:
            monkeypatch.setattr(
                torch.cuda, "get_device_capability", lambda *args, **kw: (major, minor)
            )
            monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.version, "hip", "rocm-x.y" if hip else None)
        _rnn_precision._is_tensor_core_capable.cache_clear()

    def test_set_get_global_roundtrip(self):
        try:
            for mode in ("ieee", "tf32", "tf32x3"):
                set_recurrent_matmul_precision(mode)
                assert get_recurrent_matmul_precision() == mode
        finally:
            self._restore_precision()

    def test_set_auto_clears_override(self, monkeypatch):
        # Pin to a known GPU regime so the "auto" resolution is deterministic
        # regardless of what the host actually has.
        self._patch_device_capability(monkeypatch, has_cuda=True, major=9)
        try:
            set_recurrent_matmul_precision("ieee")
            assert get_recurrent_matmul_precision() == "ieee"
            set_recurrent_matmul_precision("auto")
            # falls back to torch.get_float32_matmul_precision mapping →
            # preset → concrete value at the patched compute capability.
            preset = {"highest": "ieee", "high": "high-prec", "medium": "fast"}[
                torch.get_float32_matmul_precision()
            ]
            preset_concrete = {"ieee": "ieee", "high-prec": "tf32x3", "fast": "tf32"}[
                preset
            ]
            assert get_recurrent_matmul_precision() == preset_concrete
            set_recurrent_matmul_precision("tf32")
            assert get_recurrent_matmul_precision() == "tf32"
            set_recurrent_matmul_precision(None)
            assert get_recurrent_matmul_precision() == preset_concrete
        finally:
            self._restore_precision()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="recurrent matmul precision"):
            set_recurrent_matmul_precision("fp16")
        with pytest.raises(ValueError, match="recurrent_matmul_precision"):
            LSTMModule(
                input_size=3,
                hidden_size=8,
                num_layers=1,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_matmul_precision="fp16",
            )
        with pytest.raises(ValueError, match="recurrent_matmul_precision"):
            GRUModule(
                input_size=3,
                hidden_size=8,
                num_layers=1,
                in_keys=["obs", "hidden"],
                out_keys=["feat", ("next", "hidden")],
                recurrent_matmul_precision="fp16",
            )

    def test_torch_precision_drives_auto_ampere(self, monkeypatch):
        """``torch.set_float32_matmul_precision`` maps via presets on Ampere+."""
        self._patch_device_capability(monkeypatch, has_cuda=True, major=8)
        try:
            set_recurrent_matmul_precision(None)
            prev = torch.get_float32_matmul_precision()
            try:
                torch.set_float32_matmul_precision("highest")
                assert get_recurrent_matmul_precision() == "ieee"
                torch.set_float32_matmul_precision("high")
                # "high" → "high-prec" preset → tf32x3 on Ampere+
                assert get_recurrent_matmul_precision() == "tf32x3"
                torch.set_float32_matmul_precision("medium")
                # "medium" → "fast" preset → tf32 on Ampere+
                assert get_recurrent_matmul_precision() == "tf32"
            finally:
                torch.set_float32_matmul_precision(prev)
        finally:
            self._restore_precision()

    def test_torch_precision_drives_auto_volta(self, monkeypatch):
        """On pre-Ampere GPUs ``"high"`` / ``"medium"`` fall back to IEEE.

        Pre-presets, ``"high"`` would have resolved to ``"tf32x3"`` even on
        V100/T4 — a software-emulated path that doesn't help when there are
        no tensor cores for ``tl.dot(..., input_precision="tf32")`` to use.
        """
        self._patch_device_capability(monkeypatch, has_cuda=True, major=7)
        try:
            set_recurrent_matmul_precision(None)
            prev = torch.get_float32_matmul_precision()
            try:
                torch.set_float32_matmul_precision("highest")
                assert get_recurrent_matmul_precision() == "ieee"
                torch.set_float32_matmul_precision("high")
                # Volta has no TF32 tensor cores → preset falls back to IEEE.
                assert get_recurrent_matmul_precision() == "ieee"
                torch.set_float32_matmul_precision("medium")
                assert get_recurrent_matmul_precision() == "ieee"
            finally:
                torch.set_float32_matmul_precision(prev)
        finally:
            self._restore_precision()

    def test_env_var_default(self, monkeypatch):
        from torchrl.modules.tensordict_module import _rnn_precision

        monkeypatch.setenv("TORCHRL_RNN_PRECISION", "tf32x3")
        # ``_read_env_default`` re-reads the env var; ``set(None)`` rebinds the
        # global override from that read.
        try:
            set_recurrent_matmul_precision(None)
            assert _rnn_precision._GLOBAL_OVERRIDE == "tf32x3"
            assert get_recurrent_matmul_precision() == "tf32x3"
        finally:
            self._restore_precision()

    def test_env_var_accepts_preset(self, monkeypatch):
        """``TORCHRL_RNN_PRECISION=fast`` resolves per-GPU at call time."""
        from torchrl.modules.tensordict_module import _rnn_precision

        monkeypatch.setenv("TORCHRL_RNN_PRECISION", "high-prec")
        self._patch_device_capability(monkeypatch, has_cuda=True, major=10)
        try:
            set_recurrent_matmul_precision(None)
            assert _rnn_precision._GLOBAL_OVERRIDE == "high-prec"
            assert get_recurrent_matmul_precision() == "tf32x3"
        finally:
            self._restore_precision()

    def test_env_var_invalid_raises(self, monkeypatch):
        from torchrl.modules.tensordict_module import _rnn_precision

        monkeypatch.setenv("TORCHRL_RNN_PRECISION", "fp16")
        with pytest.raises(ValueError, match="TORCHRL_RNN_PRECISION"):
            _rnn_precision._read_env_default()

    def test_resolve_kwarg_overrides_global(self, monkeypatch):
        from torchrl.modules.tensordict_module._rnn_precision import _resolve_precision

        # Force a deterministic preset resolution.
        self._patch_device_capability(monkeypatch, has_cuda=True, major=9)
        try:
            set_recurrent_matmul_precision("ieee")
            assert _resolve_precision("tf32") == "tf32"
            assert _resolve_precision("auto") == "ieee"
            assert _resolve_precision(None) == "ieee"
            set_recurrent_matmul_precision("high-prec")
            assert _resolve_precision("tf32") == "tf32"
            # Per-call presets resolve via the patched GPU capability.
            assert _resolve_precision("auto") == "tf32x3"
            assert _resolve_precision("fast") == "tf32"
            assert _resolve_precision("high-prec") == "tf32x3"
        finally:
            self._restore_precision()

    @pytest.mark.parametrize(
        "major,fast_expected,high_prec_expected",
        [
            (10, "tf32", "tf32x3"),  # Blackwell
            (9, "tf32", "tf32x3"),  # Hopper
            (8, "tf32", "tf32x3"),  # Ampere
            (7, "ieee", "ieee"),  # Turing
            (6, "ieee", "ieee"),  # Pascal
        ],
    )
    def test_preset_resolution_per_compute_capability(
        self, monkeypatch, major, fast_expected, high_prec_expected
    ):
        """``"fast"`` / ``"high-prec"`` map per-architecture as documented."""
        from torchrl.modules.tensordict_module._rnn_precision import _resolve_gpu_preset

        self._patch_device_capability(monkeypatch, has_cuda=True, major=major)
        try:
            assert _resolve_gpu_preset("fast") == fast_expected
            assert _resolve_gpu_preset("high-prec") == high_prec_expected
        finally:
            self._restore_precision()

    def test_preset_resolution_no_cuda(self, monkeypatch):
        """Presets fall back to IEEE when CUDA is unavailable."""
        from torchrl.modules.tensordict_module._rnn_precision import _resolve_gpu_preset

        self._patch_device_capability(monkeypatch, has_cuda=False, major=0)
        try:
            assert _resolve_gpu_preset("fast") == "ieee"
            assert _resolve_gpu_preset("high-prec") == "ieee"
        finally:
            self._restore_precision()

    def test_preset_resolution_hip(self, monkeypatch):
        """ROCm/HIP devices fall back to IEEE (no TF32 in triton's ``tl.dot``)."""
        from torchrl.modules.tensordict_module._rnn_precision import _resolve_gpu_preset

        # Even with major>=8 reported, the HIP flag forces the fallback.
        self._patch_device_capability(monkeypatch, has_cuda=True, major=9, hip=True)
        try:
            assert _resolve_gpu_preset("fast") == "ieee"
            assert _resolve_gpu_preset("high-prec") == "ieee"
        finally:
            self._restore_precision()

    def test_set_preset_globally(self, monkeypatch):
        """``set_recurrent_matmul_precision("fast")`` is honored as preset."""
        self._patch_device_capability(monkeypatch, has_cuda=True, major=8)
        try:
            set_recurrent_matmul_precision("fast")
            assert get_recurrent_matmul_precision() == "tf32"
            # Switch to a pre-Ampere card and re-resolve.
            self._patch_device_capability(monkeypatch, has_cuda=True, major=7)
            assert get_recurrent_matmul_precision() == "ieee"
        finally:
            self._restore_precision()

    def test_maybe_enable_tf32_flips_and_restores(self):
        """Replacement for the old context-manager helper. Plain flip + restore.

        compiled_autograd handles a couple of conditional ``allow_tf32``
        writes much better than the ``@contextmanager`` generator that used
        to wrap the cuBLAS calls. This test pins the contract.
        """
        from torchrl.modules.tensordict_module._rnn_precision import (
            _maybe_enable_tf32,
            _restore_tf32,
        )

        prev_outer = torch.backends.cuda.matmul.allow_tf32
        try:
            # tf32 wants True. With prev=False the helper flips and returns False.
            torch.backends.cuda.matmul.allow_tf32 = False
            prev = _maybe_enable_tf32("tf32")
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert prev is False
            _restore_tf32(prev)
            assert torch.backends.cuda.matmul.allow_tf32 is False

            # ieee wants False. Already False → no-op (returns None).
            prev = _maybe_enable_tf32("ieee")
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert prev is None
            _restore_tf32(prev)
            assert torch.backends.cuda.matmul.allow_tf32 is False

            # tf32x3 also wants cuBLAS at IEEE → no-op.
            prev = _maybe_enable_tf32("tf32x3")
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert prev is None

            # When global is True and we want ieee, the helper flips.
            torch.backends.cuda.matmul.allow_tf32 = True
            prev = _maybe_enable_tf32("ieee")
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert prev is True
            _restore_tf32(prev)
            assert torch.backends.cuda.matmul.allow_tf32 is True
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_outer

    def test_restore_tf32_with_none_is_noop(self):
        """``_restore_tf32(None)`` must leave the flag untouched."""
        from torchrl.modules.tensordict_module._rnn_precision import _restore_tf32

        prev = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            _restore_tf32(None)
            assert torch.backends.cuda.matmul.allow_tf32 is True
            torch.backends.cuda.matmul.allow_tf32 = False
            _restore_tf32(None)
            assert torch.backends.cuda.matmul.allow_tf32 is False
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("precision", ["ieee", "tf32", "tf32x3"])
    def test_lstm_triton_precision_grad_parity(self, precision):
        """Each precision produces finite, pad-comparable gradients.

        ``ieee`` should match pad/cuDNN very closely; ``tf32`` and ``tf32x3``
        get slightly looser tolerances. The test scale (B=4, T=7, H=64) is
        small enough that all three precisions agree at ``5e-3``.
        """
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
            recurrent_matmul_precision=precision,
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

        atol = 5e-3
        for k in grads_pad:
            assert torch.isfinite(
                grads_triton[k]
            ).all(), f"precision={precision} produced non-finite grad for {k}"
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=atol, rtol=atol
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    @pytest.mark.parametrize("precision", ["ieee", "tf32", "tf32x3"])
    def test_gru_triton_precision_grad_parity(self, precision):
        """GRU counterpart of ``test_lstm_triton_precision_grad_parity``."""
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
            recurrent_matmul_precision=precision,
            device=device,
        )
        triton_module.load_state_dict(pad_module.state_dict())

        obs = torch.randn(B, T, F, device=device)
        hidden = torch.zeros(B, T, 1, H, device=device)
        is_init = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        is_init[1, 3] = True
        data = TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init},
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

        atol = 5e-3
        for k in grads_pad:
            assert torch.isfinite(
                grads_triton[k]
            ).all(), f"precision={precision} produced non-finite grad for {k}"
            torch.testing.assert_close(
                grads_pad[k], grads_triton[k], atol=atol, rtol=atol
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_triton, reason=_triton_skip_reason)
    def test_lstm_module_precision_kwarg_takes_effect(self):
        """Per-module kwarg beats the process-global override.

        Uses the global setter to force ``tf32`` everywhere, then constructs
        a module pinned to ``ieee`` and asserts its stored attribute is the
        explicit override rather than the global value.
        """
        try:
            set_recurrent_matmul_precision("tf32")
            device = torch.device("cuda")
            module = LSTMModule(
                input_size=3,
                hidden_size=16,
                num_layers=1,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="triton",
                recurrent_matmul_precision="ieee",
                device=device,
            )
            assert module.recurrent_matmul_precision == "ieee"
        finally:
            self._restore_precision()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
