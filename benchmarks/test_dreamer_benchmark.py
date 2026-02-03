# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark for Dreamer world model with different RSSM implementations.

This benchmark compares:
- Loop + C++ GRU (baseline)
- Loop + Python GRU
- Scan + Python GRU
- With and without torch.compile

Run with:
    python -m pytest benchmarks/test_dreamer_benchmark.py -v --benchmark-group-by=func
"""
import argparse

import pytest
import torch
from packaging import version
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

from torch.distributions import Independent
from torch.distributions.normal import Normal

from torchrl.data import Bounded
from torchrl.modules import MLP, ObsDecoder, ObsEncoder
from torchrl.modules.models.model_based import RSSMPosterior, RSSMPrior, RSSMRollout
from torchrl.objectives.dreamer import DreamerModelLoss

TORCH_VERSION = torch.__version__
FULLGRAPH = version.parse(".".join(TORCH_VERSION.split(".")[:3])) >= version.parse(
    "2.5.0"
)


class IndependentNormal(Independent):
    """Independent Normal distribution for decoder output."""
    def __init__(self, loc, scale=1.0, event_dim=1):
        super().__init__(Normal(loc, scale), event_dim)


def _make_dreamer_world_model(
    device,
    state_dim=30,
    rssm_hidden_dim=200,
    hidden_dim=400,
    use_scan=False,
    use_python_gru=False,
):
    """Create a Dreamer world model for benchmarking."""
    action_dim = 6
    image_size = 64
    in_channels = 3
    
    # Encoder and Decoder
    encoder = ObsEncoder(in_channels=in_channels).to(device)
    decoder = ObsDecoder(latent_dim=state_dim + rssm_hidden_dim).to(device)
    
    # Compute encoder output size
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
        obs_embed_dim = encoder(dummy).shape[-1]
    
    # RSSM Prior and Posterior
    action_spec = Bounded(shape=(action_dim,), dtype=torch.float32, low=-1, high=1)
    rssm_prior = RSSMPrior(
        hidden_dim=rssm_hidden_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        action_spec=action_spec,
        action_dim=action_dim,
    ).to(device)
    
    rssm_posterior = RSSMPosterior(
        hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        obs_embed_dim=obs_embed_dim,
    ).to(device)
    
    # Replace GRU with Python version if requested
    if use_python_gru:
        from torchrl.modules.tensordict_module.rnn import GRUCell as PythonGRUCell
        old_rnn = rssm_prior.rnn
        python_rnn = PythonGRUCell(old_rnn.input_size, old_rnn.hidden_size)
        python_rnn.load_state_dict(old_rnn.state_dict())
        rssm_prior.rnn = python_rnn
    
    # Reward module
    reward_module = MLP(
        out_features=1,
        depth=2,
        num_cells=hidden_dim,
        activation_class=torch.nn.ELU,
    ).to(device)
    
    # Build RSSM Rollout
    rssm_rollout = RSSMRollout(
        TensorDictModule(
            rssm_prior,
            in_keys={"state": "state", "belief": "belief", "action": "action", "noise": "prior_noise"},
            out_keys=[("next", "prior_mean"), ("next", "prior_std"), "_", ("next", "belief")],
            out_to_in_map=True,
        ),
        TensorDictModule(
            rssm_posterior,
            in_keys={"belief": ("next", "belief"), "obs_embedding": ("next", "encoded_latents"), "noise": "posterior_noise"},
            out_keys=[("next", "posterior_mean"), ("next", "posterior_std"), ("next", "state")],
            out_to_in_map=True,
        ),
        use_scan=use_scan,
    )
    
    # Build decoder with probabilistic output
    decoder_module = ProbabilisticTensorDictSequential(
        TensorDictModule(
            decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=["loc"],
        ),
        ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=[("next", "reco_pixels")],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 3},
        ),
    )
    
    # Build transition model
    transition_model = TensorDictSequential(
        TensorDictModule(
            encoder,
            in_keys=[("next", "pixels")],
            out_keys=[("next", "encoded_latents")],
        ),
        rssm_rollout,
    )
    
    # Build reward model
    reward_model = TensorDictModule(
        reward_module,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )
    
    # Combined world model
    world_model = TensorDictSequential(
        transition_model,
        decoder_module,
        reward_model,
    )
    
    return world_model


def _create_tensordict(device, batch_size=16, temporal_size=50):
    """Create a tensordict with fake data for benchmarking."""
    state_dim = 30
    rssm_hidden_dim = 200
    action_dim = 6
    image_size = 64
    
    td = TensorDict(
        {
            "state": torch.randn(batch_size, temporal_size, state_dim, device=device),
            "belief": torch.randn(batch_size, temporal_size, rssm_hidden_dim, device=device),
            "action": torch.randn(batch_size, temporal_size, action_dim, device=device),
            "next": {
                "pixels": torch.randn(batch_size, temporal_size, 3, image_size, image_size, device=device),
                "reward": torch.randn(batch_size, temporal_size, 1, device=device),
            },
        },
        batch_size=[batch_size, temporal_size],
        device=device,
    )
    return td


def _maybe_compile(fn, compile_mode, td, warmup=3):
    """Optionally compile a function with warmup."""
    if compile_mode:
        if isinstance(compile_mode, str):
            fn = torch.compile(fn, mode=compile_mode, fullgraph=False)
        else:
            fn = torch.compile(fn, fullgraph=False)
        
        for _ in range(warmup):
            try:
                fn(td.clone())
            except Exception:
                pass  # Some configs may fail during warmup
    
    return fn


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(
    "use_scan,use_python_gru",
    [
        (False, False),  # Loop + C++ GRU (baseline)
        (False, True),   # Loop + Python GRU
        (True, True),    # Scan + Python GRU
    ],
    ids=["loop_cpp_gru", "loop_python_gru", "scan_python_gru"],
)
def test_dreamer_world_model_loss(
    benchmark,
    backward,
    compile,
    use_scan,
    use_python_gru,
    batch_size=16,
    temporal_size=50,
):
    """Benchmark DreamerModelLoss with different RSSM implementations."""
    if compile:
        torch._dynamo.reset_code_caches()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create world model
    world_model = _make_dreamer_world_model(
        device=device,
        use_scan=use_scan,
        use_python_gru=use_python_gru,
    )
    
    # Create loss module
    loss = DreamerModelLoss(world_model)
    
    # Create test data
    td = _create_tensordict(device, batch_size=batch_size, temporal_size=temporal_size)
    
    # Initialize (handles lazy modules)
    try:
        loss(td.clone())
    except Exception as e:
        pytest.skip(f"Configuration failed: {e}")
    
    # Maybe compile
    if compile:
        try:
            loss = _maybe_compile(loss, compile, td)
        except Exception as e:
            pytest.skip(f"Compilation failed: {e}")
    
    if backward:
        def loss_and_bw(td):
            loss_td, _ = loss(td)
            total_loss = (
                loss_td["loss_model_kl"]
                + loss_td["loss_model_reco"]
                + loss_td["loss_model_reward"]
            )
            total_loss.backward()
        
        benchmark.pedantic(
            loss_and_bw,
            args=(td.clone(),),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=3,
            rounds=20,
        )
    else:
        def forward_only(td):
            return loss(td)
        
        benchmark.pedantic(
            forward_only,
            args=(td.clone(),),
            iterations=1,
            warmup_rounds=3,
            rounds=20,
        )


@pytest.mark.parametrize(
    "use_scan,use_python_gru",
    [
        (False, False),  # Loop + C++ GRU (baseline)
        (False, True),   # Loop + Python GRU
        (True, True),    # Scan + Python GRU
    ],
    ids=["loop_cpp_gru", "loop_python_gru", "scan_python_gru"],
)
def test_rssm_rollout_only(
    benchmark,
    use_scan,
    use_python_gru,
    batch_size=16,
    temporal_size=50,
):
    """Benchmark RSSMRollout alone without the full world model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    state_dim = 30
    rssm_hidden_dim = 200
    action_dim = 6
    obs_embed_dim = 1024
    
    action_spec = Bounded(shape=(action_dim,), dtype=torch.float32, low=-1, high=1)
    rssm_prior = RSSMPrior(
        hidden_dim=rssm_hidden_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        action_spec=action_spec,
        action_dim=action_dim,
    ).to(device)
    
    rssm_posterior = RSSMPosterior(
        hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        obs_embed_dim=obs_embed_dim,
    ).to(device)
    
    # Replace GRU with Python version if requested
    if use_python_gru:
        from torchrl.modules.tensordict_module.rnn import GRUCell as PythonGRUCell
        old_rnn = rssm_prior.rnn
        python_rnn = PythonGRUCell(old_rnn.input_size, old_rnn.hidden_size)
        python_rnn.load_state_dict(old_rnn.state_dict())
        rssm_prior.rnn = python_rnn
    
    rssm_rollout = RSSMRollout(
        TensorDictModule(
            rssm_prior,
            in_keys={"state": "state", "belief": "belief", "action": "action", "noise": "prior_noise"},
            out_keys=[("next", "prior_mean"), ("next", "prior_std"), "_", ("next", "belief")],
            out_to_in_map=True,
        ),
        TensorDictModule(
            rssm_posterior,
            in_keys={"belief": ("next", "belief"), "obs_embedding": ("next", "encoded_latents"), "noise": "posterior_noise"},
            out_keys=[("next", "posterior_mean"), ("next", "posterior_std"), ("next", "state")],
            out_to_in_map=True,
        ),
        use_scan=use_scan,
    )
    
    td = TensorDict(
        {
            "state": torch.randn(batch_size, temporal_size, state_dim, device=device),
            "belief": torch.randn(batch_size, temporal_size, rssm_hidden_dim, device=device),
            "action": torch.randn(batch_size, temporal_size, action_dim, device=device),
            "next": {
                "encoded_latents": torch.randn(batch_size, temporal_size, obs_embed_dim, device=device),
            },
        },
        batch_size=[batch_size, temporal_size],
        device=device,
    )
    
    # Initialize
    _ = rssm_rollout(td.clone())
    
    benchmark.pedantic(
        rssm_rollout,
        args=(td.clone(),),
        iterations=1,
        warmup_rounds=3,
        rounds=50,
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main(
        [__file__, "--capture", "no", "--exitfirst", "--benchmark-group-by", "func", "-v"]
        + unknown
    )

