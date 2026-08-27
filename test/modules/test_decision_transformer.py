# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from _modules_common import _has_transformers
from tensordict import TensorDict, unravel_key_list
from tensordict.nn import TensorDictModule
from torchrl.modules import (
    DecisionTransformerInferenceWrapper,
    DTActor,
    OnlineDTActor,
    ProbabilisticActor,
    TanhDelta,
    TanhNormal,
)
from torchrl.modules.models.decision_transformer import DecisionTransformer


@pytest.mark.skipif(
    not _has_transformers, reason="transformers needed for TestDecisionTransformer"
)
class TestDecisionTransformer:
    def test_init(self):
        DecisionTransformer(
            3,
            4,
        )
        with pytest.raises(TypeError):
            DecisionTransformer(3, 4, config="some_str")
        DecisionTransformer(
            3,
            4,
            config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_exec(self, batch_dims, T=5):
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        model = DecisionTransformer(
            3,
            4,
            config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        out = model(observations, actions, r2go)
        assert out.shape == torch.Size([*batch_dims, T, 16])

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_dtactor(self, batch_dims, T=5):
        dtactor = DTActor(
            3,
            4,
            transformer_config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        out = dtactor(observations, actions, r2go)
        assert out.shape == torch.Size([*batch_dims, T, 4])

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_onlinedtactor(self, batch_dims, T=5):
        dtactor = OnlineDTActor(
            3,
            4,
            transformer_config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        mu, sig = dtactor(observations, actions, r2go)
        assert mu.shape == torch.Size([*batch_dims, T, 4])
        assert sig.shape == torch.Size([*batch_dims, T, 4])
        assert (dtactor.log_std_min < sig.log()).all()
        assert (dtactor.log_std_max > sig.log()).all()


@pytest.mark.skipif(
    not _has_transformers, reason="transformers needed to test DT classes"
)
class TestDecisionTransformerInferenceWrapper:
    @pytest.mark.parametrize("online", [True, False])
    def test_dt_inference_wrapper(self, online):
        action_key = ("nested", ("action",))
        if online:
            dtactor = OnlineDTActor(
                state_dim=4, action_dim=2, transformer_config=DTActor.default_config()
            )
            in_keys = ["loc", "scale"]
            actor_module = TensorDictModule(
                dtactor,
                in_keys=["observation", action_key, "return_to_go"],
                out_keys=in_keys,
            )
            dist_class = TanhNormal
        else:
            dtactor = DTActor(
                state_dim=4, action_dim=2, transformer_config=DTActor.default_config()
            )
            in_keys = ["param"]
            actor_module = TensorDictModule(
                dtactor,
                in_keys=["observation", action_key, "return_to_go"],
                out_keys=in_keys,
            )
            dist_class = TanhDelta
        dist_kwargs = {
            "low": -1.0,
            "high": 1.0,
        }
        actor = ProbabilisticActor(
            in_keys=in_keys,
            out_keys=[action_key],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
        )
        inference_actor = DecisionTransformerInferenceWrapper(actor)
        sequence_length = 20
        td = TensorDict(
            {
                "observation": torch.randn(1, sequence_length, 4),
                action_key: torch.randn(1, sequence_length, 2),
                "return_to_go": torch.randn(1, sequence_length, 1),
            },
            [1],
        )
        with pytest.raises(
            ValueError,
            match="The value of out_action_key",
        ):
            result = inference_actor(td)
        inference_actor.set_tensor_keys(action=action_key, out_action=action_key)
        result = inference_actor(td)
        # checks that the seq length has disappeared
        assert result.get(action_key).shape == torch.Size([1, 2])
        assert inference_actor.out_keys == unravel_key_list(
            sorted([action_key, *in_keys, "observation", "return_to_go"], key=str)
        )
        assert set(result.keys(True, True)) - set(td.keys(True, True)) == set(
            inference_actor.out_keys
        ) - set(inference_actor.in_keys)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
