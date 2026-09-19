# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dense, window-aligned replay feasibility probe, not a collector feature.

This intentionally compacts after collection. It demonstrates real replay
allocation savings and learner parity, but makes no collection-peak claim.
The sampler selects whole windows; it does not insert artificial reset flags.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import set_recurrent_mode
from torchrl.testing._state_candidates import _make_candidate


def _allocated_bytes(td):
    storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in td.values(True, True)
    }
    return sum(storages.values())


def _probe(container, storage_cls):
    torch.manual_seed(4)
    module = _make_candidate("gtrxl", container, state_key=("state",))
    spec = module.transformer.state_spec
    state = spec.zero([2])
    state.get("memory").normal_()
    state.get("valid").fill_(True)
    steps = []
    for t in range(8):
        is_init = torch.zeros(2, 1, dtype=torch.bool)
        if t == 4:
            is_init[0] = True
            state[0] = spec.zero()
        td = TensorDict(
            {"observation": torch.randn(2, 7), "is_init": is_init, "state": state}, [2]
        )
        module(td)
        steps.append(td.clone())
        state = td.get(("next", "state"))
    full = torch.stack(steps, 1)
    records = TensorDict(
        {
            "transitions": full.exclude("state", ("next", "state")),
            "initial_state": full.get("state")[:, 0].clone(),
        },
        [2],
    )
    with tempfile.TemporaryDirectory() as directory:
        kwargs = {"scratch_dir": directory} if storage_cls is LazyMemmapStorage else {}
        storage = storage_cls(4, **kwargs)
        rb = TensorDictReplayBuffer(storage=storage, batch_size=2)
        rb.extend(records)
        sample = rb.sample()
        transitions = sample["transitions"].clone()
        initial = sample["initial_state"]
        expanded = initial.unsqueeze(-1).expand(transitions.batch_size)
        # Expanding memory is a view. Only the much smaller validity tensor is
        # materialized. Real episode resets invalidate history; there are no
        # sampler-inserted slice boundaries in this whole-window representation.
        learner_state = type(initial).from_dict(
            {
                "memory": expanded.get("memory"),
                "valid": expanded.get("valid") & ~transitions["is_init"],
            },
            batch_size=transitions.batch_size,
        )
        transitions.set("state", learner_state)
        with set_recurrent_mode(True):
            output = module(transitions)["embed"]
        error = (output - sample["transitions", "embed"]).abs().max().item()
        torch.testing.assert_close(
            output, sample["transitions", "embed"], atol=2e-5, rtol=2e-5
        )
        compact_bytes = _allocated_bytes(storage._storage)
        compact_memory_bytes = (
            storage._storage.get(("initial_state", "memory")).untyped_storage().nbytes()
        )
    with tempfile.TemporaryDirectory() as directory:
        kwargs = {"scratch_dir": directory} if storage_cls is LazyMemmapStorage else {}
        storage = storage_cls(4, **kwargs)
        rb = TensorDictReplayBuffer(storage=storage)
        rb.extend(full)
        full_bytes = _allocated_bytes(storage._storage)
        full_memory_bytes = (
            storage._storage.get(("state", "memory")).untyped_storage().nbytes()
        )
    return {
        "container": container,
        "storage": storage_cls.__name__,
        "capacity_windows": 4,
        "window_length": 8,
        "full_allocated_bytes": full_bytes,
        "compact_allocated_bytes": compact_bytes,
        "full_root_memory_bytes": full_memory_bytes,
        "compact_memory_bytes": compact_memory_bytes,
        "max_output_error": error,
        "sampled_state_class": type(initial).__name__,
        "memory_expansion_is_view": expanded.get("memory").untyped_storage().data_ptr()
        == initial.get("memory").untyped_storage().data_ptr(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        _probe(container, storage_cls)
        for container in ("td", "tc", "ttd")
        for storage_cls in (LazyTensorStorage, LazyMemmapStorage)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
