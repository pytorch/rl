# Flaky Test Report - 2026-09-06

## Summary

- **Flaky tests**: 33
- **Newly flaky** (last 7 days): 33
- **Resolved**: 0
- **Total tests analyzed**: 31588
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...DreamerV3::test_dreamer_v3_actor_loss_cuda_graph[device0]` 🆕 | 20.0% (4/20) | 4 | 0.32 | 2026-09-05 |
| `...dreamer_v3_full_learner_cuda_graph_matches_eager[device0]` 🆕 | 20.0% (4/20) | 4 | 0.32 | 2026-09-05 |
| `..._update_weights[MultiProcessWeightSyncScheme-False-False]` 🆕 | 13.4% (22/164) | 22 | 0.27 | 2026-09-05 |
| `..._block_gru_triton_gradient_parity[SiLU-4-1-2-3-dtype4-48]` 🆕 | 11.1% (8/72) | 8 | 0.22 | 2026-09-05 |
| `...opy::test_ppo_entropy_mc_when_analytic_is_nonfinite[True]` 🆕 | 9.1% (6/66) | 6 | 0.18 | 2026-09-05 |
| `...joco::test_microduck_collision_meshes_use_runtime_proxies` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...dreamer_v3_actor_loss[ValueEstimators.TD0-True-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...dreamer_v3_actor_loss[ValueEstimators.TD0-True-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamer_v3_actor_loss[ValueEstimators.TD0-False-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamer_v3_actor_loss[ValueEstimators.TD0-False-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...dreamer_v3_actor_loss[ValueEstimators.TD1-True-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...dreamer_v3_actor_loss[ValueEstimators.TD1-True-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamer_v3_actor_loss[ValueEstimators.TD1-False-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamer_v3_actor_loss[ValueEstimators.TD1-False-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...er_v3_actor_loss[ValueEstimators.TDLambda-True-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...er_v3_actor_loss[ValueEstimators.TDLambda-True-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...r_v3_actor_loss[ValueEstimators.TDLambda-False-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...r_v3_actor_loss[ValueEstimators.TDLambda-False-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamerV3::test_dreamer_v3_actor_loss[None-True-3-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |
| `...reamerV3::test_dreamer_v3_actor_loss[None-True-5-device0]` 🆕 | 7.4% (12/162) | 12 | 0.15 | 2026-09-05 |


### Newly Flaky Tests

- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss_cuda_graph[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_full_learner_cuda_graph_matches_eager[device0]`
- `test/test_collectors.py::TestCollectorGeneric::test_update_weights[MultiProcessWeightSyncScheme-False-False]`
- `test/modules/test_dreamer_components.py::test_public_block_gru_triton_gradient_parity[SiLU-4-1-2-3-dtype4-48]`
- `test/objectives/test_ppo.py::TestObjectiveEntropy::test_ppo_entropy_mc_when_analytic_is_nonfinite[True]`
- `test/libs/test_mujoco.py::TestMujoco::test_microduck_collision_meshes_use_runtime_proxies`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD0-True-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD0-True-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD0-False-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD0-False-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD1-True-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD1-True-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD1-False-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TD1-False-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TDLambda-True-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TDLambda-True-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TDLambda-False-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[ValueEstimators.TDLambda-False-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[None-True-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[None-True-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[None-False-3-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss[None-False-5-device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_continuation_lambda_and_weights[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_categorical_value_exposes_decoded_value[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_slow_critic_checkpoint_and_online_bootstrap[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_actor_loss_reinforce[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_reinforce_return_normalization[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_reparam_return_normalization[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_reparam_return_statistics_update[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_return_statistics_checkpoint[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_legacy_retnorm_checkpoint_migrates[device0]`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_value_loss_sync_gamma[device0]`
- `test/test_trainer.py::TestTrainerCheckpointComponents::test_torch_checkpoint_roundtrips_optimizer_state`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-09-06T06:26:45.438471+00:00*