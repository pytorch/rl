# Flaky Test Report - 2026-09-09

## Summary

- **Flaky tests**: 72
- **Newly flaky** (last 7 days): 72
- **Resolved**: 0
- **Total tests analyzed**: 31826
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...mposite::test_device_cast[dest1-shape0-dtype0-None-False]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...osite::test_device_cast[dest1-shape0-dtype0-device1-True]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...site::test_device_cast[dest1-shape0-dtype0-device1-False]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...omposite::test_device_cast[dest1-shape0-dtype1-None-True]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...mposite::test_device_cast[dest1-shape0-dtype1-None-False]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...osite::test_device_cast[dest1-shape0-dtype1-device1-True]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |
| `...site::test_device_cast[dest1-shape0-dtype1-device1-False]` 🆕 | 40.0% (6/15) | 6 | 0.80 | 2026-09-08 |


### Newly Flaky Tests

- `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]`
- `test/objectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]`
- `test/objectives/test_dt.py::TestDT::test_dt[device1]`
- `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]`
- `test/objectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]`
- `test/objectives/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype0-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype0-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype0-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype0-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype1-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype1-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype1-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype1-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype2-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype2-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype2-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-dtype2-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-None-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-None-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-None-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape0-None-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype0-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype0-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype0-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype0-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype1-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype1-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype1-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype1-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype2-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype2-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype2-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-dtype2-device1-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-None-None-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-None-None-False]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-None-device1-True]`
- `test/test_specs.py::TestComposite::test_device_cast[dest1-shape1-None-device1-False]`
- `test/objectives/test_dreamer_v3.py::test_dreamer_v3_native_replay_collection_smoke[cuda-async-True-True-True-frames]`
- `test/objectives/test_dreamer_v3.py::test_dreamer_v3_native_replay_collection_smoke[cuda-sync-True-True-True-frames]`
- `test/objectives/test_dreamer_v3.py::test_dreamer_v3_native_replay_collection_smoke[cuda-async-True-False-False-time]`
- `test/objectives/test_dreamer_v3.py::test_dreamer_v3_native_replay_collection_smoke[cuda-async-True-False-False-warmup]`
- `test/objectives/test_dreamer_v3.py::test_dreamer_v3_native_replay_collection_smoke[cuda-sync-True-False-False-reset_records]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_state_estimator_resets_and_posterior_rng[compile]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cpu-compile]`
- `test/rb/test_prioritized.py::test_cuda_prioritized_replay_buffer_samples_on_cuda`
- `test/rb/test_prioritized.py::test_tensordict_prioritized_replay_buffer_memmap_storage_cuda_sampler`
- `test/rb/test_prioritized.py::test_cuda_prioritized_replay_buffer_weight_matches_cpu_formula`
- `test/rb/test_rb_core.py::TestSequenceUnit::test_non_cpu_storage`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tensor-LazyMemmapStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tensor-LazyMemmapStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tensor-LazyTensorStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tensor-LazyTensorStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tc-LazyMemmapStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tc-LazyMemmapStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tc-LazyTensorStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[tc-LazyTensorStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[td-LazyMemmapStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[td-LazyMemmapStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[td-LazyTensorStorage-device_data0-device_storage0]`
- `test/rb/test_storages.py::TestStorages::test_storage_device[td-LazyTensorStorage-device_data3-auto]`
- `test/rb/test_storages.py::TestSharedStorageInit::test_prioritized_memmap_cuda_sampler_after_multiprocess_writes`
- `test/rb/test_writers.py::TestWriterGeneration::test_generation_cuda_data_into_cuda_storage`
- `test/test_checkpoint.py::test_cuda_map_location_and_rng`
- `test/test_custom_envs.py::TestCustomEnvs::test_financial_env_device`
- `test/objectives/test_dreamer_v3.py::TestDreamerV3CompileStrategy::test_compiled_step_selects_the_scan_for_untouched_rollouts`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-09-09T06:28:00.198636+00:00*