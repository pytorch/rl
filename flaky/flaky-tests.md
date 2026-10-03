# Flaky Test Report - 2026-10-03

## Summary

- **Flaky tests**: 5732
- **Newly flaky** (last 7 days): 1
- **Resolved**: 0
- **Total tests analyzed**: 32486
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...re::test_static_batch_pads_slices_and_owns_results[False]` 🆕 | 76.2% (48/63) | 48 | 0.48 | 2026-10-01 |
| `...st_storage_device[td-LazyMemmapStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...replay_collection_smoke[cuda-async-True-True-True-frames]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...device[td-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...st_storage_device[td-LazyTensorStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...prioritized_memmap_cuda_sampler_after_multiprocess_writes` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...b/test_rb_core.py::TestSequenceUnit::test_non_cpu_storage` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...ce[tensor-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...torage_device[tensor-LazyMemmapStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...ce[tensor-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...torage_device[tensor-LazyTensorStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...device[tc-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...st_storage_device[tc-LazyMemmapStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...device[tc-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...st_storage_device[tc-LazyTensorStorage-device_data3-auto]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...device[td-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...erGeneration::test_generation_cuda_data_into_cuda_storage` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `test/test_checkpoint.py::test_cuda_map_location_and_rng` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...custom_envs.py::TestCustomEnvs::test_financial_env_device` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |
| `...st_trainer.py::test_fql_update_matches_recipe[False-cuda]` | 20.0% (13/65) | 13 | 0.40 | 2026-10-01 |


### Newly Flaky Tests

- `test/test_inference_server.py::TestInferenceServerCore::test_static_batch_pads_slices_and_owns_results[False]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-03T06:27:45.227946+00:00*