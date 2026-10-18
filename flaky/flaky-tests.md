# Flaky Test Report - 2026-10-18

## Summary

- **Flaky tests**: 5735
- **Newly flaky** (last 7 days): 0
- **Resolved**: 0
- **Total tests analyzed**: 32551
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...replay_collection_smoke[cuda-async-True-True-True-frames]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `..._replay_collection_smoke[cuda-sync-True-True-True-frames]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...replay_collection_smoke[cuda-async-True-False-False-time]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...play_collection_smoke[cuda-async-True-False-False-warmup]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...ollection_smoke[cuda-sync-True-False-False-reset_records]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...eamer_v3_process_inference_collection_smoke[cuda:0-False]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...reamer_v3_process_inference_collection_smoke[cuda:0-True]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...d.py::test_cuda_prioritized_replay_buffer_samples_on_cuda` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...ict_prioritized_replay_buffer_memmap_storage_cuda_sampler` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...cuda_prioritized_replay_buffer_weight_matches_cpu_formula` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...b/test_rb_core.py::TestSequenceUnit::test_non_cpu_storage` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...ce[tensor-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...torage_device[tensor-LazyMemmapStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...ce[tensor-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...torage_device[tensor-LazyTensorStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...device[tc-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...st_storage_device[tc-LazyMemmapStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...device[tc-LazyTensorStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...st_storage_device[tc-LazyTensorStorage-device_data3-auto]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |
| `...device[td-LazyMemmapStorage-device_data0-device_storage0]` | 20.0% (15/75) | 15 | 0.40 | 2026-10-16 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-18T06:24:45.359166+00:00*