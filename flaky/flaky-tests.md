# Flaky Test Report - 2026-10-22

## Summary

- **Flaky tests**: 5736
- **Newly flaky** (last 7 days): 2
- **Resolved**: 0
- **Total tests analyzed**: 32760
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...replay_collection_smoke[cuda-async-True-True-True-frames]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `..._replay_collection_smoke[cuda-sync-True-True-True-frames]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...replay_collection_smoke[cuda-async-True-False-False-time]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...play_collection_smoke[cuda-async-True-False-False-warmup]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...ollection_smoke[cuda-sync-True-False-False-reset_records]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...eamer_v3_process_inference_collection_smoke[cuda:0-False]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...reamer_v3_process_inference_collection_smoke[cuda:0-True]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...d.py::test_cuda_prioritized_replay_buffer_samples_on_cuda` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...ict_prioritized_replay_buffer_memmap_storage_cuda_sampler` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...cuda_prioritized_replay_buffer_weight_matches_cpu_formula` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...b/test_rb_core.py::TestSequenceUnit::test_non_cpu_storage` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...ce[tensor-LazyMemmapStorage-device_data0-device_storage0]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...torage_device[tensor-LazyMemmapStorage-device_data3-auto]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...ce[tensor-LazyTensorStorage-device_data0-device_storage0]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...torage_device[tensor-LazyTensorStorage-device_data3-auto]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...device[tc-LazyMemmapStorage-device_data0-device_storage0]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...st_storage_device[tc-LazyMemmapStorage-device_data3-auto]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...device[tc-LazyTensorStorage-device_data0-device_storage0]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...st_storage_device[tc-LazyTensorStorage-device_data3-auto]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |
| `...device[td-LazyMemmapStorage-device_data0-device_storage0]` | 21.1% (16/76) | 16 | 0.42 | 2026-10-21 |


### Newly Flaky Tests

- `test/rb/test_storages.py::TestStorageExport::test_device_storage[cuda-ListStorage-True]`
- `test/libs/test_datasets.py::TestHuggingFaceDatasetAdapter::test_import_reads_unformatted_values[columns]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-22T06:28:54.142320+00:00*