# Flaky Test Report - 2026-04-25

## Summary

- **Flaky tests**: 49
- **Newly flaky** (last 7 days): 49
- **Resolved**: 0
- **Total tests analyzed**: 27176
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...est_collector_rb_multisync[LazyTensorStorage-False-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multisync[LazyTensorStorage-False-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multisync[LazyTensorStorage-True-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...:test_collector_rb_multisync[LazyTensorStorage-True-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...est_collector_rb_multisync[LazyMemmapStorage-False-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multisync[LazyMemmapStorage-False-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multisync[LazyMemmapStorage-True-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...:test_collector_rb_multisync[LazyMemmapStorage-True-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...st_collector_rb_multiasync[LazyTensorStorage-False-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...est_collector_rb_multiasync[LazyTensorStorage-False-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...est_collector_rb_multiasync[LazyTensorStorage-True-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multiasync[LazyTensorStorage-True-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...st_collector_rb_multiasync[LazyMemmapStorage-False-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...est_collector_rb_multiasync[LazyMemmapStorage-False-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...est_collector_rb_multiasync[LazyMemmapStorage-True-False]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...test_collector_rb_multiasync[LazyMemmapStorage-True-True]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...ulti_collector_and_replay_buffer[True-MultiSyncCollector]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...lti_collector_and_replay_buffer[True-MultiAsyncCollector]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...lti_collector_and_replay_buffer[False-MultiSyncCollector]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |
| `...ti_collector_and_replay_buffer[False-MultiAsyncCollector]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-24 |


### Newly Flaky Tests

- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyTensorStorage-False-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyTensorStorage-False-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyTensorStorage-True-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyTensorStorage-True-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyMemmapStorage-False-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyMemmapStorage-False-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyMemmapStorage-True-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multisync[LazyMemmapStorage-True-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyTensorStorage-False-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyTensorStorage-False-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyTensorStorage-True-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyTensorStorage-True-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyMemmapStorage-False-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyMemmapStorage-False-True]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyMemmapStorage-True-False]`
- `test/test_collectors.py::TestCollectorRB::test_collector_rb_multiasync[LazyMemmapStorage-True-True]`
- `test/test_collectors.py::TestCollectorRB::test_parallel_env_with_multi_collector_and_replay_buffer[True-MultiSyncCollector]`
- `test/test_collectors.py::TestCollectorRB::test_parallel_env_with_multi_collector_and_replay_buffer[True-MultiAsyncCollector]`
- `test/test_collectors.py::TestCollectorRB::test_parallel_env_with_multi_collector_and_replay_buffer[False-MultiSyncCollector]`
- `test/test_collectors.py::TestCollectorRB::test_parallel_env_with_multi_collector_and_replay_buffer[False-MultiAsyncCollector]`
- `test/test_collectors.py::TestCollectorRB::test_collector_postproc_zeros[True-True-collector_class1]`
- `test/test_collectors.py::TestCollectorRB::test_collector_postproc_zeros[True-True-MultiAsyncCollector]`
- `test/test_collectors.py::TestCollectorRB::test_collector_postproc_zeros[False-True-collector_class1]`
- `test/test_collectors.py::TestCollectorRB::test_collector_postproc_zeros[False-True-MultiAsyncCollector]`
- `test/test_collectors.py::TestAsyncCollection::test_pause`
- `test/test_collectors.py::TestAsyncCollection::test_start_multi[MultiAsyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_multi[MultiAsyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_multi[MultiSyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_multi[MultiSyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[None-MultiAsyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[None-MultiAsyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[None-MultiSyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[None-MultiSyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[MultiProcessWeightSyncScheme-MultiAsyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[MultiProcessWeightSyncScheme-MultiAsyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[MultiProcessWeightSyncScheme-MultiSyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[MultiProcessWeightSyncScheme-MultiSyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[SharedMemWeightSyncScheme-MultiAsyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[SharedMemWeightSyncScheme-MultiAsyncCollector-1000000000]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[SharedMemWeightSyncScheme-MultiSyncCollector--1]`
- `test/test_collectors.py::TestAsyncCollection::test_start_update_policy[SharedMemWeightSyncScheme-MultiSyncCollector-1000000000]`
- `test/test_collectors.py::TestInitRandomFramesWithStart::test_init_random_frames_with_start[MultiSyncCollector]`
- `test/test_collectors.py::TestInitRandomFramesWithStart::test_init_random_frames_with_start[MultiAsyncCollector]`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_completeness_multi_collector`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_collector_rb`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_collector_rb_start`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_collector_batched_env_rb`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_async_collector_rb`
- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_async_ndim2[True]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-04-25T06:40:22.296506+00:00*