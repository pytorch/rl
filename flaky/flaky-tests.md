# Flaky Test Report - 2026-04-17

## Summary

- **Flaky tests**: 32
- **Newly flaky** (last 7 days): 32
- **Resolved**: 0
- **Total tests analyzed**: 27175
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ffer::test_trajs_per_batch_multi_collector_batched_env_rb` 🆕 | 22.2% (20/90) | 20 | 0.44 | 2026-04-12 |
| `test/test_rb.py::TestSamplers::test_slice_sampler_errors` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...plers::test_slice_sampler_prioritized[False-False-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...lers::test_slice_sampler_prioritized[False-False-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...mplers::test_slice_sampler_prioritized[False-True-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...plers::test_slice_sampler_prioritized[False-True-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...mplers::test_slice_sampler_prioritized[True-False-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...plers::test_slice_sampler_prioritized[True-False-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...amplers::test_slice_sampler_prioritized[True-True-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...mplers::test_slice_sampler_prioritized[True-True-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...::test_slice_sampler_prioritized_span[False-False-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...:test_slice_sampler_prioritized_span[False-False-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...s::test_slice_sampler_prioritized_span[False-True-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...::test_slice_sampler_prioritized_span[False-True-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...::test_slice_sampler_prioritized_span[span1-False-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...:test_slice_sampler_prioritized_span[span1-False-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...s::test_slice_sampler_prioritized_span[span1-True-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...::test_slice_sampler_prioritized_span[span1-True-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...::test_slice_sampler_prioritized_span[span2-False-True-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |
| `...:test_slice_sampler_prioritized_span[span2-False-False-2]` 🆕 | 13.3% (20/150) | 20 | 0.27 | 2026-04-12 |


### Newly Flaky Tests

- `test/test_collectors.py::TestTrajsPerBatchReplayBuffer::test_trajs_per_batch_multi_collector_batched_env_rb`
- `test/test_rb.py::TestSamplers::test_slice_sampler_errors`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[False-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[False-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[False-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[False-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[True-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[True-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[True-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized[True-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[False-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[False-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[False-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[False-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span1-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span1-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span1-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span1-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span2-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span2-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span2-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span2-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[3-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[3-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[3-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[3-True-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span4-False-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span4-False-False-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span4-True-True-2]`
- `test/test_rb.py::TestSamplers::test_slice_sampler_prioritized_span[span4-True-False-2]`
- `test/test_rb.py::TestRBMultidim::test_done_slicesampler[True]`
- `test/test_rb.py::TestRBMultidim::test_done_slicesampler[False]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-04-17T06:44:52.562850+00:00*