# Flaky Test Report - 2026-11-05

## Summary

- **Flaky tests**: 53
- **Newly flaky** (last 7 days): 53
- **Resolved**: 0
- **Total tests analyzed**: 32892
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `.../test_examples.py::test_example[replay-buffer-checkpoint]` 🆕 | 40.0% (8/20) | 8 | 0.80 | 2026-11-02 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 8.5% (13/153) | 13 | 0.17 | 2026-11-05 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 8.5% (13/153) | 13 | 0.17 | 2026-11-05 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 8.5% (13/153) | 13 | 0.17 | 2026-11-05 |
| `...TestDataConfigs::test_complex_replay_buffer_configuration` 🆕 | 6.1% (10/164) | 10 | 0.12 | 2026-11-03 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch0]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch1]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch2]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch0]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch1]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch2]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch0]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch1]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch2]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch0]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch1]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch2]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch0]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch1]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch2]` 🆕 | 6.1% (10/165) | 10 | 0.12 | 2026-11-05 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[replay-buffer-checkpoint]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
- `test/test_configs.py::TestDataConfigs::test_complex_replay_buffer_configuration`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-1-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-True-3-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-1-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[0-1-False-3-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-1-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-True-3-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-1-device0-keys1-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys0-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys0-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys0-batch2]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys1-batch0]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys1-batch1]`
- `test/transforms/test_observation_transforms.py::TestObservationNorm::test_observationnorm[1-2-False-3-device0-keys1-batch2]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-11-05T06:19:49.261603+00:00*