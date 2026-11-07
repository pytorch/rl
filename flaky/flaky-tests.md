# Flaky Test Report - 2026-11-07

## Summary

- **Flaky tests**: 52
- **Newly flaky** (last 7 days): 52
- **Resolved**: 0
- **Total tests analyzed**: 32939
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...TestDataConfigs::test_complex_replay_buffer_configuration` 🆕 | 8.7% (14/161) | 14 | 0.17 | 2026-11-07 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...tion.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...ploration.py::TestEGreedy::test_bounded_compile_annealing` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch1]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys0-batch2]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch1]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-1-device0-keys1-batch2]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch1]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys0-batch2]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch1]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...rm::test_observationnorm[0-1-True-3-device0-keys1-batch2]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch1]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...m::test_observationnorm[0-1-False-1-device0-keys0-batch2]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |
| `...m::test_observationnorm[0-1-False-1-device0-keys1-batch0]` 🆕 | 7.4% (12/161) | 12 | 0.15 | 2026-11-06 |


### Newly Flaky Tests

- `test/test_configs.py::TestDataConfigs::test_complex_replay_buffer_configuration`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[0.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_fullgraph[1.0]`
- `test/test_exploration.py::TestEGreedy::test_bounded_compile_annealing`
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

*Generated at 2026-11-07T06:16:25.435009+00:00*