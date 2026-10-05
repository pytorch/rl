# Flaky Test Report - 2026-10-05

## Summary

- **Flaky tests**: 3
- **Newly flaky** (last 7 days): 3
- **Resolved**: 0
- **Total tests analyzed**: 107
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ripts/test_examples.py::test_example[distributed-ray-dqn]` 🆕 | 7.1% (2/28) | 2 | 0.06 | 2026-10-04 |
| `...scripts/test_examples.py::test_example[ray-wandb-monitor]` 🆕 | 7.1% (2/28) | 2 | 0.06 | 2026-10-04 |
| `...ts/test_examples.py::test_example[services-ray-collector]` 🆕 | 7.1% (2/28) | 2 | 0.06 | 2026-10-04 |


### Newly Flaky Tests

- `.github/unittest/examples/scripts/test_examples.py::test_example[distributed-ray-dqn]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[ray-wandb-monitor]`
- `.github/unittest/examples/scripts/test_examples.py::test_example[services-ray-collector]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-05T06:34:59.603013+00:00*