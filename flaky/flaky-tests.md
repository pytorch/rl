# Flaky Test Report - 2026-07-03

## Summary

- **Flaky tests**: 17
- **Newly flaky** (last 7 days): 17
- **Resolved**: 0
- **Total tests analyzed**: 29446
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...test_configs.py::TestHydraParsing::test_simple_env_config` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...est_configs.py::TestHydraParsing::test_batched_env_config` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...py::TestHydraParsing::test_batched_env_with_one_transform` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...y::TestHydraParsing::test_batched_env_with_two_transforms` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...gs.py::TestHydraParsing::test_simple_config_instantiation` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `test/test_configs.py::TestHydraParsing::test_env_parsing` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `..._configs.py::TestHydraParsing::test_env_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...gs.py::TestHydraParsing::test_collector_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...figs.py::TestHydraParsing::test_trainer_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `....py::TestHydraParsing::test_dqn_trainer_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...stHydraParsing::test_dqn_trainer_parsing_with_hook_config` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...ydraParsing::test_dqn_trainer_iql_style_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...draParsing::test_dqn_trainer_qmix_style_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...py::TestHydraParsing::test_ddpg_trainer_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `....py::TestHydraParsing::test_cql_trainer_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |
| `...:TestHydraParsing::test_transformed_env_parsing_with_file` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-06-27 |


### Newly Flaky Tests

- `test/test_configs.py::TestHydraParsing::test_simple_env_config`
- `test/test_configs.py::TestHydraParsing::test_batched_env_config`
- `test/test_configs.py::TestHydraParsing::test_batched_env_with_one_transform`
- `test/test_configs.py::TestHydraParsing::test_batched_env_with_two_transforms`
- `test/test_configs.py::TestHydraParsing::test_simple_config_instantiation`
- `test/test_configs.py::TestHydraParsing::test_env_parsing`
- `test/test_configs.py::TestHydraParsing::test_env_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_collector_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_hook_config`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_iql_style_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_qmix_style_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_ddpg_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_cql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_transformed_env_parsing_with_file`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-07-03T07:15:04.387050+00:00*