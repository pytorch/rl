# Flaky Test Report - 2026-08-11

## Summary

- **Flaky tests**: 20
- **Newly flaky** (last 7 days): 20
- **Resolved**: 0
- **Total tests analyzed**: 31295
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...test_configs.py::TestHydraParsing::test_simple_env_config` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...est_configs.py::TestHydraParsing::test_batched_env_config` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...py::TestHydraParsing::test_batched_env_with_one_transform` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...y::TestHydraParsing::test_batched_env_with_two_transforms` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...gs.py::TestHydraParsing::test_simple_config_instantiation` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `test/test_configs.py::TestHydraParsing::test_env_parsing` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `..._configs.py::TestHydraParsing::test_env_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...gs.py::TestHydraParsing::test_collector_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...figs.py::TestHydraParsing::test_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `....py::TestHydraParsing::test_a2c_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...estHydraParsing::test_reinforce_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `....py::TestHydraParsing::test_dqn_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...stHydraParsing::test_dqn_trainer_parsing_with_hook_config` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...ydraParsing::test_dqn_trainer_iql_style_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...draParsing::test_dqn_trainer_qmix_style_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...py::TestHydraParsing::test_ddpg_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `....py::TestHydraParsing::test_cql_trainer_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...:TestHydraParsing::test_transformed_env_parsing_with_file` 🆕 | 46.4% (70/151) | 70 | 0.93 | 2026-08-07 |
| `...test_wrapped_class_kwargs_have_config_fields[LBFGSConfig]` 🆕 | 53.9% (76/141) | 76 | 0.92 | 2026-08-07 |


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
- `test/test_configs.py::TestHydraParsing::test_a2c_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_reinforce_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_hook_config`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_iql_style_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_qmix_style_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_ddpg_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_cql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_transformed_env_parsing_with_file`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[LBFGSConfig]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-11T06:23:40.390482+00:00*