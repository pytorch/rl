# Flaky Test Report - 2026-10-29

## Summary

- **Flaky tests**: 96
- **Newly flaky** (last 7 days): 21
- **Resolved**: 0
- **Total tests analyzed**: 32822
- **CI runs analyzed**: 60

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/objectives/test_dt.py::TestOnlineDT::test_odt[device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...bjectives/test_dt.py::TestOnlineDT::test_seq_odt[device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_dt[device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `test/objectives/test_dt.py::TestDT::test_seq_dt[device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[0.1-True-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[0.1-False-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ectives/test_dt.py::TestGAIL::test_gail[1.0-True-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ctives/test_dt.py::TestGAIL::test_gail[1.0-False-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[0.1-True-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[0.1-False-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...ves/test_dt.py::TestGAIL::test_seq_gail[1.0-True-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...es/test_dt.py::TestGAIL::test_seq_gail[1.0-False-device1]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype0-None-True]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype0-None-False]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype0-device1-True]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype0-device1-False]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...omposite::test_device_cast[dest1-shape0-dtype1-None-True]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...mposite::test_device_cast[dest1-shape0-dtype1-None-False]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...osite::test_device_cast[dest1-shape0-dtype1-device1-True]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |
| `...site::test_device_cast[dest1-shape0-dtype1-device1-False]` | 53.8% (7/13) | 7 | 0.92 | 2026-10-24 |


### Newly Flaky Tests

- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_simple_env_config`
- `test/test_configs.py::TestHydraParsing::test_batched_env_config`
- `test/test_configs.py::TestHydraParsing::test_batched_env_with_one_transform`
- `test/test_configs.py::TestHydraParsing::test_batched_env_with_two_transforms`
- `test/test_configs.py::TestHydraParsing::test_simple_config_instantiation`
- `test/test_configs.py::TestHydraParsing::test_env_parsing`
- `test/test_configs.py::TestHydraParsing::test_env_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_collector_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_ppo_ewma_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_a2c_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_reinforce_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_dqn_resume`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_parsing_with_hook_config`
- `test/test_configs.py::TestHydraParsing::test_dqn_trainer_qmix_style_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_ddpg_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_iql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_cql_trainer_parsing_with_file`
- `test/test_configs.py::TestHydraParsing::test_transformed_env_parsing_with_file`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-10-29T06:18:49.467271+00:00*