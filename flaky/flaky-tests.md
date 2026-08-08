# Flaky Test Report - 2026-08-08

## Summary

- **Flaky tests**: 129
- **Newly flaky** (last 7 days): 129
- **Resolved**: 0
- **Total tests analyzed**: 31295
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...test_configs.py::TestHydraParsing::test_simple_env_config` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...est_configs.py::TestHydraParsing::test_batched_env_config` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...py::TestHydraParsing::test_batched_env_with_one_transform` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...y::TestHydraParsing::test_batched_env_with_two_transforms` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...gs.py::TestHydraParsing::test_simple_config_instantiation` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `test/test_configs.py::TestHydraParsing::test_env_parsing` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `..._configs.py::TestHydraParsing::test_env_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...gs.py::TestHydraParsing::test_collector_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...figs.py::TestHydraParsing::test_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `....py::TestHydraParsing::test_a2c_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...estHydraParsing::test_reinforce_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `....py::TestHydraParsing::test_dqn_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...stHydraParsing::test_dqn_trainer_parsing_with_hook_config` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...ydraParsing::test_dqn_trainer_iql_style_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...draParsing::test_dqn_trainer_qmix_style_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...py::TestHydraParsing::test_ddpg_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `....py::TestHydraParsing::test_iql_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `....py::TestHydraParsing::test_cql_trainer_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `...:TestHydraParsing::test_transformed_env_parsing_with_file` 🆕 | 62.3% (86/138) | 86 | 0.75 | 2026-08-07 |
| `test/test_configs.py::TestEnvConfigs::test_gym_env_config` 🆕 | 14.5% (20/138) | 20 | 0.29 | 2026-08-05 |


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
- `test/test_configs.py::TestEnvConfigs::test_gym_env_config`
- `test/test_configs.py::TestEnvConfigs::test_batched_env_config[ParallelEnv]`
- `test/test_configs.py::TestEnvConfigs::test_batched_env_config[SerialEnv]`
- `test/test_configs.py::TestEnvConfigs::test_batched_env_config[AsyncEnvPool]`
- `test/test_configs.py::TestDataConfigs::test_round_robin_writer_config`
- `test/test_configs.py::TestDataConfigs::test_random_sampler_config`
- `test/test_configs.py::TestDataConfigs::test_tensor_storage_config`
- `test/test_configs.py::TestDataConfigs::test_tensordict_replay_buffer_config`
- `test/test_configs.py::TestDataConfigs::test_list_storage_config`
- `test/test_configs.py::TestDataConfigs::test_replay_buffer_config`
- `test/test_configs.py::TestDataConfigs::test_tensordict_replay_buffer_config_optional_fields`
- `test/test_configs.py::TestDataConfigs::test_tensor_dict_max_value_writer_config`
- `test/test_configs.py::TestDataConfigs::test_tensor_dict_round_robin_writer_config`
- `test/test_configs.py::TestDataConfigs::test_immutable_dataset_writer_config`
- `test/test_configs.py::TestDataConfigs::test_prioritized_sampler_config`
- `test/test_configs.py::TestDataConfigs::test_sampler_without_replacement_config`
- `test/test_configs.py::TestDataConfigs::test_lazy_stack_storage_config`
- `test/test_configs.py::TestDataConfigs::test_lazy_memmap_storage_config`
- `test/test_configs.py::TestDataConfigs::test_lazy_tensor_storage_config`
- `test/test_configs.py::TestModuleConfigs::test_mlp_config`
- `test/test_configs.py::TestModuleConfigs::test_convnet_config`
- `test/test_configs.py::TestModuleConfigs::test_tanh_normal_model_config`
- `test/test_configs.py::TestModuleConfigs::test_tanh_normal_model_config_defaults`
- `test/test_configs.py::TestModuleConfigs::test_tensordict_sequential_config`
- `test/test_configs.py::TestModuleConfigs::test_tanh_module_config`
- `test/test_configs.py::TestModuleConfigs::test_value_model_config`
- `test/test_configs.py::TestModuleConfigs::test_qvalue_model_config`
- `test/test_configs.py::TestModuleConfigs::test_additive_gaussian_module_config`
- `test/test_configs.py::TestCollectorsConfig::test_generic_collector_backend_fields`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[async-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[async-False]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[multi_sync-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[multi_sync-False]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[multi_async-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_config[multi_async-False]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[async-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[async-False]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[multi_sync-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[multi_sync-False]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[multi_async-True]`
- `test/test_configs.py::TestCollectorsConfig::test_collector_auto_configures_exploration_modules[multi_async-False]`
- `test/test_configs.py::TestLossConfigs::test_ppo_loss_config[clip]`
- `test/test_configs.py::TestLossConfigs::test_ppo_loss_config[kl]`
- `test/test_configs.py::TestLossConfigs::test_ppo_loss_config[ppo]`
- `test/test_configs.py::TestLossConfigs::test_a2c_loss_config`
- `test/test_configs.py::TestLossConfigs::test_reinforce_loss_config`
- `test/test_configs.py::TestLoggerConfigs::test_wandb_logger_config_instantiation`
- `test/test_configs.py::TestLoggerConfigs::test_trackio_logger_config_instantiation`
- `test/test_configs.py::TestTrainerConfigs::test_hook_config`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[BatchSubSamplerConfig-kwargs0-BatchSubSampler]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[ClearCudaCacheConfig-kwargs1-ClearCudaCache]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[LogScalarConfig-kwargs2-LogScalar]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[LogTimingConfig-kwargs3-LogTiming]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[RewardNormalizerConfig-kwargs4-RewardNormalizer]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[SelectKeysConfig-kwargs5-SelectKeys]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[CountFramesLogConfig-kwargs6-CountFramesLog]`
- `test/test_configs.py::TestTrainerConfigs::test_individual_hook_configs[EarlyStoppingConfig-kwargs7-EarlyStopping]`
- `test/test_configs.py::TestTransformConfigs::test_init_tracker_config`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels0-distributional0-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels0-distributional0-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels0-distributional1-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels0-distributional1-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels1-distributional0-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels1-distributional0-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels1-distributional1-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding0-from_pixels1-distributional1-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels0-distributional0-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels0-distributional0-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels0-distributional1-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels0-distributional1-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels1-distributional0-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels1-distributional0-noisy1-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels1-distributional1-noisy0-device0]`
- `test/test_helpers.py::test_dqn_maker[categorical_action_encoding1-from_pixels1-distributional1-noisy1-device0]`
- `test/test_helpers.py::test_transformed_env_constructor_with_state_dict[from_pixels0]`
- `test/test_helpers.py::test_transformed_env_constructor_with_state_dict[from_pixels1]`
- `test/test_render.py::TestSotaCheckpointFactories::test_dqn_cartpole_checkpoint_render_factories`
- `test/test_render.py::TestSotaCheckpointFactories::test_mujoco_playground_ppo_uses_scalar_proof_and_eval_envs`
- `test/test_configs.py::TestConfigClassParity::test_wrapped_class_kwargs_have_config_fields[LBFGSConfig]`
- `test/test_render.py::TestSotaCheckpointFactories::test_ppo_inverted_pendulum_checkpoint_render_factories`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[a2c_atari]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[cql_online-single]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[ddpg]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[diffusion_bc]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[discrete_cql_online]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[discrete_sac]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[dreamer]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[iql_marl]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[mappo_ippo]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[ppo_atari]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[qmix_vdn]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[redq-single]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[sac]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[td3-single]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[vla_grpo]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[a2c_mujoco]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[cql_online]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[crossq]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[ddpg-single]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[discrete_cql_offline]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[dqn_atari]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[dreamer_v3]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[impala_single_node]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[maddpg_iddpg]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[marl_sac]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[ppo_mujoco]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[redq]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[rnd_mujoco]`
- `.github/unittest/linux_sota/scripts/test_sota.py::test_commands[td3]`
- `test/rb/test_rb_core.py::TestSequenceUnit::test_metadata_flows_into_tensordict_sample`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-08T06:21:00.376436+00:00*