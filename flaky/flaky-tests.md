# Flaky Test Report - 2026-04-11

## Summary

- **Flaky tests**: 31
- **Newly flaky** (last 7 days): 31
- **Resolved**: 0
- **Total tests analyzed**: 29738
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `..._transforms.py::TestTokenizer::test_transform_no_env[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `....py::TestTokenizer::test_transform_no_env[NonTensorStack]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...forms.py::TestTokenizer::test_single_trans_env_check[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...forms.py::TestTokenizer::test_serial_trans_env_check[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...rms.py::TestTokenizer::test_parallel_trans_env_check[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...forms.py::TestTokenizer::test_trans_serial_env_check[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...rms.py::TestTokenizer::test_trans_parallel_env_check[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...transforms.py::TestTokenizer::test_transform_compose[str]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...key_transforms.py::TestTokenizer::test_transform_model[3]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...key_transforms.py::TestTokenizer::test_transform_model[5]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...key_transforms.py::TestTokenizer::test_transform_model[7]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...test_key_transforms.py::TestTokenizer::test_transform_env` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...sforms.py::TestTokenizer::test_transform_rb[ReplayBuffer]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `...:TestTokenizer::test_transform_rb[TensorDictReplayBuffer]` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `..._key_transforms.py::TestTokenizer::test_transform_inverse` 🆕 | 5.7% (8/140) | 8 | 0.11 | 2026-04-09 |
| `..._reset.py::TestNonTensorEnv::test_from_text_env_tokenizer` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-04-09 |
| `...:TestNonTensorEnv::test_from_text_env_tokenizer_catframes` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-04-09 |
| `...eset.py::TestNonTensorEnv::test_from_text_rb_slicesampler` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-04-09 |
| `test/test_actors.py::test_lmhead_actorvalueoperator[device0]` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-04-09 |
| `test/test_modules.py::TestDecisionTransformer::test_init` 🆕 | 5.3% (8/150) | 8 | 0.11 | 2026-04-09 |


### Newly Flaky Tests

- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_no_env[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_no_env[NonTensorStack]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_single_trans_env_check[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_serial_trans_env_check[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_parallel_trans_env_check[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_trans_serial_env_check[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_trans_parallel_env_check[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_compose[str]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_model[3]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_model[5]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_model[7]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_env`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_rb[ReplayBuffer]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_rb[TensorDictReplayBuffer]`
- `test/transforms/test_key_transforms.py::TestTokenizer::test_transform_inverse`
- `test/envs/test_auto_reset.py::TestNonTensorEnv::test_from_text_env_tokenizer`
- `test/envs/test_auto_reset.py::TestNonTensorEnv::test_from_text_env_tokenizer_catframes`
- `test/envs/test_auto_reset.py::TestNonTensorEnv::test_from_text_rb_slicesampler`
- `test/test_actors.py::test_lmhead_actorvalueoperator[device0]`
- `test/test_modules.py::TestDecisionTransformer::test_init`
- `test/test_modules.py::TestDecisionTransformer::test_exec[batch_dims0]`
- `test/test_modules.py::TestDecisionTransformer::test_exec[batch_dims1]`
- `test/test_modules.py::TestDecisionTransformer::test_exec[batch_dims2]`
- `test/test_modules.py::TestDecisionTransformer::test_dtactor[batch_dims0]`
- `test/test_modules.py::TestDecisionTransformer::test_dtactor[batch_dims1]`
- `test/test_modules.py::TestDecisionTransformer::test_dtactor[batch_dims2]`
- `test/test_modules.py::TestDecisionTransformer::test_onlinedtactor[batch_dims0]`
- `test/test_modules.py::TestDecisionTransformer::test_onlinedtactor[batch_dims1]`
- `test/test_modules.py::TestDecisionTransformer::test_onlinedtactor[batch_dims2]`
- `test/test_tensordictmodules.py::TestDecisionTransformerInferenceWrapper::test_dt_inference_wrapper[True]`
- `test/test_tensordictmodules.py::TestDecisionTransformerInferenceWrapper::test_dt_inference_wrapper[False]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-04-11T06:30:39.379356+00:00*