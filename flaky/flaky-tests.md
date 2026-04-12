# Flaky Test Report - 2026-04-12

## Summary

- **Flaky tests**: 62
- **Newly flaky** (last 7 days): 62
- **Resolved**: 0
- **Total tests analyzed**: 27072
- **CI runs analyzed**: 15

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test/test_rb.py::TestSamplers::test_slice_sampler_errors` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...plers::test_slice_sampler_prioritized[False-False-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...lers::test_slice_sampler_prioritized[False-False-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...mplers::test_slice_sampler_prioritized[False-True-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...plers::test_slice_sampler_prioritized[False-True-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...mplers::test_slice_sampler_prioritized[True-False-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...plers::test_slice_sampler_prioritized[True-False-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...amplers::test_slice_sampler_prioritized[True-True-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...mplers::test_slice_sampler_prioritized[True-True-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...::test_slice_sampler_prioritized_span[False-False-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...:test_slice_sampler_prioritized_span[False-False-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...s::test_slice_sampler_prioritized_span[False-True-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...::test_slice_sampler_prioritized_span[False-True-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...::test_slice_sampler_prioritized_span[span1-False-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...:test_slice_sampler_prioritized_span[span1-False-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...s::test_slice_sampler_prioritized_span[span1-True-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...::test_slice_sampler_prioritized_span[span1-True-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...::test_slice_sampler_prioritized_span[span2-False-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...:test_slice_sampler_prioritized_span[span2-False-False-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |
| `...s::test_slice_sampler_prioritized_span[span2-True-True-2]` 🆕 | 6.7% (10/150) | 10 | 0.13 | 2026-04-11 |


### Newly Flaky Tests

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

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-04-12T06:39:48.189768+00:00*