# Flaky Test Report - 2026-07-17

## Summary

- **Flaky tests**: 4
- **Newly flaky** (last 7 days): 4
- **Resolved**: 0
- **Total tests analyzed**: 30947
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ayTransport::test_ray_owned_inference_with_nccl_transport` 🆕 | 20.0% (12/60) | 12 | 0.40 | 2026-07-16 |
| `...ibuted.py::TestRayRB::test_ray_replay_with_gloo_transport` 🆕 | 20.0% (12/60) | 12 | 0.40 | 2026-07-16 |
| `...ayTransport::test_ray_owned_inference_with_gloo_transport` 🆕 | 9.1% (12/132) | 12 | 0.18 | 2026-07-16 |
| `...rver::test_process_server_with_distributed_gloo_transport` 🆕 | 9.1% (12/132) | 12 | 0.18 | 2026-07-16 |


### Newly Flaky Tests

- `test/test_inference_server.py::TestRayTransport::test_ray_owned_inference_with_nccl_transport`
- `test/rb/test_rb_distributed.py::TestRayRB::test_ray_replay_with_gloo_transport`
- `test/test_inference_server.py::TestRayTransport::test_ray_owned_inference_with_gloo_transport`
- `test/test_inference_server.py::TestProcessInferenceServer::test_process_server_with_distributed_gloo_transport`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-07-17T06:53:56.458531+00:00*