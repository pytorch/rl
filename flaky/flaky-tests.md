# Flaky Test Report - 2026-07-19

## Summary

- **Flaky tests**: 4
- **Newly flaky** (last 7 days): 4
- **Resolved**: 0
- **Total tests analyzed**: 30975
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ayTransport::test_ray_owned_inference_with_nccl_transport` 🆕 | 20.0% (13/65) | 13 | 0.40 | 2026-07-17 |
| `...ibuted.py::TestRayRB::test_ray_replay_with_gloo_transport` 🆕 | 20.0% (13/65) | 13 | 0.40 | 2026-07-17 |
| `...ayTransport::test_ray_owned_inference_with_gloo_transport` 🆕 | 9.1% (13/143) | 13 | 0.18 | 2026-07-17 |
| `...rver::test_process_server_with_distributed_gloo_transport` 🆕 | 9.1% (13/143) | 13 | 0.18 | 2026-07-17 |


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

*Generated at 2026-07-19T07:02:31.754624+00:00*