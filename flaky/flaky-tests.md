# Flaky Test Report - 2026-07-16

## Summary

- **Flaky tests**: 4
- **Newly flaky** (last 7 days): 4
- **Resolved**: 0
- **Total tests analyzed**: 30918
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ayTransport::test_ray_owned_inference_with_nccl_transport` 🆕 | 20.0% (6/30) | 6 | 0.40 | 2026-07-15 |
| `...ibuted.py::TestRayRB::test_ray_replay_with_gloo_transport` 🆕 | 20.0% (6/30) | 6 | 0.40 | 2026-07-15 |
| `...ayTransport::test_ray_owned_inference_with_gloo_transport` 🆕 | 9.1% (6/66) | 6 | 0.18 | 2026-07-15 |
| `...rver::test_process_server_with_distributed_gloo_transport` 🆕 | 9.1% (6/66) | 6 | 0.18 | 2026-07-15 |


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

*Generated at 2026-07-16T07:01:36.021432+00:00*