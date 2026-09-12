window.BENCHMARK_DATA = {
  "lastUpdate": 1789228422226,
  "repoUrl": "https://github.com/pytorch/rl",
  "entries": {
    "Async CPU (v1)": [
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bedcc833257604d97868c447087c80921f1379b2",
          "message": "[CI] Track async collection across merges and recover benchmark publication (#4287)",
          "timestamp": "2026-09-07T22:01:57+01:00",
          "tree_id": "fb1cd0d1f31013091e551509bb157b87527ebbab",
          "url": "https://github.com/pytorch/rl/commit/bedcc833257604d97868c447087c80921f1379b2"
        },
        "date": 1788817240362,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 83.00504563532915,
            "range": "82.9-83.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3207.51 ms; process-tree RSS: 6682.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 84.1781294904139,
            "range": "83.4-84.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3163.52 ms; process-tree RSS: 6683.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 478.10879791889374,
            "range": "467.4-479.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 582.76 ms; process-tree RSS: 6717.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 489.6342148723182,
            "range": "479.4-492.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 564.99 ms; process-tree RSS: 6715.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.64919718072196,
            "range": "467.7-479.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 571.91 ms; process-tree RSS: 6722.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 484.42284781423945,
            "range": "477.6-486.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 555.52 ms; process-tree RSS: 6716.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.97215628666976,
            "range": "461.2-465.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.70 ms; process-tree RSS: 6671.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1729.1687342318444,
            "range": "1670.9-1737.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 157.65 ms; process-tree RSS: 6691.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.14514072298647,
            "range": "191.5-192.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2207.058123599732,
            "range": "2194.9-2211.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2559.431383926744,
            "range": "2558.3-2566.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 195.53775375771582,
            "range": "190.3-195.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2144.031730061451,
            "range": "1984.1-2168.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2412.1531142877716,
            "range": "2337.0-2439.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2193.441325546298,
            "range": "2179.3-2197.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b3af48309e211373536cbe4becfbc18505f1683d",
          "message": "[CI] Stop failing benchmark publication on performance alerts (#4292)",
          "timestamp": "2026-09-08T06:46:34+01:00",
          "tree_id": "f027b3fec5afdd468c94389b80cb07904383595f",
          "url": "https://github.com/pytorch/rl/commit/b3af48309e211373536cbe4becfbc18505f1683d"
        },
        "date": 1788848914264,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 83.09627590145249,
            "range": "82.9-84.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3230.36 ms; process-tree RSS: 6695.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 84.51734679016104,
            "range": "83.7-84.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3187.76 ms; process-tree RSS: 6682.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 477.18347027194153,
            "range": "470.2-484.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 563.94 ms; process-tree RSS: 6733.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 489.9098520900363,
            "range": "487.6-491.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 566.16 ms; process-tree RSS: 6730.94 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 470.61808234278413,
            "range": "467.0-471.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 587.67 ms; process-tree RSS: 6719.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 491.9114578648693,
            "range": "477.8-493.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 558.30 ms; process-tree RSS: 6720.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.95161152315546,
            "range": "464.7-469.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.39 ms; process-tree RSS: 6692.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1737.0094188386015,
            "range": "1701.1-1741.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.21 ms; process-tree RSS: 6689.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.2606836324408,
            "range": "190.6-191.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2174.2657015833765,
            "range": "2146.7-2216.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2551.072041874402,
            "range": "2544.9-2562.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.19338331681124,
            "range": "189.4-202.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2160.9848908116815,
            "range": "2138.5-2202.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2346.9924270151587,
            "range": "2344.8-2381.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2193.96937916136,
            "range": "2179.5-2200.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2446fc19ee3c84b5815d24217fbb475b1d131d8",
          "message": "[CI] Run the full benchmark suite after merging labelled pull requests (#4293)",
          "timestamp": "2026-09-08T06:54:32+01:00",
          "tree_id": "0835048819db5479b3473148c9d55144a4e0ce23",
          "url": "https://github.com/pytorch/rl/commit/a2446fc19ee3c84b5815d24217fbb475b1d131d8"
        },
        "date": 1788849190089,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 83.29811758312105,
            "range": "82.4-83.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3216.98 ms; process-tree RSS: 6689.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.33258985038336,
            "range": "82.9-84.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3168.10 ms; process-tree RSS: 6706.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 476.7430982381544,
            "range": "469.9-477.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 584.47 ms; process-tree RSS: 6725.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 487.51108666384164,
            "range": "483.0-487.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 581.47 ms; process-tree RSS: 6704.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.746312008894,
            "range": "474.7-477.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 576.72 ms; process-tree RSS: 6718.56 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 486.5405047846961,
            "range": "477.3-488.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 574.47 ms; process-tree RSS: 6736.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 469.29645388660015,
            "range": "468.9-469.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 549.53 ms; process-tree RSS: 6696.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1730.8551188634133,
            "range": "1709.1-1851.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 151.80 ms; process-tree RSS: 6682.78 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.6141422850225,
            "range": "189.6-190.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2216.613838055538,
            "range": "2158.5-2234.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2553.604796834511,
            "range": "2545.1-2556.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 196.50798690477438,
            "range": "191.0-206.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2125.094695381289,
            "range": "2085.1-2134.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2374.5364506000356,
            "range": "2308.0-2382.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2193.881181400684,
            "range": "2181.7-2205.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "740b45173e831053c8649042bb9f376596f60753",
          "message": "[Test] Track direct process-slot collection in continuous benchmarks (#4289)",
          "timestamp": "2026-09-08T06:55:16+01:00",
          "tree_id": "996181ff290fb3a8e3d71e6173ece1a40615c100",
          "url": "https://github.com/pytorch/rl/commit/740b45173e831053c8649042bb9f376596f60753"
        },
        "date": 1788849380295,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 83.0403847978416,
            "range": "82.9-83.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3171.08 ms; process-tree RSS: 6697.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.78613326213114,
            "range": "83.7-84.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3199.91 ms; process-tree RSS: 6710.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 472.3442682342988,
            "range": "463.9-473.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 580.65 ms; process-tree RSS: 6724.42 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 492.0493330525726,
            "range": "485.2-496.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 547.86 ms; process-tree RSS: 6708.78 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 471.11428375499713,
            "range": "469.9-477.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 580.80 ms; process-tree RSS: 6720.65 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 485.5778819149628,
            "range": "483.4-494.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 559.05 ms; process-tree RSS: 6727.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 472.1862879413804,
            "range": "465.5-474.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 550.17 ms; process-tree RSS: 6681.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1720.723732032471,
            "range": "1703.5-1727.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 152.71 ms; process-tree RSS: 6696.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.74473024063127,
            "range": "191.6-192.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2244.2377021799853,
            "range": "2180.3-2247.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2563.657098510543,
            "range": "2559.5-2583.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 195.02740083558612,
            "range": "187.9-196.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2153.6941197373144,
            "range": "2138.9-2163.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2347.9303117550103,
            "range": "2287.2-2361.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2198.589925977239,
            "range": "2194.8-2200.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5cca10a441a48b4116614ba7ef45ea58965367b1",
          "message": "[Feature] CUDA-graph static inference batches (#4269)",
          "timestamp": "2026-09-08T06:59:05+01:00",
          "tree_id": "d0da341c3beefd661bac546748c97cdc04bb7c83",
          "url": "https://github.com/pytorch/rl/commit/5cca10a441a48b4116614ba7ef45ea58965367b1"
        },
        "date": 1788849715994,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.92653527097113,
            "range": "82.7-83.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3210.57 ms; process-tree RSS: 6703.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.85269327046494,
            "range": "83.7-84.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3172.50 ms; process-tree RSS: 6708.15 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 466.6465901389223,
            "range": "466.4-468.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 583.02 ms; process-tree RSS: 6725.33 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 478.7343924768634,
            "range": "471.1-486.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 571.70 ms; process-tree RSS: 6726.87 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 476.63981405645643,
            "range": "468.7-480.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 584.34 ms; process-tree RSS: 6731.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 483.0498400776259,
            "range": "480.8-487.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 566.00 ms; process-tree RSS: 6724.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 470.0128954463864,
            "range": "461.5-477.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 547.93 ms; process-tree RSS: 6673.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1809.0926789472926,
            "range": "1758.4-1849.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 151.96 ms; process-tree RSS: 6685.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.86694154413846,
            "range": "189.1-191.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2187.4533397133505,
            "range": "2175.0-2190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2528.3070516721814,
            "range": "2521.0-2555.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.8960686161756,
            "range": "188.2-199.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2108.1943585640306,
            "range": "2059.1-2196.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2317.282810478995,
            "range": "2271.3-2346.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2184.9389465648337,
            "range": "2176.0-2187.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "402147d710ac41981ff3ada88572fcbe81ab09fd",
          "message": "[Performance] Let AsyncEnvPool workers host multiple environments (#4271)",
          "timestamp": "2026-09-08T06:20:52Z",
          "tree_id": "dd9532ec1be2f6dce5cacaff0d74e03a98333bf2",
          "url": "https://github.com/pytorch/rl/commit/402147d710ac41981ff3ada88572fcbe81ab09fd"
        },
        "date": 1788851336174,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.78143974789992,
            "range": "81.4-82.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3284.74 ms; process-tree RSS: 6708.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.1629726055108,
            "range": "82.1-82.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3232.06 ms; process-tree RSS: 6706.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 487.2995682209467,
            "range": "483.6-511.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 572.44 ms; process-tree RSS: 2789.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 505.04165451635134,
            "range": "499.6-514.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 541.27 ms; process-tree RSS: 2800.60 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 480.966907689618,
            "range": "477.6-483.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 569.27 ms; process-tree RSS: 2792.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 497.15052979013353,
            "range": "492.0-521.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 549.29 ms; process-tree RSS: 2792.70 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 467.0237422565277,
            "range": "462.0-469.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 580.75 ms; process-tree RSS: 6766.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 479.1245398228553,
            "range": "475.6-488.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 571.07 ms; process-tree RSS: 6757.51 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.5691560519635,
            "range": "462.0-468.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 551.07 ms; process-tree RSS: 6701.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1729.700219641078,
            "range": "1687.6-1745.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.85 ms; process-tree RSS: 6681.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.1461629489206,
            "range": "188.9-190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2233.6672949508343,
            "range": "2212.4-2235.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2626.1733240385497,
            "range": "2620.8-2637.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.8578880424202,
            "range": "1223.5-1232.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.7399006164549,
            "range": "1220.0-1229.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.67421483442098,
            "range": "188.6-191.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1901.0081844677159,
            "range": "1895.4-1960.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2390.3700878353384,
            "range": "2379.6-2401.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2236.4171936115545,
            "range": "2232.1-2244.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "303c50429547c8ae28ce2cbb7fd0960f207ab4f2",
          "message": "[Performance] Run inference from environment processes (#4272)",
          "timestamp": "2026-09-08T07:23:53+01:00",
          "tree_id": "d0869de7c9b96cb590ca59244d6b277acb1528e4",
          "url": "https://github.com/pytorch/rl/commit/303c50429547c8ae28ce2cbb7fd0960f207ab4f2"
        },
        "date": 1788852361819,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 860.7978154753349,
            "range": "835.7-887.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 429.28 ms; process-tree RSS: 7561.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 826.0883182767465,
            "range": "814.4-830.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 443.13 ms; process-tree RSS: 7613.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.23009258680132,
            "range": "81.8-82.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3224.66 ms; process-tree RSS: 6693.56 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.11107983851196,
            "range": "82.7-83.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3189.51 ms; process-tree RSS: 6713.04 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 481.8984501635922,
            "range": "474.9-485.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 574.53 ms; process-tree RSS: 2764.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 508.1586829283123,
            "range": "498.7-528.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 544.63 ms; process-tree RSS: 2763.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 501.3210809208843,
            "range": "491.3-509.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 555.92 ms; process-tree RSS: 2777.19 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 514.643110261423,
            "range": "490.8-526.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 532.93 ms; process-tree RSS: 2776.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 473.2028694722521,
            "range": "470.9-481.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 584.99 ms; process-tree RSS: 6745.87 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 488.62976199404164,
            "range": "484.7-492.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.18 ms; process-tree RSS: 6726.08 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 469.1642573061799,
            "range": "465.6-471.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.04 ms; process-tree RSS: 6684.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1736.369297954363,
            "range": "1706.8-1806.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.57 ms; process-tree RSS: 6694.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.52105811449732,
            "range": "190.4-191.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2255.9961193346353,
            "range": "2252.4-2358.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2645.9847848201975,
            "range": "2626.1-2660.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1228.1790395553346,
            "range": "1227.4-1236.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.2920943411161,
            "range": "1224.5-1226.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.28759030781595,
            "range": "186.1-196.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1944.708208156109,
            "range": "1940.3-2009.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2400.328107037397,
            "range": "2387.4-2479.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2272.407657399985,
            "range": "2263.5-2275.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d670c59b26285501dde5cab18fc666248e59e1b1",
          "message": "[Feature] Add native stream replay (#4274)",
          "timestamp": "2026-09-08T07:34:57+01:00",
          "tree_id": "c53f65850a9d2e0cce9fce72655c2eef106197cf",
          "url": "https://github.com/pytorch/rl/commit/d670c59b26285501dde5cab18fc666248e59e1b1"
        },
        "date": 1788852681431,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 885.8372526045194,
            "range": "848.0-935.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 426.80 ms; process-tree RSS: 7559.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 823.4845486028097,
            "range": "790.5-835.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 445.22 ms; process-tree RSS: 7601.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.66597472038337,
            "range": "82.4-83.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3225.01 ms; process-tree RSS: 6705.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.37123820112008,
            "range": "83.2-84.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3203.66 ms; process-tree RSS: 6694.65 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 496.184503494049,
            "range": "484.9-501.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 553.75 ms; process-tree RSS: 2770.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 501.6887598148345,
            "range": "501.5-503.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 546.18 ms; process-tree RSS: 2770.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 484.1842622162737,
            "range": "483.1-500.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 565.77 ms; process-tree RSS: 2770.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 493.13731570654295,
            "range": "487.6-499.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 560.78 ms; process-tree RSS: 2764.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.148816040746,
            "range": "471.2-479.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 578.19 ms; process-tree RSS: 6728.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 477.9068467166622,
            "range": "474.2-487.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 585.25 ms; process-tree RSS: 6747.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.042324814646,
            "range": "460.8-465.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.20 ms; process-tree RSS: 6695.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1778.9732694339057,
            "range": "1757.6-1909.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 147.33 ms; process-tree RSS: 6704.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.9779062392516,
            "range": "189.6-190.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2259.799091325212,
            "range": "2221.6-2281.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2628.2991627381853,
            "range": "2626.8-2634.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.7223734334125,
            "range": "1226.1-1233.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.2500067003,
            "range": "1223.1-1225.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 193.09075997616065,
            "range": "188.4-198.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1978.0843246992144,
            "range": "1952.2-1982.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2380.340531376775,
            "range": "2349.4-2423.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2245.2325917350645,
            "range": "2236.5-2255.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "c7def7982b33efb570b46a535c1de083bef45f37",
          "message": "[Feature] Route async collection into replay (#4275)",
          "timestamp": "2026-09-08T08:03:55+01:00",
          "tree_id": "b9c45ccec9cb190b4479a396837e9a295d157c0c",
          "url": "https://github.com/pytorch/rl/commit/c7def7982b33efb570b46a535c1de083bef45f37"
        },
        "date": 1788854124942,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 931.9414863681611,
            "range": "919.3-991.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 374.99 ms; process-tree RSS: 7447.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1001.7651601048606,
            "range": "996.4-1023.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.94 ms; process-tree RSS: 7465.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.72021416887823,
            "range": "82.0-83.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3255.63 ms; process-tree RSS: 6698.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.90211041608838,
            "range": "82.3-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3200.32 ms; process-tree RSS: 6695.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 497.57131311837895,
            "range": "488.0-508.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 543.93 ms; process-tree RSS: 2778.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 511.8037734096614,
            "range": "507.1-516.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 536.19 ms; process-tree RSS: 2776.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 499.23011914596697,
            "range": "493.9-502.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 542.74 ms; process-tree RSS: 2776.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 507.7089392409764,
            "range": "506.4-512.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 535.73 ms; process-tree RSS: 2774.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 478.7438528786492,
            "range": "476.2-486.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 562.86 ms; process-tree RSS: 6721.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 495.12467869675163,
            "range": "494.2-499.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.20 ms; process-tree RSS: 6734.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 470.7653107667225,
            "range": "464.4-473.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.43 ms; process-tree RSS: 6705.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1727.2040031379872,
            "range": "1709.4-1781.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.64 ms; process-tree RSS: 6686.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.4270876685536,
            "range": "189.8-190.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2238.3445546422413,
            "range": "2219.8-2340.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2649.4384447944994,
            "range": "2641.6-2660.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.8709348950129,
            "range": "1225.8-1234.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.726990648249,
            "range": "1223.7-1228.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 194.34182871668307,
            "range": "191.6-203.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1996.078905717204,
            "range": "1931.2-2015.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2452.162523814651,
            "range": "2433.2-2511.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2274.2917366915185,
            "range": "2254.9-2279.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "ffa8471a3d49339e780fb8b7fbf83360e406b418",
          "message": "[Performance] Use native replay collection in DreamerV3 (#4265)",
          "timestamp": "2026-09-08T08:04:13+01:00",
          "tree_id": "1f163e91791410417486537fd3080b1b017e1526",
          "url": "https://github.com/pytorch/rl/commit/ffa8471a3d49339e780fb8b7fbf83360e406b418"
        },
        "date": 1788854391203,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 921.9340138354627,
            "range": "902.2-967.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 376.36 ms; process-tree RSS: 7449.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 997.0615185196973,
            "range": "996.4-1024.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.15 ms; process-tree RSS: 7447.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.23652640329263,
            "range": "81.7-82.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3249.27 ms; process-tree RSS: 6714.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.64163790909342,
            "range": "82.6-83.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3209.13 ms; process-tree RSS: 6718.92 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 505.4032164310738,
            "range": "489.7-520.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 534.95 ms; process-tree RSS: 2761.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 519.4417974556621,
            "range": "514.1-528.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 526.82 ms; process-tree RSS: 2762.13 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 511.0778331341972,
            "range": "489.3-513.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 529.55 ms; process-tree RSS: 2764.06 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 513.2960893076189,
            "range": "509.6-532.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 542.47 ms; process-tree RSS: 2779.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 474.3050643733749,
            "range": "466.2-476.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 578.12 ms; process-tree RSS: 6721.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 496.27150467348264,
            "range": "486.7-504.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 559.58 ms; process-tree RSS: 6725.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.8968658500092,
            "range": "465.8-472.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.03 ms; process-tree RSS: 6700.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1719.8697946370976,
            "range": "1708.9-1941.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.97 ms; process-tree RSS: 6678.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.77934982712384,
            "range": "189.6-190.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2230.6393908674436,
            "range": "2186.5-2231.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2638.06237955826,
            "range": "2624.4-2638.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.7289963724406,
            "range": "1222.9-1235.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.1452515581925,
            "range": "1223.7-1229.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 193.4975221183469,
            "range": "192.6-202.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1948.198160511657,
            "range": "1825.5-1963.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2475.372036512062,
            "range": "2394.0-2476.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2252.129804681807,
            "range": "2248.4-2259.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "53e877e2137db36cf4a85de0e7581737ba63fa88",
          "message": "[Performance] Compile the complete DreamerV3 learner step (#4268)",
          "timestamp": "2026-09-08T08:05:02+01:00",
          "tree_id": "6cf05f7cdc41d2458d118868249dd6e3a6aa7cb2",
          "url": "https://github.com/pytorch/rl/commit/53e877e2137db36cf4a85de0e7581737ba63fa88"
        },
        "date": 1788854728077,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 904.9029932942093,
            "range": "893.1-983.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 377.37 ms; process-tree RSS: 7460.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1017.5005776973699,
            "range": "1008.5-1019.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.88 ms; process-tree RSS: 7469.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.55571606173324,
            "range": "81.0-82.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3250.48 ms; process-tree RSS: 6705.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.32404743244577,
            "range": "82.0-83.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3224.59 ms; process-tree RSS: 6693.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 489.4163541357764,
            "range": "488.0-498.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 558.85 ms; process-tree RSS: 2776.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 498.57404123854997,
            "range": "497.9-503.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 533.91 ms; process-tree RSS: 2775.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 496.4302657063245,
            "range": "494.6-500.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 551.18 ms; process-tree RSS: 2775.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 513.1367299896266,
            "range": "500.0-520.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 540.22 ms; process-tree RSS: 2774.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 478.57548596750985,
            "range": "474.5-480.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 578.69 ms; process-tree RSS: 6732.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 494.1322418768508,
            "range": "494.1-503.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 575.08 ms; process-tree RSS: 6728.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.3817568940099,
            "range": "464.3-476.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 555.08 ms; process-tree RSS: 6674.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1699.3676531811584,
            "range": "1688.9-1752.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.02 ms; process-tree RSS: 6689.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.53044006715697,
            "range": "188.7-189.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2244.117144975527,
            "range": "2225.3-2279.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2633.789602616316,
            "range": "2618.6-2643.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1233.8376634347792,
            "range": "1233.0-1236.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.7046289295977,
            "range": "1227.6-1230.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.95800636886736,
            "range": "188.2-195.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1961.940109233375,
            "range": "1736.4-1968.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2413.6471418442275,
            "range": "2411.6-2434.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2242.968669968257,
            "range": "2230.0-2262.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e963df811902d2d8c37cdaea5818389855a6aebd",
          "message": "[Feature] Support configurable DreamerV3 environments and run controls (#4280)",
          "timestamp": "2026-09-08T08:12:57Z",
          "tree_id": "dde565eea4f7b3d51ebad6fea1030eb2236b2cad",
          "url": "https://github.com/pytorch/rl/commit/e963df811902d2d8c37cdaea5818389855a6aebd"
        },
        "date": 1788859517553,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 915.9690183570682,
            "range": "903.7-998.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.98 ms; process-tree RSS: 7467.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 995.9181245280057,
            "range": "990.8-1001.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.54 ms; process-tree RSS: 7466.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.87670089535975,
            "range": "81.9-82.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3219.99 ms; process-tree RSS: 6691.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 84.28692763295851,
            "range": "83.5-84.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3189.01 ms; process-tree RSS: 6689.27 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 498.8134703521496,
            "range": "485.6-504.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 549.32 ms; process-tree RSS: 2775.33 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 509.0269750136206,
            "range": "508.4-533.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 527.48 ms; process-tree RSS: 2776.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 489.61506438291366,
            "range": "486.6-492.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 565.52 ms; process-tree RSS: 2776.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 508.49067221949167,
            "range": "506.4-533.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 534.22 ms; process-tree RSS: 2775.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 491.11147838523686,
            "range": "478.7-494.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 565.57 ms; process-tree RSS: 6723.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 501.0844254805637,
            "range": "500.7-508.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 550.69 ms; process-tree RSS: 6729.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.94890420370257,
            "range": "466.2-468.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.28 ms; process-tree RSS: 6698.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1790.091337487498,
            "range": "1749.2-1812.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 149.05 ms; process-tree RSS: 6704.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.93344159701874,
            "range": "190.9-192.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2278.0167248250928,
            "range": "2230.2-2293.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2665.2274646583755,
            "range": "2643.1-2667.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1222.9316265997377,
            "range": "1222.2-1226.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.421005205787,
            "range": "1223.1-1231.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.82936904710516,
            "range": "191.0-192.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1960.6809822636335,
            "range": "1817.0-1980.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2420.6531869756964,
            "range": "2414.3-2431.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2277.4187725838965,
            "range": "2261.6-2282.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "db7c2e97dafbdf310a1d3943bec35021ebe3dd81",
          "message": "[Test] Baseline benchmarks for async collectors and DreamerV3 (#4312)",
          "timestamp": "2026-09-09T09:58:47+01:00",
          "tree_id": "baf547d13c55b319c1dcf45e39a522e89528c2f2",
          "url": "https://github.com/pytorch/rl/commit/db7c2e97dafbdf310a1d3943bec35021ebe3dd81"
        },
        "date": 1788948789522,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 994.8534443694253,
            "range": "965.7-1008.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 370.73 ms; process-tree RSS: 7459.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 996.0858785946575,
            "range": "979.3-1012.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 372.77 ms; process-tree RSS: 7460.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 973.6765545090408,
            "range": "970.8-990.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 375.21 ms; process-tree RSS: 7449.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 999.0859641556519,
            "range": "972.2-1022.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.53 ms; process-tree RSS: 7462.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.86010226245382,
            "range": "81.6-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3259.53 ms; process-tree RSS: 6743.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.47498991034489,
            "range": "82.2-82.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3198.95 ms; process-tree RSS: 6730.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 496.2158609938161,
            "range": "490.2-501.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 553.18 ms; process-tree RSS: 2782.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 515.064844943748,
            "range": "503.7-525.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 542.72 ms; process-tree RSS: 2779.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 480.38967303731374,
            "range": "478.0-485.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 563.98 ms; process-tree RSS: 2780.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 511.26704960277306,
            "range": "501.9-516.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 554.16 ms; process-tree RSS: 2773.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 482.7379879647494,
            "range": "477.2-482.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 567.58 ms; process-tree RSS: 6730.44 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 489.2931941062773,
            "range": "486.8-497.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 558.78 ms; process-tree RSS: 6739.87 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 463.93835820662576,
            "range": "462.6-471.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 557.53 ms; process-tree RSS: 6704.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1741.536784304634,
            "range": "1684.2-1771.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 151.03 ms; process-tree RSS: 6711.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.53541743250662,
            "range": "189.4-192.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2240.314561166843,
            "range": "2216.2-2287.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2620.20400156051,
            "range": "2619.8-2632.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.711215752779,
            "range": "1228.2-1234.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1229.5110360680944,
            "range": "1228.2-1230.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 188.2985379545109,
            "range": "186.5-188.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1905.964988435905,
            "range": "1902.8-1959.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2405.0043819694915,
            "range": "2402.0-2422.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2233.9719669791407,
            "range": "2231.5-2238.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7f348ee4cdf1ac598fd22059af46e24260e85bf",
          "message": "[BugFix] Async DreamerV3: explicit exploration, stored episode flags, cross-episode replay sequences (#4310)",
          "timestamp": "2026-09-09T10:00:30+01:00",
          "tree_id": "46797f58ed9134dda4573a56b110776a6792db2d",
          "url": "https://github.com/pytorch/rl/commit/b7f348ee4cdf1ac598fd22059af46e24260e85bf"
        },
        "date": 1788949103272,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 978.4928560250534,
            "range": "968.2-981.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 383.71 ms; process-tree RSS: 7481.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1007.165539226154,
            "range": "1005.6-1019.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.09 ms; process-tree RSS: 7479.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1007.7788233011255,
            "range": "976.9-1008.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 376.74 ms; process-tree RSS: 7486.49 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 996.1394363218388,
            "range": "993.1-1015.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.39 ms; process-tree RSS: 7475.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.3205717374336,
            "range": "82.0-82.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3265.08 ms; process-tree RSS: 6698.92 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.35348021235805,
            "range": "81.3-82.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3220.17 ms; process-tree RSS: 6709.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 502.41359195189466,
            "range": "494.7-502.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 543.77 ms; process-tree RSS: 2787.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 510.7465409063514,
            "range": "508.3-513.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 563.50 ms; process-tree RSS: 2786.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 476.9911576646119,
            "range": "476.8-487.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 568.54 ms; process-tree RSS: 2796.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 508.10724331194405,
            "range": "500.8-508.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 539.84 ms; process-tree RSS: 2796.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 474.48856157999387,
            "range": "473.2-477.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 571.43 ms; process-tree RSS: 6750.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 485.92959675284015,
            "range": "480.8-489.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 560.23 ms; process-tree RSS: 6750.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 465.0796962929864,
            "range": "463.5-467.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.09 ms; process-tree RSS: 6701.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1763.5846323634025,
            "range": "1726.3-1879.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.86 ms; process-tree RSS: 6719.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.9156424112526,
            "range": "189.2-189.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2245.59658739342,
            "range": "2214.0-2285.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2617.2767059003218,
            "range": "2609.5-2620.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.9476398493653,
            "range": "1226.0-1229.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.163676392244,
            "range": "1224.0-1227.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 188.91293059905342,
            "range": "187.0-189.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1970.6051867659573,
            "range": "1912.3-1972.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2402.327149324076,
            "range": "2363.1-2412.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2234.8212842555395,
            "range": "2227.7-2246.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5076ae186cdc29c4f9f14e36405fdba4dff2b6bd",
          "message": "[Performance] Chunk process-slot worker results in AsyncBatchedCollector (#4305)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T11:28:10+01:00",
          "tree_id": "875733f3bf4fb2745762c16c617861cfb48c8aca",
          "url": "https://github.com/pytorch/rl/commit/5076ae186cdc29c4f9f14e36405fdba4dff2b6bd"
        },
        "date": 1788953903214,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1545.1680716506792,
            "range": "1508.1-1576.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 278.69 ms; process-tree RSS: 7587.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1603.9395738258372,
            "range": "1572.7-1643.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 323.45 ms; process-tree RSS: 7613.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1608.4405582271147,
            "range": "1576.1-1615.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 271.00 ms; process-tree RSS: 7757.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1644.884585074976,
            "range": "1623.4-1659.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 314.22 ms; process-tree RSS: 7645.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 995.8410362099985,
            "range": "983.0-1003.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 381.67 ms; process-tree RSS: 7457.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1022.1630370748413,
            "range": "1000.3-1026.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 374.70 ms; process-tree RSS: 7453.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.56503271286667,
            "range": "82.0-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3218.99 ms; process-tree RSS: 6696.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.93103014136318,
            "range": "82.8-83.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3166.19 ms; process-tree RSS: 6696.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 493.59236182559806,
            "range": "492.8-510.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 551.00 ms; process-tree RSS: 2764.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 523.7184501354653,
            "range": "502.2-524.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 535.20 ms; process-tree RSS: 2765.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 492.7066684019121,
            "range": "483.9-496.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 566.64 ms; process-tree RSS: 2763.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 510.6065071004321,
            "range": "504.4-515.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 545.97 ms; process-tree RSS: 2776.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 476.6919550631519,
            "range": "476.0-481.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 589.48 ms; process-tree RSS: 6736.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 496.910574240135,
            "range": "490.4-502.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 557.06 ms; process-tree RSS: 6729.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.86797715402315,
            "range": "461.3-468.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.45 ms; process-tree RSS: 6710.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1700.005077768823,
            "range": "1694.6-1703.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.58 ms; process-tree RSS: 6700.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.40527656635012,
            "range": "191.1-193.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2327.565222872454,
            "range": "2293.1-2364.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2646.9764396568103,
            "range": "2646.3-2658.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.0031263233823,
            "range": "1223.4-1232.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.0977188816726,
            "range": "1225.4-1231.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.41193152940986,
            "range": "191.9-193.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1957.7143302065517,
            "range": "1955.6-2004.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2466.62139444899,
            "range": "2455.2-2505.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2271.9491619069827,
            "range": "2267.0-2272.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e11e38e50ccc89e006096c42ec32408a98131324",
          "message": "[Performance] Serve process-slot inference passes from pinned staging batches (#4306)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T12:13:05+01:00",
          "tree_id": "6f41274772cb4d7f2e88ee9dd653414365353492",
          "url": "https://github.com/pytorch/rl/commit/e11e38e50ccc89e006096c42ec32408a98131324"
        },
        "date": 1788956589568,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1674.8007793393517,
            "range": "1673.4-1775.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 278.73 ms; process-tree RSS: 7615.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1724.133448844689,
            "range": "1713.9-1770.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 285.69 ms; process-tree RSS: 7593.04 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1736.6052196721917,
            "range": "1656.0-1815.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.89 ms; process-tree RSS: 7616.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1762.0646700927634,
            "range": "1676.1-1768.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 275.64 ms; process-tree RSS: 7620.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1019.9253787333189,
            "range": "1017.4-1024.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.29 ms; process-tree RSS: 7471.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1036.1649757388625,
            "range": "1029.7-1057.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 361.10 ms; process-tree RSS: 7457.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.97166057339108,
            "range": "81.6-82.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3268.83 ms; process-tree RSS: 6707.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.37222664668566,
            "range": "82.8-83.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3182.83 ms; process-tree RSS: 6709.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 497.57821225277553,
            "range": "492.0-503.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 559.29 ms; process-tree RSS: 2787.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 526.5583169949043,
            "range": "509.8-533.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 522.12 ms; process-tree RSS: 2793.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 494.99499251659904,
            "range": "490.9-498.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 538.91 ms; process-tree RSS: 2786.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 520.2174146266876,
            "range": "506.7-535.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 527.99 ms; process-tree RSS: 2790.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 484.1499852513183,
            "range": "476.3-485.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 563.07 ms; process-tree RSS: 6751.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 499.8112307184785,
            "range": "491.1-503.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 538.73 ms; process-tree RSS: 6751.19 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 469.89470633835884,
            "range": "464.0-472.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.01 ms; process-tree RSS: 6700.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1734.645462979456,
            "range": "1720.8-1747.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.34 ms; process-tree RSS: 6708.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.47569237944217,
            "range": "190.7-191.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2278.3212198104825,
            "range": "2251.1-2287.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2647.4398382259624,
            "range": "2631.0-2650.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1228.54504152145,
            "range": "1223.5-1231.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.800185610614,
            "range": "1218.7-1231.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.04892542399645,
            "range": "185.5-194.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1904.4434064403156,
            "range": "1892.0-1965.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2413.5873446398823,
            "range": "2413.1-2455.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2268.410606563731,
            "range": "2245.0-2277.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8fb9586eedd544ad7ff2249b07a7a6e9d007f3ed",
          "message": "[Performance] Keep the DreamerV3 replay write-back off the learner stream (#4307)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T13:09:30+01:00",
          "tree_id": "5d11090585cafa84b06ca40bbd03a47419acacb1",
          "url": "https://github.com/pytorch/rl/commit/8fb9586eedd544ad7ff2249b07a7a6e9d007f3ed"
        },
        "date": 1788960356985,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1759.9924551323238,
            "range": "1759.0-1862.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 278.68 ms; process-tree RSS: 7599.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1711.0121384528816,
            "range": "1629.7-1728.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 265.64 ms; process-tree RSS: 7589.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1680.016089828961,
            "range": "1677.5-1812.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 282.63 ms; process-tree RSS: 7575.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1710.7647380173112,
            "range": "1621.8-1768.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 265.84 ms; process-tree RSS: 7586.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1028.0208155744876,
            "range": "1021.4-1041.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.69 ms; process-tree RSS: 7461.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1038.1738133305078,
            "range": "1011.6-1040.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.54 ms; process-tree RSS: 7448.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.50325910043033,
            "range": "80.8-82.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3306.37 ms; process-tree RSS: 6705.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.55876237542185,
            "range": "82.1-82.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3223.49 ms; process-tree RSS: 6707.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 498.54456122815384,
            "range": "495.0-505.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 541.59 ms; process-tree RSS: 2772.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 527.2887874293983,
            "range": "512.5-543.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 533.39 ms; process-tree RSS: 2771.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 484.1580344571229,
            "range": "475.7-486.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 575.35 ms; process-tree RSS: 2776.35 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 514.2026278354407,
            "range": "502.8-519.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 539.89 ms; process-tree RSS: 2778.53 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.0674971747986,
            "range": "470.8-476.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 569.94 ms; process-tree RSS: 6727.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 494.2369323916407,
            "range": "491.1-497.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.48 ms; process-tree RSS: 6718.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.7593417420992,
            "range": "459.5-468.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.67 ms; process-tree RSS: 6705.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1746.1465870006512,
            "range": "1689.8-1757.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 149.82 ms; process-tree RSS: 6699.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.94419240778922,
            "range": "189.6-190.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2239.556658063869,
            "range": "2222.3-2256.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2633.2784074059405,
            "range": "2622.5-2654.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1227.9292017272512,
            "range": "1227.2-1233.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.5443398222442,
            "range": "1226.3-1230.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 195.3959259074386,
            "range": "189.3-195.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1952.7176462524085,
            "range": "1868.4-1996.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2417.7919111145434,
            "range": "2398.2-2490.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2259.2382748920945,
            "range": "2249.0-2259.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a08aa5859fb4e2ed3b32fe2bd9b8b60616a7b93",
          "message": "[Performance] DreamerV3: freeze the setup heap before the training loop (#4308)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T14:04:07+01:00",
          "tree_id": "f5fead42a3db7becb7b1f6e4cfdd8328810490bb",
          "url": "https://github.com/pytorch/rl/commit/0a08aa5859fb4e2ed3b32fe2bd9b8b60616a7b93"
        },
        "date": 1788963034322,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1700.9694028946478,
            "range": "1685.4-1709.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.51 ms; process-tree RSS: 7619.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1743.7118718835604,
            "range": "1712.4-1782.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 273.38 ms; process-tree RSS: 7581.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1802.0876460319391,
            "range": "1743.2-1823.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 275.92 ms; process-tree RSS: 7594.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1715.8796132430134,
            "range": "1686.4-1812.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 283.36 ms; process-tree RSS: 7611.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1024.3444809027662,
            "range": "1020.7-1040.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 370.81 ms; process-tree RSS: 7464.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1036.5504093119862,
            "range": "970.0-1058.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.72 ms; process-tree RSS: 7447.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.64199886372978,
            "range": "81.5-82.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3241.71 ms; process-tree RSS: 6702.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.4037011889557,
            "range": "82.0-84.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3223.22 ms; process-tree RSS: 6711.92 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 490.9503649023057,
            "range": "471.9-517.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 561.39 ms; process-tree RSS: 2785.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 503.94510869244124,
            "range": "501.6-539.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 535.44 ms; process-tree RSS: 2784.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 489.29216236359053,
            "range": "487.9-490.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 548.96 ms; process-tree RSS: 2784.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 511.11525285285126,
            "range": "506.6-515.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 546.87 ms; process-tree RSS: 2778.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 481.86059891926936,
            "range": "478.0-484.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 574.03 ms; process-tree RSS: 6733.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 494.5967870598336,
            "range": "484.9-495.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.67 ms; process-tree RSS: 6735.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 467.15313740747877,
            "range": "466.3-471.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.13 ms; process-tree RSS: 6712.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1758.3075166483252,
            "range": "1749.1-1772.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 150.39 ms; process-tree RSS: 6696.27 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.06871456968062,
            "range": "189.1-191.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2236.711447734057,
            "range": "2232.9-2238.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2635.817447088181,
            "range": "2624.0-2638.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.676751653405,
            "range": "1228.4-1234.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.3360201605815,
            "range": "1220.8-1230.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.7629147082659,
            "range": "187.9-192.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1981.439258614995,
            "range": "1943.6-1987.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2457.395732420192,
            "range": "2420.7-2460.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2245.770423940615,
            "range": "2242.0-2258.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "iliam.souami@gmail.com",
            "name": "Iliam Souami",
            "username": "Iliamsou"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9b0587e271df94a0dd0566c44a8d5f04233b6e52",
          "message": "[Feature] Add LBForaging environment wrapper (#4301)\n\nCo-authored-by: Vincent Moens <vincentmoens@gmail.com>",
          "timestamp": "2026-09-09T15:14:46+01:00",
          "tree_id": "25f47b5c871ffc1d96d14d99a930cd601f1662fc",
          "url": "https://github.com/pytorch/rl/commit/9b0587e271df94a0dd0566c44a8d5f04233b6e52"
        },
        "date": 1788967212277,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1690.0251774637798,
            "range": "1613.0-1814.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 284.84 ms; process-tree RSS: 7600.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1733.6989368033373,
            "range": "1685.4-1790.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 247.36 ms; process-tree RSS: 7561.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1739.4060810167757,
            "range": "1696.3-1855.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 284.65 ms; process-tree RSS: 7591.17 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1696.205550747145,
            "range": "1685.2-1732.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 286.01 ms; process-tree RSS: 7592.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1029.1808056075633,
            "range": "1017.8-1031.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.99 ms; process-tree RSS: 7449.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1046.6941857354893,
            "range": "1037.1-1061.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 363.26 ms; process-tree RSS: 7462.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.00042688461252,
            "range": "80.8-82.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3249.84 ms; process-tree RSS: 6712.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.5607916611902,
            "range": "82.4-84.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3260.79 ms; process-tree RSS: 6723.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 489.3560633381632,
            "range": "484.9-492.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 579.31 ms; process-tree RSS: 2774.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 505.9491339660576,
            "range": "500.4-508.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 539.18 ms; process-tree RSS: 2771.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 489.2702611268601,
            "range": "477.1-492.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 558.92 ms; process-tree RSS: 2772.35 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 509.0350991044526,
            "range": "503.4-512.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 541.16 ms; process-tree RSS: 2792.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 478.90093992752327,
            "range": "477.7-479.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 572.07 ms; process-tree RSS: 6767.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 491.47417903662836,
            "range": "487.4-503.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.37 ms; process-tree RSS: 6750.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.74937419995865,
            "range": "465.8-467.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.83 ms; process-tree RSS: 6711.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1873.777532852737,
            "range": "1721.3-1897.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 148.33 ms; process-tree RSS: 6702.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.12677484572268,
            "range": "189.7-190.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2266.7857192563442,
            "range": "2199.4-2270.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2627.7494644331355,
            "range": "2619.0-2646.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.4520166048424,
            "range": "1230.2-1233.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1222.8056122180415,
            "range": "1206.5-1223.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.53797201472898,
            "range": "189.1-194.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1955.7215885088883,
            "range": "1895.0-1987.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2420.393176822425,
            "range": "2396.7-2456.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2239.2218571089616,
            "range": "2214.2-2258.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2706e86d2ad7a81ad7df21e3a0587dddc65daa2",
          "message": "[Feature] Fast asynchronous defaults for AsyncBatchedCollector and DreamerV3 (#4313)",
          "timestamp": "2026-09-09T16:53:46+01:00",
          "tree_id": "1b7a8f8b087b991e912db98ef6572865ed5d997b",
          "url": "https://github.com/pytorch/rl/commit/a2706e86d2ad7a81ad7df21e3a0587dddc65daa2"
        },
        "date": 1788971452658,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1747.510432897315,
            "range": "1693.0-1781.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 286.21 ms; process-tree RSS: 7580.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1724.7891498743497,
            "range": "1686.5-1727.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 263.59 ms; process-tree RSS: 7570.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1760.2514340388177,
            "range": "1726.2-1792.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 281.95 ms; process-tree RSS: 7733.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1674.4695174695323,
            "range": "1630.6-1699.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.12 ms; process-tree RSS: 7621.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 992.3381445876025,
            "range": "991.9-1014.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 371.79 ms; process-tree RSS: 7458.27 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1037.7101549655226,
            "range": "1013.4-1051.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.49 ms; process-tree RSS: 7472.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.44876445078195,
            "range": "81.4-81.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3240.52 ms; process-tree RSS: 6718.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.31430257158691,
            "range": "82.7-83.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3208.52 ms; process-tree RSS: 6714.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 495.1645077500628,
            "range": "480.5-500.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 560.56 ms; process-tree RSS: 2781.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 519.2793735382828,
            "range": "519.1-527.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 530.08 ms; process-tree RSS: 2781.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 488.4847558884562,
            "range": "477.6-492.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 559.15 ms; process-tree RSS: 2781.90 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 505.73520964216107,
            "range": "503.0-509.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 526.80 ms; process-tree RSS: 2786.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 479.34619987006,
            "range": "478.1-479.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 561.40 ms; process-tree RSS: 6742.70 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 492.94285699962694,
            "range": "489.7-494.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 559.80 ms; process-tree RSS: 6732.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.7700283541243,
            "range": "461.3-467.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 561.95 ms; process-tree RSS: 6695.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1717.5207825677387,
            "range": "1693.8-1723.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.71 ms; process-tree RSS: 6717.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.97247238766593,
            "range": "189.8-190.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2259.510684823897,
            "range": "2251.6-2272.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2652.6686241408734,
            "range": "2646.0-2652.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.1626363904363,
            "range": "1228.8-1232.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.8963701667676,
            "range": "1226.0-1230.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.212872230485,
            "range": "187.9-193.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1932.2106754060378,
            "range": "1879.5-1977.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2411.85926841725,
            "range": "2396.6-2485.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2255.258728170762,
            "range": "2245.6-2258.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "crystalinecohomology@gmail.com",
            "name": "ゆり",
            "username": "yurekami"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "588010e30b5b99f9129477412f092285f3857132",
          "message": "[BugFix] Clamp negative MultiCategorical projections (#4317)\n\nCo-authored-by: yurekami <yurekami@users.noreply.github.com>",
          "timestamp": "2026-09-09T20:46:16+01:00",
          "tree_id": "66382fb2cd2f135fee5c48b1df4780fd1a60bd4d",
          "url": "https://github.com/pytorch/rl/commit/588010e30b5b99f9129477412f092285f3857132"
        },
        "date": 1788986279537,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1739.6022516660403,
            "range": "1590.7-1806.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 274.13 ms; process-tree RSS: 7591.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1740.831524021399,
            "range": "1634.0-1770.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 275.84 ms; process-tree RSS: 7567.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1673.9225624564767,
            "range": "1613.4-1796.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 291.78 ms; process-tree RSS: 7573.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1707.3182219834675,
            "range": "1663.9-1787.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 265.40 ms; process-tree RSS: 7574.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1029.5456786844975,
            "range": "1025.8-1031.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.63 ms; process-tree RSS: 7454.44 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1051.5766156239206,
            "range": "1026.9-1067.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.80 ms; process-tree RSS: 7457.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.60211717530012,
            "range": "82.6-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3206.22 ms; process-tree RSS: 6704.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.26922905133449,
            "range": "82.7-83.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3207.84 ms; process-tree RSS: 6686.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 489.2993458425794,
            "range": "488.5-516.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 554.88 ms; process-tree RSS: 2778.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 517.8474133689934,
            "range": "510.9-548.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 535.27 ms; process-tree RSS: 2776.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 513.7360175210237,
            "range": "512.0-533.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 542.71 ms; process-tree RSS: 2777.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 510.8389025092104,
            "range": "510.6-521.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 537.77 ms; process-tree RSS: 2773.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 486.3077223194296,
            "range": "480.3-488.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 576.65 ms; process-tree RSS: 6740.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 505.08722147753605,
            "range": "494.9-512.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 540.58 ms; process-tree RSS: 6740.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 465.0189486774986,
            "range": "465.0-480.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 555.45 ms; process-tree RSS: 6679.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1767.2681836614313,
            "range": "1715.6-1788.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 151.47 ms; process-tree RSS: 6697.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.70733979765853,
            "range": "190.6-191.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2299.367003126045,
            "range": "2283.1-2302.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2662.455764070962,
            "range": "2645.2-2666.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.5796771268538,
            "range": "1225.5-1231.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.4974318930997,
            "range": "1224.6-1229.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 199.2828735346279,
            "range": "190.4-203.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1998.3310189122667,
            "range": "1937.8-2024.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2440.7230949162795,
            "range": "2423.7-2462.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2276.127070830759,
            "range": "2266.2-2307.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "apaninga@berkeley.edu",
            "name": "theap06",
            "username": "theap06"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e219c9727bd9b97aef5c830539b49eea6de125f7",
          "message": "[BugFix] DreamerV3: reduction-safe two-hot decode and dtype-stable RSSM scan carry (#4290)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T18:54:53-07:00",
          "tree_id": "28b2533ddede1dd32dcc821cf4f8f468d49f2dd0",
          "url": "https://github.com/pytorch/rl/commit/e219c9727bd9b97aef5c830539b49eea6de125f7"
        },
        "date": 1789007866138,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1793.1779323082296,
            "range": "1763.3-1802.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 280.32 ms; process-tree RSS: 7639.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1782.226657978913,
            "range": "1741.6-1801.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 265.93 ms; process-tree RSS: 7591.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1725.1688292081035,
            "range": "1718.4-1808.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 279.70 ms; process-tree RSS: 8020.81 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1764.5158786178142,
            "range": "1718.3-1768.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 261.35 ms; process-tree RSS: 7749.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1044.4081136611178,
            "range": "1028.9-1045.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.14 ms; process-tree RSS: 7476.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1030.1447982694967,
            "range": "1013.5-1049.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.49 ms; process-tree RSS: 7481.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.6792506960849,
            "range": "81.2-83.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3264.22 ms; process-tree RSS: 6726.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.5095035310347,
            "range": "82.4-83.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3218.07 ms; process-tree RSS: 6720.90 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 497.8769152365627,
            "range": "489.9-503.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 541.90 ms; process-tree RSS: 2793.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 506.1059475775414,
            "range": "502.1-507.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 555.59 ms; process-tree RSS: 2792.65 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 490.78023627049146,
            "range": "487.5-512.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 550.12 ms; process-tree RSS: 2789.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 503.93880935602886,
            "range": "500.2-513.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 538.57 ms; process-tree RSS: 2787.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 485.1480079238454,
            "range": "475.4-487.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 573.79 ms; process-tree RSS: 6752.51 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 492.92414798684547,
            "range": "485.1-493.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 550.23 ms; process-tree RSS: 6766.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.4508686081732,
            "range": "463.6-467.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.22 ms; process-tree RSS: 6701.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1794.790626166332,
            "range": "1737.7-1901.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 147.36 ms; process-tree RSS: 6693.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.31425722198478,
            "range": "189.8-191.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2257.4797596059466,
            "range": "2246.3-2263.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2635.7402937290526,
            "range": "2628.7-2643.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1225.0371904661406,
            "range": "1222.9-1226.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.5907461935572,
            "range": "1225.3-1228.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.12626389164888,
            "range": "189.2-198.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1984.7724189680287,
            "range": "1913.4-1994.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2452.794587860725,
            "range": "2433.5-2452.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2233.7155662029913,
            "range": "2230.8-2238.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "apaninga@berkeley.edu",
            "name": "theap06",
            "username": "theap06"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4722f653d9eb73e9ac21d5895b00fced44d1026",
          "message": "[BugFix] TensorDictPrimer fills non-float specs as float32 (#4318)\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-09-10T00:12:14-07:00",
          "tree_id": "3836c2d343958669b65f909a124a63707c2d86ed",
          "url": "https://github.com/pytorch/rl/commit/d4722f653d9eb73e9ac21d5895b00fced44d1026"
        },
        "date": 1789026580409,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1730.936680883657,
            "range": "1701.0-1749.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 300.42 ms; process-tree RSS: 7587.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1687.2101639855978,
            "range": "1684.9-1766.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 220.31 ms; process-tree RSS: 7562.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1655.0414684460789,
            "range": "1635.1-1875.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 290.54 ms; process-tree RSS: 7606.70 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1792.674002400091,
            "range": "1670.6-1809.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.07 ms; process-tree RSS: 7601.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1034.283173244676,
            "range": "1012.3-1041.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 373.46 ms; process-tree RSS: 7466.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1066.9689222909637,
            "range": "1050.3-1071.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.17 ms; process-tree RSS: 7477.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.49288501951237,
            "range": "81.3-82.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3288.23 ms; process-tree RSS: 6716.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.04993271023524,
            "range": "82.2-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3227.38 ms; process-tree RSS: 6721.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 490.16433516601654,
            "range": "483.8-491.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 575.31 ms; process-tree RSS: 2804.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 513.3732946113777,
            "range": "509.6-523.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 532.36 ms; process-tree RSS: 2804.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 490.90900499133375,
            "range": "489.8-517.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 561.00 ms; process-tree RSS: 2802.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 524.3869400861162,
            "range": "518.2-530.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 516.80 ms; process-tree RSS: 2793.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 480.94950215508493,
            "range": "469.7-487.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 583.46 ms; process-tree RSS: 6744.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 496.12849352010767,
            "range": "495.6-500.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 551.91 ms; process-tree RSS: 6756.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.8867994922546,
            "range": "464.4-465.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.57 ms; process-tree RSS: 6701.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1732.6798483902378,
            "range": "1688.1-1912.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.08 ms; process-tree RSS: 6691.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.44941249525675,
            "range": "189.9-192.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2227.544254713941,
            "range": "2186.6-2262.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2626.973200797978,
            "range": "2622.8-2645.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.0566618623775,
            "range": "1231.5-1232.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.1710950581466,
            "range": "1225.5-1226.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.92342097371844,
            "range": "185.9-200.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1968.053195954691,
            "range": "1955.9-1982.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2418.533642769019,
            "range": "2410.7-2431.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2239.203404438493,
            "range": "2238.6-2268.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "573dcadf9633e8540042f4114e71c014e3fd67c7",
          "message": "[Performance] Send at least 64 transitions per worker message by default (#4315)",
          "timestamp": "2026-09-10T08:25:35+01:00",
          "tree_id": "c7f4dfbeba80e4f3e21380cee6d452ed81b4421d",
          "url": "https://github.com/pytorch/rl/commit/573dcadf9633e8540042f4114e71c014e3fd67c7"
        },
        "date": 1789027642764,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1753.1124834363516,
            "range": "1746.0-1804.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 272.09 ms; process-tree RSS: 7609.65 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1677.0841983530506,
            "range": "1670.3-1746.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 277.08 ms; process-tree RSS: 7594.06 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1656.3833299611724,
            "range": "1631.5-1808.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.27 ms; process-tree RSS: 7983.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1729.3120336876423,
            "range": "1674.0-1767.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 262.90 ms; process-tree RSS: 7755.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1021.3310220267572,
            "range": "992.4-1036.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 377.48 ms; process-tree RSS: 7450.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1020.5527381799658,
            "range": "1015.0-1058.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 370.67 ms; process-tree RSS: 7475.81 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.02958542363528,
            "range": "81.0-81.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3290.60 ms; process-tree RSS: 6702.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.24824294016283,
            "range": "82.2-83.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3197.20 ms; process-tree RSS: 6684.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 495.353395522523,
            "range": "490.8-514.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 549.98 ms; process-tree RSS: 2776.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 504.7484767483084,
            "range": "502.0-523.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 534.51 ms; process-tree RSS: 2776.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 493.4094041050241,
            "range": "485.4-498.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 555.13 ms; process-tree RSS: 2774.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 510.41597048345545,
            "range": "496.4-511.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 543.03 ms; process-tree RSS: 2774.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.59927868530497,
            "range": "471.4-477.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 570.72 ms; process-tree RSS: 6742.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 496.4559440375551,
            "range": "491.6-500.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 554.50 ms; process-tree RSS: 6736.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.01549693580563,
            "range": "464.0-464.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.38 ms; process-tree RSS: 6698.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1725.006495853851,
            "range": "1712.6-1774.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 157.10 ms; process-tree RSS: 6713.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.9460426268186,
            "range": "189.5-190.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2240.9957458416056,
            "range": "2207.2-2257.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2639.2495512461683,
            "range": "2629.6-2644.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.960781286855,
            "range": "1229.6-1232.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.7292545209466,
            "range": "1225.9-1228.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.36650756925877,
            "range": "186.4-198.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1913.468664126647,
            "range": "1884.0-1958.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2399.5022972954202,
            "range": "2359.6-2428.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2248.832244718652,
            "range": "2246.1-2249.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1c972b314be763841daa238c9a6b0c7bde25c1ce",
          "message": "[CI] Fix DreamerV3 process benchmark exchange (#4320)",
          "timestamp": "2026-09-10T09:55:37+01:00",
          "tree_id": "4df515cdcff163653fd157b0aa338ebb535a3fbe",
          "url": "https://github.com/pytorch/rl/commit/1c972b314be763841daa238c9a6b0c7bde25c1ce"
        },
        "date": 1789035021163,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1734.516035903343,
            "range": "1706.4-1859.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 276.82 ms; process-tree RSS: 7783.13 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1700.7175623026694,
            "range": "1697.7-1726.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 273.35 ms; process-tree RSS: 7971.15 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1771.997371609075,
            "range": "1643.2-1839.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 291.21 ms; process-tree RSS: 7945.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1716.5534106573316,
            "range": "1676.4-1768.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 270.15 ms; process-tree RSS: 7676.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1024.0894429478033,
            "range": "1016.3-1029.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 376.06 ms; process-tree RSS: 7463.19 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1026.6514207194837,
            "range": "968.9-1035.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.03 ms; process-tree RSS: 7456.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.80349056706888,
            "range": "81.4-82.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3273.42 ms; process-tree RSS: 6713.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.10090321221547,
            "range": "81.9-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3211.71 ms; process-tree RSS: 6708.56 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 487.9855224226572,
            "range": "481.4-503.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 563.32 ms; process-tree RSS: 2778.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 511.6325521522291,
            "range": "511.6-522.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 541.47 ms; process-tree RSS: 2776.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 506.23002835715823,
            "range": "490.2-506.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 562.66 ms; process-tree RSS: 2777.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 513.068901325993,
            "range": "507.7-521.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 540.78 ms; process-tree RSS: 2781.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 482.61851628735303,
            "range": "474.9-483.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 579.48 ms; process-tree RSS: 6715.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 496.3280425135656,
            "range": "494.5-504.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 559.72 ms; process-tree RSS: 6743.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 465.4318226498136,
            "range": "465.3-467.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.59 ms; process-tree RSS: 6696.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1723.086958815633,
            "range": "1702.9-1728.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.44 ms; process-tree RSS: 6705.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.03730099097092,
            "range": "190.0-190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2220.3796203425013,
            "range": "2216.7-2224.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2621.2174221531313,
            "range": "2617.5-2634.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.2095675447977,
            "range": "1222.6-1231.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.3912296917254,
            "range": "1221.5-1228.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 186.66809855682078,
            "range": "185.5-191.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1987.507702781148,
            "range": "1844.8-2010.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2457.308892444739,
            "range": "2452.2-2470.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2252.634303203829,
            "range": "2250.9-2254.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "apaninga@berkeley.edu",
            "name": "theap06",
            "username": "theap06"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2acf99c7aebf0a72378d377b0738ae865606109",
          "message": "[Feature] Transformer Module  (#4193)\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-09-11T00:06:23-07:00",
          "tree_id": "d5dfb75760d34d2d152a1483ffceec5a558458a0",
          "url": "https://github.com/pytorch/rl/commit/a2acf99c7aebf0a72378d377b0738ae865606109"
        },
        "date": 1789114857571,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1608.0404509989937,
            "range": "1588.7-1802.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 285.20 ms; process-tree RSS: 7611.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1706.2689325350277,
            "range": "1649.4-1717.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 295.49 ms; process-tree RSS: 7564.90 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1731.2192947909869,
            "range": "1687.7-1804.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 292.19 ms; process-tree RSS: 7969.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1769.2526309037808,
            "range": "1742.5-1811.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 252.03 ms; process-tree RSS: 7775.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1042.6887899733806,
            "range": "1011.9-1056.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.11 ms; process-tree RSS: 7444.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1050.639478798071,
            "range": "1036.3-1074.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.22 ms; process-tree RSS: 7450.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.41440074441762,
            "range": "81.4-83.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3266.83 ms; process-tree RSS: 6708.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.93865920095254,
            "range": "82.5-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3220.28 ms; process-tree RSS: 6705.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 513.4740896321222,
            "range": "507.8-517.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 546.63 ms; process-tree RSS: 2778.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 516.0143145252757,
            "range": "515.9-529.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 559.21 ms; process-tree RSS: 2780.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 495.6861585702619,
            "range": "487.3-522.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 547.49 ms; process-tree RSS: 2782.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 517.851170600194,
            "range": "512.3-523.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 533.31 ms; process-tree RSS: 2781.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 476.62438635220917,
            "range": "474.5-480.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 579.21 ms; process-tree RSS: 6726.24 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 497.36631398309675,
            "range": "494.8-497.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.50 ms; process-tree RSS: 6748.37 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 467.92851221239613,
            "range": "464.2-473.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.63 ms; process-tree RSS: 6693.49 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1754.2745284463713,
            "range": "1729.2-1769.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 150.71 ms; process-tree RSS: 6696.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.43436397924717,
            "range": "190.8-191.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2298.6218890816886,
            "range": "2248.0-2332.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2669.877917954756,
            "range": "2669.3-2674.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.4365544433347,
            "range": "1225.4-1231.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.2661970705433,
            "range": "1226.4-1230.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.01533518073208,
            "range": "188.9-190.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1976.6447264576382,
            "range": "1935.3-1981.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2437.0740344444653,
            "range": "2421.7-2499.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2276.7663702680898,
            "range": "2274.5-2292.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ed803e488dbfd0632453e2fbc304831d233540d8",
          "message": "[BugFix] TransformedEnv wraps a transformed env without a transform (#4336)",
          "timestamp": "2026-09-11T19:44:22+01:00",
          "tree_id": "6071759645edc36a8512f94b5d1e4da5c36ba4e9",
          "url": "https://github.com/pytorch/rl/commit/ed803e488dbfd0632453e2fbc304831d233540d8"
        },
        "date": 1789157052950,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1795.4255548070946,
            "range": "1716.5-1821.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 281.45 ms; process-tree RSS: 7630.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1724.1298288542869,
            "range": "1665.9-1748.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 284.52 ms; process-tree RSS: 7618.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1813.425520520349,
            "range": "1694.0-1843.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 280.23 ms; process-tree RSS: 7623.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1767.8236111176368,
            "range": "1765.0-1793.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 284.46 ms; process-tree RSS: 7604.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1017.102324287105,
            "range": "1014.0-1027.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 372.40 ms; process-tree RSS: 7476.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1039.5380827959111,
            "range": "1019.4-1047.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 372.87 ms; process-tree RSS: 7476.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.2262412090976,
            "range": "81.8-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3271.89 ms; process-tree RSS: 6732.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.81416707410162,
            "range": "82.6-83.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3187.42 ms; process-tree RSS: 6728.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 504.2007346338475,
            "range": "483.3-514.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 546.35 ms; process-tree RSS: 2801.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 516.09746290922,
            "range": "507.5-527.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 526.07 ms; process-tree RSS: 2801.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 494.1378212912559,
            "range": "487.4-496.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 544.58 ms; process-tree RSS: 2799.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 510.17505586871295,
            "range": "509.3-540.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 533.20 ms; process-tree RSS: 2798.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 478.13429220023477,
            "range": "474.0-485.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 573.63 ms; process-tree RSS: 6748.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 503.3953512555405,
            "range": "488.6-513.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 555.46 ms; process-tree RSS: 6756.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 468.7526722106602,
            "range": "463.3-469.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.60 ms; process-tree RSS: 6701.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1709.7373007170202,
            "range": "1706.1-1765.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.53 ms; process-tree RSS: 6702.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.891509611246,
            "range": "189.8-191.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2251.0716200071256,
            "range": "2235.0-2278.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2645.660860084733,
            "range": "2640.0-2646.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.1890875132515,
            "range": "1228.9-1229.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.9854003721055,
            "range": "1225.7-1228.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.52633636119322,
            "range": "186.9-195.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1889.9293099445417,
            "range": "1849.6-1969.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2459.2591066940536,
            "range": "2424.9-2502.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2269.799007324999,
            "range": "2262.2-2277.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "44c871df8915eb8d2959c14828e88e2658478c9a",
          "message": "[BugFix] ParallelEnv without buffers keeps the state of workers it does not reset (#4339)",
          "timestamp": "2026-09-12T06:28:14+01:00",
          "tree_id": "0af9eb3dc30224936ea1921eeddd075bc233704f",
          "url": "https://github.com/pytorch/rl/commit/44c871df8915eb8d2959c14828e88e2658478c9a"
        },
        "date": 1789194979488,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1708.4576371127537,
            "range": "1684.7-1790.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 295.66 ms; process-tree RSS: 7605.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1711.323014012122,
            "range": "1697.3-1722.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 285.48 ms; process-tree RSS: 7585.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1744.9584240312327,
            "range": "1692.0-1745.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 280.74 ms; process-tree RSS: 7608.24 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1779.4781432883065,
            "range": "1748.5-1787.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 264.83 ms; process-tree RSS: 7595.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1025.8587812237172,
            "range": "1017.7-1049.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 371.74 ms; process-tree RSS: 7461.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1051.530943411311,
            "range": "1040.9-1051.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.55 ms; process-tree RSS: 7450.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.17508091773752,
            "range": "81.5-83.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3233.16 ms; process-tree RSS: 6691.19 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 83.06617105089725,
            "range": "82.5-83.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3230.04 ms; process-tree RSS: 6704.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 490.9309559819322,
            "range": "489.0-498.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 559.71 ms; process-tree RSS: 2774.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 524.8197973311023,
            "range": "517.6-528.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 536.42 ms; process-tree RSS: 2773.65 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 489.16942936133995,
            "range": "485.4-493.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 549.43 ms; process-tree RSS: 2773.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 516.6069885031611,
            "range": "515.8-532.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 525.42 ms; process-tree RSS: 2774.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 485.21902010167776,
            "range": "479.9-489.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 567.63 ms; process-tree RSS: 6751.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 505.1748590624496,
            "range": "503.5-512.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 546.02 ms; process-tree RSS: 6741.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.87824673898126,
            "range": "465.8-467.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.79 ms; process-tree RSS: 6697.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1730.6466391821134,
            "range": "1688.1-1735.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 152.19 ms; process-tree RSS: 6709.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.8901796311214,
            "range": "190.7-192.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2274.1154290047575,
            "range": "2264.3-2298.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2682.13005444326,
            "range": "2678.9-2683.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.0873809147704,
            "range": "1228.3-1232.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.3493997294715,
            "range": "1226.3-1230.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.62564316500138,
            "range": "191.2-193.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1991.6310186434027,
            "range": "1976.5-2035.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2487.2607193857457,
            "range": "2438.7-2490.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2282.4746542948214,
            "range": "2267.6-2297.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46c5daee9e28e47560a6adfa839a5ce0d9176e25",
          "message": "[Feature] Shared buffers for ParallelEnv workers that report their own metadata (#4341)",
          "timestamp": "2026-09-12T08:49:35+01:00",
          "tree_id": "4eb265147caf5bbc95988160dfdc1572bf29ae97",
          "url": "https://github.com/pytorch/rl/commit/46c5daee9e28e47560a6adfa839a5ce0d9176e25"
        },
        "date": 1789203707830,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1782.6317323373955,
            "range": "1699.9-1819.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 287.74 ms; process-tree RSS: 7607.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1689.2339719685442,
            "range": "1643.0-1735.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 295.64 ms; process-tree RSS: 7611.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1734.4434607088263,
            "range": "1644.3-1793.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 290.05 ms; process-tree RSS: 7613.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1728.4115123114118,
            "range": "1649.5-1815.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 274.97 ms; process-tree RSS: 7609.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1031.707703577719,
            "range": "1010.4-1045.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.92 ms; process-tree RSS: 7502.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1049.4584182208487,
            "range": "1007.9-1058.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.31 ms; process-tree RSS: 7498.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.83702946220937,
            "range": "81.4-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3266.17 ms; process-tree RSS: 6724.81 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.83126509487063,
            "range": "82.3-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3232.57 ms; process-tree RSS: 6728.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 488.83212130679993,
            "range": "486.9-506.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 566.43 ms; process-tree RSS: 2813.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 524.6658588110168,
            "range": "494.6-526.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 536.36 ms; process-tree RSS: 2815.35 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 501.9743056314398,
            "range": "491.9-509.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 550.79 ms; process-tree RSS: 2807.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 501.8915104919581,
            "range": "499.2-508.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 549.55 ms; process-tree RSS: 2815.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 480.7839938751815,
            "range": "477.2-482.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 577.10 ms; process-tree RSS: 6751.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 491.17400697026596,
            "range": "487.0-505.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.13 ms; process-tree RSS: 6759.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 466.0887950615578,
            "range": "464.1-468.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.80 ms; process-tree RSS: 6692.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1841.415760567385,
            "range": "1727.5-1937.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 148.62 ms; process-tree RSS: 6693.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.22931157624728,
            "range": "189.7-190.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2248.5688073641318,
            "range": "2225.7-2263.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2639.278267810452,
            "range": "2635.3-2639.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.1270921881132,
            "range": "1229.3-1234.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.294192178009,
            "range": "1224.6-1225.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 186.36568619508927,
            "range": "185.5-189.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1958.8595122031654,
            "range": "1825.2-1989.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2400.3200864338723,
            "range": "2395.7-2421.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2252.004064519869,
            "range": "2240.5-2259.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7ef78484cda129fb09e40a405457b3376cd98c66",
          "message": "[Doc] Document NoisyLinear reset_noise caller contract (#4357)",
          "timestamp": "2026-09-12T13:54:44+01:00",
          "tree_id": "280a1b58db5cd35b7dc2e8b32895bfd5a02a24d2",
          "url": "https://github.com/pytorch/rl/commit/7ef78484cda129fb09e40a405457b3376cd98c66"
        },
        "date": 1789221724588,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1721.8034164274538,
            "range": "1676.7-1782.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 281.19 ms; process-tree RSS: 7605.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1726.2280353371555,
            "range": "1710.5-1768.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 269.32 ms; process-tree RSS: 7592.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1664.2389878388817,
            "range": "1663.4-1748.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 297.13 ms; process-tree RSS: 7626.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1730.665101595547,
            "range": "1661.2-1753.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 286.37 ms; process-tree RSS: 7603.17 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1002.9298204913014,
            "range": "975.5-1019.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.67 ms; process-tree RSS: 7479.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1031.2914844928475,
            "range": "1023.4-1043.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 363.37 ms; process-tree RSS: 7477.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.24169997028359,
            "range": "80.9-81.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3281.45 ms; process-tree RSS: 6720.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.97333471527281,
            "range": "82.5-83.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3187.70 ms; process-tree RSS: 6711.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 484.7806447093419,
            "range": "480.9-493.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 569.89 ms; process-tree RSS: 2787.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 509.68801540645063,
            "range": "509.5-527.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 545.78 ms; process-tree RSS: 2786.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 484.93911105942226,
            "range": "480.5-507.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 569.07 ms; process-tree RSS: 2791.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 512.4837080040672,
            "range": "499.2-537.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 529.64 ms; process-tree RSS: 2790.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 477.1410901711804,
            "range": "468.5-483.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 570.35 ms; process-tree RSS: 6734.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 489.66104985020445,
            "range": "489.3-495.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 559.10 ms; process-tree RSS: 6736.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 464.42692678183187,
            "range": "462.5-468.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.58 ms; process-tree RSS: 6701.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1722.8897381476816,
            "range": "1691.6-1837.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 155.98 ms; process-tree RSS: 6693.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.25908379066263,
            "range": "188.9-189.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2245.833304265128,
            "range": "2242.7-2276.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2613.0877554143267,
            "range": "2607.0-2614.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1220.9741592882485,
            "range": "1220.8-1224.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.0640889896838,
            "range": "1224.4-1226.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.77786344531998,
            "range": "186.0-189.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1942.7237974178079,
            "range": "1936.8-1943.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2443.942023160549,
            "range": "2423.5-2455.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2234.587046322596,
            "range": "2228.7-2235.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "85978465+yupengtang@users.noreply.github.com",
            "name": "David",
            "username": "yupengtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "021cf60155dd76bdf7089243e8a333f487537323",
          "message": "[BugFix] Only forward the applicable SACLossConfig fields to each SAC variant (#4338)",
          "timestamp": "2026-09-12T14:05:16+01:00",
          "tree_id": "8dc0180cb8f6a2875af83801f7acd4f964077336",
          "url": "https://github.com/pytorch/rl/commit/021cf60155dd76bdf7089243e8a333f487537323"
        },
        "date": 1789222352470,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1816.8043308148617,
            "range": "1808.1-1827.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 289.50 ms; process-tree RSS: 7575.08 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1685.7541419937402,
            "range": "1672.5-1755.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 256.53 ms; process-tree RSS: 7793.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1818.9169551727732,
            "range": "1673.3-1826.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.55 ms; process-tree RSS: 7953.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1676.1314596521158,
            "range": "1662.6-1690.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 293.01 ms; process-tree RSS: 7757.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 999.8807105208773,
            "range": "989.0-1009.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 370.40 ms; process-tree RSS: 7456.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1035.4943015374763,
            "range": "1023.7-1043.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.52 ms; process-tree RSS: 7459.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.77968450434508,
            "range": "81.4-82.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3271.51 ms; process-tree RSS: 6708.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.88866574004591,
            "range": "82.4-83.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3173.37 ms; process-tree RSS: 6693.37 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 492.6670152054585,
            "range": "482.8-527.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 559.67 ms; process-tree RSS: 2770.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 506.4406375414292,
            "range": "500.3-522.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 540.20 ms; process-tree RSS: 2768.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 503.45433663774077,
            "range": "488.4-505.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 560.09 ms; process-tree RSS: 2767.60 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 500.2028781841898,
            "range": "495.3-514.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 543.81 ms; process-tree RSS: 2764.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 479.7988480063977,
            "range": "479.0-481.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 590.65 ms; process-tree RSS: 6724.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 503.6670111247239,
            "range": "491.2-504.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 548.96 ms; process-tree RSS: 6725.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 467.01700029034276,
            "range": "464.3-473.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.33 ms; process-tree RSS: 6707.27 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1735.1327425771822,
            "range": "1720.7-1743.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.62 ms; process-tree RSS: 6715.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.3895099854066,
            "range": "189.6-190.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2260.1520817055975,
            "range": "2259.1-2291.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2642.525070301535,
            "range": "2634.8-2648.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.6262738988924,
            "range": "1220.0-1232.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.9764021203723,
            "range": "1223.3-1228.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.50685027871327,
            "range": "190.3-200.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1954.1890454391735,
            "range": "1950.3-1997.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2441.5320956279274,
            "range": "2429.6-2448.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2248.864665395075,
            "range": "2240.2-2249.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e3ff3568640a0aafbf7efa76cdec116cc479a6aa",
          "message": "[BugFix] Import ale_py for classic Atari environment ids (#4358)",
          "timestamp": "2026-09-12T15:41:00+01:00",
          "tree_id": "f0bdebf9bd2d7e5142a93e3e280d5c3420de4bc6",
          "url": "https://github.com/pytorch/rl/commit/e3ff3568640a0aafbf7efa76cdec116cc479a6aa"
        },
        "date": 1789228361185,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1677.995759914337,
            "range": "1660.7-1762.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 289.46 ms; process-tree RSS: 7631.44 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1672.4969955491163,
            "range": "1663.0-1703.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 278.86 ms; process-tree RSS: 8023.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1809.4932593464387,
            "range": "1782.6-1814.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 273.85 ms; process-tree RSS: 7972.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1728.3401909285324,
            "range": "1681.5-1735.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 289.07 ms; process-tree RSS: 7773.24 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1027.7636138268106,
            "range": "1003.7-1037.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 381.83 ms; process-tree RSS: 7481.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1036.929147049599,
            "range": "1000.2-1071.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.53 ms; process-tree RSS: 7468.33 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.19327879171016,
            "range": "81.0-82.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3224.87 ms; process-tree RSS: 6708.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.40664146438122,
            "range": "82.2-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3226.53 ms; process-tree RSS: 6727.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 496.9660161622126,
            "range": "483.9-513.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 548.69 ms; process-tree RSS: 2790.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 510.5260243230906,
            "range": "504.7-515.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 528.42 ms; process-tree RSS: 2789.42 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 485.75779952929673,
            "range": "480.5-490.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 565.69 ms; process-tree RSS: 2787.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 508.83620404574924,
            "range": "507.2-512.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 538.26 ms; process-tree RSS: 2789.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 480.35829857939234,
            "range": "479.7-484.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 570.77 ms; process-tree RSS: 6745.11 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 497.2312655416938,
            "range": "473.2-502.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 560.55 ms; process-tree RSS: 6751.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 465.465124842113,
            "range": "464.3-467.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 553.87 ms; process-tree RSS: 6706.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1706.4594399206696,
            "range": "1689.0-1708.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.32 ms; process-tree RSS: 6704.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.01225385685137,
            "range": "190.0-190.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2312.506083030846,
            "range": "2309.4-2326.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2616.312640070243,
            "range": "2615.2-2633.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.2659273389936,
            "range": "1221.8-1234.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.4564968464092,
            "range": "1224.7-1228.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 196.4044364490127,
            "range": "189.2-197.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1921.4434904267816,
            "range": "1815.4-2035.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2432.5264350564735,
            "range": "2413.6-2448.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2245.701127415929,
            "range": "2228.9-2248.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d282ef6fe84b03487950c0035d2d930fecdf9096",
          "message": "[BugFix] Stop caching Tokenizer parent.device (#4352)",
          "timestamp": "2026-09-12T15:41:52+01:00",
          "tree_id": "cffecb57437f99c9187fa7dc259132b94e546c26",
          "url": "https://github.com/pytorch/rl/commit/d282ef6fe84b03487950c0035d2d930fecdf9096"
        },
        "date": 1789228360294,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1717.0941386501215,
            "range": "1665.0-1801.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 291.51 ms; process-tree RSS: 7598.98 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1757.7323418153917,
            "range": "1745.3-1778.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 288.96 ms; process-tree RSS: 7578.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1779.2086754661175,
            "range": "1778.1-1850.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 278.71 ms; process-tree RSS: 7632.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1698.1295186268098,
            "range": "1656.3-1767.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 294.13 ms; process-tree RSS: 7616.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1026.863225322172,
            "range": "992.3-1044.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 377.19 ms; process-tree RSS: 7462.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1060.945768584176,
            "range": "1057.8-1061.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.47 ms; process-tree RSS: 7460.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 82.2216718666091,
            "range": "81.7-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3276.92 ms; process-tree RSS: 6726.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.71934908591406,
            "range": "82.5-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3206.69 ms; process-tree RSS: 6716.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 487.5818736804344,
            "range": "487.6-518.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 557.75 ms; process-tree RSS: 2778.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 510.40239329676643,
            "range": "500.2-515.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 527.44 ms; process-tree RSS: 2778.33 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 490.14626181762355,
            "range": "484.6-509.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 564.95 ms; process-tree RSS: 2778.84 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 505.4008993404615,
            "range": "498.5-513.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 540.41 ms; process-tree RSS: 2779.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 472.4291006606137,
            "range": "471.7-485.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 571.19 ms; process-tree RSS: 6742.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 494.4568241711265,
            "range": "494.1-514.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 562.18 ms; process-tree RSS: 6737.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 469.04091995388967,
            "range": "465.8-472.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 550.79 ms; process-tree RSS: 6705.27 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1797.6704746712549,
            "range": "1726.5-1819.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 150.81 ms; process-tree RSS: 6709.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.18696160246327,
            "range": "190.0-190.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2238.3841769147557,
            "range": "2207.8-2320.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2634.1141061909407,
            "range": "2624.7-2644.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.017539964069,
            "range": "1229.5-1234.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.0811729932075,
            "range": "1225.0-1227.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 185.8784099347555,
            "range": "185.7-190.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1985.8229963082656,
            "range": "1806.9-2025.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2442.686307721979,
            "range": "2411.3-2501.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2243.405279596145,
            "range": "2236.7-2266.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "08280d5c2a71ceec80aa49b68cea440a5d2635a0",
          "message": "[BugFix] Accept History on ChatEnv.reset (#4355)",
          "timestamp": "2026-09-12T15:41:12+01:00",
          "tree_id": "2bf1ac5306a888f1c5cb7bf0db08003963129165",
          "url": "https://github.com/pytorch/rl/commit/08280d5c2a71ceec80aa49b68cea440a5d2635a0"
        },
        "date": 1789228377549,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1757.5833827163974,
            "range": "1599.4-1798.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 295.02 ms; process-tree RSS: 7586.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1753.070249167999,
            "range": "1726.8-1765.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 259.24 ms; process-tree RSS: 7585.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1807.1695009679793,
            "range": "1718.4-1829.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 286.89 ms; process-tree RSS: 7914.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1728.73047971033,
            "range": "1702.2-1775.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 280.58 ms; process-tree RSS: 7629.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1022.7039599980291,
            "range": "1004.3-1056.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.82 ms; process-tree RSS: 7477.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1017.621876803303,
            "range": "996.3-1064.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 359.46 ms; process-tree RSS: 7482.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.8435353854468,
            "range": "81.7-82.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3219.90 ms; process-tree RSS: 6707.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.37960150404929,
            "range": "82.1-82.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3219.77 ms; process-tree RSS: 6724.57 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 485.01578222412013,
            "range": "482.3-508.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 563.96 ms; process-tree RSS: 2795.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 522.0213016552852,
            "range": "508.3-524.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 536.04 ms; process-tree RSS: 2793.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 488.49348002527756,
            "range": "484.4-496.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 563.71 ms; process-tree RSS: 2792.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 531.2640408020782,
            "range": "504.0-542.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 521.97 ms; process-tree RSS: 2793.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 482.6857744404639,
            "range": "474.3-483.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 565.82 ms; process-tree RSS: 6756.35 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 490.85756807758946,
            "range": "481.0-497.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 556.86 ms; process-tree RSS: 6757.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 467.4586609986426,
            "range": "467.1-479.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 552.60 ms; process-tree RSS: 6705.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1750.2434372404089,
            "range": "1694.8-1792.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 154.89 ms; process-tree RSS: 6696.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.4451884604114,
            "range": "189.0-189.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2206.014355172729,
            "range": "2200.5-2243.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2631.11756191701,
            "range": "2623.2-2640.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1232.982915254084,
            "range": "1227.6-1233.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.0759568720543,
            "range": "1216.0-1231.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.7599118722897,
            "range": "187.1-203.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1952.4097715800376,
            "range": "1952.1-1980.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2422.11822254753,
            "range": "2421.9-2427.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2249.5758193026145,
            "range": "2238.0-2260.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8975d8500cfbb51e87862b9ea313357eedebe9a9",
          "message": "[BugFix] Count dropped dialog turns in LLMCollector last-step mode (#4354)",
          "timestamp": "2026-09-12T15:41:22+01:00",
          "tree_id": "3911aee5d940561b57cb6ac0d903eba65dea05c5",
          "url": "https://github.com/pytorch/rl/commit/8975d8500cfbb51e87862b9ea313357eedebe9a9"
        },
        "date": 1789228398165,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 1785.7587210364009,
            "range": "1681.2-1823.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 285.46 ms; process-tree RSS: 7580.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 1732.064995136739,
            "range": "1696.6-1736.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 269.75 ms; process-tree RSS: 7577.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 1718.688811835264,
            "range": "1676.5-1833.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 292.48 ms; process-tree RSS: 7909.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1741.9374361973776,
            "range": "1701.0-1749.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 275.94 ms; process-tree RSS: 7660.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1019.0530092049348,
            "range": "1010.8-1027.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 377.40 ms; process-tree RSS: 7468.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1054.238787002374,
            "range": "1029.9-1058.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.26 ms; process-tree RSS: 7462.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.92946108307896,
            "range": "81.7-82.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3233.12 ms; process-tree RSS: 6719.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.98474313576531,
            "range": "82.1-83.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3230.68 ms; process-tree RSS: 6707.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 490.74891049948224,
            "range": "486.0-499.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 570.12 ms; process-tree RSS: 2790.83 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 514.4772938302375,
            "range": "511.4-532.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 530.98 ms; process-tree RSS: 2789.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 492.6828129479861,
            "range": "482.6-505.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 550.71 ms; process-tree RSS: 2789.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 509.33052993299674,
            "range": "509.0-514.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 539.43 ms; process-tree RSS: 2787.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 475.3715090550356,
            "range": "473.0-477.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 573.60 ms; process-tree RSS: 6748.87 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 492.36151381238557,
            "range": "491.2-494.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 572.28 ms; process-tree RSS: 6742.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 468.0172329539666,
            "range": "466.3-468.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 550.05 ms; process-tree RSS: 6700.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 1766.4893983631944,
            "range": "1698.9-1813.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 153.92 ms; process-tree RSS: 6706.84 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.49415976353757,
            "range": "188.8-189.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2227.941810102916,
            "range": "2226.9-2231.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2616.7799578633203,
            "range": "2606.1-2646.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.5595386911473,
            "range": "1229.4-1232.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.017658233056,
            "range": "1222.5-1227.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 188.04752907467733,
            "range": "187.0-189.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1888.5918279033172,
            "range": "1777.5-1965.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2402.9183753419943,
            "range": "2395.6-2432.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2237.3140391246634,
            "range": "2230.6-2245.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      }
    ],
    "Async GPU (v1)": [
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bedcc833257604d97868c447087c80921f1379b2",
          "message": "[CI] Track async collection across merges and recover benchmark publication (#4287)",
          "timestamp": "2026-09-07T22:01:57+01:00",
          "tree_id": "fb1cd0d1f31013091e551509bb157b87527ebbab",
          "url": "https://github.com/pytorch/rl/commit/bedcc833257604d97868c447087c80921f1379b2"
        },
        "date": 1788817260624,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 80.92066681357122,
            "range": "80.5-81.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3325.17 ms; process-tree RSS: 23736.75 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 82.0483854605767,
            "range": "81.5-82.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3289.04 ms; process-tree RSS: 23731.68 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 828.1077617472397,
            "range": "824.5-829.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 423.79 ms; process-tree RSS: 23815.00 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 827.1468116696517,
            "range": "814.6-830.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 419.16 ms; process-tree RSS: 23811.11 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 834.7290875904632,
            "range": "830.9-838.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 441.56 ms; process-tree RSS: 23806.52 MiB; CUDA peak: 31.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 844.4307448750315,
            "range": "820.6-850.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 420.31 ms; process-tree RSS: 23801.78 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1678.0560075116277,
            "range": "1669.7-1679.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.36 ms; process-tree RSS: 23686.89 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4855.4096696185525,
            "range": "4716.4-4935.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.95 ms; process-tree RSS: 23682.91 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.06369811754638,
            "range": "191.8-193.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2180.481860810571,
            "range": "2158.7-2230.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2592.878128982831,
            "range": "2583.0-2594.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.1368453491702,
            "range": "189.4-195.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2136.133332402003,
            "range": "2124.0-2208.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2356.4601858929923,
            "range": "2337.9-2372.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2201.843643246694,
            "range": "2200.2-2209.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b3af48309e211373536cbe4becfbc18505f1683d",
          "message": "[CI] Stop failing benchmark publication on performance alerts (#4292)",
          "timestamp": "2026-09-08T06:46:34+01:00",
          "tree_id": "f027b3fec5afdd468c94389b80cb07904383595f",
          "url": "https://github.com/pytorch/rl/commit/b3af48309e211373536cbe4becfbc18505f1683d"
        },
        "date": 1788848931860,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 80.89078035786567,
            "range": "80.8-81.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3329.85 ms; process-tree RSS: 23746.86 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 81.50947464226887,
            "range": "81.3-82.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3268.17 ms; process-tree RSS: 23734.63 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 837.8362799196549,
            "range": "826.3-839.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 420.40 ms; process-tree RSS: 23822.40 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 829.7796809750727,
            "range": "829.4-857.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 410.88 ms; process-tree RSS: 23818.83 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 828.4419680328933,
            "range": "810.3-832.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 406.93 ms; process-tree RSS: 23815.72 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 830.2416810102169,
            "range": "823.4-831.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 419.02 ms; process-tree RSS: 23811.60 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1681.3105035142225,
            "range": "1665.1-1694.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 254.72 ms; process-tree RSS: 23683.45 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4731.073624105957,
            "range": "4711.7-4884.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.80 ms; process-tree RSS: 23678.91 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.9055184936447,
            "range": "192.8-193.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2233.8633943581635,
            "range": "2200.2-2304.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2563.097929963449,
            "range": "2558.8-2576.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 197.1140610780093,
            "range": "190.8-205.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2099.6734206103074,
            "range": "2049.5-2194.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2382.6488049628156,
            "range": "2349.0-2442.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2195.050940401548,
            "range": "2193.7-2203.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2446fc19ee3c84b5815d24217fbb475b1d131d8",
          "message": "[CI] Run the full benchmark suite after merging labelled pull requests (#4293)",
          "timestamp": "2026-09-08T06:54:32+01:00",
          "tree_id": "0835048819db5479b3473148c9d55144a4e0ce23",
          "url": "https://github.com/pytorch/rl/commit/a2446fc19ee3c84b5815d24217fbb475b1d131d8"
        },
        "date": 1788849209994,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 81.06470672708355,
            "range": "80.3-81.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3271.06 ms; process-tree RSS: 23740.36 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 81.93463720343085,
            "range": "81.9-82.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3259.72 ms; process-tree RSS: 23738.77 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 812.631417687403,
            "range": "808.8-836.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 426.24 ms; process-tree RSS: 23816.43 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 837.855094362132,
            "range": "834.9-843.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 403.90 ms; process-tree RSS: 23808.47 MiB; CUDA peak: 31.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 822.9599155610836,
            "range": "799.8-829.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 409.28 ms; process-tree RSS: 23809.94 MiB; CUDA peak: 31.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 825.0723734108191,
            "range": "816.3-848.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 409.06 ms; process-tree RSS: 23807.36 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1683.9414003526347,
            "range": "1681.6-1695.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 254.86 ms; process-tree RSS: 23685.61 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4727.199218985904,
            "range": "4700.8-4942.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.89 ms; process-tree RSS: 23682.45 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.1080750322137,
            "range": "191.5-192.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2249.222710693598,
            "range": "2205.6-2268.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2589.790908678325,
            "range": "2584.3-2606.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.4511441646031,
            "range": "190.7-192.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2179.4465121481926,
            "range": "2121.3-2197.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2366.5590215878624,
            "range": "2324.5-2387.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2220.12745391795,
            "range": "2192.4-2251.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "740b45173e831053c8649042bb9f376596f60753",
          "message": "[Test] Track direct process-slot collection in continuous benchmarks (#4289)",
          "timestamp": "2026-09-08T06:55:16+01:00",
          "tree_id": "996181ff290fb3a8e3d71e6173ece1a40615c100",
          "url": "https://github.com/pytorch/rl/commit/740b45173e831053c8649042bb9f376596f60753"
        },
        "date": 1788849400467,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 80.42705949239814,
            "range": "80.3-80.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3336.93 ms; process-tree RSS: 23736.66 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 81.77726563998552,
            "range": "81.2-82.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3277.22 ms; process-tree RSS: 23737.31 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 826.0476442016837,
            "range": "802.3-828.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 414.55 ms; process-tree RSS: 23812.18 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 836.8550267289158,
            "range": "821.9-846.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 432.37 ms; process-tree RSS: 23807.82 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 834.358149619186,
            "range": "828.9-838.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 418.97 ms; process-tree RSS: 23807.12 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 817.5990945331608,
            "range": "801.5-848.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 422.36 ms; process-tree RSS: 23802.76 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1665.9467845968597,
            "range": "1664.2-1671.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 254.95 ms; process-tree RSS: 23684.18 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4705.866046770779,
            "range": "4701.5-4706.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.23 ms; process-tree RSS: 23682.95 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.9299921918354,
            "range": "190.1-192.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2241.735606349012,
            "range": "2167.4-2260.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2563.303614535559,
            "range": "2562.6-2567.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.2055760848926,
            "range": "190.9-196.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2208.0482630354813,
            "range": "2038.4-2236.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2312.99980736709,
            "range": "2308.6-2387.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2236.7414507481035,
            "range": "2196.5-2237.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5cca10a441a48b4116614ba7ef45ea58965367b1",
          "message": "[Feature] CUDA-graph static inference batches (#4269)",
          "timestamp": "2026-09-08T06:59:05+01:00",
          "tree_id": "d0da341c3beefd661bac546748c97cdc04bb7c83",
          "url": "https://github.com/pytorch/rl/commit/5cca10a441a48b4116614ba7ef45ea58965367b1"
        },
        "date": 1788849737588,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 80.59274327527706,
            "range": "79.6-80.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3325.21 ms; process-tree RSS: 23736.20 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 80.97380203926573,
            "range": "80.6-81.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3281.57 ms; process-tree RSS: 23733.79 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 746.3525582838927,
            "range": "730.4-751.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 471.99 ms; process-tree RSS: 23807.43 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 756.606357918514,
            "range": "737.3-763.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 442.15 ms; process-tree RSS: 23806.27 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 800.1031876830641,
            "range": "790.5-806.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 406.90 ms; process-tree RSS: 23805.77 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 732.379442676643,
            "range": "730.7-748.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 441.74 ms; process-tree RSS: 23814.09 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 726.5355300401951,
            "range": "725.9-729.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 466.49 ms; process-tree RSS: 23811.86 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 811.2677233962456,
            "range": "811.2-830.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 423.11 ms; process-tree RSS: 23802.33 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1651.6860702547408,
            "range": "1650.3-1698.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.94 ms; process-tree RSS: 23686.22 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4718.698694231091,
            "range": "4711.5-4735.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.67 ms; process-tree RSS: 23679.75 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.9916936839477,
            "range": "190.6-191.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2237.8173078248847,
            "range": "2216.5-2256.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2545.231146738669,
            "range": "2516.3-2555.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 193.46720158173935,
            "range": "191.5-196.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2162.169649405634,
            "range": "2106.0-2185.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2346.7457649554194,
            "range": "2312.5-2348.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2173.0531180490275,
            "range": "2146.8-2186.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "402147d710ac41981ff3ada88572fcbe81ab09fd",
          "message": "[Performance] Let AsyncEnvPool workers host multiple environments (#4271)",
          "timestamp": "2026-09-08T06:20:52Z",
          "tree_id": "dd9532ec1be2f6dce5cacaff0d74e03a98333bf2",
          "url": "https://github.com/pytorch/rl/commit/402147d710ac41981ff3ada88572fcbe81ab09fd"
        },
        "date": 1788851358312,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.18895965236585,
            "range": "76.3-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3497.77 ms; process-tree RSS: 23742.52 MiB; CUDA peak: 27.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 78.19052883163863,
            "range": "77.4-78.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3435.13 ms; process-tree RSS: 23740.97 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 851.1963778721204,
            "range": "851.1-853.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 336.42 ms; process-tree RSS: 7433.69 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 849.133373898151,
            "range": "839.4-862.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 345.07 ms; process-tree RSS: 7430.60 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 760.2904130589831,
            "range": "754.1-762.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 437.44 ms; process-tree RSS: 7415.86 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 758.3335323218615,
            "range": "738.3-773.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 404.48 ms; process-tree RSS: 7409.29 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 830.331258101265,
            "range": "826.8-838.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 414.62 ms; process-tree RSS: 23820.23 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 759.4370683410535,
            "range": "748.7-769.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 428.79 ms; process-tree RSS: 23857.80 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 767.357536616805,
            "range": "766.8-770.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 443.44 ms; process-tree RSS: 23856.65 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 831.8121156342181,
            "range": "817.9-832.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 382.79 ms; process-tree RSS: 23808.98 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1691.5865643297786,
            "range": "1678.3-1692.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.14 ms; process-tree RSS: 23692.77 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4869.014129007405,
            "range": "4850.2-4896.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.47 ms; process-tree RSS: 23681.70 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.69519869757085,
            "range": "190.7-190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2249.3786316682954,
            "range": "2211.0-2251.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2631.2080918682045,
            "range": "2619.1-2667.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1227.0222646208022,
            "range": "1223.9-1229.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.7077253409857,
            "range": "1224.3-1230.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.61147647265943,
            "range": "189.0-191.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1971.7037938810993,
            "range": "1953.2-1972.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2425.25928769153,
            "range": "2413.1-2432.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2245.687207062395,
            "range": "2243.9-2250.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "303c50429547c8ae28ce2cbb7fd0960f207ab4f2",
          "message": "[Performance] Run inference from environment processes (#4272)",
          "timestamp": "2026-09-08T07:23:53+01:00",
          "tree_id": "d0869de7c9b96cb590ca59244d6b277acb1528e4",
          "url": "https://github.com/pytorch/rl/commit/303c50429547c8ae28ce2cbb7fd0960f207ab4f2"
        },
        "date": 1788852385858,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 688.2198396011311,
            "range": "687.6-699.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 537.24 ms; process-tree RSS: 26090.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 662.7170134895288,
            "range": "659.2-664.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 537.92 ms; process-tree RSS: 26303.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 667.320223845162,
            "range": "666.0-670.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 541.58 ms; process-tree RSS: 26276.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 696.5050187574853,
            "range": "682.1-755.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 518.06 ms; process-tree RSS: 26080.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.08793359795905,
            "range": "76.1-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3497.15 ms; process-tree RSS: 23743.32 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.4133762922315,
            "range": "77.0-78.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3496.95 ms; process-tree RSS: 23740.35 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 836.6843351220435,
            "range": "832.5-860.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 347.89 ms; process-tree RSS: 7425.03 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 837.179362439834,
            "range": "816.7-840.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 345.55 ms; process-tree RSS: 7431.81 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 736.8457296533568,
            "range": "717.4-740.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 467.93 ms; process-tree RSS: 7425.71 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 756.66349965798,
            "range": "747.7-765.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 439.22 ms; process-tree RSS: 7414.62 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 822.4780857662445,
            "range": "808.0-843.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 423.37 ms; process-tree RSS: 23822.34 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 758.7096037471979,
            "range": "742.0-758.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 467.91 ms; process-tree RSS: 23858.73 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 755.2751822630545,
            "range": "754.2-758.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 422.40 ms; process-tree RSS: 23856.76 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 839.4132964702728,
            "range": "811.2-845.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 395.21 ms; process-tree RSS: 23809.08 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1682.0055166758596,
            "range": "1678.4-1701.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.53 ms; process-tree RSS: 23693.70 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4848.1719210307765,
            "range": "4777.7-4895.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.34 ms; process-tree RSS: 23686.88 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.06356025955878,
            "range": "188.8-190.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2234.795210142484,
            "range": "2154.4-2257.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2612.6763661312716,
            "range": "2610.0-2622.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1228.1363838733444,
            "range": "1219.5-1229.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.447501008935,
            "range": "1224.2-1228.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.55593377218904,
            "range": "186.5-193.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1899.2056789239816,
            "range": "1799.9-1959.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2411.54801607475,
            "range": "2385.5-2418.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2233.7689486368777,
            "range": "2217.8-2264.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d670c59b26285501dde5cab18fc666248e59e1b1",
          "message": "[Feature] Add native stream replay (#4274)",
          "timestamp": "2026-09-08T07:34:57+01:00",
          "tree_id": "c53f65850a9d2e0cce9fce72655c2eef106197cf",
          "url": "https://github.com/pytorch/rl/commit/d670c59b26285501dde5cab18fc666248e59e1b1"
        },
        "date": 1788852702309,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 702.7981442786892,
            "range": "687.0-767.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 501.05 ms; process-tree RSS: 26061.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 670.4603627427705,
            "range": "665.0-675.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 535.39 ms; process-tree RSS: 26302.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 676.833954852744,
            "range": "672.9-683.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 531.75 ms; process-tree RSS: 26278.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 694.271877937224,
            "range": "694.3-698.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 505.86 ms; process-tree RSS: 26089.49 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.1143258733445,
            "range": "76.8-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3505.72 ms; process-tree RSS: 23748.13 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 78.15700568499422,
            "range": "77.4-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3426.80 ms; process-tree RSS: 23745.08 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 849.8131367969296,
            "range": "840.9-862.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 337.26 ms; process-tree RSS: 7434.45 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 864.7330712188678,
            "range": "862.0-874.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 332.49 ms; process-tree RSS: 7429.04 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 766.7303057312921,
            "range": "756.4-772.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 432.24 ms; process-tree RSS: 7423.68 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 765.3405172797842,
            "range": "743.5-766.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 444.36 ms; process-tree RSS: 7418.90 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 837.0221816370715,
            "range": "816.1-853.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 388.30 ms; process-tree RSS: 23824.90 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 754.3976498556931,
            "range": "735.1-778.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 430.65 ms; process-tree RSS: 23864.31 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 769.2968506282373,
            "range": "741.7-770.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 437.31 ms; process-tree RSS: 23862.78 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 829.6845396698432,
            "range": "823.8-846.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 415.93 ms; process-tree RSS: 23815.59 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1684.1987369241167,
            "range": "1654.4-1708.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.67 ms; process-tree RSS: 23695.92 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4784.643656284997,
            "range": "4690.7-4821.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.70 ms; process-tree RSS: 23689.19 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.43039171903337,
            "range": "190.4-191.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2204.854234212079,
            "range": "2204.0-2274.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2620.637060286605,
            "range": "2620.5-2644.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1233.1541211968342,
            "range": "1232.2-1233.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.4629320855886,
            "range": "1221.0-1225.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.45143260475697,
            "range": "188.2-192.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1924.7234927945588,
            "range": "1813.0-1946.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2485.5675634716627,
            "range": "2391.9-2492.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2248.9897316849806,
            "range": "2243.7-2265.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9f8385869b7e5c5bfb28d1767495551183e3b98",
          "message": "[BugFix] Bound process-slot collection and stop orphaned workers (#4297)",
          "timestamp": "2026-09-08T07:55:53+01:00",
          "tree_id": "a2105435ca31e99ba36cd82633ba3bb816f03125",
          "url": "https://github.com/pytorch/rl/commit/f9f8385869b7e5c5bfb28d1767495551183e3b98"
        },
        "date": 1788853617646,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 976.3852819219898,
            "range": "970.8-1116.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 358.37 ms; process-tree RSS: 25089.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 989.0181940566438,
            "range": "958.2-1101.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.50 ms; process-tree RSS: 25107.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 983.8536075795874,
            "range": "956.7-1066.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 360.37 ms; process-tree RSS: 25109.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 994.3458542761305,
            "range": "984.2-1057.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.09 ms; process-tree RSS: 25089.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.51229807528728,
            "range": "77.4-77.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3542.71 ms; process-tree RSS: 23747.93 MiB; CUDA peak: 27.40 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.76083155955902,
            "range": "77.5-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3433.43 ms; process-tree RSS: 23743.27 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 843.9625624073883,
            "range": "834.5-863.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 342.70 ms; process-tree RSS: 7433.11 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 853.1860094392125,
            "range": "845.1-853.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 347.44 ms; process-tree RSS: 7426.55 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 744.9627958304815,
            "range": "740.4-757.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 448.79 ms; process-tree RSS: 7421.25 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 749.7633403866573,
            "range": "739.5-757.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 431.64 ms; process-tree RSS: 7429.71 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 824.262739782268,
            "range": "815.0-844.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 408.33 ms; process-tree RSS: 23830.35 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 744.0218047797546,
            "range": "731.4-763.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 424.84 ms; process-tree RSS: 23870.16 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 764.8995919441938,
            "range": "736.8-771.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 446.95 ms; process-tree RSS: 23866.34 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 838.5757165864321,
            "range": "825.5-848.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 376.25 ms; process-tree RSS: 23813.11 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1670.811014945757,
            "range": "1660.8-1683.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.66 ms; process-tree RSS: 23693.71 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4662.579099524269,
            "range": "4658.1-4823.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 58.04 ms; process-tree RSS: 23686.84 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 946.0255864537431,
            "range": "945.7-972.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 371.80 ms; process-tree RSS: 47105.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1010.9449151567426,
            "range": "999.5-1047.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 365.30 ms; process-tree RSS: 45808.32 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.3283520190966,
            "range": "189.2-189.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2244.9155857262153,
            "range": "2190.6-2262.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2615.324529324509,
            "range": "2605.1-2619.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.5116324803141,
            "range": "1228.6-1229.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.9610514423632,
            "range": "1219.2-1229.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.45290869315906,
            "range": "187.6-190.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1877.0404096024795,
            "range": "1847.3-1934.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2395.2821082684873,
            "range": "2340.9-2396.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2241.870656809806,
            "range": "2236.2-2242.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "c7def7982b33efb570b46a535c1de083bef45f37",
          "message": "[Feature] Route async collection into replay (#4275)",
          "timestamp": "2026-09-08T08:03:55+01:00",
          "tree_id": "b9c45ccec9cb190b4479a396837e9a295d157c0c",
          "url": "https://github.com/pytorch/rl/commit/c7def7982b33efb570b46a535c1de083bef45f37"
        },
        "date": 1788854147571,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 995.4794035934317,
            "range": "982.0-1043.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 350.64 ms; process-tree RSS: 25093.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 979.439970132376,
            "range": "978.0-1083.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.05 ms; process-tree RSS: 25109.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1004.3900841067664,
            "range": "977.5-1072.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.02 ms; process-tree RSS: 25107.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 980.997934921582,
            "range": "979.1-992.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.46 ms; process-tree RSS: 25090.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.63801249763695,
            "range": "76.2-77.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3485.22 ms; process-tree RSS: 23746.42 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.73644368104236,
            "range": "77.6-78.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3461.95 ms; process-tree RSS: 23744.56 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 867.908660855033,
            "range": "867.7-876.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 336.98 ms; process-tree RSS: 7436.84 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 885.7319380503033,
            "range": "864.3-891.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 339.77 ms; process-tree RSS: 7424.41 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 789.5190306271918,
            "range": "781.1-795.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 378.51 ms; process-tree RSS: 7416.21 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 795.7414435716262,
            "range": "780.9-807.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 388.44 ms; process-tree RSS: 7418.13 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 880.0484530614632,
            "range": "874.5-881.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 321.20 ms; process-tree RSS: 23820.56 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 780.6418376506115,
            "range": "780.0-794.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 376.13 ms; process-tree RSS: 23871.79 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 797.4621359390643,
            "range": "794.7-807.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 368.04 ms; process-tree RSS: 23865.26 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 884.6175292745719,
            "range": "883.2-884.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 324.81 ms; process-tree RSS: 23816.58 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1666.3904700409223,
            "range": "1659.0-1678.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.25 ms; process-tree RSS: 23691.48 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4739.832777495907,
            "range": "4700.0-4807.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.74 ms; process-tree RSS: 23692.84 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 966.9905170790506,
            "range": "952.1-972.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.32 ms; process-tree RSS: 47094.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1103.2033932400075,
            "range": "1069.0-1108.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 342.61 ms; process-tree RSS: 45813.82 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.12300964851448,
            "range": "189.9-191.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2224.2890346823033,
            "range": "2209.9-2227.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2623.7054777453523,
            "range": "2614.2-2624.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1233.3948407905104,
            "range": "1231.7-1236.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.1189946902066,
            "range": "1223.2-1228.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.84709245866316,
            "range": "184.5-189.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1986.073293315794,
            "range": "1908.8-1988.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2394.3156803958977,
            "range": "2392.9-2419.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2232.6350591671667,
            "range": "2229.0-2251.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "ffa8471a3d49339e780fb8b7fbf83360e406b418",
          "message": "[Performance] Use native replay collection in DreamerV3 (#4265)",
          "timestamp": "2026-09-08T08:04:13+01:00",
          "tree_id": "1f163e91791410417486537fd3080b1b017e1526",
          "url": "https://github.com/pytorch/rl/commit/ffa8471a3d49339e780fb8b7fbf83360e406b418"
        },
        "date": 1788854413852,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 981.827335789024,
            "range": "974.5-1002.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.55 ms; process-tree RSS: 25086.76 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 987.3415828270363,
            "range": "981.5-992.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 357.76 ms; process-tree RSS: 25103.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 978.8463550293014,
            "range": "977.8-995.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.25 ms; process-tree RSS: 25103.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 983.2188130499948,
            "range": "955.5-993.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.85 ms; process-tree RSS: 25079.56 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.09593408638754,
            "range": "77.1-78.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3467.38 ms; process-tree RSS: 23749.53 MiB; CUDA peak: 27.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.86529576710653,
            "range": "77.7-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3398.75 ms; process-tree RSS: 23748.54 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 865.1424849963614,
            "range": "864.8-871.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 336.35 ms; process-tree RSS: 7419.48 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 902.408171204438,
            "range": "882.4-910.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 322.23 ms; process-tree RSS: 7435.53 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 786.103987608733,
            "range": "783.7-793.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 368.75 ms; process-tree RSS: 7421.58 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 795.9417424089987,
            "range": "789.0-819.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 378.70 ms; process-tree RSS: 7408.01 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 869.6342527334092,
            "range": "842.6-876.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 346.80 ms; process-tree RSS: 23827.12 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 797.5279990370567,
            "range": "790.0-798.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 373.22 ms; process-tree RSS: 23864.35 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 788.2046552697312,
            "range": "786.6-825.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 372.43 ms; process-tree RSS: 23863.54 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 891.267827396494,
            "range": "867.2-899.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 339.20 ms; process-tree RSS: 23817.79 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1688.5585701880852,
            "range": "1684.0-1689.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.27 ms; process-tree RSS: 23694.49 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4832.755807211985,
            "range": "4768.1-4902.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.12 ms; process-tree RSS: 23693.25 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 959.6401310522989,
            "range": "948.1-960.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.05 ms; process-tree RSS: 47096.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1103.7538658193394,
            "range": "1057.4-1106.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 347.58 ms; process-tree RSS: 45810.77 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.3310401871713,
            "range": "189.8-191.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2236.521520605362,
            "range": "2203.9-2250.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2635.676373570596,
            "range": "2626.6-2639.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.625439688226,
            "range": "1228.7-1231.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1223.8542600308924,
            "range": "1221.1-1227.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.6745969054517,
            "range": "184.1-193.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1902.1140530248304,
            "range": "1867.9-1904.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2438.3193747982473,
            "range": "2412.7-2460.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2256.622383434773,
            "range": "2235.7-2265.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "53e877e2137db36cf4a85de0e7581737ba63fa88",
          "message": "[Performance] Compile the complete DreamerV3 learner step (#4268)",
          "timestamp": "2026-09-08T08:05:02+01:00",
          "tree_id": "6cf05f7cdc41d2458d118868249dd6e3a6aa7cb2",
          "url": "https://github.com/pytorch/rl/commit/53e877e2137db36cf4a85de0e7581737ba63fa88"
        },
        "date": 1788854750187,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1002.822660808118,
            "range": "999.7-1137.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 351.95 ms; process-tree RSS: 25088.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1002.9007238465878,
            "range": "997.9-1007.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 350.50 ms; process-tree RSS: 25093.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1008.2531060150806,
            "range": "986.2-1011.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 353.34 ms; process-tree RSS: 25094.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1005.0948917694774,
            "range": "1004.5-1024.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 353.33 ms; process-tree RSS: 25081.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.84534360828596,
            "range": "77.2-78.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3405.17 ms; process-tree RSS: 23752.80 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 78.33628591197154,
            "range": "77.9-78.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3440.46 ms; process-tree RSS: 23743.12 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 897.7397321834138,
            "range": "893.3-899.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 315.04 ms; process-tree RSS: 7436.26 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 912.8649036374871,
            "range": "890.1-914.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 316.19 ms; process-tree RSS: 7428.20 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 796.5454613663128,
            "range": "782.5-806.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 360.91 ms; process-tree RSS: 7424.95 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 803.0038397503197,
            "range": "801.1-810.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 363.79 ms; process-tree RSS: 7423.87 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 881.8763315485434,
            "range": "877.7-897.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.10 ms; process-tree RSS: 23832.95 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 811.9565151423924,
            "range": "804.4-813.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 355.86 ms; process-tree RSS: 23864.59 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 811.7487722910336,
            "range": "790.6-812.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 359.95 ms; process-tree RSS: 23862.82 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 898.9239751849617,
            "range": "880.0-903.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 323.88 ms; process-tree RSS: 23819.98 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1660.0447363250332,
            "range": "1658.1-1665.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.11 ms; process-tree RSS: 23694.03 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4708.227227527847,
            "range": "4699.1-4835.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.19 ms; process-tree RSS: 23690.76 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 977.7088735750796,
            "range": "969.2-982.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.71 ms; process-tree RSS: 47099.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1131.660063709946,
            "range": "1086.0-1146.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 330.93 ms; process-tree RSS: 45798.54 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 193.27932881506814,
            "range": "192.4-193.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2312.293004755094,
            "range": "2231.9-2318.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2677.775806622805,
            "range": "2668.9-2683.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.6600699189876,
            "range": "1225.3-1228.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.6724485104933,
            "range": "1216.9-1230.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 194.05266111784633,
            "range": "192.2-194.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1850.6399363803203,
            "range": "1830.2-2007.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2444.913861273467,
            "range": "2435.5-2463.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2296.614301704628,
            "range": "2276.0-2298.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b013a6a94ed04819bcdeaea3b8a0619d5aeb46f9",
          "message": "[CI] Label benchmark trend points with the benchmarked commit (#4298)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-08T08:05:46+01:00",
          "tree_id": "652f5cdfc0ab4de5dc181ff0629c59ed43a1cfe0",
          "url": "https://github.com/pytorch/rl/commit/b013a6a94ed04819bcdeaea3b8a0619d5aeb46f9"
        },
        "date": 1788855267711,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 996.1496452301136,
            "range": "989.1-1084.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 353.80 ms; process-tree RSS: 25094.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1006.315015885191,
            "range": "986.0-1010.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.38 ms; process-tree RSS: 25106.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1012.131931255013,
            "range": "1007.2-1013.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 354.51 ms; process-tree RSS: 25107.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1001.6396253767601,
            "range": "1000.6-1002.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 351.04 ms; process-tree RSS: 25092.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.85976919218206,
            "range": "77.7-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3421.11 ms; process-tree RSS: 23756.75 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 78.75891969387546,
            "range": "78.2-79.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3397.14 ms; process-tree RSS: 23745.08 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 896.056364409797,
            "range": "895.5-911.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 316.81 ms; process-tree RSS: 7422.30 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 894.1068538000726,
            "range": "886.8-903.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 330.78 ms; process-tree RSS: 7436.32 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 800.1430493245426,
            "range": "787.8-809.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 350.59 ms; process-tree RSS: 7428.60 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 806.584293148314,
            "range": "794.3-810.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 358.05 ms; process-tree RSS: 7422.50 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 888.4941784351488,
            "range": "882.3-888.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.09 ms; process-tree RSS: 23830.41 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 801.4008618545529,
            "range": "798.1-835.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 394.24 ms; process-tree RSS: 23870.57 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 811.5452230385844,
            "range": "803.6-816.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 366.52 ms; process-tree RSS: 23870.34 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 905.0201343116893,
            "range": "875.0-907.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 327.86 ms; process-tree RSS: 23823.51 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1658.116104119462,
            "range": "1657.9-1671.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.57 ms; process-tree RSS: 23696.61 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4694.58245414078,
            "range": "4686.3-4734.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 58.01 ms; process-tree RSS: 23690.31 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 966.8780577783417,
            "range": "964.8-968.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 362.54 ms; process-tree RSS: 47099.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1143.6344747835894,
            "range": "1116.9-1147.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 314.25 ms; process-tree RSS: 45807.98 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.79632210906163,
            "range": "191.6-192.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2321.9942064475604,
            "range": "2279.4-2329.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2672.767083526616,
            "range": "2668.5-2690.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1225.1404965157294,
            "range": "1225.1-1228.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.5479098314363,
            "range": "1226.8-1230.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.82659313585924,
            "range": "191.5-193.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1982.1995844708374,
            "range": "1859.3-2002.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2450.1837475697453,
            "range": "2447.2-2458.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2292.9930286853037,
            "range": "2276.2-2294.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ffa09de8417026ca2ebb09517da4e2a5fe1fe9bd",
          "message": "[Feature] Compose DreamerV3 reconstruction heads in the public model loss (#4283)",
          "timestamp": "2026-09-08T08:08:52+01:00",
          "tree_id": "6376fd639efea2f2bb241339f70e242321754bd0",
          "url": "https://github.com/pytorch/rl/commit/ffa09de8417026ca2ebb09517da4e2a5fe1fe9bd"
        },
        "date": 1788855601391,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1000.320443471327,
            "range": "986.6-1074.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 353.26 ms; process-tree RSS: 25092.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 998.9701486013848,
            "range": "965.2-1013.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 352.92 ms; process-tree RSS: 25107.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1003.5968233896758,
            "range": "1002.3-1089.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 356.14 ms; process-tree RSS: 25109.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 987.2560743803174,
            "range": "986.9-995.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 356.00 ms; process-tree RSS: 25089.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.25288348592416,
            "range": "77.2-77.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3454.37 ms; process-tree RSS: 23744.37 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.53149626007945,
            "range": "77.2-78.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3436.40 ms; process-tree RSS: 23742.79 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 888.9876305661145,
            "range": "867.7-903.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 336.01 ms; process-tree RSS: 7439.86 MiB; CUDA peak: 48.55 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 889.4041549947132,
            "range": "887.7-893.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 324.32 ms; process-tree RSS: 7428.04 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 811.1469255361718,
            "range": "803.3-813.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 359.16 ms; process-tree RSS: 7427.43 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 805.353111031369,
            "range": "781.4-814.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 361.05 ms; process-tree RSS: 7427.81 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 884.5642764091928,
            "range": "878.8-897.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 323.59 ms; process-tree RSS: 23831.66 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 818.4591739704027,
            "range": "794.7-824.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 356.05 ms; process-tree RSS: 23871.84 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 806.081926550471,
            "range": "797.7-816.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 351.36 ms; process-tree RSS: 23868.38 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 888.7124169867642,
            "range": "871.5-897.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 328.24 ms; process-tree RSS: 23815.71 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1660.844576543579,
            "range": "1658.4-1662.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.45 ms; process-tree RSS: 23695.70 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4685.89976837277,
            "range": "4659.8-4704.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 58.26 ms; process-tree RSS: 23692.21 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 963.3972158610154,
            "range": "952.5-972.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 362.32 ms; process-tree RSS: 47120.06 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1128.5939884063384,
            "range": "1069.3-1138.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 341.09 ms; process-tree RSS: 45809.52 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.45138013190206,
            "range": "191.2-191.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2286.5447761301534,
            "range": "2238.5-2309.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2675.323770543928,
            "range": "2652.8-2677.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1233.6092453801098,
            "range": "1231.8-1233.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1229.8445561751905,
            "range": "1226.5-1230.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 194.9433097097478,
            "range": "190.6-196.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1993.0963036808507,
            "range": "1980.2-1993.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2486.831353705024,
            "range": "2415.6-2492.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2283.129866793309,
            "range": "2281.2-2294.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4bcbe8b3067d02f71121926d72d374a008f74595",
          "message": "[Feature] Resume DreamerV3 with native replay checkpoints (#4281)",
          "timestamp": "2026-09-08T09:13:04+01:00",
          "tree_id": "91f7d036b35d7a80eb6ac16f07fedb6cc10c1556",
          "url": "https://github.com/pytorch/rl/commit/4bcbe8b3067d02f71121926d72d374a008f74595"
        },
        "date": 1788863227439,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 986.4103498383145,
            "range": "980.0-994.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.36 ms; process-tree RSS: 25090.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 972.531074443018,
            "range": "968.8-996.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.55 ms; process-tree RSS: 25107.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 988.5060102532037,
            "range": "981.5-990.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.23 ms; process-tree RSS: 25109.56 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 983.6360822068908,
            "range": "981.0-998.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.32 ms; process-tree RSS: 25087.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.30847095202637,
            "range": "76.5-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3486.50 ms; process-tree RSS: 23752.26 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.83671045460515,
            "range": "77.8-77.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3425.07 ms; process-tree RSS: 23746.15 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 869.4028688090328,
            "range": "868.6-878.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 329.62 ms; process-tree RSS: 7424.93 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 886.3325365886557,
            "range": "876.1-886.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 317.09 ms; process-tree RSS: 7433.49 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 770.3330781358583,
            "range": "760.1-800.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 373.16 ms; process-tree RSS: 7428.14 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 785.1035681830793,
            "range": "776.0-790.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 361.76 ms; process-tree RSS: 7420.07 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 862.3271605734275,
            "range": "847.9-885.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 351.24 ms; process-tree RSS: 23833.90 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 796.6338733579154,
            "range": "788.1-811.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 360.52 ms; process-tree RSS: 23876.44 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 807.635291928918,
            "range": "804.8-808.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 356.72 ms; process-tree RSS: 23875.50 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 875.9107219716585,
            "range": "875.1-889.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 337.85 ms; process-tree RSS: 23819.70 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1673.2096106915874,
            "range": "1660.4-1685.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.84 ms; process-tree RSS: 23694.40 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4716.186135963674,
            "range": "4642.9-4891.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.07 ms; process-tree RSS: 23695.13 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 971.5140984990255,
            "range": "947.6-972.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.05 ms; process-tree RSS: 47110.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1105.8973487006147,
            "range": "1041.5-1136.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 312.64 ms; process-tree RSS: 45823.05 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.9592625309215,
            "range": "190.5-191.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2268.6022744618963,
            "range": "2261.6-2291.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2625.0965099486443,
            "range": "2618.9-2632.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.8083561311596,
            "range": "1223.3-1229.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.273666250188,
            "range": "1225.1-1231.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.81712741216793,
            "range": "191.1-198.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1886.9266084721025,
            "range": "1859.3-1947.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2385.2542964443996,
            "range": "2381.8-2423.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2233.5052918659153,
            "range": "2230.7-2240.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "db7c2e97dafbdf310a1d3943bec35021ebe3dd81",
          "message": "[Test] Baseline benchmarks for async collectors and DreamerV3 (#4312)",
          "timestamp": "2026-09-09T09:58:47+01:00",
          "tree_id": "baf547d13c55b319c1dcf45e39a522e89528c2f2",
          "url": "https://github.com/pytorch/rl/commit/db7c2e97dafbdf310a1d3943bec35021ebe3dd81"
        },
        "date": 1788948816111,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 998.4761829836066,
            "range": "994.0-1118.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 357.92 ms; process-tree RSS: 25113.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 1001.8370453049341,
            "range": "985.5-1005.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 363.65 ms; process-tree RSS: 25114.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 992.954157297455,
            "range": "992.6-1003.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 359.92 ms; process-tree RSS: 25093.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1101.8552899487045,
            "range": "978.4-1131.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 355.96 ms; process-tree RSS: 25114.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 996.6421849789482,
            "range": "986.9-1069.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 357.49 ms; process-tree RSS: 25111.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1082.2712915663428,
            "range": "1056.7-1095.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 356.32 ms; process-tree RSS: 25096.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.1030768012754,
            "range": "76.8-77.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3454.41 ms; process-tree RSS: 23762.69 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.94611498036613,
            "range": "77.9-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3437.37 ms; process-tree RSS: 23758.31 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 888.9395561286138,
            "range": "874.3-909.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 326.54 ms; process-tree RSS: 7433.68 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 908.9266527144911,
            "range": "889.0-918.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 316.85 ms; process-tree RSS: 7427.43 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 806.8241152148272,
            "range": "799.1-824.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 363.11 ms; process-tree RSS: 7421.13 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 808.4953559080285,
            "range": "795.1-830.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 384.79 ms; process-tree RSS: 7431.67 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 879.4801257619616,
            "range": "860.6-896.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 325.22 ms; process-tree RSS: 23843.37 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 804.4823437018808,
            "range": "797.6-815.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 350.73 ms; process-tree RSS: 23879.40 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 812.6834019199155,
            "range": "804.7-838.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 356.00 ms; process-tree RSS: 23877.74 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 888.1216781929164,
            "range": "887.6-892.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 342.31 ms; process-tree RSS: 23834.70 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1693.694746113798,
            "range": "1659.3-1706.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 253.18 ms; process-tree RSS: 23701.59 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4840.233190522154,
            "range": "4699.2-4888.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.32 ms; process-tree RSS: 23694.62 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1060.7253923697908,
            "range": "972.1-1063.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.94 ms; process-tree RSS: 47119.87 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1124.3109399912423,
            "range": "1116.2-1159.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.39 ms; process-tree RSS: 45821.07 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.248502719339335,
            "range": "52.1-52.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8071.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 83.68707658333949,
            "range": "83.5-84.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8061.41 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.6876694905597,
            "range": "190.8-191.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2274.069471369813,
            "range": "2267.8-2299.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2673.9500749392164,
            "range": "2665.8-2689.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.883463263839,
            "range": "1226.9-1232.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1230.319084816903,
            "range": "1225.8-1232.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 192.69359118766957,
            "range": "190.8-194.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1999.3713734291086,
            "range": "1839.9-2018.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2454.1971349258893,
            "range": "2434.6-2475.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2281.967847173151,
            "range": "2281.9-2284.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7f348ee4cdf1ac598fd22059af46e24260e85bf",
          "message": "[BugFix] Async DreamerV3: explicit exploration, stored episode flags, cross-episode replay sequences (#4310)",
          "timestamp": "2026-09-09T10:00:30+01:00",
          "tree_id": "46797f58ed9134dda4573a56b110776a6792db2d",
          "url": "https://github.com/pytorch/rl/commit/b7f348ee4cdf1ac598fd22059af46e24260e85bf"
        },
        "date": 1788949126750,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 986.7323771844906,
            "range": "972.8-990.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 363.45 ms; process-tree RSS: 25111.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 999.3186159679922,
            "range": "992.4-1007.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 357.91 ms; process-tree RSS: 25112.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 983.5260500761874,
            "range": "981.0-989.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.03 ms; process-tree RSS: 25093.36 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 987.4376285236888,
            "range": "980.0-990.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 364.43 ms; process-tree RSS: 25112.81 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1004.0766088842803,
            "range": "980.7-1084.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 361.92 ms; process-tree RSS: 25109.64 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1104.1125649487938,
            "range": "1043.7-1106.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.45 ms; process-tree RSS: 25094.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.75722472421302,
            "range": "76.5-76.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3459.00 ms; process-tree RSS: 23757.04 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.85949646771031,
            "range": "77.4-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3391.19 ms; process-tree RSS: 23754.92 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 873.6723290496087,
            "range": "866.6-899.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 320.43 ms; process-tree RSS: 7434.35 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 898.1877266239592,
            "range": "873.2-898.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 332.52 ms; process-tree RSS: 7433.58 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 782.6039893285208,
            "range": "762.2-789.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 363.71 ms; process-tree RSS: 7427.04 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 797.9908031700949,
            "range": "776.6-799.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 383.23 ms; process-tree RSS: 7412.27 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 873.8542559979682,
            "range": "862.5-874.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 333.62 ms; process-tree RSS: 23836.45 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 789.8929750405737,
            "range": "780.4-822.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 358.33 ms; process-tree RSS: 23880.19 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 793.9296443319771,
            "range": "767.9-809.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 380.62 ms; process-tree RSS: 23875.80 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 886.83647906835,
            "range": "868.5-887.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 332.89 ms; process-tree RSS: 23827.57 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1662.9335648623123,
            "range": "1657.0-1678.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.48 ms; process-tree RSS: 23701.60 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4724.435728781671,
            "range": "4696.8-4748.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.31 ms; process-tree RSS: 23697.02 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 954.6779312796781,
            "range": "945.2-957.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 372.94 ms; process-tree RSS: 47124.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1086.2560406387445,
            "range": "1056.9-1152.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 294.59 ms; process-tree RSS: 45819.64 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.84511156621122,
            "range": "52.4-53.0",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8077.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 83.18893889764604,
            "range": "82.9-84.1",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8071.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.82625989542774,
            "range": "189.3-190.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2214.5810764678063,
            "range": "2174.2-2223.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2614.189474803263,
            "range": "2611.8-2625.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1227.3317812827424,
            "range": "1225.7-1231.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.6323917856746,
            "range": "1217.4-1229.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.28850050947074,
            "range": "184.9-197.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1955.1008642649863,
            "range": "1928.0-1965.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2362.5961438487566,
            "range": "2357.9-2388.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2244.913004386397,
            "range": "2230.5-2269.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5076ae186cdc29c4f9f14e36405fdba4dff2b6bd",
          "message": "[Performance] Chunk process-slot worker results in AsyncBatchedCollector (#4305)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T11:28:10+01:00",
          "tree_id": "875733f3bf4fb2745762c16c617861cfb48c8aca",
          "url": "https://github.com/pytorch/rl/commit/5076ae186cdc29c4f9f14e36405fdba4dff2b6bd"
        },
        "date": 1788953928174,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 2987.8152749823553,
            "range": "2510.8-7456.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 215.89 ms; process-tree RSS: 25470.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 2811.768708454003,
            "range": "2443.1-3395.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 245.42 ms; process-tree RSS: 25720.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 2945.8467664360637,
            "range": "2878.1-3254.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 187.06 ms; process-tree RSS: 25569.92 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 3076.3461233683624,
            "range": "2912.5-3448.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 140.18 ms; process-tree RSS: 25502.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1049.5955708904462,
            "range": "972.4-1054.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 361.00 ms; process-tree RSS: 25098.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 959.7886803268026,
            "range": "949.3-1084.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 367.04 ms; process-tree RSS: 25116.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 984.8481667373568,
            "range": "971.1-1102.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.23 ms; process-tree RSS: 25118.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1061.3611379270046,
            "range": "983.7-1095.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.91 ms; process-tree RSS: 25098.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.04009872068083,
            "range": "76.3-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3529.35 ms; process-tree RSS: 23755.14 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.66173370406997,
            "range": "77.6-77.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3457.63 ms; process-tree RSS: 23752.95 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 894.1459766934546,
            "range": "884.1-903.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 326.90 ms; process-tree RSS: 7433.39 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 878.858879126163,
            "range": "874.2-881.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 341.58 ms; process-tree RSS: 7435.43 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 785.1729118943517,
            "range": "771.2-799.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 387.70 ms; process-tree RSS: 7421.55 MiB; CUDA peak: 54.08 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 796.542426508924,
            "range": "773.7-802.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 378.93 ms; process-tree RSS: 7417.58 MiB; CUDA peak: 45.96 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 865.4544798508406,
            "range": "855.9-880.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 334.06 ms; process-tree RSS: 23843.12 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 794.8521524541281,
            "range": "787.1-800.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 372.52 ms; process-tree RSS: 23882.75 MiB; CUDA peak: 70.33 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 802.9562704479184,
            "range": "790.5-809.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 365.13 ms; process-tree RSS: 23869.63 MiB; CUDA peak: 62.21 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 885.3177300417501,
            "range": "876.9-886.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 349.06 ms; process-tree RSS: 23827.51 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1656.0006344943056,
            "range": "1654.0-1680.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.79 ms; process-tree RSS: 23697.72 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4834.40941511615,
            "range": "4699.5-4863.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.64 ms; process-tree RSS: 23696.60 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1054.3718832546242,
            "range": "1049.6-1072.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.26 ms; process-tree RSS: 47108.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1105.1154854743909,
            "range": "1093.8-1112.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 286.30 ms; process-tree RSS: 45828.77 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 51.71691605538064,
            "range": "51.3-51.9",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8070.82 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 82.84619218805152,
            "range": "82.7-83.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8065.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.11177920629962,
            "range": "189.0-190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2235.861325662362,
            "range": "2213.1-2281.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2616.1736633452183,
            "range": "2611.2-2634.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1228.0773236530874,
            "range": "1225.3-1231.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.6290210364427,
            "range": "1222.1-1228.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 190.13187618416558,
            "range": "189.3-191.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1920.1791971233724,
            "range": "1875.3-2006.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2417.7881348099727,
            "range": "2404.7-2448.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2238.2773325837284,
            "range": "2224.0-2244.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e11e38e50ccc89e006096c42ec32408a98131324",
          "message": "[Performance] Serve process-slot inference passes from pinned staging batches (#4306)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T12:13:05+01:00",
          "tree_id": "6f41274772cb4d7f2e88ee9dd653414365353492",
          "url": "https://github.com/pytorch/rl/commit/e11e38e50ccc89e006096c42ec32408a98131324"
        },
        "date": 1788956614262,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 6429.439600255789,
            "range": "4644.1-21150.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 105.49 ms; process-tree RSS: 25455.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4927.2189138484855,
            "range": "4698.8-5618.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 112.55 ms; process-tree RSS: 25634.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 5102.517779097973,
            "range": "3900.5-5522.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 106.00 ms; process-tree RSS: 25502.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 4808.6002868035375,
            "range": "4252.4-5261.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 85.92 ms; process-tree RSS: 25527.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 984.0339341027421,
            "range": "970.2-1015.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 362.39 ms; process-tree RSS: 25081.26 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 982.1610510324138,
            "range": "978.3-1104.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.76 ms; process-tree RSS: 25163.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1042.7937779763156,
            "range": "988.6-1051.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 366.68 ms; process-tree RSS: 25157.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 981.1355515318738,
            "range": "976.1-1058.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.57 ms; process-tree RSS: 25080.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.35247825843615,
            "range": "76.4-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3504.74 ms; process-tree RSS: 23753.32 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.5145712610186,
            "range": "76.8-77.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3440.50 ms; process-tree RSS: 23754.27 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 872.561557174943,
            "range": "860.1-894.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 333.36 ms; process-tree RSS: 7466.70 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 893.3124675431086,
            "range": "875.9-893.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 323.10 ms; process-tree RSS: 7452.96 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 782.9454443707002,
            "range": "778.9-788.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 370.75 ms; process-tree RSS: 7453.51 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 796.7703616787898,
            "range": "769.5-799.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 363.29 ms; process-tree RSS: 7446.21 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 878.4216086131279,
            "range": "868.0-884.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 333.78 ms; process-tree RSS: 23836.09 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 787.1192243493432,
            "range": "780.9-800.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 363.20 ms; process-tree RSS: 23905.79 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 796.158060905479,
            "range": "785.3-796.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 359.83 ms; process-tree RSS: 23913.70 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 874.1989394549613,
            "range": "870.4-882.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 328.91 ms; process-tree RSS: 23825.68 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1656.887747005451,
            "range": "1654.9-1706.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.86 ms; process-tree RSS: 23700.77 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4759.275275592926,
            "range": "4681.6-4764.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.88 ms; process-tree RSS: 23695.17 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1029.1577641000742,
            "range": "944.5-1065.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 375.26 ms; process-tree RSS: 47113.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1118.9035629714701,
            "range": "1067.0-1139.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 283.17 ms; process-tree RSS: 45863.53 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.410831901752445,
            "range": "52.1-52.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8071.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 83.08214182024817,
            "range": "81.8-83.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8068.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.66403934822458,
            "range": "190.7-190.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2220.758487916858,
            "range": "2191.0-2231.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2637.602064058913,
            "range": "2612.8-2639.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.2920561066928,
            "range": "1227.1-1237.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.5995092691467,
            "range": "1228.2-1228.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 191.53434125464358,
            "range": "190.8-194.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1890.6801988269344,
            "range": "1794.4-1936.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2403.7618271650686,
            "range": "2396.7-2489.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2261.124998033883,
            "range": "2227.6-2267.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8fb9586eedd544ad7ff2249b07a7a6e9d007f3ed",
          "message": "[Performance] Keep the DreamerV3 replay write-back off the learner stream (#4307)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T13:09:30+01:00",
          "tree_id": "5d11090585cafa84b06ca40bbd03a47419acacb1",
          "url": "https://github.com/pytorch/rl/commit/8fb9586eedd544ad7ff2249b07a7a6e9d007f3ed"
        },
        "date": 1788960384577,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4695.46427001287,
            "range": "4581.5-5927.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 92.84 ms; process-tree RSS: 25461.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4841.851946900813,
            "range": "4195.9-8017.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 139.18 ms; process-tree RSS: 25688.05 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4944.817884701705,
            "range": "4910.2-5328.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 88.47 ms; process-tree RSS: 25564.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5844.125184875235,
            "range": "5260.1-5906.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 82.17 ms; process-tree RSS: 25539.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 985.4808196573188,
            "range": "963.2-991.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.06 ms; process-tree RSS: 25083.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 986.0134568882564,
            "range": "980.2-989.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 361.44 ms; process-tree RSS: 25157.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 998.0020661370197,
            "range": "976.8-1100.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 363.18 ms; process-tree RSS: 25160.92 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1076.7181536460248,
            "range": "1037.0-1117.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.05 ms; process-tree RSS: 25084.44 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.35015592913302,
            "range": "76.0-76.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3462.32 ms; process-tree RSS: 23766.81 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.7727286770415,
            "range": "76.9-78.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3434.85 ms; process-tree RSS: 23756.82 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 875.9131727356096,
            "range": "859.2-884.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 342.98 ms; process-tree RSS: 7463.18 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 878.9951665120426,
            "range": "874.6-896.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 326.81 ms; process-tree RSS: 7459.00 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 769.9966695086697,
            "range": "763.4-782.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 373.39 ms; process-tree RSS: 7459.02 MiB; CUDA peak: 53.41 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 784.057312445423,
            "range": "779.4-794.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 367.96 ms; process-tree RSS: 7437.97 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 873.6921537692656,
            "range": "863.4-876.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 332.99 ms; process-tree RSS: 23842.07 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 776.5247610435432,
            "range": "775.5-787.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 373.48 ms; process-tree RSS: 23904.07 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 804.5237958208294,
            "range": "783.8-816.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 374.56 ms; process-tree RSS: 23899.95 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 878.2673801174918,
            "range": "868.8-898.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 332.85 ms; process-tree RSS: 23837.70 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1682.4536351761137,
            "range": "1654.3-1685.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.42 ms; process-tree RSS: 23696.34 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4708.3698046174395,
            "range": "4707.4-4817.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.24 ms; process-tree RSS: 23690.70 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 958.3286078249091,
            "range": "958.1-960.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 376.22 ms; process-tree RSS: 47112.90 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1101.4928518638446,
            "range": "1094.6-1117.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 291.25 ms; process-tree RSS: 45856.07 MiB; CUDA peak: 70.91 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.71806048695341,
            "range": "52.5-53.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8069.74 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.3621706198126,
            "range": "83.8-84.9",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8066.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.55949798622453,
            "range": "187.7-190.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2242.285868199279,
            "range": "2206.3-2243.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2618.6779408528955,
            "range": "2618.1-2634.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.3774986928247,
            "range": "1230.9-1231.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.7642006917888,
            "range": "1220.0-1230.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 188.73046283305584,
            "range": "184.8-193.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1885.9202911442483,
            "range": "1826.3-1921.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2412.1221358116427,
            "range": "2379.0-2438.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2226.9930013627513,
            "range": "2223.4-2234.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a08aa5859fb4e2ed3b32fe2bd9b8b60616a7b93",
          "message": "[Performance] DreamerV3: freeze the setup heap before the training loop (#4308)\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-09T14:04:07+01:00",
          "tree_id": "f5fead42a3db7becb7b1f6e4cfdd8328810490bb",
          "url": "https://github.com/pytorch/rl/commit/0a08aa5859fb4e2ed3b32fe2bd9b8b60616a7b93"
        },
        "date": 1788963061745,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4786.12548586646,
            "range": "3641.9-22749.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 112.24 ms; process-tree RSS: 25463.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 6932.047748965724,
            "range": "6612.4-8232.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 159.39 ms; process-tree RSS: 25519.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 5299.061131078563,
            "range": "4007.7-5360.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 86.56 ms; process-tree RSS: 25658.09 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5269.568860453086,
            "range": "4941.9-6416.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 89.23 ms; process-tree RSS: 25521.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 980.4604637397406,
            "range": "975.3-1024.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 363.75 ms; process-tree RSS: 25081.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 970.8074907270071,
            "range": "970.0-973.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 370.35 ms; process-tree RSS: 25160.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 980.249197285234,
            "range": "970.6-982.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 362.44 ms; process-tree RSS: 25161.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1030.7424035300432,
            "range": "1007.1-1117.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 362.56 ms; process-tree RSS: 25088.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.80734727370758,
            "range": "76.5-77.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3480.78 ms; process-tree RSS: 23757.24 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.59228218072238,
            "range": "77.2-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3471.62 ms; process-tree RSS: 23753.77 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 869.3887032823017,
            "range": "852.0-878.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 349.19 ms; process-tree RSS: 7461.89 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 882.7680396722592,
            "range": "875.3-884.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 323.16 ms; process-tree RSS: 7462.66 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 780.6695098715097,
            "range": "776.3-794.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 369.78 ms; process-tree RSS: 7452.13 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 779.8011976502355,
            "range": "773.2-782.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 366.67 ms; process-tree RSS: 7426.79 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 874.6973682709408,
            "range": "858.4-886.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 343.48 ms; process-tree RSS: 23833.29 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 798.1203131796811,
            "range": "785.2-806.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 374.59 ms; process-tree RSS: 23905.26 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 791.5216914622074,
            "range": "788.1-809.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 368.32 ms; process-tree RSS: 23904.29 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 873.2979677390969,
            "range": "872.6-892.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 329.17 ms; process-tree RSS: 23825.30 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1663.9413857928105,
            "range": "1659.3-1680.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 256.31 ms; process-tree RSS: 23696.48 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4763.1757838203275,
            "range": "4741.6-4874.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.21 ms; process-tree RSS: 23698.50 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 936.9427861029001,
            "range": "922.8-1021.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 374.06 ms; process-tree RSS: 47117.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1095.7892709226123,
            "range": "1069.3-1096.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 303.21 ms; process-tree RSS: 45859.93 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.632114704207076,
            "range": "51.9-52.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8069.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 83.83039945831257,
            "range": "83.7-85.0",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8065.84 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.1501387915725,
            "range": "188.7-190.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2232.7279436869526,
            "range": "2202.5-2254.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2618.6537624525454,
            "range": "2611.3-2624.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.9497028640774,
            "range": "1217.9-1231.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1221.3035761202354,
            "range": "1219.9-1225.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 187.91092004221187,
            "range": "187.8-190.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1827.6987252755894,
            "range": "1773.3-1857.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2409.998108739741,
            "range": "2402.2-2411.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2236.542984594553,
            "range": "2228.0-2237.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "iliam.souami@gmail.com",
            "name": "Iliam Souami",
            "username": "Iliamsou"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9b0587e271df94a0dd0566c44a8d5f04233b6e52",
          "message": "[Feature] Add LBForaging environment wrapper (#4301)\n\nCo-authored-by: Vincent Moens <vincentmoens@gmail.com>",
          "timestamp": "2026-09-09T15:14:46+01:00",
          "tree_id": "25f47b5c871ffc1d96d14d99a930cd601f1662fc",
          "url": "https://github.com/pytorch/rl/commit/9b0587e271df94a0dd0566c44a8d5f04233b6e52"
        },
        "date": 1788967241330,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4655.755358386737,
            "range": "4642.7-4732.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 99.36 ms; process-tree RSS: 25436.62 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4435.5319706526725,
            "range": "3023.8-4758.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 94.18 ms; process-tree RSS: 25454.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4422.543639793447,
            "range": "4214.3-4618.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 117.21 ms; process-tree RSS: 25501.24 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5281.625068556144,
            "range": "4338.7-6412.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 92.92 ms; process-tree RSS: 25508.66 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 989.7668345277935,
            "range": "980.7-1041.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.20 ms; process-tree RSS: 25090.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 989.1905048376611,
            "range": "973.9-1102.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.88 ms; process-tree RSS: 25165.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 992.4593952805963,
            "range": "980.5-1076.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 356.64 ms; process-tree RSS: 25163.50 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1007.9554359278943,
            "range": "986.4-1031.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.40 ms; process-tree RSS: 25091.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.8617735152978,
            "range": "76.6-77.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3504.19 ms; process-tree RSS: 23755.23 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.62136777973679,
            "range": "77.4-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3417.26 ms; process-tree RSS: 23753.50 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 874.0401052995859,
            "range": "865.8-883.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 319.36 ms; process-tree RSS: 7458.41 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 891.7586619286324,
            "range": "887.4-891.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 341.16 ms; process-tree RSS: 7454.10 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 779.6252226590681,
            "range": "776.8-798.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 379.48 ms; process-tree RSS: 7446.96 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 776.00489439629,
            "range": "763.5-793.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 367.87 ms; process-tree RSS: 7426.86 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 866.5674408020482,
            "range": "866.3-871.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 329.03 ms; process-tree RSS: 23829.45 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 791.7450098286854,
            "range": "790.6-793.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 367.67 ms; process-tree RSS: 23907.44 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 793.0728365349814,
            "range": "790.6-797.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 356.35 ms; process-tree RSS: 23928.80 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 878.452274976457,
            "range": "855.0-897.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.83 ms; process-tree RSS: 23824.98 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1657.7742565473202,
            "range": "1654.3-1659.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.60 ms; process-tree RSS: 23699.91 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4731.148009600499,
            "range": "4706.1-4814.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.22 ms; process-tree RSS: 23694.62 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1071.710686408875,
            "range": "1051.7-1079.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.74 ms; process-tree RSS: 47121.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1123.5843901015023,
            "range": "1109.6-1127.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 348.74 ms; process-tree RSS: 45868.93 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.67502274547757,
            "range": "52.2-52.8",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8072.55 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.31562695821559,
            "range": "84.0-84.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8063.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.94154681892894,
            "range": "189.5-190.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2224.5079524177527,
            "range": "2223.6-2240.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2631.2501729571195,
            "range": "2629.8-2645.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1228.5987574892936,
            "range": "1222.0-1231.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.3657816544408,
            "range": "1225.7-1231.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.11141479888357,
            "range": "187.1-189.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1904.8864062375949,
            "range": "1795.0-1948.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2414.4672045440775,
            "range": "2362.9-2447.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2248.4388385410475,
            "range": "2243.7-2255.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1c972b314be763841daa238c9a6b0c7bde25c1ce",
          "message": "[CI] Fix DreamerV3 process benchmark exchange (#4320)",
          "timestamp": "2026-09-10T09:55:37+01:00",
          "tree_id": "4df515cdcff163653fd157b0aa338ebb535a3fbe",
          "url": "https://github.com/pytorch/rl/commit/1c972b314be763841daa238c9a6b0c7bde25c1ce"
        },
        "date": 1789035049947,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4963.251581221636,
            "range": "3881.7-4966.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 126.20 ms; process-tree RSS: 25402.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4181.103143394876,
            "range": "3936.5-6827.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 224.13 ms; process-tree RSS: 25454.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 5080.240513666567,
            "range": "5023.6-6441.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 85.42 ms; process-tree RSS: 25481.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 4722.229535377951,
            "range": "3834.0-5237.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 80.13 ms; process-tree RSS: 25521.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 981.4127564097686,
            "range": "974.5-1126.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 358.85 ms; process-tree RSS: 25077.53 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 992.8843113258796,
            "range": "984.1-1009.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 356.63 ms; process-tree RSS: 25163.48 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1019.3021231549953,
            "range": "997.3-1127.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 353.94 ms; process-tree RSS: 25168.34 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 993.1451819915015,
            "range": "985.1-1115.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.76 ms; process-tree RSS: 25083.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.48839373825206,
            "range": "77.4-77.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3479.74 ms; process-tree RSS: 23754.80 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 78.08466030589398,
            "range": "77.9-78.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3429.46 ms; process-tree RSS: 23753.51 MiB; CUDA peak: 27.94 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 897.0158265818428,
            "range": "894.1-913.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 322.37 ms; process-tree RSS: 7454.81 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 900.5551802198942,
            "range": "896.5-908.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 329.77 ms; process-tree RSS: 7453.32 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 807.5771258216038,
            "range": "801.7-817.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 355.94 ms; process-tree RSS: 7445.34 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 801.3305812582091,
            "range": "799.6-808.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 351.89 ms; process-tree RSS: 7426.04 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 891.358539318037,
            "range": "889.2-900.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 333.06 ms; process-tree RSS: 23839.16 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 811.8269789343052,
            "range": "811.7-828.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 371.13 ms; process-tree RSS: 23913.49 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 805.9497039421323,
            "range": "798.1-810.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 371.84 ms; process-tree RSS: 23914.43 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 892.7588411476078,
            "range": "890.5-902.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 324.89 ms; process-tree RSS: 23825.23 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1663.3332374861222,
            "range": "1662.5-1679.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.05 ms; process-tree RSS: 23695.47 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4785.856120109956,
            "range": "4737.1-4822.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.01 ms; process-tree RSS: 23688.79 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 966.4325965029846,
            "range": "955.0-1069.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 375.87 ms; process-tree RSS: 47115.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1143.6848784056886,
            "range": "1134.5-1161.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 262.01 ms; process-tree RSS: 45859.79 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 125.79377839695839,
            "range": "125.6-127.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9612.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 192.78894787381464,
            "range": "192.2-197.7",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9593.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 53.97164785207964,
            "range": "54.0-54.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8077.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 85.78807770327228,
            "range": "85.6-86.1",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8070.22 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 192.43475490050795,
            "range": "192.3-192.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2291.5192358994536,
            "range": "2274.6-2312.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2679.5518289417428,
            "range": "2678.1-2681.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1233.0679688379817,
            "range": "1220.8-1237.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.559736961182,
            "range": "1225.6-1229.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 195.3417848367289,
            "range": "189.9-197.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1962.8244348966143,
            "range": "1956.6-1990.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2463.994489507259,
            "range": "2440.9-2470.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2277.1156960732565,
            "range": "2268.9-2296.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "apaninga@berkeley.edu",
            "name": "theap06",
            "username": "theap06"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a2acf99c7aebf0a72378d377b0738ae865606109",
          "message": "[Feature] Transformer Module  (#4193)\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-09-11T00:06:23-07:00",
          "tree_id": "d5dfb75760d34d2d152a1483ffceec5a558458a0",
          "url": "https://github.com/pytorch/rl/commit/a2acf99c7aebf0a72378d377b0738ae865606109"
        },
        "date": 1789114877523,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4896.822585241005,
            "range": "4654.7-5456.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 105.54 ms; process-tree RSS: 25433.47 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 6080.876621072509,
            "range": "4721.4-11111.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 117.76 ms; process-tree RSS: 25471.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4200.3737774800675,
            "range": "3804.2-4746.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 99.77 ms; process-tree RSS: 25499.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 4760.27331016048,
            "range": "4530.2-5313.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 99.90 ms; process-tree RSS: 25531.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 986.1098179257514,
            "range": "981.1-1048.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 367.79 ms; process-tree RSS: 25083.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 975.6064414892558,
            "range": "973.5-985.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 368.00 ms; process-tree RSS: 25162.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 982.080416924106,
            "range": "981.0-1046.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 364.46 ms; process-tree RSS: 25169.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1068.1487508748505,
            "range": "1052.5-1110.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 359.37 ms; process-tree RSS: 25083.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.01847281233451,
            "range": "76.3-77.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3474.03 ms; process-tree RSS: 23772.29 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.64182767720266,
            "range": "76.6-78.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3486.10 ms; process-tree RSS: 23759.52 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 876.7174412008806,
            "range": "863.0-884.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 329.37 ms; process-tree RSS: 7469.79 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 878.3726975757808,
            "range": "864.7-891.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 334.84 ms; process-tree RSS: 7463.64 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 771.5810129950321,
            "range": "770.1-778.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 372.97 ms; process-tree RSS: 7461.07 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 789.0438432898721,
            "range": "783.8-790.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 373.71 ms; process-tree RSS: 7432.38 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 865.8021483408131,
            "range": "861.9-872.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 341.50 ms; process-tree RSS: 23844.39 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 796.3018887990014,
            "range": "784.4-798.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 371.78 ms; process-tree RSS: 23913.86 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 795.7865353536736,
            "range": "772.2-796.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 367.69 ms; process-tree RSS: 23917.56 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 875.514540921734,
            "range": "867.7-885.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 332.38 ms; process-tree RSS: 23838.90 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1675.7833892529288,
            "range": "1651.9-1704.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.73 ms; process-tree RSS: 23701.48 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4736.439877907731,
            "range": "4697.3-4844.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.72 ms; process-tree RSS: 23696.81 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 945.1792796439777,
            "range": "942.0-967.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 380.41 ms; process-tree RSS: 47134.69 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1138.3078776380364,
            "range": "1080.7-1139.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 282.47 ms; process-tree RSS: 45881.80 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 118.69751780991108,
            "range": "117.8-122.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9608.46 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 184.5727675952552,
            "range": "183.8-189.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9591.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.18148269514149,
            "range": "51.9-52.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8073.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 83.46822337568322,
            "range": "83.0-84.8",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8074.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.63489073848856,
            "range": "191.1-192.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2284.626991950983,
            "range": "2252.9-2317.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2623.934032084573,
            "range": "2617.6-2645.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.0955050185783,
            "range": "1218.8-1234.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1228.4751361386595,
            "range": "1206.6-1230.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 188.6593339922531,
            "range": "188.5-192.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1929.0055648914915,
            "range": "1890.8-1940.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2411.7782156780213,
            "range": "2398.9-2437.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2243.762940502935,
            "range": "2236.2-2267.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ed803e488dbfd0632453e2fbc304831d233540d8",
          "message": "[BugFix] TransformedEnv wraps a transformed env without a transform (#4336)",
          "timestamp": "2026-09-11T19:44:22+01:00",
          "tree_id": "6071759645edc36a8512f94b5d1e4da5c36ba4e9",
          "url": "https://github.com/pytorch/rl/commit/ed803e488dbfd0632453e2fbc304831d233540d8"
        },
        "date": 1789157078797,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 5733.41378402045,
            "range": "4822.4-22457.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 104.66 ms; process-tree RSS: 25477.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 6251.094666475902,
            "range": "5336.5-17772.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 151.72 ms; process-tree RSS: 25507.79 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4243.3259868676105,
            "range": "4211.0-4343.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 93.78 ms; process-tree RSS: 25511.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 4270.296044652388,
            "range": "4080.1-5282.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 116.99 ms; process-tree RSS: 25545.08 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 989.9430171108542,
            "range": "964.3-1019.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.55 ms; process-tree RSS: 25091.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 977.6007038250692,
            "range": "970.5-983.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 363.83 ms; process-tree RSS: 25169.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 970.2272705649422,
            "range": "951.1-1045.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 369.75 ms; process-tree RSS: 25164.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 968.3903726550252,
            "range": "966.7-1070.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.62 ms; process-tree RSS: 25090.67 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.57348804582952,
            "range": "77.3-77.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3504.39 ms; process-tree RSS: 23760.04 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.72936928786193,
            "range": "77.7-78.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3453.44 ms; process-tree RSS: 23756.84 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 901.1457104079118,
            "range": "853.0-904.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 323.40 ms; process-tree RSS: 7457.88 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 888.6715562373403,
            "range": "877.3-903.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 324.79 ms; process-tree RSS: 7449.48 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 774.7632803670534,
            "range": "773.2-789.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 378.14 ms; process-tree RSS: 7446.04 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 782.9096984361826,
            "range": "768.7-785.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 380.55 ms; process-tree RSS: 7436.75 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 875.721623694514,
            "range": "873.7-887.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 333.00 ms; process-tree RSS: 23838.52 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 786.9999378379432,
            "range": "783.8-796.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 364.69 ms; process-tree RSS: 23910.24 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 801.0078537390374,
            "range": "792.3-808.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 367.20 ms; process-tree RSS: 23907.66 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 876.7082957587351,
            "range": "872.9-899.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.97 ms; process-tree RSS: 23825.28 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1662.6822829316568,
            "range": "1660.3-1671.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.70 ms; process-tree RSS: 23700.81 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4648.191946611349,
            "range": "4617.4-4657.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 58.12 ms; process-tree RSS: 23699.08 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1034.0524049943808,
            "range": "936.1-1049.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 377.36 ms; process-tree RSS: 47117.51 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1119.4834846636068,
            "range": "1108.2-1141.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 271.39 ms; process-tree RSS: 45869.23 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 120.36430070948609,
            "range": "119.6-123.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9602.38 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 190.2738443298023,
            "range": "190.2-191.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9599.81 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 53.18439790605381,
            "range": "53.0-53.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8070.30 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.67693445408462,
            "range": "84.2-85.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8082.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.77254171656227,
            "range": "191.2-191.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2311.2270753256134,
            "range": "2281.5-2327.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2662.56048530059,
            "range": "2630.0-2667.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1227.837488370509,
            "range": "1226.4-1235.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1223.4839368456378,
            "range": "1223.2-1228.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.6813592489116,
            "range": "186.7-190.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 2036.9526241083583,
            "range": "1957.7-2055.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2459.684831815702,
            "range": "2442.4-2460.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2259.885060568089,
            "range": "2256.9-2261.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "44c871df8915eb8d2959c14828e88e2658478c9a",
          "message": "[BugFix] ParallelEnv without buffers keeps the state of workers it does not reset (#4339)",
          "timestamp": "2026-09-12T06:28:14+01:00",
          "tree_id": "0af9eb3dc30224936ea1921eeddd075bc233704f",
          "url": "https://github.com/pytorch/rl/commit/44c871df8915eb8d2959c14828e88e2658478c9a"
        },
        "date": 1789195007319,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4157.830920676267,
            "range": "4055.2-4511.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 118.82 ms; process-tree RSS: 25429.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4690.613973337401,
            "range": "4386.5-6244.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 221.24 ms; process-tree RSS: 25481.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4492.181141571,
            "range": "4224.5-5184.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 92.14 ms; process-tree RSS: 25546.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5219.333254443856,
            "range": "5184.6-5313.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 83.09 ms; process-tree RSS: 25536.90 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 980.6517233034068,
            "range": "979.6-989.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 363.31 ms; process-tree RSS: 25089.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 977.8679929757454,
            "range": "971.8-985.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 360.89 ms; process-tree RSS: 25166.96 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 996.657501755491,
            "range": "977.8-1066.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.22 ms; process-tree RSS: 25162.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1079.0211827093126,
            "range": "1061.6-1118.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 349.25 ms; process-tree RSS: 25093.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.78954339701859,
            "range": "76.3-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3473.89 ms; process-tree RSS: 23767.95 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.86186035638283,
            "range": "77.7-78.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3426.27 ms; process-tree RSS: 23757.76 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 885.4703070948583,
            "range": "862.8-891.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 332.69 ms; process-tree RSS: 7458.54 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 880.9184814649301,
            "range": "875.2-888.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 331.75 ms; process-tree RSS: 7459.90 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 775.6277080623234,
            "range": "765.1-788.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 370.91 ms; process-tree RSS: 7452.74 MiB; CUDA peak: 53.41 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 780.041041722389,
            "range": "777.0-793.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 371.22 ms; process-tree RSS: 7446.59 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 875.1423219430363,
            "range": "865.8-885.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 339.84 ms; process-tree RSS: 23841.69 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 782.2401683327805,
            "range": "779.0-786.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 370.55 ms; process-tree RSS: 23913.36 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 792.0696409595222,
            "range": "784.9-808.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 371.66 ms; process-tree RSS: 23906.37 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 873.99949942193,
            "range": "864.9-877.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.57 ms; process-tree RSS: 23836.81 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1659.349349253463,
            "range": "1656.9-1666.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.73 ms; process-tree RSS: 23703.36 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4732.9277944474325,
            "range": "4677.1-4749.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.49 ms; process-tree RSS: 23696.12 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1041.5274159854634,
            "range": "1030.2-1070.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.28 ms; process-tree RSS: 47117.63 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1120.78449678605,
            "range": "1085.3-1151.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 276.03 ms; process-tree RSS: 45869.40 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 120.4446555657358,
            "range": "118.2-120.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9618.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 190.2617302568134,
            "range": "179.9-190.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9596.78 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.55441847776755,
            "range": "52.5-52.7",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8065.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.20099892428988,
            "range": "84.2-84.8",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8075.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 188.92243223072725,
            "range": "188.7-189.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2220.4626742080723,
            "range": "2202.4-2228.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2631.40465197507,
            "range": "2611.9-2643.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1230.3738312092328,
            "range": "1225.6-1232.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.0474858132113,
            "range": "1220.3-1226.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 194.09505979481668,
            "range": "186.9-202.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1948.0129929801342,
            "range": "1811.0-1999.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2391.04734526147,
            "range": "2385.6-2408.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2240.6862917519934,
            "range": "2237.8-2249.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vincentmoens@gmail.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46c5daee9e28e47560a6adfa839a5ce0d9176e25",
          "message": "[Feature] Shared buffers for ParallelEnv workers that report their own metadata (#4341)",
          "timestamp": "2026-09-12T08:49:35+01:00",
          "tree_id": "4eb265147caf5bbc95988160dfdc1572bf29ae97",
          "url": "https://github.com/pytorch/rl/commit/46c5daee9e28e47560a6adfa839a5ce0d9176e25"
        },
        "date": 1789203735826,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 10971.443539597582,
            "range": "5943.5-21050.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 172.23 ms; process-tree RSS: 25457.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4753.303479773123,
            "range": "4047.8-4816.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 112.72 ms; process-tree RSS: 25507.78 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4673.852395224082,
            "range": "4142.0-5153.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 81.92 ms; process-tree RSS: 25487.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5313.312550256431,
            "range": "4253.0-6226.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 90.72 ms; process-tree RSS: 25545.59 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1024.1618984886513,
            "range": "980.0-1089.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.50 ms; process-tree RSS: 25078.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 975.0656618404721,
            "range": "962.3-1069.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 366.67 ms; process-tree RSS: 25166.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1112.5106400805798,
            "range": "1010.5-1117.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 365.10 ms; process-tree RSS: 25162.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 988.179898687255,
            "range": "962.7-1028.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.65 ms; process-tree RSS: 25085.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.07350706849296,
            "range": "76.1-77.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3529.09 ms; process-tree RSS: 23767.30 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.95262993524204,
            "range": "77.8-78.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3454.50 ms; process-tree RSS: 23765.42 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 888.5127680138326,
            "range": "876.8-893.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 336.88 ms; process-tree RSS: 7459.74 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 877.8784166220173,
            "range": "874.6-896.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 331.75 ms; process-tree RSS: 7456.76 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 776.3689939838866,
            "range": "750.7-791.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 366.34 ms; process-tree RSS: 7443.93 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 771.6052656112562,
            "range": "769.0-780.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 386.97 ms; process-tree RSS: 7424.25 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 864.528004886671,
            "range": "857.5-889.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 347.48 ms; process-tree RSS: 23841.39 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 784.0875388163047,
            "range": "777.6-785.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 370.21 ms; process-tree RSS: 23907.98 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 794.7216130235857,
            "range": "782.5-797.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 362.85 ms; process-tree RSS: 23922.45 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 874.0242126744126,
            "range": "865.1-876.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 342.48 ms; process-tree RSS: 23837.78 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1688.8218404630475,
            "range": "1677.8-1696.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.04 ms; process-tree RSS: 23700.70 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4843.591651299283,
            "range": "4670.0-4851.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 55.45 ms; process-tree RSS: 23695.32 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1006.2913828609345,
            "range": "942.6-1071.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.67 ms; process-tree RSS: 47140.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1105.1827899861032,
            "range": "1074.0-1113.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 279.14 ms; process-tree RSS: 45874.97 MiB; CUDA peak: 70.91 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 121.9895829754682,
            "range": "121.3-127.0",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9608.42 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 187.4871331233322,
            "range": "185.6-192.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9598.14 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.541240937648276,
            "range": "52.2-52.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8074.07 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.47314151012522,
            "range": "84.1-84.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8071.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.06067188783248,
            "range": "189.5-190.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2225.834285620222,
            "range": "2152.8-2234.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2599.8617234090048,
            "range": "2597.0-2627.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1229.277148350795,
            "range": "1221.0-1230.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1225.0501811770357,
            "range": "1221.3-1226.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.61734491879554,
            "range": "186.5-192.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1891.3065253347377,
            "range": "1887.1-1952.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2407.1080320521837,
            "range": "2328.8-2411.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2222.2998290990845,
            "range": "2210.6-2235.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7ef78484cda129fb09e40a405457b3376cd98c66",
          "message": "[Doc] Document NoisyLinear reset_noise caller contract (#4357)",
          "timestamp": "2026-09-12T13:54:44+01:00",
          "tree_id": "280a1b58db5cd35b7dc2e8b32895bfd5a02a24d2",
          "url": "https://github.com/pytorch/rl/commit/7ef78484cda129fb09e40a405457b3376cd98c66"
        },
        "date": 1789221747658,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4767.1771658548605,
            "range": "4101.5-4769.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 101.75 ms; process-tree RSS: 25405.23 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4935.638169679087,
            "range": "4870.1-5722.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 110.54 ms; process-tree RSS: 25438.45 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4677.202525154743,
            "range": "4342.7-5197.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 89.10 ms; process-tree RSS: 25507.49 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5538.713143823989,
            "range": "4718.7-6067.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 76.99 ms; process-tree RSS: 25495.91 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1096.8463799579622,
            "range": "983.3-1112.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 355.85 ms; process-tree RSS: 25086.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 978.3278007159985,
            "range": "974.7-1095.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 361.11 ms; process-tree RSS: 25160.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 1070.2267301770837,
            "range": "979.2-1078.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 359.45 ms; process-tree RSS: 25156.17 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1042.8351025926622,
            "range": "975.8-1056.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 366.81 ms; process-tree RSS: 25081.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.74438217205102,
            "range": "76.1-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3472.96 ms; process-tree RSS: 23762.62 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.57818020127317,
            "range": "77.1-77.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3443.68 ms; process-tree RSS: 23761.28 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 893.8015777080764,
            "range": "887.2-894.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 324.83 ms; process-tree RSS: 7456.03 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 896.0275073725614,
            "range": "884.4-896.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 337.07 ms; process-tree RSS: 7446.75 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 785.8486420965255,
            "range": "773.5-790.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 364.64 ms; process-tree RSS: 7454.09 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 782.3454597239037,
            "range": "766.6-807.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 384.43 ms; process-tree RSS: 7447.32 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 890.9302025539022,
            "range": "888.2-891.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.86 ms; process-tree RSS: 23847.02 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 793.9940023335383,
            "range": "787.2-799.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 376.28 ms; process-tree RSS: 23910.65 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 785.6499966260576,
            "range": "768.9-803.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 370.74 ms; process-tree RSS: 23903.38 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 882.2111759523042,
            "range": "870.9-890.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 337.03 ms; process-tree RSS: 23831.63 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1656.4789303119308,
            "range": "1652.8-1682.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.44 ms; process-tree RSS: 23701.99 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4682.211000578652,
            "range": "4674.4-4761.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.49 ms; process-tree RSS: 23696.12 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1056.5508504170807,
            "range": "949.0-1076.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 376.61 ms; process-tree RSS: 47108.39 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1130.7836622087714,
            "range": "1126.9-1133.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 270.61 ms; process-tree RSS: 45867.36 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 119.69320837632412,
            "range": "118.9-120.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9616.52 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 188.02313113221038,
            "range": "186.9-188.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9591.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.31109353789848,
            "range": "51.6-52.7",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8072.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.29885235170399,
            "range": "84.0-84.7",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8065.15 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.15875531078385,
            "range": "189.7-190.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2270.6241131904717,
            "range": "2255.3-2289.8",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2632.393516619771,
            "range": "2624.4-2633.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1226.4693706715477,
            "range": "1226.3-1230.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.3724176072178,
            "range": "1223.5-1224.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 185.90609821108274,
            "range": "185.4-186.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1959.5132341028159,
            "range": "1827.9-2026.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2392.049931416599,
            "range": "2389.5-2415.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2232.94706503708,
            "range": "2231.5-2243.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "85978465+yupengtang@users.noreply.github.com",
            "name": "David",
            "username": "yupengtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "021cf60155dd76bdf7089243e8a333f487537323",
          "message": "[BugFix] Only forward the applicable SACLossConfig fields to each SAC variant (#4338)",
          "timestamp": "2026-09-12T14:05:16+01:00",
          "tree_id": "8dc0180cb8f6a2875af83801f7acd4f964077336",
          "url": "https://github.com/pytorch/rl/commit/021cf60155dd76bdf7089243e8a333f487537323"
        },
        "date": 1789222378318,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4607.986546978865,
            "range": "4433.2-6370.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 96.05 ms; process-tree RSS: 25899.70 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4964.974866043405,
            "range": "4878.5-4984.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 130.00 ms; process-tree RSS: 25785.78 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 5219.138581172673,
            "range": "4601.8-5229.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 94.66 ms; process-tree RSS: 25588.32 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5482.969569899702,
            "range": "4345.2-5766.1",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 92.71 ms; process-tree RSS: 25547.16 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 985.3892401584078,
            "range": "966.2-986.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.54 ms; process-tree RSS: 25085.01 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 979.3889658041766,
            "range": "975.2-981.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 364.81 ms; process-tree RSS: 25164.95 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 999.5962246236312,
            "range": "985.4-1114.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 360.81 ms; process-tree RSS: 25161.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1073.3328620312052,
            "range": "1052.1-1091.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 360.75 ms; process-tree RSS: 25081.93 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.2519708693861,
            "range": "76.9-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3459.40 ms; process-tree RSS: 23769.43 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.28821106661306,
            "range": "77.2-77.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3450.84 ms; process-tree RSS: 23759.12 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 888.9294594134263,
            "range": "877.9-896.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 324.90 ms; process-tree RSS: 7464.83 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 870.5083972725739,
            "range": "869.8-876.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 343.66 ms; process-tree RSS: 7454.85 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 768.4273331979782,
            "range": "766.4-786.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 365.82 ms; process-tree RSS: 7451.80 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 776.2237616356234,
            "range": "770.3-786.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 371.06 ms; process-tree RSS: 7445.46 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 873.3644603325831,
            "range": "861.5-879.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 341.03 ms; process-tree RSS: 23838.61 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 789.8011874350211,
            "range": "774.9-792.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 372.92 ms; process-tree RSS: 23907.88 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 804.6784653490689,
            "range": "784.7-805.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 366.75 ms; process-tree RSS: 24348.43 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 871.7678488688656,
            "range": "871.1-894.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 322.41 ms; process-tree RSS: 23835.93 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1656.5740030343632,
            "range": "1655.2-1657.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.58 ms; process-tree RSS: 23698.60 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4761.35697558107,
            "range": "4730.9-4804.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.62 ms; process-tree RSS: 23699.57 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1050.7847254172716,
            "range": "943.8-1061.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 369.57 ms; process-tree RSS: 47119.03 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1137.406556081052,
            "range": "1102.3-1152.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 276.30 ms; process-tree RSS: 45870.66 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 122.09103638792365,
            "range": "121.3-125.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9612.00 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 189.86777554155194,
            "range": "179.4-190.9",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9599.72 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.997363385364224,
            "range": "52.8-53.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8081.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.07628539615143,
            "range": "84.0-84.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8078.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.7633337197138,
            "range": "190.7-192.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2232.720091249286,
            "range": "2231.4-2335.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2644.73242676426,
            "range": "2637.7-2650.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.3283633628923,
            "range": "1229.3-1231.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1224.991918583237,
            "range": "1222.9-1228.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 189.38830498397627,
            "range": "185.9-190.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1922.4002777476783,
            "range": "1914.4-1967.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2424.6807093897755,
            "range": "2416.3-2477.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2242.995733536556,
            "range": "2237.7-2250.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e3ff3568640a0aafbf7efa76cdec116cc479a6aa",
          "message": "[BugFix] Import ale_py for classic Atari environment ids (#4358)",
          "timestamp": "2026-09-12T15:41:00+01:00",
          "tree_id": "f0bdebf9bd2d7e5142a93e3e280d5c3420de4bc6",
          "url": "https://github.com/pytorch/rl/commit/e3ff3568640a0aafbf7efa76cdec116cc479a6aa"
        },
        "date": 1789228385718,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 6348.807619838584,
            "range": "4114.4-22199.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 160.47 ms; process-tree RSS: 25468.58 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4490.5371183167435,
            "range": "4328.5-4641.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 98.34 ms; process-tree RSS: 25464.89 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4920.865963806339,
            "range": "4168.8-5086.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 82.01 ms; process-tree RSS: 25516.12 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5177.551095084687,
            "range": "4993.5-5311.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 91.74 ms; process-tree RSS: 25539.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1039.083109086735,
            "range": "938.4-1096.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 358.56 ms; process-tree RSS: 25089.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1086.2820843291274,
            "range": "981.6-1122.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 355.54 ms; process-tree RSS: 25161.75 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 985.4326010287751,
            "range": "976.5-1084.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 364.03 ms; process-tree RSS: 25163.31 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 982.0264079639003,
            "range": "936.1-1035.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 364.02 ms; process-tree RSS: 25091.25 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 76.88408152715702,
            "range": "76.8-77.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3489.64 ms; process-tree RSS: 23759.86 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.85686424910148,
            "range": "77.6-78.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3437.77 ms; process-tree RSS: 23758.20 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 877.635989550983,
            "range": "845.5-884.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 334.20 ms; process-tree RSS: 7459.34 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 896.1298491029755,
            "range": "875.1-896.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 333.90 ms; process-tree RSS: 7442.20 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 777.9513135946365,
            "range": "776.1-791.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 363.27 ms; process-tree RSS: 7448.71 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 781.8129707126448,
            "range": "765.8-796.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 367.95 ms; process-tree RSS: 7440.89 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 872.9739225808951,
            "range": "867.4-890.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 339.97 ms; process-tree RSS: 23834.27 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 789.2104900568931,
            "range": "783.8-789.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 365.71 ms; process-tree RSS: 23908.79 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 795.7924216500116,
            "range": "786.2-810.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 382.74 ms; process-tree RSS: 23901.82 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 880.7371042955305,
            "range": "868.9-900.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 331.38 ms; process-tree RSS: 23828.98 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1667.1542903023214,
            "range": "1645.3-1694.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 254.63 ms; process-tree RSS: 23703.20 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4701.749091982943,
            "range": "4662.6-4837.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.94 ms; process-tree RSS: 23696.51 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1071.2076520362923,
            "range": "1065.4-1073.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 370.92 ms; process-tree RSS: 47117.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1134.7379861843644,
            "range": "1079.6-1150.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 284.69 ms; process-tree RSS: 45873.40 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 122.8774362556799,
            "range": "119.2-123.8",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9614.86 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 190.88380597819233,
            "range": "188.1-191.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9589.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.079861641884605,
            "range": "51.7-52.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8069.43 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.45962692751019,
            "range": "82.6-84.6",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8076.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 190.0899819795896,
            "range": "189.8-191.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2248.346500882287,
            "range": "2230.9-2294.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2623.4461429695148,
            "range": "2616.1-2628.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1227.656672537543,
            "range": "1224.1-1230.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1227.2616961904423,
            "range": "1223.6-1227.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 194.40872816377936,
            "range": "190.0-197.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1963.9403520323026,
            "range": "1937.9-1978.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2437.784945799447,
            "range": "2406.7-2468.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2245.2638231696283,
            "range": "2241.2-2246.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d282ef6fe84b03487950c0035d2d930fecdf9096",
          "message": "[BugFix] Stop caching Tokenizer parent.device (#4352)",
          "timestamp": "2026-09-12T15:41:52+01:00",
          "tree_id": "cffecb57437f99c9187fa7dc259132b94e546c26",
          "url": "https://github.com/pytorch/rl/commit/d282ef6fe84b03487950c0035d2d930fecdf9096"
        },
        "date": 1789228387996,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 5272.298677529069,
            "range": "4516.5-5911.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 115.72 ms; process-tree RSS: 25473.10 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 4781.8668545446935,
            "range": "4270.3-11069.2",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 94.70 ms; process-tree RSS: 25464.61 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 4411.124791243311,
            "range": "4127.4-5067.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 88.98 ms; process-tree RSS: 25569.06 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5359.038968558951,
            "range": "5238.7-5652.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 74.39 ms; process-tree RSS: 25529.97 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1003.286815616568,
            "range": "964.1-1066.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.00 ms; process-tree RSS: 25095.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1001.6439765430882,
            "range": "998.8-1001.9",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 358.24 ms; process-tree RSS: 25162.73 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 987.4671936686381,
            "range": "971.4-997.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 364.61 ms; process-tree RSS: 25175.77 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1058.0254400604792,
            "range": "946.8-1095.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 357.57 ms; process-tree RSS: 25098.37 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.03495720107601,
            "range": "76.9-77.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3469.48 ms; process-tree RSS: 23759.70 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.97397814876089,
            "range": "77.4-78.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3402.23 ms; process-tree RSS: 23757.37 MiB; CUDA peak: 27.76 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 885.3122917841546,
            "range": "876.7-898.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 327.61 ms; process-tree RSS: 7467.91 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 893.922509896295,
            "range": "891.8-900.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 325.43 ms; process-tree RSS: 7466.86 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 784.767909316757,
            "range": "781.1-787.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 376.78 ms; process-tree RSS: 7464.10 MiB; CUDA peak: 53.43 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 807.4424277345349,
            "range": "796.1-811.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 366.17 ms; process-tree RSS: 7433.73 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 870.084763504524,
            "range": "867.0-877.6",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 327.85 ms; process-tree RSS: 23843.93 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 790.2501809674786,
            "range": "780.1-799.8",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 372.00 ms; process-tree RSS: 23916.95 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 800.2783086608562,
            "range": "791.7-815.5",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 366.23 ms; process-tree RSS: 23950.16 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 882.445106200731,
            "range": "881.6-885.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 343.58 ms; process-tree RSS: 23831.20 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1658.8514433038695,
            "range": "1655.6-1661.4",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.38 ms; process-tree RSS: 23703.34 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4709.330682418853,
            "range": "4681.1-4759.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 57.38 ms; process-tree RSS: 23709.31 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 1054.6303625891828,
            "range": "1054.0-1089.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 365.25 ms; process-tree RSS: 47111.29 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1109.7414955155346,
            "range": "1053.3-1117.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 322.82 ms; process-tree RSS: 45872.99 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 127.43708333844134,
            "range": "126.5-129.8",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9613.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 193.7455494989299,
            "range": "190.9-195.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9598.85 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 53.97961544060713,
            "range": "53.7-54.2",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8075.88 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 86.28010164863593,
            "range": "85.1-86.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8072.71 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 191.19637536872577,
            "range": "189.7-192.1",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2230.992759249198,
            "range": "2189.1-2253.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2637.3407866952393,
            "range": "2634.1-2643.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1225.1528796374475,
            "range": "1223.0-1228.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.2221249981947,
            "range": "1219.2-1229.2",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 197.62623936784618,
            "range": "189.1-198.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1924.744664011425,
            "range": "1820.5-1985.3",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2442.837501087494,
            "range": "2395.1-2455.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2242.942638758149,
            "range": "2239.5-2261.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "neos960518@gmail.com",
            "name": "Yeonwoo Sung",
            "username": "YeonwooSung"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "08280d5c2a71ceec80aa49b68cea440a5d2635a0",
          "message": "[BugFix] Accept History on ChatEnv.reset (#4355)",
          "timestamp": "2026-09-12T15:41:12+01:00",
          "tree_id": "2bf1ac5306a888f1c5cb7bf0db08003963129165",
          "url": "https://github.com/pytorch/rl/commit/08280d5c2a71ceec80aa49b68cea440a5d2635a0"
        },
        "date": 1789228404539,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-slow-reset]",
            "value": 4754.835062450707,
            "range": "3704.4-5825.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 107.00 ms; process-tree RSS: 25448.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-chunked-uniform]",
            "value": 3918.715642783326,
            "range": "2958.5-4661.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 203.18 ms; process-tree RSS: 25493.28 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-slow-reset]",
            "value": 5041.715791592774,
            "range": "4255.6-5159.3",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 77.22 ms; process-tree RSS: 25484.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-integrated-uniform]",
            "value": 5514.250819253741,
            "range": "4105.7-7660.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting; 64 transitions/message. Batch p95: 88.24 ms; process-tree RSS: 25553.53 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-slow-reset]",
            "value": 1000.2271746437438,
            "range": "987.8-1069.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 362.36 ms; process-tree RSS: 25082.18 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-slow-reset]",
            "value": 1083.4676450544878,
            "range": "987.5-1106.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 356.51 ms; process-tree RSS: 25162.99 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-static-uniform]",
            "value": 986.9736385490897,
            "range": "976.7-1016.0",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker; process acting. Batch p95: 363.78 ms; process-tree RSS: 25163.20 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-process-slots-uniform]",
            "value": 1005.9880110099075,
            "range": "1000.4-1095.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 361.09 ms; process-tree RSS: 25087.21 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-slow-reset]",
            "value": 77.06291267242322,
            "range": "76.6-77.7",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3542.49 ms; process-tree RSS: 23757.84 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-queue-uniform]",
            "value": 77.65925858145003,
            "range": "77.0-77.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 3501.38 ms; process-tree RSS: 23756.61 MiB; CUDA peak: 27.58 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-slow-reset]",
            "value": 874.1986185407941,
            "range": "858.5-874.5",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 323.95 ms; process-tree RSS: 7466.69 MiB; CUDA peak: 48.37 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-grouped-uniform]",
            "value": 901.1773796219254,
            "range": "884.8-911.3",
            "unit": "frames/s",
            "extra": "Execution: eager; 4 envs/worker. Batch p95: 333.68 ms; process-tree RSS: 7461.52 MiB; CUDA peak: 48.19 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-slow-reset]",
            "value": 765.1788460437682,
            "range": "753.7-766.2",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 384.65 ms; process-tree RSS: 7445.90 MiB; CUDA peak: 53.41 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-integrated-uniform]",
            "value": 777.5136651766214,
            "range": "768.6-783.6",
            "unit": "frames/s",
            "extra": "Execution: graph; 4 envs/worker. Batch p95: 357.95 ms; process-tree RSS: 7443.29 MiB; CUDA peak: 45.31 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-slow-reset]",
            "value": 874.9117297155982,
            "range": "874.8-902.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 333.52 ms; process-tree RSS: 23842.57 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-slow-reset]",
            "value": 772.5816098173583,
            "range": "765.0-773.4",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 373.67 ms; process-tree RSS: 23906.68 MiB; CUDA peak: 69.68 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-static-uniform]",
            "value": 800.5714053388833,
            "range": "777.4-801.7",
            "unit": "frames/s",
            "extra": "Execution: graph; 1 envs/worker. Batch p95: 357.27 ms; process-tree RSS: 23916.68 MiB; CUDA peak: 61.56 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[async-shm-uniform]",
            "value": 877.2075597497522,
            "range": "869.8-886.1",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 326.54 ms; process-tree RSS: 23828.88 MiB; CUDA peak: 32.12 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-slow-reset]",
            "value": 1675.5006730616137,
            "range": "1664.8-1675.9",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 255.40 ms; process-tree RSS: 23691.76 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels[parallel-uniform]",
            "value": 4689.167818561178,
            "range": "4680.4-4817.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 56.78 ms; process-tree RSS: 23700.72 MiB; CUDA peak: 24.83 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-process-slots]",
            "value": 971.7487868932413,
            "range": "944.0-1060.8",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker; process acting. Batch p95: 368.75 ms; process-tree RSS: 47120.40 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_collectors_benchmark.py::test_async_collection_pixels_64_envs[async-shm]",
            "value": 1115.0925553718296,
            "range": "1103.1-1119.0",
            "unit": "frames/s",
            "extra": "Execution: eager; 1 envs/worker. Batch p95: 257.85 ms; process-tree RSS: 45869.93 MiB; CUDA peak: 71.07 MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio16]",
            "value": 123.24504948506137,
            "range": "122.0-123.5",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9607.68 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[process-ratio2]",
            "value": 188.6756874502821,
            "range": "186.4-196.4",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; process inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 9602.02 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio16]",
            "value": 52.77604074539123,
            "range": "52.7-53.3",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 16; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8075.80 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_dreamer_v3_benchmark.py::test_dreamer_v3_async_training[thread-ratio2]",
            "value": 84.64751993100421,
            "range": "84.4-84.7",
            "unit": "frames/s",
            "extra": "Execution: eager learner on cuda:0; thread inference; train ratio 2; 8 envs; inference batch <= 1. Batch p95: - ms; process-tree RSS: 8066.54 MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[queue]",
            "value": 189.088372811495,
            "range": "189.0-189.6",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_dispatch[shm]",
            "value": 2205.583179228786,
            "range": "2200.2-2228.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_fast_step_slow_reset[shm]",
            "value": 2609.308344406496,
            "range": "2602.4-2617.7",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[16-processes-of-4]",
            "value": 1231.416920548099,
            "range": "1222.9-1233.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_multi_env_workers[64-processes]",
            "value": 1226.2578605256608,
            "range": "1225.7-1227.0",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[queue]",
            "value": 186.6628162891471,
            "range": "186.3-193.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_per_env_dispatch[shm]",
            "value": 1967.8035062109434,
            "range": "1954.8-1973.9",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[default]",
            "value": 2415.623193034613,
            "range": "2398.3-2511.4",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          },
          {
            "name": "benchmarks/test_envs_benchmark.py::test_async_env_pool_step_latency_jitter[pinned]",
            "value": 2227.6361006713882,
            "range": "2224.2-2230.5",
            "unit": "frames/s",
            "extra": "Execution: pool. Batch p95: - ms; process-tree RSS: - MiB; CUDA peak: - MiB. Three independent run medians."
          }
        ]
      }
    ]
  }
}