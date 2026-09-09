window.BENCHMARK_DATA = {
  "lastUpdate": 1788953924893,
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
      }
    ]
  }
}