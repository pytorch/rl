window.BENCHMARK_DATA = {
  "lastUpdate": 1788849206610,
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
      }
    ]
  }
}