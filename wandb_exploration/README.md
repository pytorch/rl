# W&B Logging Exploration for RL Experiments

This directory contains demonstration scripts exploring different W&B logging strategies for RL experiments with separate training and inference pipelines using Ray actors.

## Overview

All demos use a common backbone (`common.py`) that provides:
- **WandbRun**: Thread-safe wrapper for W&B runs via Ray actors
- **TrainingPipelineActor / InferencePipelineActor**: Centralized logging actors
- **Dummy metric generators**: Realistic training and inference metrics

## Demos

### Demo 1: Centralized Logging via Ray Actors
**File:** `demo1_centralized_actors.py`

Shows how to structure logging where each pipeline owns its W&B run:
- Training actor owns `project="rl-training-demo"`
- Inference actor owns `project="rl-inference-demo"`
- Workers send metrics back to their respective actor

```bash
uv run python personal/vincent/wandb_exploration/demo1_centralized_actors.py
```

### Demo 2: Project Indicates Job Type
**File:** `demo2_project_per_job.py`

Uses separate projects for each job type:
- `project="rl-agent"` for training runs
- `project="inference-only"` for inference runs

```bash
uv run python personal/vincent/wandb_exploration/demo2_project_per_job.py
```

### Demo 3: Hierarchical Metrics in Single Project
**File:** `demo3_hierarchical_metrics.py`

Single project with metric path hierarchy:
- Training: `training/loss`, `training/grads/grad_norm_0`
- Inference: `inference/e2e/speed`, `inference/reward/mean`

```bash
uv run python personal/vincent/wandb_exploration/demo3_hierarchical_metrics.py
```

### Demo 4: Advanced Extensions
**File:** `demo4_extensions.py`

Combines all patterns with advanced features:
- `job_type` metadata for filtering
- `group` parameter to link related runs
- Artifact logging (checkpoints, trajectories)
- Cross-project comparison with consistent naming

```bash
uv run python personal/vincent/wandb_exploration/demo4_extensions.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ray Cluster                               │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐           │
│  │ TrainingPipelineActor│    │InferencePipelineActor│           │
│  │                      │    │                      │           │
│  │  wandb.init(...)     │    │  wandb.init(...)     │           │
│  │  wandb.log(...)      │    │  wandb.log(...)      │           │
│  │  wandb.finish()      │    │  wandb.finish()      │           │
│  └──────────┬───────────┘    └──────────┬───────────┘           │
│             │                           │                        │
│       ┌─────┴─────┐               ┌─────┴─────┐                 │
│       │           │               │           │                 │
│  ┌────┴────┐ ┌────┴────┐    ┌────┴────┐ ┌────┴────┐            │
│  │Worker 0 │ │Worker 1 │    │Worker 0 │ │Worker 1 │            │
│  │(metrics)│ │(metrics)│    │(metrics)│ │(metrics)│            │
│  └─────────┘ └─────────┘    └─────────┘ └─────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
            ┌───────────────────────────────────────┐
            │           Weights & Biases             │
            │                                        │
            │  ┌─────────────┐  ┌─────────────┐     │
            │  │rl-agent    │  │inference-only│     │
            │  │(training)   │  │(inference)   │     │
            │  └─────────────┘  └─────────────┘     │
            └───────────────────────────────────────┘
```

## Key W&B Concepts

### Projects
- Namespace for related runs
- Use for job type separation (Option 2) or unified view (Option 3)

### Metric Path Hierarchy
- Use `/` in metric names: `training/loss`, `inference/e2e/speed`
- W&B auto-groups panels by path prefix
- Great for single-project multi-pipeline setups

### Groups
- `group=experiment_id` links related runs
- View grouped runs together in W&B UI (View → Group by → Group)

### Job Types
- `job_type="training"` or `job_type="inference"`
- Filter runs in W&B UI by job type
- Useful within a single project

### Tags
- Flexible filtering: `["seed-42", "experiment-1"]`
- Filter in sidebar or search: `tags: seed-42`

## Dashboard Tips

### Organizing Panels
1. **By metric prefix**: W&B auto-creates sections for `training/*` and `inference/*`
2. **Custom sections**: Create workspace sections and drag panels
3. **Panel grouping**: Use "Group by" in View menu

### Cross-Project Comparison
1. Create a **Report**
2. Add panels from multiple projects
3. Use consistent metric names (`e2e/speed`) across projects
4. Filter by tags or experiment_id

### Filtering Runs
- **By project**: Switch projects in sidebar
- **By job_type**: Job Type dropdown or `job_type: training`
- **By group**: View → Group by → Group
- **By tags**: Tag filter in sidebar or `tags: demo4`

## Comparison of Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Separate Projects** | Clean separation, independent views | Cross-project comparison requires Reports |
| **Single Project + Hierarchy** | Easy comparison, auto-panel grouping | May get cluttered with many runs |
| **job_type + group** | Best of both: filtering + grouping | Requires more config |

## Recommendations

1. **For separate teams**: Use separate projects (Demo 2)
2. **For unified experiments**: Use hierarchical metrics (Demo 3)
3. **For complex pipelines**: Use job_type + group + artifacts (Demo 4)

## Dependencies

- `ray`
- `wandb`

Make sure you're logged into W&B:
```bash
wandb login
```

