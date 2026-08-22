# DualMemory

## A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents

<p align="center">
  <strong>Bin Wen</strong><sup>1,2</sup> ·
  <strong>Ruoxuan Zhang</strong><sup>3</sup> ·
  <strong>Yang Chen</strong><sup>1,2</sup> ·
  <strong>Hongxia Xie</strong><sup>3</sup> ·
  <strong>Lan-Zhe Guo</strong><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University<br>
  <sup>2</sup>School of Intelligence Science and Technology, Nanjing University<br>
  <sup>3</sup>Jilin University
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2604.02734">
    <img src="https://img.shields.io/badge/PAPER%20-b91c1c?style=for-the-badge&labelColor=7f1d1d" alt="Paper">
  </a>
  <a href="https://wenbin08.github.io/Aligning-Progress-and-Feasibility/">
    <img src="https://img.shields.io/badge/PROJECT_PAGE-4285f4?style=for-the-badge&labelColor=374151" alt="Project Page">
  </a>
  </a>
</p>

This repository contains the public implementation of the method in:

> **Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents**


## Overview

Long-horizon agents commonly fail in two different ways:

- **Progress Drift:** the agent loses track of the global task stage and enters
  irrelevant or repetitive interaction loops.
- **Feasibility Violation:** the agent proposes an action that is locally
  impossible under the current environment state.

DualAlign treats these as separate alignment problems and assigns each one a
different memory mechanism:

| Component | Role | Representation |
| --- | --- | --- |
| **Progress Memory** | Keeps execution aligned with the current semantic stage | Retrieved procedural blueprints and stage anchors |
| **Feasibility Memory** | Blocks locally invalid actions before execution | Executable Python verifier rules induced from failed transitions |

During inference, Progress Memory guides the candidate action toward the next
task stage. Feasibility Memory checks that candidate against the current
symbolic state and requests refinement when a hard precondition is violated.

## Results

The reported evaluation uses ALFWorld, WebShop, and TextCraft.

| Method | ALFWorld Success Rate (%) | WebShop Success Rate (%) | WebShop Score (%) | TextCraft Success Rate (%) |
| --- | ---: | ---: | ---: | ---: |
| ReAct | 78.1 +/- 2.2 | 34.0 +/- 2.0 | 52.1 +/- 2.3 | 60.0 +/- 2.6 |
| ADaPT | 71.9 +/- 0.4 | 33.0 +/- 1.0 | 52.3 +/- 1.2 | 73.7 +/- 2.9 |
| StateAct | 68.9 +/- 5.6 | 25.0 +/- 6.9 | 42.0 +/- 10.7 | 69.3 +/- 1.5 |
| ExpeL | 85.3 +/- 1.1 | 32.7 +/- 3.2 | 50.5 +/- 4.0 | 89.0 +/- 1.0 |
| WALL-E 2.0 | 83.8 +/- 0.9 | 36.3 +/- 2.5 | 59.9 +/- 1.2 | 64.7 +/- 1.5 |
| AWM | 87.3 +/- 0.7 | 40.3 +/- 2.1 | 61.1 +/- 1.7 | 65.3 +/- 6.1 |
| **Ours** | **95.3 +/- 1.6** | **50.0 +/- 1.0** | **71.0 +/- 0.5** | **94.0 +/- 1.0** |

*Table 1: Main results. Values are reported as mean +/- standard deviation.*

## Installation

Clone this repository and install the method dependencies:

```bash
cd DualMemory
pip install -r requirements.txt
```

The benchmark environments are not vendored here. Follow the
[StateAct repository setup](https://github.com/ai-nikolai/StateAct) for
ALFWorld, WebShop, and TextCraft. This includes the ALFWorld package and
assets, the WebShop server and data, and the TextCraft package and data.
Use the StateAct environment variables for those installations, including
`ALFWORLD_DATA` for the ALFWorld data root. Start the WebShop server before
running its S1 or S3 command.

### API Configuration

Each environment has an empty
configuration template:

```text
dualmemory/alfworld/alfworld_runs/api_config.yaml
dualmemory/webshop/our_design/api_config.yaml
dualmemory/textcraft/our_design/api_config.yaml
```

Fill in the template locally, or configure the client through environment
variables:

```yaml
api:
  gpt:
    api_key: "YOUR_API_KEY"
    api_base: "https://api.openai.com/v1"
  other:
    api_key: "YOUR_API_KEY"
    api_base: "https://your-provider.example/v1"
```

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

Keep local credential files and environment variables out of version control.

## Reproduction

The complete public workflow is **S1 -> S2 -> S3**. Run the stages in this
order; S3 cannot start until the S1 rules and S2 library exist. The stages are
separate:

1. **S1** collects trajectories on the disjoint construction task pool and
   builds Feasibility Memory by mining, verifying, and selecting executable
   rules.
2. **S2** reads the successful S1 trajectories and extracts/indexes the
   Progress Memory library.
3. **S3** loads the S1 rules and S2 library, then evaluates DualAlign on the
   fixed evaluation task pool.


The published task files determine the task instances and order used by the
workflow:

- ALFWorld uses ExpeL-compatible suffix files,
  `tasks/alfworld_s1_tasks_suffix.json` and
  `tasks/alfworld_s3_tasks_suffix.json`. Each record contains the relative
  ALFWorld `gamefile` path.
- WebShop uses `tasks/webshop_s1_fixed.json` and
  `tasks/webshop_s3_fixed.json`, which contain `session_idx` and the complete
  target key (`asin`, `query`, `name`, and `goal_options`). Task count and
  order are read from the file; `--start_task_id` and `--num_envs` are not
  needed.
- TextCraft uses `tasks/textcraft_s1_fixed.json` and
  `tasks/textcraft_s3_fixed.json`, which contain the seed and the complete
  target goal and recipe set. Task count and seeds are read from the file;
  `--num_games` and `--seed_start` are not needed.

### Stage 1: Build Feasibility Memory

Run S1 first. This stage collects the 50 construction tasks for each
environment and writes the verified Feasibility Memory rule code together with
the trajectory logs.

Build all three environments:

```bash
bash scripts/run_s1.sh all
```

Run one environment only:

```bash
bash scripts/run_s1.sh alfworld
bash scripts/run_s1.sh webshop
bash scripts/run_s1.sh textcraft
```

Outputs are written to `runs/s1_build/<environment>/` by default. Set
`DUALIGN_OUTPUT_DIR` to use another location.

### Stage 2: Build Progress Memory

After S1 finishes, run S2. S2 consumes only the S1 trajectory outputs and
writes `progress_memory_library.json`. It does not run evaluation.

Build all three libraries:

```bash
bash scripts/run_s2.sh all
```

The command above is the complete S2 command. It reads the three S1 output
directories and writes one `progress_memory_library.json` file per environment; it
does not evaluate tasks or create a second S1 run. To build only one library,
pass `alfworld`, `webshop`, or `textcraft` instead of `all`.

Run one environment only:

```bash
bash scripts/run_s2.sh alfworld
bash scripts/run_s2.sh webshop
bash scripts/run_s2.sh textcraft
```

### Stage 3: Evaluate DualAlign

After both S1 and S2 complete, run S3. The default script performs three
independent evaluation runs (`r1`, `r2`, and `r3`), following the paper's
reported evaluation protocol:

```bash
bash scripts/run_s3.sh all
```

Or evaluate one environment:

```bash
bash scripts/run_s3.sh alfworld
bash scripts/run_s3.sh webshop
bash scripts/run_s3.sh textcraft
```

To change the number of independent evaluation runs, set
`DUALIGN_EVAL_RUNS`. For example, run once or five times with:

```bash
DUALIGN_EVAL_RUNS=1 bash scripts/run_s3.sh all
DUALIGN_EVAL_RUNS=5 bash scripts/run_s3.sh all
```

If S1/S2 used a custom output directory, point S3 to it with
`DUALIGN_S1_OUTPUT_DIR`. S3 outputs go to `runs/s3_eval/` by default. The
WebShop server must be available at `http://127.0.0.1:3000`.

The wrappers pass the published S3 manifests and the generated S1/S2 artifacts
to every environment entrypoint. They fail before an API call when either
memory artifact is missing. The task manifests reproduce the paper task pools:
50 S1 tasks per environment, followed by 134 ALFWorld, 100 WebShop, and 100
TextCraft S3 tasks.

The supplied model configuration templates match the models used for the
reported runs. Since S1/S2 regenerate LLM-produced rules and progress-memory
entries, the public workflow is protocol-reproducible but is not expected to
be bit-for-bit identical to the private run. The fixed task files, model
settings, stage order, and evaluation protocol are fully specified here.

## Citation

```bibtex
@inproceedings{wen2026dualmemory,
  title     = {Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents},
  author    = {Wen, Bin and Zhang, Ruoxuan and Chen, Yang and Xie, Hongxia and Guo, Lan-Zhe},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
