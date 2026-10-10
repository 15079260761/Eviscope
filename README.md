# EviScope

This repository contains the released EviScope datasets, local encoder models,
the main retrieval benchmark, supplementary experiments, and adapted temporal
KGQA baselines. Runtime paths are resolved from this repository directory, so
the checkout may be placed anywhere as long as the internal layout is kept.

## Layout

- `data/`: original sources and released EviScope benchmarks.
- `models/`: local Dual-Encoder and Cross-Encoder backbones.
- `main_benchmark/`: BM25, Dual Encoder, Cross-Encoder, and EviRet evaluation.
- `evidence_space/`: evidence-space ablations.
- `ambiguity_stress/`: controlled distractor-ambiguity experiment.
- `Baselines/`: adapted temporal KGQA baselines and their experiment drivers.

## Quick start

Linux, Bash, Python 3.10--3.12, CUDA, and `tmux` are expected for the supplied
GPU launch scripts. From the repository root:

```bash
bash main_benchmark/setup_env.sh
bash main_benchmark/launch_tmux.sh exp1_full_v1
```

After the main benchmark has produced its checkpoints, run the supplementary experiments:

```bash
bash evidence_space/launch_tmux.sh exp2_v2 exp1_full_v1
bash ambiguity_stress/launch_tmux.sh exp3_v1 exp1_full_v1
```

Each experiment directory contains its detailed protocol, audit, and summary
commands. Adapted baseline instructions are under `Baselines/experiments/`.

By default the virtual environment is `Eviscope/.venv`. Advanced users may
override discovered locations with `PROJECT_ROOT`, `VENV`, `VENV_DIR`, or
`PYTHON_BIN` as supported by the corresponding launcher.
