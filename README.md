# hyper

This branch is `hyper v2.0`. The repository now uses the `hyperMLT` codebase as the main project at the repository root.

## Entrypoints

Training:

```bash
python /Users/admin/git/hyper/scripts/train.py \
  --config-file /Users/admin/git/hyper/config/train/deepgreen_vorticity_inviscid_baseline.yaml
```

Multiday training:

```bash
python /Users/admin/git/hyper/scripts/multiday_train.py \
  --config-file /Users/admin/git/hyper/config/train/deepgreen_multiday_vorticity_inviscid_shared_trunk.yaml
```

Inference:

```bash
python /Users/admin/git/hyper/scripts/infer.py \
  --config-file /Users/admin/git/hyper/config/infer/deepgreen_daily_diagnostics.yaml
```

## Scope

This repository contains:

- DeepGreen and RESPINN models
- PDE-informed training workflows
- daily and multiday training
- inference and diagnostic plotting
- local implementations inside `hyperMLT/`

## Main directories

- `config/train/`
- `config/infer/`
- `hyperMLT/`
- `scripts/`
- `docs/hyper_v2_branch_plan.md`

## Migration note

This branch is the replacement track for the old `hyper` project. The migration checklist is documented in `docs/hyper_v2_branch_plan.md`.
