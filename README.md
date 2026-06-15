# hyper

This repository currently contains two code lines:

- legacy `hyper` under `src/`
- `hyperMLT` under `hyperMLT/`

The active replacement project is `hyperMLT`. It is the planned `hyper v2.0` codebase and is the project that should be used for new training, multiday training, and inference work.

## v2.0 entrypoints

Training:

```bash
python /Users/admin/git/hyper/hyperMLT/scripts/train.py \
  --config-file /Users/admin/git/hyper/hyperMLT/config/train/deepgreen_vorticity_inviscid_baseline.yaml
```

Multiday training:

```bash
python /Users/admin/git/hyper/hyperMLT/scripts/multiday_train.py \
  --config-file /Users/admin/git/hyper/hyperMLT/config/train/deepgreen_multiday_vorticity_inviscid_shared_trunk.yaml
```

Inference:

```bash
python /Users/admin/git/hyper/hyperMLT/scripts/infer.py \
  --config-file /Users/admin/git/hyper/hyperMLT/config/infer/deepgreen_daily_diagnostics.yaml
```

## Project status

`hyperMLT` is the forward path for this repository:

- standalone package and scripts
- local implementations only, without runtime imports from legacy project code
- DeepGreen and RESPINN training
- PDE-informed training
- multiday DeepGreen shared-trunk workflow
- inference and diagnostics

The legacy `hyper` tree is still present in this branch only as historical reference while `hyperMLT` is being validated. It is not the recommended entrypoint for new work.

## Main directories

- `hyperMLT/config/train/`
- `hyperMLT/config/infer/`
- `hyperMLT/scripts/`
- `hyperMLT/hyperMLT/`
- `docs/hyper_v2_branch_plan.md`

## Validation branch strategy

The intended migration path is:

1. create a dedicated `hyper v2.0` branch from this repository
2. treat `hyperMLT` as the main project on that branch
3. validate training, multiday training, inference, and diagnostics over time
4. merge only after the new project is stable enough to replace the legacy mainline

The branch promotion checklist is in:

- `docs/hyper_v2_branch_plan.md`
