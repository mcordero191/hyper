# hyperMLT

`hyperMLT` is the replacement implementation for the legacy `hyper` project.

It is intended to become `hyper v2.0` on a dedicated Git branch before any merge back into the main GitHub line.

## Scope

Implemented project areas include:

- SMR data loading and filtering
- daily training
- multiday DeepGreen shared-trunk training
- DeepGreen and RESPINN models
- physics-informed training
- inference on regular grids
- training and inference diagnostics

## Main scripts

Training:

```bash
python hyperMLT/scripts/train.py --config-file hyperMLT/config/train/deepgreen_vorticity_inviscid_baseline.yaml
```

Multiday training:

```bash
python hyperMLT/scripts/multiday_train.py --config-file hyperMLT/config/train/deepgreen_multiday_vorticity_inviscid_shared_trunk.yaml
```

Inference:

```bash
python hyperMLT/scripts/infer.py --config-file hyperMLT/config/infer/deepgreen_daily_diagnostics.yaml
```

## Config layout

- `config/train/`
  - daily training
  - multiday training
- `config/infer/`
  - daily inference
  - multiday inference

## Design rules

- no runtime imports from legacy local project code
- explicit YAML-driven workflows
- separate train, multiday-train, and infer scripts
- diagnostics and artifact output are first-class parts of the workflow

## Migration note

This directory is still a subproject in the current repository layout, but it is the codebase intended to replace legacy `hyper` on the future `v2.0` branch.
