# hyper v2.0 Branch Plan

This document defines the branch-level migration path for replacing legacy `hyper` with `hyperMLT`.

## Goal

Create a dedicated Git branch where `hyperMLT` is treated as the main project and validated over time as the replacement for legacy `hyper`.

The legacy `main` branch on GitHub remains the rollback and comparison line until `hyperMLT` is accepted as the replacement.

## Scope

In scope for the v2.0 branch:

- `hyperMLT` training workflows
- `hyperMLT` multiday workflows
- `hyperMLT` inference workflows
- `hyperMLT` plotting and diagnostics
- `hyperMLT` configs, manifests, artifacts, and documentation

Out of scope for the v2.0 branch:

- backward compatibility with legacy config files
- backward compatibility with legacy CLIs
- keeping the legacy implementation as an active supported path

## Recommended branch policy

1. Create a dedicated branch, for example `hyper-v2`.
2. Make `hyperMLT` the documented and supported project on that branch.
3. Keep the legacy code only as temporary reference while validation is ongoing.
4. Do not merge until the new training and inference paths are stable.

## Promotion steps

1. Update repository documentation so new users land on `hyperMLT`.
2. Use `hyperMLT/scripts/train.py`, `hyperMLT/scripts/multiday_train.py`, and `hyperMLT/scripts/infer.py` as the official entrypoints.
3. Validate canonical runs repeatedly:
   - daily training
   - daily PDE training
   - multiday DeepGreen
   - inference diagnostics
4. Standardize output directories and artifact expectations around `hyperMLT`.
5. Once validation is satisfactory, remove or archive the legacy implementation on the v2.0 branch.

## Validation checklist

- training completes on canonical DeepGreen configs
- multiday shared-trunk training completes
- inference HDF5 export completes
- mean-wind and time-coordinate diagnostic plots are produced
- shared trunk lineage is model-signature specific
- best-checkpoint saving works
- training history and manifests are written

## Current entrypoints

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
