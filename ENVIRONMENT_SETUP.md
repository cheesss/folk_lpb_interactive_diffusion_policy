# Environment Setup

This repository was exported from the `robodiff` conda environment on the source machine.

Relevant files:

- `robodiff_env.yaml`: full conda environment export
- `robodiff_pip_freeze.txt`: exact pip package snapshot from the same environment
- `robodiff_dynamics_min_env.yaml`: reduced environment for dynamics training only

## Full environment install

Use this when you want the closest possible reproduction of the source machine.

```bash
conda env create -f robodiff_env.yaml
conda activate robodiff
```

Then install any remaining pip-only packages if needed:

```bash
pip install -r robodiff_pip_freeze.txt
```

## Lightweight dynamics-only install

Use this when you only need offline dynamics training such as:

- `train_ood_dynamics_workspace.py`
- `train_lpb_visual_dynamics_workspace.py`

Create the reduced environment:

```bash
conda env create -f robodiff_dynamics_min_env.yaml
conda activate robodiff-dynamics
```

## Notes

- Start with `robodiff_env.yaml`. It is the primary environment definition.
- Use `robodiff_pip_freeze.txt` as a secondary compatibility snapshot.
- `pip freeze` can overlap with conda-managed packages, so apply it after the conda environment is created.
- `robodiff_dynamics_min_env.yaml` intentionally excludes robot runtime packages and visualization-heavy extras.
- `robodiff_dynamics_min_env.yaml` assumes a CUDA 12.1 compatible GPU stack because the source environment used `torch 2.1.2 + cu121`.
- `pytorch3d` is the most version-sensitive dependency in the reduced environment. If installation fails on another machine, match its wheel to the local `torch` and CUDA version first.
- CUDA, NVIDIA driver, and system library differences can still affect portability across machines.
