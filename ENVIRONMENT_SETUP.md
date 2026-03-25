# Environment Setup

This repository was exported from the `robodiff` conda environment on the source machine.

Relevant files:

- `robodiff_env.yaml`: full conda environment export
- `robodiff_pip_freeze.txt`: exact pip package snapshot from the same environment

## Recommended install flow

Create the conda environment first:

```bash
conda env create -f robodiff_env.yaml
conda activate robodiff
```

Then install any remaining pip-only packages if needed:

```bash
pip install -r robodiff_pip_freeze.txt
```

## Notes

- Start with `robodiff_env.yaml`. It is the primary environment definition.
- Use `robodiff_pip_freeze.txt` as a secondary compatibility snapshot.
- `pip freeze` can overlap with conda-managed packages, so apply it after the conda environment is created.
- CUDA, NVIDIA driver, and system library differences can still affect portability across machines.
