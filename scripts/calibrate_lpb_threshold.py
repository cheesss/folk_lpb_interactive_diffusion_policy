"""
Estimate a practical raw LPB threshold from expert-bank distances on recorded observations.

The script is intentionally lightweight: it reuses the current OOD monitor and reports
basic percentiles so the user can decide whether to keep the bank p95 default or set a
more conservative raw threshold.
"""

import argparse
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.ood_monitor import OODMonitor
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def main():
    parser = argparse.ArgumentParser(description="Calibrate LPB raw threshold from an expert bank")
    parser.add_argument("--checkpoint", required=True, help="Base policy checkpoint path")
    parser.add_argument("--bank", required=True, help="Expert latent bank .pt path")
    parser.add_argument("--dataset", required=True, help="HDF5 dataset path used for quick calibration")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of samples to score")
    args = parser.parse_args()

    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    monitor = OODMonitor(
        policy=policy,
        bank_path=args.bank,
        expected_shape_meta=cfg.task.shape_meta,
        dynamics_ckpt=None,
        threshold=None,
        device=str(device),
    )

    import h5py

    scores = []
    with h5py.File(args.dataset, "r") as file:
        demos = file["data"]
        for demo_name in demos.keys():
            obs_group = demos[demo_name]["obs"]
            length = obs_group["position"].shape[0]
            for step_idx in range(length):
                env_obs = {
                    key: obs_group[key][max(0, step_idx - cfg.n_obs_steps + 1): step_idx + 1]
                    for key in cfg.task.shape_meta["obs"].keys()
                }
                # Left-pad to n_obs_steps.
                for key, arr in env_obs.items():
                    if arr.shape[0] < cfg.n_obs_steps:
                        pad = np.repeat(arr[[0]], cfg.n_obs_steps - arr.shape[0], axis=0)
                        env_obs[key] = np.concatenate([pad, arr], axis=0)
                obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                metrics = monitor.score_current(obs_dict)
                scores.append(metrics["raw"])
                if len(scores) >= args.limit:
                    break
            if len(scores) >= args.limit:
                break

    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        raise RuntimeError("No scores were collected.")

    print("count:", int(scores.size))
    for q in [50, 75, 90, 95, 97.5, 99]:
        print(f"p{q}: {float(np.percentile(scores, q)):.6f}")
    print(f"mean: {float(scores.mean()):.6f}")
    print(f"std: {float(scores.std()):.6f}")


if __name__ == "__main__":
    main()
