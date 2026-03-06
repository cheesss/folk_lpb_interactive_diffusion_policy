"""
Verify that SonReplayImageDatasetRelative produces correct relative obs/action.
With obs_pose_repr=relative, the last obs step (base) should have position ~ 0 and quat ~ identity 6D.

Usage:
  python scripts/verify_relative_batch.py
  python scripts/verify_relative_batch.py --dataset_path /path/to/pick_tissue_2ep.hdf5
"""
import argparse
import sys
import numpy as np

# Repo root on path
sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="Verify relative pose in one batch")
    parser.add_argument(
        "--dataset_path",
        default="/media/son/son_dataset_2/IL/dataset/rb10_pick_and_place/pick_tissue_2ep.hdf5",
        help="Path to HDF5 (e.g. 2-ep subset)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Tolerance for allclose checks",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from diffusion_policy.dataset.son_replay_image_dataset_relative import (
        SonReplayImageDatasetRelative,
    )

    # Minimal config matching son_pick_and_place_tissue_image_relative + workspace
    shape_meta = {
        "obs": {
            "image0": {"shape": [3, 240, 320], "type": "rgb"},
            "image1": {"shape": [3, 240, 320], "type": "rgb"},
            "position": {"shape": [3], "type": "low_dim"},
            "quat": {"raw_shape": [4], "shape": [6], "type": "low_dim", "rotation_rep": "rotation_6d"},
            "gripper": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [10], "rotation_rep": "rotation_6d"},
    }
    pose_repr = {"obs_pose_repr": "relative", "action_pose_repr": "relative"}
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 6
    pad_before = n_obs_steps - 1
    pad_after = n_action_steps - 1

    dataset = SonReplayImageDatasetRelative(
        shape_meta=shape_meta,
        dataset_path=args.dataset_path,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        n_obs_steps=n_obs_steps,
        rotation_rep="rotation_6d",
        use_cache=True,
        seed=42,
        val_ratio=0.0,
        pose_repr=pose_repr,
    )

    sample = dataset[0]
    obs = sample["obs"]
    action = sample["action"]

    # Convert to numpy for checks
    pos = obs["position"].numpy()
    quat = obs["quat"].numpy()
    act = action.numpy()

    # Last obs step is the base; in relative mode it should be ~0 and identity
    pos_last = pos[-1]
    quat_last = quat[-1]
    # Identity 6D = first two columns of I3: [1,0,0, 0,1,0]
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    print("--- Last obs step (base) ---")
    print("position[-1]:", pos_last)
    print("quat[-1] (6D):", quat_last)
    print("expected identity_6d:", identity_6d)

    pos_ok = np.allclose(pos_last, 0.0, atol=args.atol)
    quat_ok = np.allclose(quat_last, identity_6d, atol=args.atol)

    if pos_ok:
        print("PASS: position[-1] ~ 0")
    else:
        print("FAIL: position[-1] not close to 0 (atol=%s)" % args.atol)

    if quat_ok:
        print("PASS: quat[-1] ~ identity 6D")
    else:
        print("FAIL: quat[-1] not close to identity 6D (atol=%s)" % args.atol)

    # Action shape
    print("\n--- Action ---")
    print("action shape:", act.shape)
    print("action[0] (first step):", act[0])

    if pos_ok and quat_ok:
        print("\nRelative conversion verified.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
