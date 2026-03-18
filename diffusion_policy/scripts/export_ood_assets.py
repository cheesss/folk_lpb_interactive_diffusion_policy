"""Expert demo를 latent bank(.pt)로 내보내는 오프라인 export 스크립트."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.ood_utils import (
    compute_reference_stats,
    encode_policy_obs,
    encode_policy_obs_window,
    get_lowdim_keys,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_relative_obs_dict,
)


def _get_obs_group(demo):
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise RuntimeError("No obs or observations group found in demo.")


def _load_policy_from_checkpoint(ckpt_path: str, use_ema: bool):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if use_ema and getattr(workspace, "ema_model", None) is not None:
        return workspace.ema_model, cfg
    return workspace.model, cfg


def _to_plain_container(config_or_dict):
    if OmegaConf.is_config(config_or_dict):
        return OmegaConf.to_container(config_or_dict, resolve=True)
    return config_or_dict


def _get_pose_repr_cfg(task_cfg):
    pose_repr_cfg = task_cfg.get("pose_repr", {}) if task_cfg is not None else {}
    if OmegaConf.is_config(pose_repr_cfg):
        return OmegaConf.to_container(pose_repr_cfg, resolve=True)
    if isinstance(pose_repr_cfg, dict):
        return pose_repr_cfg
    return {}


def _build_obs_rotation_helpers(shape_meta, obs_pose_repr):
    rot_quat2mat = RotationTransformer("quaternion", "matrix")
    rot_mat2target = {}
    for key, attr in shape_meta["obs"].items():
        if "rotation_rep" in attr:
            rot_mat2target[key] = RotationTransformer("matrix", attr["rotation_rep"])
    use_rot_obs_dict = "quat" in rot_mat2target
    if obs_pose_repr == "relative" and "quat" not in rot_mat2target:
        raise KeyError(
            "task.shape_meta.obs.quat.rotation_rep is required when obs_pose_repr='relative'"
        )
    return use_rot_obs_dict, rot_quat2mat, rot_mat2target


def _iter_obs_window_batches(
    dataset_path,
    shape_meta,
    n_obs_steps,
    batch_size,
    device,
    obs_pose_repr,
    use_rot_obs_dict,
    rot_quat2mat,
    rot_mat2target,
):
    obs_keys = list(shape_meta["obs"].keys())
    builder = get_real_relative_obs_dict if (use_rot_obs_dict or obs_pose_repr == "relative") else get_real_obs_dict

    with h5py.File(dataset_path, "r") as file:
        demos = file["data"]
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            obs_group = _get_obs_group(demo)
            episode_obs = {key: obs_group[key][:] for key in obs_keys}
            length = episode_obs[obs_keys[0]].shape[0]
            offsets = np.arange(-n_obs_steps + 1, 1, dtype=np.int64)

            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)
                target_indices = np.arange(start, end, dtype=np.int64)
                window_indices = np.clip(target_indices[:, None] + offsets[None, :], 0, length - 1)

                batch_obs = None
                for sample_indices in window_indices:
                    env_obs = {key: episode_obs[key][sample_indices] for key in obs_keys}
                    if builder is get_real_relative_obs_dict:
                        obs_np = builder(
                            env_obs=env_obs,
                            shape_meta=shape_meta,
                            rot_quat2mat=rot_quat2mat,
                            rot_mat2target=rot_mat2target,
                            obs_pose_repr=obs_pose_repr,
                        )
                    else:
                        obs_np = builder(env_obs=env_obs, shape_meta=shape_meta)

                    if batch_obs is None:
                        batch_obs = {key: [] for key in obs_np.keys()}
                    for key, value in obs_np.items():
                        batch_obs[key].append(value.astype(np.float32))

                yield {
                    key: torch.from_numpy(np.stack(value, axis=0)).to(device)
                    for key, value in batch_obs.items()
                }


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="son_export_ood_assets",
)
def main(cfg: OmegaConf):
    device = torch.device(cfg.export.device)
    policy, policy_cfg = _load_policy_from_checkpoint(
        cfg.reference_policy.checkpoint,
        cfg.reference_policy.use_ema,
    )
    policy.to(device).eval()
    for param in policy.parameters():
        param.requires_grad = False

    policy_task_cfg = getattr(policy_cfg, "task", None)
    shape_meta = getattr(policy_task_cfg, "shape_meta", cfg.task.shape_meta)
    expert_dataset_path = cfg.task.expert_dataset_path
    n_obs_steps = int(getattr(policy_cfg, "n_obs_steps", getattr(policy, "n_obs_steps", 1)))
    pose_repr = _get_pose_repr_cfg(policy_task_cfg)
    obs_pose_repr = pose_repr.get("obs_pose_repr", "abs")
    use_rot_obs_dict, rot_quat2mat, rot_mat2target = _build_obs_rotation_helpers(shape_meta, obs_pose_repr)

    lowdim_keys = get_lowdim_keys(shape_meta)
    current_latents = []
    step_latents = []
    with torch.no_grad():
        for obs in _iter_obs_window_batches(
            expert_dataset_path,
            shape_meta,
            n_obs_steps,
            cfg.export.batch_size,
            device,
            obs_pose_repr,
            use_rot_obs_dict,
            rot_quat2mat,
            rot_mat2target,
        ):
            current_latent, _ = encode_policy_obs_window(policy, obs, lowdim_keys)
            step_obs = {key: value[:, -1] for key, value in obs.items()}
            step_latent, _ = encode_policy_obs(policy, step_obs, lowdim_keys)
            current_latents.append(current_latent.detach().cpu())
            step_latents.append(step_latent.detach().cpu())

    current_latents = torch.cat(current_latents, dim=0)
    step_latents = torch.cat(step_latents, dim=0)
    current_stats = compute_reference_stats(current_latents.to(device), chunk_size=cfg.export.chunk_size)
    step_stats = compute_reference_stats(step_latents.to(device), chunk_size=cfg.export.chunk_size)

    output_path = pathlib.Path(cfg.export.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": current_latents,
            "stats": current_stats,
            "current_latents": current_latents,
            "current_stats": current_stats,
            "step_latents": step_latents,
            "step_stats": step_stats,
            "lowdim_keys": lowdim_keys,
            "shape_meta": _to_plain_container(shape_meta),
            "n_obs_steps": n_obs_steps,
            "pose_repr": pose_repr,
            "reference_policy_checkpoint": cfg.reference_policy.checkpoint,
            "expert_dataset_path": expert_dataset_path,
        },
        output_path,
    )
    print(f"Saved OOD reference bank to {output_path}")


if __name__ == "__main__":
    main()
