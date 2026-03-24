from __future__ import annotations

import os
from typing import Dict, Optional

import dill
import h5py
import hydra
import numpy as np
import torch
from einops import rearrange

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_relative_obs_dict,
)
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class LPBOriginalPlanner:
    """
    Original-LPB-style planner adapted to the folk repository structure.
    """

    def __init__(
        self,
        policy,
        bank_path: Optional[str],
        dynamics_ckpt: str,
        expected_shape_meta: Optional[dict] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        chunk_size: int = 4096,
        action_horizon: Optional[int] = None,
        action_start_idx: Optional[int] = None,
        demo_dataset_path: Optional[str] = None,
    ):
        del bank_path
        self.policy = policy
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.expected_shape_meta = expected_shape_meta or {}
        self.demo_dataset_path = demo_dataset_path

        payload = torch.load(open(dynamics_ckpt, "rb"), pickle_module=dill, map_location=self.device)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=os.path.dirname(os.path.dirname(dynamics_ckpt)))
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.workspace = workspace
        self.dyn_model = workspace.model.to(self.device).eval()
        self.dyn_normalizer = workspace.normalizer.to(self.device)
        self.frameskip = int(cfg.frameskip)
        self.original_img_size = int(cfg.task.original_img_size)
        self.view_names = list(cfg.task.view_names)
        self.lowdim_keys = list(cfg.task.lowdim_keys)
        self.original_action_dim = int(cfg.task.original_action_dim)
        self.threshold = float(threshold if threshold is not None else cfg.lpb.default_threshold)
        self.action_start_idx = (
            int(action_start_idx)
            if action_start_idx is not None
            else max(int(getattr(self.policy, "n_obs_steps", 1)) - 1, 0)
        )
        self.action_horizon = int(action_horizon if action_horizon is not None else self.frameskip)

        self.policy_action_normalizer = None
        self.pose_repr = cfg.task.pose_repr
        self.rot_quat2mat = RotationTransformer("quaternion", "matrix")
        self.rot_mat2target = {}
        for key, attr in self.expected_shape_meta.get("obs", {}).items():
            if "rotation_rep" in attr:
                self.rot_mat2target[key] = RotationTransformer("matrix", attr["rotation_rep"])
        self.demo_visual_latents = None
        if self.demo_dataset_path is not None:
            self._build_demo_latents()

    def set_policy_action_normalizer(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer

    def _build_demo_latents(self):
        if self.demo_dataset_path is None:
            raise RuntimeError("demo_dataset_path is required for LPBOriginalPlanner")

        obs_pose_repr = self.pose_repr.get("obs_pose_repr", "abs")
        use_relative = obs_pose_repr == "relative"
        demo_visual_latents = []
        with h5py.File(self.demo_dataset_path, "r") as file:
            demos = file["data"]
            for demo_name in demos.keys():
                obs_group = demos[demo_name]["obs"]
                length = obs_group[self.view_names[0]].shape[0]
                for idx in range(length):
                    env_obs = {
                        key: obs_group[key][idx:idx + 1]
                        for key in self.expected_shape_meta["obs"].keys()
                    }
                    if use_relative:
                        obs_dict_np = get_real_relative_obs_dict(
                            env_obs=env_obs,
                            shape_meta=self.expected_shape_meta,
                            rot_quat2mat=self.rot_quat2mat,
                            rot_mat2target=self.rot_mat2target,
                            obs_pose_repr=obs_pose_repr,
                        )
                    else:
                        obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=self.expected_shape_meta)
                    obs_dict = dict_apply(
                        obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
                    )
                    obs_wm = self._prepare_obs_for_dyn(obs_dict, batch_size=1)
                    with torch.no_grad():
                        encode_obs = self.dyn_model.encode_obs(obs_wm)
                    visual_latent = encode_obs["visual"].reshape(1, -1)
                    demo_visual_latents.append(visual_latent.cpu())
        self.demo_visual_latents = torch.cat(demo_visual_latents, dim=0)

    def _prepare_obs_for_dyn(self, current_obs: Dict[str, torch.Tensor], batch_size: int):
        proprio_arrays = []
        for key in self.lowdim_keys:
            value = current_obs[key].to(self.device)
            if value.ndim >= 3:
                value = value[:, -1]
            proprio_arrays.append(value)
        proprio = torch.cat(proprio_arrays, dim=-1) if proprio_arrays else torch.zeros((1, 0), device=self.device)

        visual = {}
        for view_name in self.view_names:
            img = current_obs[view_name].to(self.device)
            if img.ndim >= 5:
                img = img[:, -1]
            visual[view_name] = self.dyn_normalizer[view_name].normalize(img).unsqueeze(1)
        current_obs_wm = {"visual": visual, "proprio": proprio}
        current_obs_wm["proprio"] = self.dyn_normalizer["state"].normalize(current_obs_wm["proprio"]).unsqueeze(1)
        current_obs_wm["proprio"] = current_obs_wm["proprio"].expand(batch_size, -1, -1)
        current_obs_wm["visual"] = {
            key: value.expand(batch_size, -1, -1, -1, -1)
            for key, value in current_obs_wm["visual"].items()
        }
        return current_obs_wm

    def _compute_nn_reward(self, visual_latent: torch.Tensor) -> torch.Tensor:
        current_visual_latent = visual_latent.reshape(visual_latent.size(0), -1)
        device = current_visual_latent.device
        global_min_cost = None
        for start in range(0, self.demo_visual_latents.shape[0], self.chunk_size):
            demo_chunk = self.demo_visual_latents[start:start + self.chunk_size].to(device, non_blocking=True)
            dist = torch.cdist(current_visual_latent, demo_chunk, p=2)
            cost = dist.min(dim=-1).values
            if global_min_cost is None:
                global_min_cost = cost
            else:
                global_min_cost = torch.minimum(global_min_cost, cost)
        return -global_min_cost

    def compute_current_reward(self, current_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.demo_visual_latents is None:
            raise RuntimeError("demo latents are not initialized")
        with torch.no_grad():
            current_obs_wm = self._prepare_obs_for_dyn(current_obs, 1)
            encode_obs = self.dyn_model.encode_obs(current_obs_wm)
            current_visual_latent = encode_obs["visual"]
            reward = self._compute_nn_reward(current_visual_latent.squeeze(1))
        return reward

    def compute_loss(self, sample: torch.Tensor, current_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.policy_action_normalizer is None:
            raise RuntimeError("policy_action_normalizer must be set before calling compute_loss")
        if self.demo_visual_latents is None:
            raise RuntimeError("demo latents are not initialized")

        action_slice = sample[
            :,
            self.action_start_idx : self.action_start_idx + self.action_horizon,
            : self.policy.action_dim,
        ]
        init_actions_unnormalized = self.policy_action_normalizer.unnormalize(action_slice)
        init_actions = self.dyn_normalizer["act"].normalize(init_actions_unnormalized)
        action_batch = rearrange(
            init_actions,
            "b (h f) a -> b h (f a)",
            f=self.frameskip,
            h=max(init_actions.shape[1] // self.frameskip, 1),
        )
        current_obs_wm = self._prepare_obs_for_dyn(current_obs, sample.shape[0])
        act_0 = action_batch[:, :1, :]
        z = self.dyn_model.encode(current_obs_wm, act_0)
        z_pred = self.dyn_model.predict(z[:, :1])
        z_new = z_pred[:, -1:, ...]
        z_obs, _ = self.dyn_model.separate_emb(z_new)
        reward = self._compute_nn_reward(z_obs["visual"])
        cost = -reward
        return cost.mean()
