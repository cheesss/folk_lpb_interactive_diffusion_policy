from typing import Dict
import copy
import os
import shutil

import numpy as np
import torch
import zarr
from einops import rearrange
from filelock import FileLock

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.son_replay_image_dataset_relative import (
    _convert_robomimic_to_replay,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer

register_codecs()


def _pos_rot_to_pose_mat(pos, rot_mat):
    lead = rot_mat.shape[:-2]
    pose = np.zeros(lead + (4, 4), dtype=pos.dtype)
    pose[..., :3, :3] = rot_mat
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def _pose_mat_to_pos_rot(pose_mat, rot_mat_to_target):
    pos = pose_mat[..., :3, 3].copy()
    rot_mat = pose_mat[..., :3, :3].copy()
    rot = rot_mat_to_target.forward(rot_mat)
    return pos, rot


class SonLPBDynamicsDatasetRelative(BaseImageDataset):
    """
    Original-LPB-style dataset:
    returns (obs, act, state), where obs contains two frames (history + future),
    act contains two chunks of actions, and state contains low-dim states for the same frames.
    """

    def __init__(
        self,
        dataset_path,
        shape_meta,
        num_hist=1,
        num_pred=1,
        frameskip=6,
        view_names=None,
        use_crop=False,
        train=True,
        original_img_size=320,
        cropped_img_size=288,
        pose_repr=None,
    ):
        self.view_names = view_names or ["image0", "image1"]
        self.shape_obs = shape_meta["obs"]
        self.pose_repr = pose_repr or {}
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "abs")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "abs")
        self.action_gripper_repr = self.pose_repr.get("action_gripper_repr", "abs")
        self.original_img_size = original_img_size
        self.cropped_img_size = cropped_img_size
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.num_frames = num_hist + num_pred
        self.train = train
        self.use_crop = use_crop

        rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
        cache_zarr_path = dataset_path + ".lpb_dyn_rel.zarr.zip"
        cache_lock_path = cache_zarr_path + ".lock"
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    replay_buffer = _convert_robomimic_to_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_path=dataset_path,
                        abs_action=True,
                        rotation_transformer=rotation_transformer,
                    )
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception:
                    if os.path.exists(cache_zarr_path):
                        shutil.rmtree(cache_zarr_path)
                    raise
            else:
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )

        self.replay_buffer = replay_buffer
        self.episode_ends = replay_buffer.episode_ends[:]
        self.episode_start_indices = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_end_indices = self.episode_ends - 1

        self.imgs = {view_name: np.array(replay_buffer[view_name]) for view_name in self.view_names}
        state_arrays = []
        self.lowdim_keys = []
        for key, meta in self.shape_obs.items():
            if meta.get("type") == "rgb" or "image" in key:
                continue
            if key in replay_buffer.data:
                state_arrays.append(np.array(replay_buffer[key]))
                self.lowdim_keys.append(key)
        self.states = np.concatenate(state_arrays, axis=1) if state_arrays else np.zeros((len(replay_buffer), 0))
        self.proprio_dim = self.states.shape[1]
        self.original_action_dim = int(np.prod(shape_meta["action"]["shape"]))
        self.actions = np.array(replay_buffer["action"])
        self.action_dim = self.original_action_dim * frameskip

        self.valid_anchor_indices = []
        for start, end in zip(self.episode_start_indices, self.episode_end_indices):
            anchor_start = start
            anchor_end = end - num_pred * self.frameskip
            if anchor_end >= anchor_start:
                self.valid_anchor_indices.extend(np.arange(anchor_start, anchor_end))
        self.valid_anchor_indices = np.asarray(self.valid_anchor_indices)

        self.rot_quat2mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.rot_6d2mat = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")
        self.rot_mat2target = {}
        for key, attr in self.shape_obs.items():
            if "rotation_rep" in attr:
                self.rot_mat2target[key] = RotationTransformer(from_rep="matrix", to_rep=attr["rotation_rep"])
        self.rot_mat2target["action"] = RotationTransformer(
            from_rep="matrix", to_rep=shape_meta["action"]["rotation_rep"]
        )

    def __len__(self):
        return len(self.valid_anchor_indices)

    def __getitem__(self, idx):
        start = self.valid_anchor_indices[idx]
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip :] = [obs_indices[-1] - 1] * self.frameskip

        obs = {"visual": {}}
        for view_name in self.view_names:
            obs["visual"][view_name] = self.imgs[view_name][obs_indices]
            obs["visual"][view_name] = np.moveaxis(obs["visual"][view_name], -1, 1).astype(np.float32) / 255.0
            obs["visual"][view_name] = torch.from_numpy(obs["visual"][view_name])

        proprio = self._build_proprio(obs_indices)
        obs["proprio"] = torch.from_numpy(proprio.astype(np.float32))
        act = self._build_actions(start, obs_indices, action_indices)
        state = torch.from_numpy(proprio.astype(np.float32))
        return obs, torch.from_numpy(act.astype(np.float32)), state

    def _split_state_dict(self, state_rows):
        obs_dict = {}
        cursor = 0
        for key in self.lowdim_keys:
            dim = int(np.prod(self.shape_obs[key]["shape"]))
            obs_dict[key] = state_rows[:, cursor: cursor + dim]
            cursor += dim
        return obs_dict

    def _merge_state_dict(self, obs_dict):
        return np.concatenate([obs_dict[key] for key in self.lowdim_keys], axis=-1)

    def _build_proprio(self, obs_indices):
        obs_dict = self._split_state_dict(self.states[obs_indices].copy())
        current_pos = obs_dict["position"][0].copy()
        current_rot_mat = self.rot_quat2mat.forward(obs_dict["quat"][0].copy())
        base_pose_mat = _pos_rot_to_pose_mat(current_pos[None], current_rot_mat[None])[0]

        if self.obs_pose_repr == "relative":
            obs_rot_mat = self.rot_quat2mat.forward(obs_dict["quat"])
            obs_pose_mat = _pos_rot_to_pose_mat(obs_dict["position"], obs_rot_mat)
            rel_pose_mat = convert_pose_mat_rep(
                obs_pose_mat, base_pose_mat, pose_rep="relative", backward=False
            )
            obs_dict["position"], obs_dict["quat"] = _pose_mat_to_pos_rot(
                rel_pose_mat, self.rot_mat2target["quat"]
            )
        else:
            obs_dict["quat"] = self.rot_mat2target["quat"].forward(
                self.rot_quat2mat.forward(obs_dict["quat"])
            )

        obs_dict["position"] = obs_dict["position"].astype(np.float32)
        obs_dict["quat"] = obs_dict["quat"].astype(np.float32)
        obs_dict["gripper"] = obs_dict["gripper"].astype(np.float32)
        return self._merge_state_dict(obs_dict)

    def _build_actions(self, start, obs_indices, action_indices):
        action = self.actions[action_indices].copy().astype(np.float32)
        if self.action_pose_repr != "relative" and self.action_gripper_repr != "relative":
            return action

        current_state = self._split_state_dict(self.states[[start]])
        current_pos = current_state["position"][0].astype(np.float32)
        current_rot_mat = self.rot_quat2mat.forward(current_state["quat"][0].astype(np.float32))
        current_gripper = current_state["gripper"][0].astype(np.float32)
        base_pose_mat = _pos_rot_to_pose_mat(current_pos[None], current_rot_mat[None])[0]

        action_pos = action[..., :3]
        action_rot = action[..., 3:9]
        action_gripper = action[..., 9:10]
        if self.action_pose_repr == "relative":
            action_rot_mat = self.rot_6d2mat.forward(action_rot)
            action_pose_mat = _pos_rot_to_pose_mat(action_pos, action_rot_mat)
            rel_pose_mat = convert_pose_mat_rep(
                action_pose_mat, base_pose_mat, pose_rep="relative", backward=False
            )
            action_pos, action_rot = _pose_mat_to_pos_rot(
                rel_pose_mat, self.rot_mat2target["action"]
            )
        if self.action_gripper_repr == "relative":
            action_gripper = action_gripper - current_gripper
        return np.concatenate(
            [action_pos.astype(np.float32), action_rot.astype(np.float32), action_gripper.astype(np.float32)],
            axis=-1,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        act_stat = array_to_stats(self.actions)
        action_normalizers = [
            get_range_normalizer_from_stat(array_to_stats(self.actions[..., :3])),
            get_identity_normalizer_from_stat(array_to_stats(self.actions[..., 3:9])),
            get_range_normalizer_from_stat(array_to_stats(self.actions[..., 9:10])),
        ]
        normalizer["act"] = concatenate_normalizer(action_normalizers)
        state_stat = array_to_stats(self.states)
        state_normalizers = []
        cursor = 0
        for key in self.lowdim_keys:
            dim = int(np.prod(self.shape_obs[key]["shape"]))
            stat = array_to_stats(self.states[..., cursor: cursor + dim])
            if key == "position":
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "quat":
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == "gripper":
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            state_normalizers.append(this_normalizer)
            cursor += dim
        normalizer["state"] = concatenate_normalizer(state_normalizers)
        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
        return normalizer
