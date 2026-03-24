"""Relative-pose OOD dynamics dataset for LPB steering training."""

from typing import Dict, Iterable, List, Sequence
import copy
import hashlib
import json
import os
import multiprocessing

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
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


class SonReplayOODDynamicsDatasetRelative(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        expert_dataset_path: str = None,
        rollout_dataset_paths: Sequence[str] = tuple(),
        action_horizon: int = 6,
        seed: int = 42,
        val_ratio: float = 0.0,
        rotation_rep: str = "rotation_6d",
        use_cache: bool = False,
        pose_repr: dict = None,
    ):
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")

        self.shape_meta = shape_meta
        self.dataset_paths = _resolve_dataset_paths(expert_dataset_path, rollout_dataset_paths)
        self.seed = seed
        self.val_ratio = val_ratio
        self.rotation_rep = rotation_rep
        self.action_horizon = action_horizon
        self.pose_repr = pose_repr or {}
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "abs")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "abs")
        self.action_gripper_repr = self.pose_repr.get("action_gripper_repr", "abs")

        self.rot_action_to_target = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.rot_quat2mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.rot_6d2mat = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")
        self.rot_mat2target = {}
        for key, attr in shape_meta["obs"].items():
            if "rotation_rep" in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep="matrix", to_rep=attr["rotation_rep"]
                )
        if "rotation_rep" in shape_meta["action"]:
            self.rot_mat2target["action"] = RotationTransformer(
                from_rep="matrix", to_rep=shape_meta["action"]["rotation_rep"]
            )

        if use_cache:
            cache_zarr_path = _get_cache_path(self.dataset_paths)
            cache_lock_path = cache_zarr_path + ".lock"
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    replay_buffer = _convert_hdf5s_to_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_paths=self.dataset_paths,
                        rotation_transformer=self.rot_action_to_target,
                    )
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=zarr.MemoryStore(),
                        )
        else:
            replay_buffer = _convert_hdf5s_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_paths=self.dataset_paths,
                rotation_transformer=self.rot_action_to_target,
            )

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=action_horizon + 1,
            pad_before=0,
            pad_after=0,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.action_horizon + 1,
            pad_before=0,
            pad_after=0,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        action_normalizers = [
            get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., :3])
            ),
            get_identity_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., 3:9])
            ),
            get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer["action"][..., 9:10])
            ),
        ]
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key == "position":
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "quat":
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == "gripper":
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        obs_dict, next_obs_dict = self._build_obs_pair(data)
        action = self._build_action_chunk(data, obs_dict)
        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "next_obs": dict_apply(next_obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action.astype(np.float32)),
        }

    def _build_obs_pair(self, data: Dict[str, np.ndarray]):
        obs_dict = {}
        next_obs_dict = {}

        for key in self.rgb_keys:
            current = np.moveaxis(data[key][0], -1, 0).astype(np.float32) / 255.0
            nxt = np.moveaxis(data[key][-1], -1, 0).astype(np.float32) / 255.0
            obs_dict[key] = current
            next_obs_dict[key] = nxt

        current_pos = data["position"][0].astype(np.float32)
        current_quat = data["quat"][0].astype(np.float32)
        current_gripper = data["gripper"][0].astype(np.float32)

        next_pos = data["position"][-1].astype(np.float32)
        next_quat = data["quat"][-1].astype(np.float32)
        next_gripper = data["gripper"][-1].astype(np.float32)

        current_rot_mat = self.rot_quat2mat.forward(current_quat)
        next_rot_mat = self.rot_quat2mat.forward(next_quat)
        base_pose_mat = _pos_rot_to_pose_mat(current_pos[None], current_rot_mat[None])[0]
        current_pose_mat = _pos_rot_to_pose_mat(current_pos[None], current_rot_mat[None])[0]
        next_pose_mat = _pos_rot_to_pose_mat(next_pos[None], next_rot_mat[None])[0]

        if self.obs_pose_repr == "relative":
            current_pose_rel = convert_pose_mat_rep(
                current_pose_mat, base_pose_mat, pose_rep="relative", backward=False
            )
            next_pose_rel = convert_pose_mat_rep(
                next_pose_mat, base_pose_mat, pose_rep="relative", backward=False
            )
            obs_dict["position"], obs_dict["quat"] = _pose_mat_to_pos_rot(
                current_pose_rel, self.rot_mat2target["quat"]
            )
            next_obs_dict["position"], next_obs_dict["quat"] = _pose_mat_to_pos_rot(
                next_pose_rel, self.rot_mat2target["quat"]
            )
        else:
            obs_dict["position"] = current_pos.astype(np.float32)
            obs_dict["quat"] = self.rot_mat2target["quat"].forward(current_rot_mat).astype(np.float32)
            next_obs_dict["position"] = next_pos.astype(np.float32)
            next_obs_dict["quat"] = self.rot_mat2target["quat"].forward(next_rot_mat).astype(np.float32)

        obs_dict["position"] = obs_dict["position"].astype(np.float32)
        obs_dict["quat"] = obs_dict["quat"].astype(np.float32)
        next_obs_dict["position"] = next_obs_dict["position"].astype(np.float32)
        next_obs_dict["quat"] = next_obs_dict["quat"].astype(np.float32)
        obs_dict["gripper"] = current_gripper.astype(np.float32)
        next_obs_dict["gripper"] = next_gripper.astype(np.float32)
        return obs_dict, next_obs_dict

    def _build_action_chunk(self, data: Dict[str, np.ndarray], obs_dict: Dict[str, np.ndarray]):
        action = data["action"][: self.action_horizon].astype(np.float32)
        if self.action_pose_repr != "relative" and self.action_gripper_repr != "relative":
            return action

        current_pos = data["position"][0].astype(np.float32)
        current_rot_mat = self.rot_quat2mat.forward(data["quat"][0].astype(np.float32))
        current_gripper = data["gripper"][0].astype(np.float32)
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
            [
                action_pos.astype(np.float32),
                action_rot.astype(np.float32),
                action_gripper.astype(np.float32),
            ],
            axis=-1,
        )


def _resolve_dataset_paths(expert_dataset_path: str, rollout_dataset_paths: Iterable[str]) -> List[str]:
    dataset_paths = []
    if expert_dataset_path:
        dataset_paths.append(expert_dataset_path)
    dataset_paths.extend(list(rollout_dataset_paths or []))
    if len(dataset_paths) == 0:
        raise ValueError("At least one dataset path is required.")
    return dataset_paths


def _get_cache_path(dataset_paths: Sequence[str]) -> str:
    payload = json.dumps(sorted(dataset_paths)).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    base_dir = os.path.dirname(dataset_paths[0]) or "."
    return os.path.join(base_dir, f".ood_dynamics_rel_cache_{digest}.zarr.zip")


def _get_obs_group(demo):
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise RuntimeError("No obs or observations group found in demo.")


def _convert_actions(raw_actions, rotation_transformer):
    if raw_actions.shape[-1] == 10:
        return raw_actions.astype(np.float32)
    pos = raw_actions[..., :3]
    rot = raw_actions[..., 3:6]
    gripper = raw_actions[..., 6:]
    rot = rotation_transformer.forward(rot)
    return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)


def _convert_hdf5s_to_replay(
    store,
    shape_meta,
    dataset_paths,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episode_ends = []
    episode_starts = []
    prev_end = 0
    total_steps = 0
    episode_specs = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as file:
            demos = file["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                episode_length = demo["actions"].shape[0]
                episode_starts.append(prev_end)
                prev_end += episode_length
                episode_ends.append(prev_end)
                total_steps += episode_length
                episode_specs.append((dataset_path, i))

    _ = meta_group.array(
        "episode_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        this_data = []
        for dataset_path in dataset_paths:
            with h5py.File(dataset_path, "r") as file:
                demos = file["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    obs_group = _get_obs_group(demo)
                    if key == "action":
                        arr = demo["actions"][:].astype(np.float32)
                    else:
                        arr = obs_group[key][:].astype(np.float32)
                    this_data.append(arr)
        this_data = np.concatenate(this_data, axis=0)
        if key == "action":
            this_data = _convert_actions(this_data, rotation_transformer)
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

    with tqdm(total=total_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
        for key in rgb_keys:
            c, h, w = tuple(shape_meta["obs"][key]["shape"])
            img_arr = data_group.require_dataset(
                name=key,
                shape=(total_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=Jpeg2k(level=50),
                dtype=np.uint8,
            )
            for episode_idx, (dataset_path, demo_idx) in enumerate(episode_specs):
                with h5py.File(dataset_path, "r") as file:
                    demo = file["data"][f"demo_{demo_idx}"]
                    obs_group = _get_obs_group(demo)
                    hdf5_arr = obs_group[key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        img_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                        _ = img_arr[zarr_idx]
                        pbar.update(1)

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
