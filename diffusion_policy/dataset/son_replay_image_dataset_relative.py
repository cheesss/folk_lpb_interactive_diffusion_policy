from typing import Dict
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    concatenate_normalizer
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
register_codecs()


def _pos_rot_to_pose_mat(pos, rot_mat):
    """(..., 3), (..., 3, 3) -> (..., 4, 4) homogeneous pose."""
    lead = rot_mat.shape[:-2]
    pose = np.zeros(lead + (4, 4), dtype=pos.dtype)
    pose[..., :3, :3] = rot_mat
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def _pose_mat_to_pos_rot(pose_mat, rot_mat_to_target):
    """(..., 4, 4) -> pos (..., 3), rot in target rep (e.g. 6D)."""
    pos = pose_mat[..., :3, 3].copy()
    rot_mat = pose_mat[..., :3, :3].copy()
    rot = rot_mat_to_target.forward(rot_mat)
    return pos, rot


class SonReplayImageDatasetRelative(BaseImageDataset):
    """단일 팔 단팔 relative pose 표현 데이터셋.
    obs: position[3], quat[4→6D], gripper[1], image0, image1
    action: pos[3] + rot_6d[6] + gripper[1] = [10]
    pose_repr에 따라 obs/action을 현재 스텝 기준 relative로 변환한다.
    """
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            rotation_rep='rotation_6d',
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            pose_repr: dict={},
        ):
        # action의 axis_angle 회전 → rotation_6d 변환기
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=True,
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=True,
                rotation_transformer=rotation_transformer)

        # shape_meta에서 rgb/lowdim 키 분류
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        # obs step 수만큼만 각 key에서 샘플링
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.pose_repr = pose_repr
        self.obs_pose_repr = pose_repr.get('obs_pose_repr', 'abs')    # 'abs' or 'relative'
        self.action_pose_repr = pose_repr.get('action_pose_repr', 'abs')  # 'abs' or 'relative'
        self.action_gripper_repr = pose_repr.get('action_gripper_repr', 'abs')  # 'abs' or 'relative' (delta from current obs gripper)

        # quaternion/6D → rotation matrix 변환기
        self.rot_quat2mat = RotationTransformer(from_rep='quaternion', to_rep='matrix')
        self.rot_6d2mat = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

        # 각 obs key와 action의 target rotation representation으로 변환하는 변환기
        self.rot_mat2target = dict()
        for key, attr in obs_shape_meta.items():
            if 'rotation_rep' in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix', to_rep=attr['rotation_rep'])
        self.rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix', to_rep=shape_meta['action']['rotation_rep'])

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # 전체 데이터셋을 순회해 lowdim + action 데이터 수집 (image는 skip)
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        # (N, T, D) → (N*T, D)
        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B * T, D)

        # action [10]: pos(3) → range, rot_6d(6) → identity, gripper(1) → range
        action_normalizers = [
            get_range_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., :3])),        # position
            get_identity_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., 3:9])),       # rotation 6d
            get_range_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., 9:10])),      # gripper
        ]
        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs normalizer: position/gripper → range, quat(6d) → identity
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key == 'position':
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == 'quat':
                # rotation_6d는 SO(3) 위에서 bounded → identity
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key == 'gripper':
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                print("UNKNOWN KEY in get_normalizer:", key)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        print("--------------- action normalizers -----------------")
        print("action scale:", normalizer['action'].params_dict['scale'].data)
        print("action offset:", normalizer['action'].params_dict['offset'].data)
        print("--------------- obs normalizers -----------------")
        for key in self.lowdim_keys:
            print(f"{key} scale:", normalizer[key].params_dict['scale'].data)
            print(f"{key} offset:", normalizer[key].params_dict['offset'].data)

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # obs는 n_obs_steps 만큼만 사용
        T_slice = slice(self.n_obs_steps)

        # image: (T,H,W,C) uint8 → (T,C,H,W) float32 [0,1]
        obs_dict = dict()
        for key in self.rgb_keys:
            if key not in data:
                continue
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # 마지막 obs 스텝의 pose/gripper를 relative 변환의 기준(base)으로 사용
        # replay_buffer의 quat은 raw quaternion [4]로 저장됨
        current_pos = copy.copy(obs_dict['position'][-1])
        current_rot_mat = copy.copy(self.rot_quat2mat.forward(obs_dict['quat'][-1]))
        current_gripper = np.asarray(obs_dict['gripper'][-1], dtype=np.float32)  # (1,) for gripper delta

        if self.obs_pose_repr == 'relative':
            # obs: pose matrix로 만든 뒤 convert_pose_mat_rep(relative) 적용
            obs_rot_mat = self.rot_quat2mat.forward(obs_dict['quat'])
            obs_pose_mat = _pos_rot_to_pose_mat(obs_dict['position'], obs_rot_mat)
            base_pose_mat = _pos_rot_to_pose_mat(
                current_pos[None], current_rot_mat[None]
            )[0]
            rel_pose_mat = convert_pose_mat_rep(
                obs_pose_mat, base_pose_mat, pose_rep='relative', backward=False
            )
            obs_dict['position'], obs_dict['quat'] = _pose_mat_to_pos_rot(
                rel_pose_mat, self.rot_mat2target.get('quat')
            )
        else:
            # abs: quat [4] → rotation_6d [6] 변환만 수행
            quat_transformer = self.rot_mat2target.get('quat', self.rot_quat2mat)
            obs_dict['quat'] = quat_transformer.forward(
                self.rot_quat2mat.forward(obs_dict['quat'])
            )


        obs_dict['position'] = obs_dict['position'].astype(np.float32)
        obs_dict['quat'] = obs_dict['quat'].astype(np.float32)

        if self.action_pose_repr == 'relative':
            # replay_buffer action: pos(3) + rot_6d(6) + gripper(1) = [10]
            action_pos = data['action'][..., :3]
            action_rot = data['action'][..., 3:9]
            action_gripper = data['action'][..., 9:10].astype(np.float32)
            if self.action_gripper_repr == 'relative':
                action_gripper = action_gripper - current_gripper  # delta from current obs gripper

            # action pose matrix로 만든 뒤 convert_pose_mat_rep(relative) 적용
            action_rot_mat = self.rot_6d2mat.forward(action_rot)
            action_pose_mat = _pos_rot_to_pose_mat(action_pos, action_rot_mat)
            base_pose_mat = _pos_rot_to_pose_mat(
                current_pos[None], current_rot_mat[None]
            )[0]
            rel_pose_mat = convert_pose_mat_rep(
                action_pose_mat, base_pose_mat, pose_rep='relative', backward=False
            )
            action_pos, action_rot = _pose_mat_to_pos_rot(
                rel_pose_mat, self.rot_mat2target['action']
            )
            data['action'] = np.concatenate([
                action_pos.astype(np.float32),
                action_rot.astype(np.float32),
                action_gripper.astype(np.float32),
            ], axis=-1)
        elif self.action_gripper_repr == 'relative':
            # pose는 absolute 유지, gripper만 current obs 기준 delta로 변환
            action_gripper = data['action'][..., 9:10].astype(np.float32) - current_gripper
            data['action'] = np.concatenate([
                data['action'][..., :9].astype(np.float32),
                action_gripper.astype(np.float32),
            ], axis=-1)
            
        print("--------------- action -----------------")
        print("action_pos:", action_pos)
        print("action_rot:", action_rot)
        print("action_gripper:", action_gripper)
        print("data['action']:")
        print(data['action'])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True
        elif raw_actions.shape[-1] == 18:
            # dual arm with 6D rotation, no gripper - already in correct format
            actions = raw_actions.astype(np.float32)
            return actions
        elif raw_actions.shape[-1] == 10:
            # single arm: already pos(3) + rot_6d(6) + gripper(1)
            actions = raw_actions.astype(np.float32)
            return actions

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                if 'quat' in key:
                    assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['raw_shape'])
                else:
                    assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
