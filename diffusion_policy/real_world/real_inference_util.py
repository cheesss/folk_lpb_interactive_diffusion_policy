from typing import Dict, Callable, Tuple
import copy
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer


def _pos_rot_to_pose_mat(pos, rot_mat):
    """(..., 3), (..., 3, 3) -> (..., 4, 4) homogeneous pose matrix."""
    lead = rot_mat.shape[:-2]
    mat = np.zeros(lead + (4, 4), dtype=pos.dtype)
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape   # t: image수 (time step)
            co,ho,wo = shape   # cfg에 정의된 shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            # 이거는 pushT 전용; env_obs에는 3D pose 저장이라서 x,y만 사용
            # if 'pose' in key and shape == (2,):
            #     # take X,Y coordinates
            #     this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
    return obs_dict_np

# obs에서 image의 해상도 출력 (width, height)
def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_relative_obs_dict(
        env_obs: Dict[str, np.ndarray],
        shape_meta: dict,
        rot_quat2mat: RotationTransformer,
        rot_mat2target: dict,
        obs_pose_repr: str = 'relative',
        ) -> Dict[str, np.ndarray]:
    """단일 팔 단팔용 obs dict 생성.
    obs_pose_repr == 'relative': position/quat을 마지막 obs 스텝 기준 relative로 변환하고
    quat을 rotation_6d(6D)로 출력한다.
    obs_pose_repr == 'abs': 이미지/gripper는 그대로, quat[4]만 6D로 변환.
    """
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']

    # 이미지 및 나머지 lowdim(position/quat 제외) 수집
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if obs_type == 'rgb':
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi, hi),
                    output_res=(wo, ho),
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)  # THWC → TCHW
        elif obs_type == 'low_dim' and key not in ('position', 'quat'):
            obs_dict_np[key] = env_obs[key]

    if obs_pose_repr == 'relative':
        # 마지막 스텝을 기준(base)으로 relative pose 계산
        current_pos = copy.copy(env_obs['position'][-1])               # (3,)
        current_rot_mat = copy.copy(rot_quat2mat.forward(env_obs['quat'][-1]))  # (3,3)
        base_pose_mat = _pos_rot_to_pose_mat(
            current_pos[None], current_rot_mat[None])[0]               # (4,4)

        obs_rot_mat = rot_quat2mat.forward(env_obs['quat'])             # (T,3,3)
        obs_pose_mat = _pos_rot_to_pose_mat(env_obs['position'], obs_rot_mat)   # (T,4,4)
        rel_pose_mat = convert_pose_mat_rep(
            obs_pose_mat, base_pose_mat, pose_rep='relative', backward=False)

        obs_dict_np['position'] = rel_pose_mat[..., :3, 3].astype(np.float32)
        obs_dict_np['quat'] = rot_mat2target['quat'].forward(
            rel_pose_mat[..., :3, :3]).astype(np.float32)              # (T,6)
    else:
        # abs: quat[4] → rotation_6d[6]만 변환
        obs_dict_np['position'] = env_obs['position'].astype(np.float32)
        obs_dict_np['quat'] = rot_mat2target['quat'].forward(
            rot_quat2mat.forward(env_obs['quat'])).astype(np.float32)  # (T,6)

    return obs_dict_np


def get_real_relative_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray],
        action_pose_repr: str,
        action_gripper_repr: str,
        rot_quat2mat: RotationTransformer,
        rot_6d2mat: RotationTransformer,
        rot_mat2target: dict,
        ) -> np.ndarray:
    """단일 팔 단팔용 relative action → absolute action 변환.
    action: (T, 10) = pos(3) + rot_6d(6) + gripper(1), policy 출력값.
    두 플래그 모두 'abs'이면 action을 그대로 반환(no-op).
    """
    if action_pose_repr != 'relative' and action_gripper_repr != 'relative':
        return action

    action_pos = action[..., :3].copy()
    action_rot_6d = action[..., 3:9].copy()
    action_gripper = action[..., 9:10].copy()

    # 현재 obs 마지막 스텝으로 base_pose_mat 생성
    current_pos = env_obs['position'][-1]
    current_rot_mat = rot_quat2mat.forward(env_obs['quat'][-1])
    base_pose_mat = _pos_rot_to_pose_mat(
        current_pos[None], current_rot_mat[None])[0].astype(np.float32)  # (4,4)

    if action_pose_repr == 'relative':
        # relative pose → absolute pose (backward=True)
        action_rot_mat = rot_6d2mat.forward(action_rot_6d)                    # (T,3,3)
        rel_pose_mat = _pos_rot_to_pose_mat(action_pos, action_rot_mat)       # (T,4,4)
        abs_pose_mat = convert_pose_mat_rep(
            rel_pose_mat.astype(np.float64),
            base_pose_mat.astype(np.float64),
            pose_rep='relative', backward=True)
        action_pos = abs_pose_mat[..., :3, 3].astype(np.float32)
        action_rot_6d = rot_mat2target['action'].forward(
            abs_pose_mat[..., :3, :3]).astype(np.float32)                     # (T,6)

    if action_gripper_repr == 'relative':
        # delta gripper → absolute gripper
        action_gripper = (action_gripper + env_obs['gripper'][-1]).astype(np.float32)

    return np.concatenate([action_pos, action_rot_6d, action_gripper], axis=-1)
