#!/home/vision/anaconda3/envs/robodiff/bin/python

"""
원본 rb10_eval_real_robot.py를 건드리지 않고,
실시간 OOD 모니터링과 시각화를 분리한 엔트리포인트.
"""

import time
import sys
import threading
import queue
import termios
import tty
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import imageio.v2 as imageio
from omegaconf import OmegaConf

from diffusion_policy.real_world.ood_monitor import OODMonitor
from diffusion_policy.real_world.rb10_real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
    get_real_relative_action,
    get_real_relative_obs_dict,
)
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)

LPB_DEFAULT_PRESETS = {
    "pusht": {"guidance_scale": 0.05, "threshold": 3.2, "guidance_num_steps": 10},
    "square": {"guidance_scale": 0.05, "threshold": 5.0, "guidance_num_steps": 10},
    "tool_hang": {"guidance_scale": 0.05, "threshold": 1.4, "guidance_num_steps": 10},
    "transport": {"guidance_scale": 0.2, "threshold": 2.8, "guidance_num_steps": 10},
    "libero": {"guidance_scale": 0.2, "threshold": 1.1, "guidance_num_steps": 10},
    # Real-robot default: paper uses 16-step DDIM with guidance over the final five steps.
    "real_robot_default": {"guidance_scale": 0.05, "threshold": None, "guidance_num_steps": 5},
    "son_pick_and_place_image": {"guidance_scale": 0.05, "threshold": None, "guidance_num_steps": 5},
    "son_pick_and_place_tissue_image_relative": {
        "guidance_scale": 0.05,
        "threshold": None,
        "guidance_num_steps": 5,
    },
}


def _get_last_step_obs(obs_dict):
    return dict_apply(obs_dict, lambda x: x[:, -1])


def _get_rgb_keys(shape_meta):
    rgb_keys = []
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "rgb":
            rgb_keys.append(key)
    return rgb_keys


def _normalize_vis_image(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


class _TerminalKeyReader:
    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._queue = queue.Queue()
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _reader(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not self._stop.is_set():
                ch = sys.stdin.read(1)
                if ch:
                    self._queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_key(self):
        if not self._enabled:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def _get_pose_repr_cfg(task_cfg):
    pose_repr_cfg = task_cfg.get("pose_repr", {})
    if OmegaConf.is_config(pose_repr_cfg):
        return OmegaConf.to_container(pose_repr_cfg, resolve=True)
    if isinstance(pose_repr_cfg, dict):
        return pose_repr_cfg
    return {}


def _resolve_lpb_defaults(task_name: str):
    normalized = (task_name or "").lower()
    if normalized in LPB_DEFAULT_PRESETS:
        return LPB_DEFAULT_PRESETS[normalized]
    for key, preset in LPB_DEFAULT_PRESETS.items():
        if key == "real_robot_default":
            continue
        if key in normalized:
            return preset
    return LPB_DEFAULT_PRESETS["real_robot_default"]


def _validate_obs_dict_np(obs_dict_np, shape_meta):
    expected_obs = shape_meta["obs"]
    actual_keys = set(obs_dict_np.keys())
    expected_keys = set(expected_obs.keys())

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"Observation key mismatch. missing={missing}, extra={extra}")

    for key, attr in expected_obs.items():
        expected_type = attr.get("type", "low_dim")
        expected_shape = tuple(attr["shape"])
        value = obs_dict_np[key]

        if value.ndim != len(expected_shape) + 1:
            raise ValueError(
                f"Observation rank mismatch for '{key}': "
                f"expected {len(expected_shape) + 1} dims (T + feature), got {value.ndim} with shape {tuple(value.shape)}"
            )

        actual_feature_shape = tuple(value.shape[1:])
        if actual_feature_shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch for '{key}': "
                f"expected per-step shape {expected_shape}, got {actual_feature_shape}"
            )

        if expected_type == "rgb":
            if value.dtype not in (np.float32, np.float64, np.uint8):
                raise ValueError(f"Unexpected dtype for rgb key '{key}': {value.dtype}")
        elif expected_type == "low_dim":
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"Unexpected dtype for low_dim key '{key}': {value.dtype}")
        else:
            raise RuntimeError(f"Unsupported obs type in shape_meta: {expected_type}")


def _compute_plot_bounds(values, min_span=0.15, padding_ratio=0.2):
    if len(values) == 0:
        return 0.0, 1.0

    values_np = np.asarray(values, dtype=np.float32)
    y_min = float(values_np.min())
    y_max = float(values_np.max())
    span = max(y_max - y_min, float(min_span))
    pad = span * float(padding_ratio)
    center = 0.5 * (y_min + y_max)
    lower = center - 0.5 * span - pad
    upper = center + 0.5 * span + pad
    if upper <= lower + 1e-6:
        upper = lower + 1.0
    return lower, upper


def _compute_bound_series(values):
    lower_series = []
    upper_series = []
    for idx in range(1, len(values) + 1):
        lower, upper = _compute_plot_bounds(values[:idx])
        lower_series.append(lower)
        upper_series.append(upper)
    return lower_series, upper_series


def _draw_polyline(canvas, values, x0, y0, x1, y1, y_min, y_span, color, thickness):
    if len(values) < 2:
        return
    xs = np.linspace(x0, x1, num=len(values))
    ys = []
    for value in values:
        t = np.clip((value - y_min) / y_span, 0.0, 1.0)
        ys.append(y1 - t * (y1 - y0))
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _downsample_series(values, max_points):
    if max_points <= 0 or len(values) <= max_points:
        return list(values)
    indices = np.linspace(0, len(values) - 1, num=max_points).round().astype(np.int64)
    indices = np.unique(indices)
    return [values[idx] for idx in indices]


def _draw_metric_plot(values, width, height, color):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    outer_margin = 16
    x0 = outer_margin
    y0 = outer_margin
    x1 = width - outer_margin - 1
    y1 = height - outer_margin - 1

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (205, 212, 220), 1)
    y_min, y_max = _compute_plot_bounds(values)
    y_span = max(y_max - y_min, 1e-6)
    lower_series, upper_series = _compute_bound_series(values)

    tick_values = np.linspace(y_min, y_max, num=5)
    for tick in tick_values:
        y_tick = int(y1 - ((tick - y_min) / y_span) * (y1 - y0))
        cv2.line(canvas, (x0, y_tick), (x1, y_tick), (228, 232, 236), 1, cv2.LINE_AA)

    _draw_polyline(
        canvas=canvas,
        values=upper_series,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        y_min=y_min,
        y_span=y_span,
        color=(0, 0, 255),
        thickness=1,
    )
    _draw_polyline(
        canvas=canvas,
        values=lower_series,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        y_min=y_min,
        y_span=y_span,
        color=(255, 0, 0),
        thickness=1,
    )

    if len(values) >= 2:
        xs = np.linspace(x0, x1, num=len(values))
        ys = []
        for value in values:
            t = np.clip((value - y_min) / y_span, 0.0, 1.0)
            ys.append(y1 - t * (y1 - y0))
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), 4, color, -1, cv2.LINE_AA)
    elif len(values) == 1:
        y_val = int(y1 - np.clip((values[0] - y_min) / y_span, 0.0, 1.0) * (y1 - y0))
        cv2.circle(canvas, (x1, y_val), 4, color, -1, cv2.LINE_AA)

    return canvas


def _resize_keep_aspect(image, target_height):
    src_h, src_w = image.shape[:2]
    if src_h == target_height:
        return image
    target_width = max(1, int(round(src_w * (target_height / src_h))))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _render_ood_dashboard(
    frame_bgr,
    current_series,
    predicted_series,
    current_score,
    predicted_score,
    recent_window,
):
    dashboard_height = 760
    border_color = (180, 180, 180)
    panel_width = 1280
    panel = np.full((dashboard_height, panel_width, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, dashboard_height - 1), border_color, 2)

    card_x = 18
    card_w = panel_width - 2 * card_x
    plot_margin_y = 18
    score_h = 22
    plot_gap = 14
    full_plot_h = 430
    recent_plot_h = dashboard_height - 2 * plot_margin_y - score_h - plot_gap - full_plot_h
    full_series = _downsample_series(current_series, max(2, card_w - 32))
    recent_series = current_series[-recent_window:] if recent_window > 0 else current_series

    full_plot = _draw_metric_plot(
        full_series,
        card_w,
        full_plot_h,
        color=(0, 120, 255),
    )
    recent_plot = _draw_metric_plot(
        recent_series,
        card_w,
        recent_plot_h,
        color=(0, 120, 255),
    )
    full_y0 = plot_margin_y
    recent_y0 = full_y0 + full_plot_h + plot_gap
    panel[full_y0:full_y0 + full_plot_h, card_x:card_x + card_w] = full_plot
    panel[recent_y0:recent_y0 + recent_plot_h, card_x:card_x + card_w] = recent_plot

    score_text = f"{current_score:.3f}"
    score_size, _ = cv2.getTextSize(
        score_text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        thickness=1,
    )
    score_x = card_x + (card_w - score_size[0]) // 2
    score_y = recent_y0 + recent_plot_h + score_size[1]
    cv2.putText(
        panel,
        score_text,
        (score_x, score_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(120, 120, 120),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return panel


def _compose_ood_recording_frame(wrist_frame_bgr, graph_panel_bgr):
    target_height = graph_panel_bgr.shape[0]
    wrist_panel = _resize_keep_aspect(wrist_frame_bgr, target_height)
    border_color = (180, 180, 180)
    cv2.rectangle(
        wrist_panel,
        (0, 0),
        (wrist_panel.shape[1] - 1, wrist_panel.shape[0] - 1),
        border_color,
        2,
    )

    gap = 16
    out_width = wrist_panel.shape[1] + gap + graph_panel_bgr.shape[1]
    out = np.full((target_height, out_width, 3), 248, dtype=np.uint8)
    out[:, :wrist_panel.shape[1]] = wrist_panel
    out[:, wrist_panel.shape[1] + gap:] = graph_panel_bgr
    return out


_WINDOW_LAYOUT_READY = {}


def _show_dashboard_window(window_name, frame_bgr, anchor_window="Multi Cam Vis"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame_bgr)

    frame_h, frame_w = frame_bgr.shape[:2]
    cv2.resizeWindow(window_name, frame_w, frame_h)

    if _WINDOW_LAYOUT_READY.get(window_name):
        return

    anchored = False
    try:
        anchor_x, anchor_y, anchor_w, _ = cv2.getWindowImageRect(anchor_window)
        if anchor_w > 0:
            cv2.moveWindow(window_name, anchor_x + anchor_w + 24, anchor_y)
            anchored = True
    except cv2.error:
        pass

    if not anchored:
        cv2.moveWindow(window_name, 920, 60)

    _WINDOW_LAYOUT_READY[window_name] = True


def _write_ood_video(out_path, frames_bgr, fps):
    out_path = str(out_path)
    try:
        with imageio.get_writer(out_path, fps=fps, format="FFMPEG") as writer:
            for frame_bgr in frames_bgr:
                writer.append_data(frame_bgr[..., ::-1])
        return out_path
    except Exception:
        fallback = out_path + ".gif" if not out_path.lower().endswith(".gif") else out_path
        with imageio.get_writer(fallback, fps=fps) as writer:
            for frame_bgr in frames_bgr:
                writer.append_data(frame_bgr[..., ::-1])
        return fallback


def _save_ood_video_if_needed(output_dir, episode_idx, frames_bgr, fps):
    if len(frames_bgr) == 0:
        return None
    out_dir = pathlib.Path(output_dir).joinpath("ood_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(f"episode_{episode_idx:04d}.mp4")
    return _write_ood_video(out_path, frames_bgr, fps)


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--robot_ip", "-ri", default="192.168.111.50", required=True, help="UR5 IP address")
@click.option("--match_dataset", "-m", default=None, help="Dataset used to overlay and adjust initial condition")
@click.option("--match_episode", "-me", default=None, type=int, help="Match specific episode from the match dataset")
@click.option("--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize.")
@click.option("--init_joints", "-j", is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option("--steps_per_inference", "-si", default=6, type=int, help="Action horizon for inference.")
@click.option("--max_duration", "-md", default=60, help="Max duration for each epoch in seconds.")
@click.option("--frequency", "-f", default=10, type=float, help="Control frequency in Hz.")
@click.option("--command_latency", "-cl", default=0.01, type=float, help="Latency between receiving command and executing on robot in seconds.")
@click.option("--oodf", default=None, help="OOD dynamics checkpoint path")
@click.option("--ood_bank", default=None, help="Expert latent bank path")
@click.option("--ood_threshold", default=None, type=float, help="Raw OOD threshold override")
@click.option("--ood_vis_every", default=1, type=int, help="Save OOD frames every N loops")
@click.option("--ood_plot_window", default=120, type=int, help="Number of recent OOD points to plot")
@click.option("--ood_chunk_size", default=4096, type=int, help="Chunk size for kNN OOD distance")
@click.option("--steer/--no-steer", default=True, help="Enable LPB steering during policy denoising.")
@click.option("--planner_target", default="diffusion_policy.policy.lpb_original_planner.LPBOriginalPlanner", help="Planner class used for LPB guidance.")
@click.option("--guidance_scale", default=None, type=float, help="LPB guidance scale override.")
@click.option("--guidance_num_steps", default=None, type=int, help="Number of final denoising steps that receive LPB guidance.")
@click.option("--lpb_demo_dataset", default=None, help="Expert demonstration dataset path for original-style LPB planner.")
@click.option("--debug_timing", is_flag=True, default=False, help="Print timing markers for debugging")
def main(
    input,
    output,
    robot_ip,
    match_dataset,
    match_episode,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    command_latency,
    oodf,
    ood_bank,
    ood_threshold,
    ood_vis_every,
    ood_plot_window,
    ood_chunk_size,
    steer,
    planner_target,
    guidance_scale,
    guidance_num_steps,
    lpb_demo_dataset,
    debug_timing,
):
    ckpt_path = input
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset = 0
    delta_action = False
    if "diffusion" in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda")
        policy.eval().to(device)
        policy.num_inference_steps = 16
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    pose_repr = _get_pose_repr_cfg(cfg.task)
    obs_pose_repr = pose_repr.get("obs_pose_repr", "abs")
    action_pose_repr = pose_repr.get("action_pose_repr", "abs")
    action_gripper_repr = pose_repr.get("action_gripper_repr", "abs")

    rot_quat2mat = RotationTransformer("quaternion", "matrix")
    rot_6d2mat = RotationTransformer("rotation_6d", "matrix")
    rot_mat2target = {}
    for key, attr in cfg.task.shape_meta["obs"].items():
        if "rotation_rep" in attr:
            rot_mat2target[key] = RotationTransformer("matrix", attr["rotation_rep"])

    use_repr_obs_dict = obs_pose_repr == "relative" or ("quat" in rot_mat2target)
    if obs_pose_repr == "relative" and "quat" not in rot_mat2target:
        raise KeyError(
            "task.shape_meta.obs.quat.rotation_rep is required when obs_pose_repr='relative'"
        )
    if action_pose_repr == "relative":
        action_rotation_rep = cfg.task.shape_meta["action"].get("rotation_rep", None)
        if action_rotation_rep is None:
            raise KeyError(
                "task.shape_meta.action.rotation_rep is required when action_pose_repr='relative'"
            )
        rot_mat2target["action"] = RotationTransformer("matrix", action_rotation_rep)

    preset = _resolve_lpb_defaults(cfg.task.name)
    if guidance_scale is None:
        guidance_scale = preset["guidance_scale"]
    if guidance_num_steps is None:
        guidance_num_steps = preset["guidance_num_steps"]
    if ood_threshold is None and preset["threshold"] is not None:
        ood_threshold = preset["threshold"]

    planner_allows_no_bank = "lpb_original_planner" in planner_target.lower()
    steering_ready = bool(
        steer and oodf is not None and (ood_bank is not None or planner_allows_no_bank)
    )
    if steering_ready:
        policy.initialize_planner(
            planner_target=planner_target,
            bank_path=ood_bank,
            dynamics_model_ckpt=oodf,
            threshold=ood_threshold,
            guidance_scale=guidance_scale,
            guidance_num_steps=guidance_num_steps,
            chunk_size=ood_chunk_size,
            device=str(device),
            action_horizon=steps_per_inference,
            expected_shape_meta=cfg.task.shape_meta,
            demo_dataset_path=lpb_demo_dataset or cfg.task.get("dataset_path", None),
        )
        print(
            "[LPB] steering enabled with "
            f"guidance_scale={guidance_scale}, guidance_num_steps={guidance_num_steps}, "
            f"threshold={'bank_p95' if ood_threshold is None else ood_threshold}"
        )

    ood_monitor = None
    if ood_bank is not None:
        ood_monitor = OODMonitor(
            policy=policy,
            bank_path=ood_bank,
            expected_shape_meta=cfg.task.shape_meta,
            dynamics_ckpt=oodf,
            threshold=ood_threshold,
            device=str(device),
            chunk_size=ood_chunk_size,
        )
    predicted_ood_supported = (
        ood_monitor is not None and getattr(ood_monitor, "dynamics_model", None) is not None
    )

    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    rgb_keys = _get_rgb_keys(cfg.task.shape_meta)
    if len(rgb_keys) == 0:
        raise RuntimeError("No RGB keys found in shape_meta")
    if vis_camera_idx >= len(rgb_keys):
        raise RuntimeError("vis_camera_idx out of range for RGB keys")
    vis_rgb_key = rgb_keys[vis_camera_idx]
    wrist_rgb_key = "image0" if "image0" in rgb_keys else vis_rgb_key
    episode_idx = 0

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(1)
            key_reader = _TerminalKeyReader()
            key_reader.start()

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                if use_repr_obs_dict:
                    obs_dict_np = get_real_relative_obs_dict(
                        env_obs=obs,
                        shape_meta=cfg.task.shape_meta,
                        rot_quat2mat=rot_quat2mat,
                        rot_mat2target=rot_mat2target,
                        obs_pose_repr=obs_pose_repr,
                    )
                else:
                    obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                _validate_obs_dict_np(obs_dict_np, cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(
                    obs_dict,
                    use_lpb_guidance=steering_ready,
                    current_obs=obs_dict,
                )
                action = result["action"][0].detach().to("cpu").numpy()
                action = get_real_relative_action(
                    action,
                    obs,
                    action_pose_repr,
                    action_gripper_repr,
                    rot_quat2mat,
                    rot_6d2mat,
                    rot_mat2target,
                )
                assert action.shape[-1] == 10
                if ood_monitor is not None:
                    _ = ood_monitor.score_current(obs_dict)
                    if predicted_ood_supported:
                        last_obs = _get_last_step_obs(obs_dict)
                        _ = ood_monitor.score_predicted(last_obs, result["action"])
                del result

            print("Ready!")
            stop_all = False
            ood_current_series = []
            ood_pred_series = []
            while True:
                ood_frames = []
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    episode_idx += 1
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    paused = False
                    last_pred_score = None
                    while True:
                        key = key_reader.get_key()
                        if key == "s":
                            if not paused:
                                paused = True
                                print("Paused. Press 'c' to continue, 'q' to quit.")
                        elif key == "c":
                            if paused:
                                paused = False
                                eval_t_start = time.time()
                                t_start = time.monotonic()
                                iter_idx = 0
                                print("Resumed.")
                        elif key == "q" or key == "\x03":
                            stop_all = True
                            print("Stopping and exiting.")
                            env.end_episode()
                            if ood_monitor is not None:
                                saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                                if saved is not None:
                                    print(f"Saved OOD video to {saved}")
                            break

                        if paused:
                            time.sleep(0.05)
                            continue

                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        if debug_timing:
                            print(f"[TIMING] before_get_obs t={time.time():.3f}")
                        print("get_obs")
                        obs = env.get_obs()
                        if debug_timing:
                            print(f"[TIMING] after_get_obs t={time.time():.3f}")
                        obs_timestamps = obs["timestamp"]
                        print(f"Obs latency {time.time() - obs_timestamps[-1]}")

                        with torch.no_grad():
                            if debug_timing:
                                print(f"[TIMING] before_policy_infer t={time.time():.3f}")
                            if use_repr_obs_dict:
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs,
                                    shape_meta=cfg.task.shape_meta,
                                    rot_quat2mat=rot_quat2mat,
                                    rot_mat2target=rot_mat2target,
                                    obs_pose_repr=obs_pose_repr,
                                )
                            else:
                                obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                            _validate_obs_dict_np(obs_dict_np, cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            should_run_inference = (iter_idx % steps_per_inference) == 0
                            result = None
                            action = None
                            if should_run_inference:
                                s = time.time()
                                result = policy.predict_action(
                                    obs_dict,
                                    use_lpb_guidance=steering_ready,
                                    current_obs=obs_dict,
                                )
                                action = result["action"][0].detach().to("cpu").numpy()
                                action = get_real_relative_action(
                                    action,
                                    obs,
                                    action_pose_repr,
                                    action_gripper_repr,
                                    rot_quat2mat,
                                    rot_6d2mat,
                                    rot_mat2target,
                                )
                                print("Inference latency:", time.time() - s)
                                if debug_timing:
                                    print(f"[TIMING] after_policy_infer t={time.time():.3f}")

                            if ood_monitor is not None:
                                current_metrics = ood_monitor.score_current(obs_dict)
                                predicted_metrics = None
                                if predicted_ood_supported and result is not None:
                                    last_obs = _get_last_step_obs(obs_dict)
                                    predicted_metrics = ood_monitor.score_predicted(last_obs, result["action"])

                                ood_current_series.append(current_metrics["normalized"])
                                pred_score = last_pred_score
                                if predicted_metrics is not None:
                                    pred_score = predicted_metrics["normalized"]
                                    last_pred_score = pred_score
                                    ood_pred_series.append(pred_score)
                                elif pred_score is not None and predicted_ood_supported:
                                    ood_pred_series.append(pred_score)

                                vis_frame = _render_ood_dashboard(
                                    frame_bgr=_normalize_vis_image(obs[vis_rgb_key][-1])[..., ::-1].copy(),
                                    current_series=ood_current_series,
                                    predicted_series=ood_pred_series,
                                    current_score=current_metrics["normalized"],
                                    predicted_score=pred_score,
                                    recent_window=ood_plot_window,
                                )
                                _show_dashboard_window("rb10_ood", vis_frame)
                                cv2.waitKey(1)
                                if (iter_idx % max(1, ood_vis_every)) == 0:
                                    wrist_frame = _normalize_vis_image(obs[wrist_rgb_key][-1])[..., ::-1].copy()
                                    ood_frames.append(_compose_ood_recording_frame(wrist_frame, vis_frame))
                            else:
                                _show_dashboard_window(
                                    "rb10_eval",
                                    _normalize_vis_image(obs[vis_rgb_key][-1])[..., ::-1],
                                )
                                cv2.waitKey(1)

                        if result is not None:
                            if delta_action:
                                assert len(action) == 1
                                if perv_target_pose is None:
                                    perv_target_pose = obs["robot_eef_pose"][-1]
                                this_target_pose = perv_target_pose.copy()
                                this_target_pose[[0, 1]] += action[-1]
                                perv_target_pose = this_target_pose
                                this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            else:
                                this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                                this_target_poses[:, : action.shape[-1]] = action

                            action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)

                            if np.sum(is_new) == 0:
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print("Over budget", action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            if debug_timing:
                                print(f"[TIMING] before_exec_actions t={time.time():.3f}")
                            env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)
                            if debug_timing:
                                print(f"[TIMING] after_exec_actions t={time.time():.3f}")
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print("Terminated by the timeout!")

                        if terminate:
                            env.end_episode()
                            if ood_monitor is not None:
                                saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                                if saved is not None:
                                    print(f"Saved OOD video to {saved}")
                            break

                        if debug_timing:
                            print(f"[TIMING] before_wait t={time.time():.3f}")
                        precise_wait(t_cycle_end - frame_latency)
                        if debug_timing:
                            print(f"[TIMING] after_wait t={time.time():.3f}")
                        iter_idx += 1
                    if stop_all:
                        break

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    if ood_monitor is not None:
                        saved = _save_ood_video_if_needed(output, episode_idx, ood_frames, fps=frequency)
                        if saved is not None:
                            print(f"Saved OOD video to {saved}")

                print("Stopped.")
                if stop_all:
                    break
            key_reader.stop()


if __name__ == "__main__":
    main()
