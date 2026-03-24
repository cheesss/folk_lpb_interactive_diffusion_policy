from __future__ import annotations

from typing import Dict, Optional

import dill
import hydra
import torch

from diffusion_policy.common.ood_utils import (
    chunked_min_l2,
    encode_policy_obs,
    encode_policy_obs_window,
)
from diffusion_policy.real_world.ood_monitor import (
    _infer_current_encoding_spec,
    _validate_shape_meta_compat,
)


class LPBPlanner:
    """
    Planner used by LPB-style guided denoising.

    It mirrors the original LPB control loop at a high level:
    1. measure current distance to the expert latent bank,
    2. predict a future latent under a candidate action chunk,
    3. backpropagate the future nearest-neighbor cost to the denoising sample.
    """

    def __init__(
        self,
        policy,
        bank_path: str,
        dynamics_ckpt: Optional[str],
        expected_shape_meta: Optional[dict] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        chunk_size: int = 4096,
        action_horizon: Optional[int] = None,
        action_start_idx: Optional[int] = None,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        bank_payload = torch.load(bank_path, map_location=self.device)
        self.reference_bank = bank_payload.get("current_latents", bank_payload["latents"]).to(self.device)
        self.reference_stats = bank_payload.get("current_stats", bank_payload["stats"])
        self.step_reference_bank = bank_payload.get("step_latents", self.reference_bank).to(self.device)
        self.step_reference_stats = bank_payload.get("step_stats", self.reference_stats)
        self.lowdim_keys = bank_payload["lowdim_keys"]
        self.bank_shape_meta = bank_payload.get("shape_meta")
        self.bank_n_obs_steps = bank_payload.get("n_obs_steps")
        self.bank_active_obs_keys = bank_payload.get("active_obs_keys")
        if expected_shape_meta is not None and self.bank_shape_meta is not None:
            _validate_shape_meta_compat(self.bank_shape_meta, expected_shape_meta)

        policy_n_obs_steps = getattr(self.policy, "n_obs_steps", None)
        if self.bank_n_obs_steps is not None and policy_n_obs_steps is not None:
            if int(self.bank_n_obs_steps) != int(policy_n_obs_steps):
                raise ValueError(
                    "LPB bank n_obs_steps mismatch. "
                    f"bank={self.bank_n_obs_steps}, runtime_policy={policy_n_obs_steps}"
                )

        self.current_encoding_spec = _infer_current_encoding_spec(
            policy=self.policy,
            bank_dim=int(self.reference_bank.shape[-1]),
            lowdim_keys=self.lowdim_keys,
            active_obs_keys=self.bank_active_obs_keys,
        )

        self.threshold = float(
            threshold if threshold is not None else self.reference_stats.get("p95", 1.0)
        )

        self.dynamics_model = None
        if dynamics_ckpt is not None:
            payload = torch.load(open(dynamics_ckpt, "rb"), pickle_module=dill, map_location=self.device)
            cfg = payload["cfg"]
            self.dynamics_model = hydra.utils.instantiate(cfg.model)
            self.dynamics_model.load_state_dict(payload["state_dicts"]["model"])
            self.dynamics_model.to(self.device).eval()

        max_dyn_horizon = getattr(self.dynamics_model, "max_action_horizon", None)
        if action_horizon is None:
            action_horizon = getattr(self.policy, "n_action_steps", 1)
        if max_dyn_horizon is not None:
            action_horizon = min(int(action_horizon), int(max_dyn_horizon))
        self.action_horizon = int(action_horizon)
        self.action_start_idx = (
            int(action_start_idx)
            if action_start_idx is not None
            else max(int(getattr(self.policy, "n_obs_steps", 1)) - 1, 0)
        )

    def _get_last_step_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: (value[:, -1] if value.ndim >= 3 else value)
            for key, value in obs_dict.items()
        }

    def _encode_current_for_bank(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.current_encoding_spec["mode"] == "window":
            latent, _ = encode_policy_obs_window(self.policy, obs_dict, self.lowdim_keys)
            return latent.index_select(
                1, self.current_encoding_spec["window_indices"].to(latent.device)
            )

        last_obs = self._get_last_step_obs(obs_dict)
        latent, _ = encode_policy_obs(self.policy, last_obs, self.lowdim_keys)
        return latent.index_select(
            1, self.current_encoding_spec["step_indices"].to(latent.device)
        )

    def compute_current_reward(self, current_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            latent = self._encode_current_for_bank(current_obs)
            raw = chunked_min_l2(latent, self.reference_bank, self.chunk_size)
        return -raw

    def _select_action_chunk(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 3:
            raise ValueError(f"Expected sample shape [B, T, D], got {tuple(sample.shape)}")
        start = min(self.action_start_idx, sample.shape[1] - 1)
        end = min(sample.shape[1], start + self.action_horizon)
        if end <= start:
            raise ValueError(
                f"Invalid LPB action slice start={start}, end={end}, sample_shape={tuple(sample.shape)}"
            )
        return sample[:, start:end, : self.policy.action_dim]

    def compute_loss(self, sample: torch.Tensor, current_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.dynamics_model is None:
            raise RuntimeError("LPB planner requires a dynamics checkpoint for steering.")

        last_obs = self._get_last_step_obs(current_obs)
        latent, proprio = encode_policy_obs(self.policy, last_obs, self.lowdim_keys)
        action_seq = self._select_action_chunk(sample)
        pred_latent, _ = self.dynamics_model(latent, proprio, action_seq)
        pred_latent = pred_latent.index_select(
            1, self.current_encoding_spec["step_indices"].to(pred_latent.device)
        )
        raw = chunked_min_l2(pred_latent, self.step_reference_bank, self.chunk_size)
        return raw.mean()
