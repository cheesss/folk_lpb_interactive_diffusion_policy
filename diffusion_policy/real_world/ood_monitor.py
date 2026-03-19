"""실시간 inference/teleop 중 current/predicted OOD 점수를 계산하는 런타임 모듈."""

from typing import Dict, Optional
import itertools

import dill
import hydra
import numpy as np
import torch

from diffusion_policy.common.ood_utils import (
    chunked_min_l2,
    compute_reference_distance_percentiles,
    encode_policy_obs,
    encode_policy_obs_window,
    normalize_action,
    normalize_ood_score,
    percentile_rank,
)


class OODMonitor:
    """Expert latent bank와 optional dynamics model을 묶어 OOD 점수를 계산한다."""
    def __init__(
        self,
        policy,
        bank_path: str,
        expected_shape_meta: Optional[dict] = None,
        dynamics_ckpt: Optional[str] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        chunk_size: int = 4096,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        bank_payload = torch.load(bank_path, map_location=self.device)
        self.reference_bank = bank_payload.get("current_latents", bank_payload["latents"]).to(self.device)
        self.reference_stats = bank_payload.get("current_stats", bank_payload["stats"])
        self.step_reference_bank = bank_payload.get("step_latents", self.reference_bank).to(self.device)
        self.step_reference_stats = bank_payload.get("step_stats", self.reference_stats)
        self.reference_percentiles = bank_payload.get("current_reference_percentiles")
        self.step_reference_percentiles = bank_payload.get("step_reference_percentiles")
        self.lowdim_keys = bank_payload["lowdim_keys"]
        self.bank_shape_meta = bank_payload.get("shape_meta")
        self.bank_n_obs_steps = bank_payload.get("n_obs_steps")
        self.bank_active_obs_keys = bank_payload.get("active_obs_keys")
        if self.bank_active_obs_keys is None and self.bank_shape_meta is not None:
            bank_obs_keys = list(self.bank_shape_meta.get("obs", {}).keys())
            if expected_shape_meta is None:
                self.bank_active_obs_keys = bank_obs_keys
            else:
                runtime_obs_keys = list(expected_shape_meta.get("obs", {}).keys())
                if set(bank_obs_keys) != set(runtime_obs_keys):
                    self.bank_active_obs_keys = bank_obs_keys
        if expected_shape_meta is not None and self.bank_shape_meta is not None:
            _validate_shape_meta_compat(self.bank_shape_meta, expected_shape_meta)
        policy_n_obs_steps = getattr(self.policy, "n_obs_steps", None)
        if self.bank_n_obs_steps is not None and policy_n_obs_steps is not None:
            if int(self.bank_n_obs_steps) != int(policy_n_obs_steps):
                raise ValueError(
                    "OOD bank n_obs_steps mismatch. "
                    f"bank={self.bank_n_obs_steps}, runtime_policy={policy_n_obs_steps}"
                )
        self.current_encoding_spec = _infer_current_encoding_spec(
            policy=self.policy,
            bank_dim=int(self.reference_bank.shape[-1]),
            lowdim_keys=self.lowdim_keys,
            active_obs_keys=self.bank_active_obs_keys,
        )
        self.current_encoding_mode = self.current_encoding_spec["mode"]
        if self.reference_percentiles is None:
            self.reference_percentiles = compute_reference_distance_percentiles(
                self.reference_bank,
                chunk_size=self.chunk_size,
                sample_size=min(2048, self.reference_bank.shape[0]),
            )
        else:
            self.reference_percentiles = self.reference_percentiles.to(self.device)

        if self.step_reference_percentiles is None:
            if self.step_reference_bank.data_ptr() == self.reference_bank.data_ptr():
                self.step_reference_percentiles = self.reference_percentiles
            else:
                self.step_reference_percentiles = compute_reference_distance_percentiles(
                    self.step_reference_bank,
                    chunk_size=self.chunk_size,
                    sample_size=min(2048, self.step_reference_bank.shape[0]),
                )
        else:
            self.step_reference_percentiles = self.step_reference_percentiles.to(self.device)

        self.threshold_raw = threshold if threshold is not None else self.reference_stats.get("p95", 1.0)
        self.threshold_normalized = float(
            normalize_ood_score(torch.tensor(self.threshold_raw), self.reference_stats).item()
        )
        self.threshold_percentile = float(
            percentile_rank(
                torch.tensor([self.threshold_raw], device=self.device, dtype=self.reference_percentiles.dtype),
                self.reference_percentiles,
            )[0].item()
        )
        self.step_threshold_percentile = float(
            percentile_rank(
                torch.tensor([self.threshold_raw], device=self.device, dtype=self.step_reference_percentiles.dtype),
                self.step_reference_percentiles,
            )[0].item()
        )

        self.dynamics_model = None
        if dynamics_ckpt is not None:
            payload = torch.load(open(dynamics_ckpt, "rb"), pickle_module=dill, map_location=self.device)
            cfg = payload["cfg"]
            self.dynamics_model = hydra.utils.instantiate(cfg.model)
            self.dynamics_model.load_state_dict(payload["state_dicts"]["model"])
            self.dynamics_model.to(self.device).eval()

    def score_current(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            if self.current_encoding_mode == "window":
                latent, proprio = encode_policy_obs_window(self.policy, obs_dict, self.lowdim_keys)
                latent = latent.index_select(
                    1, self.current_encoding_spec["window_indices"].to(latent.device)
                )
            else:
                last_obs = {key: value[:, -1] for key, value in obs_dict.items()}
                latent, proprio = encode_policy_obs(self.policy, last_obs, self.lowdim_keys)
                latent = latent.index_select(
                    1, self.current_encoding_spec["step_indices"].to(latent.device)
                )
            raw = chunked_min_l2(latent, self.reference_bank, self.chunk_size)
            normalized = normalize_ood_score(raw, self.reference_stats)
            percentile = percentile_rank(raw, self.reference_percentiles)
        return {
            "raw": float(raw[0].item()),
            "normalized": float(normalized[0].item()),
            "percentile": float(percentile[0].item()),
            "latent": latent,
            "proprio": proprio,
        }

    def score_predicted(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        if self.dynamics_model is None:
            return None

        with torch.no_grad():
            latent, proprio = encode_policy_obs(self.policy, obs_dict, self.lowdim_keys)
            if action_seq.ndim == 2:
                action_seq = action_seq.unsqueeze(0)
            norm_action_seq = normalize_action(self.policy, action_seq)
            pred_latent, pred_proprio = self.dynamics_model(latent, proprio, norm_action_seq)
            pred_latent = pred_latent.index_select(
                1, self.current_encoding_spec["step_indices"].to(pred_latent.device)
            )
            raw = chunked_min_l2(pred_latent, self.step_reference_bank, self.chunk_size)
            normalized = normalize_ood_score(raw, self.step_reference_stats)
            percentile = percentile_rank(raw, self.step_reference_percentiles)

        return {
            "raw": float(raw[0].item()),
            "normalized": float(normalized[0].item()),
            "percentile": float(percentile[0].item()),
            "future_proprio": pred_proprio[0].detach().cpu().tolist(),
        }


def _validate_shape_meta_compat(reference_shape_meta: dict, expected_shape_meta: dict):
    ref_obs = reference_shape_meta["obs"]
    exp_obs = expected_shape_meta["obs"]

    ref_keys = set(ref_obs.keys())
    exp_keys = set(exp_obs.keys())
    missing_in_runtime = sorted(ref_keys - exp_keys)
    if missing_in_runtime:
        raise ValueError(
            "OOD bank shape_meta mismatch. "
            f"bank_has_unknown_keys={missing_in_runtime}"
        )

    for key in sorted(ref_keys):
        ref_attr = ref_obs[key]
        exp_attr = exp_obs[key]
        ref_type = ref_attr.get("type", "low_dim")
        exp_type = exp_attr.get("type", "low_dim")
        ref_shape = tuple(ref_attr["shape"])
        exp_shape = tuple(exp_attr["shape"])
        if ref_type != exp_type or ref_shape != exp_shape:
            raise ValueError(
                f"OOD bank shape_meta mismatch for key '{key}': "
                f"bank(type={ref_type}, shape={ref_shape}) vs "
                f"runtime(type={exp_type}, shape={exp_shape})"
            )


def _infer_current_encoding_spec(policy, bank_dim: int, lowdim_keys, active_obs_keys=None) -> dict:
    feature_spec = _get_obs_encoder_feature_spec(policy)
    ordered_keys = feature_spec["ordered_keys"]
    step_dim = feature_spec["step_dim"]
    n_obs_steps = int(getattr(policy, "n_obs_steps", 1))
    window_dim = step_dim * n_obs_steps
    full_step_indices = torch.arange(step_dim, dtype=torch.long)
    full_window_indices = _repeat_step_indices(full_step_indices, step_dim, n_obs_steps)

    if bank_dim == window_dim:
        return {
            "mode": "window",
            "selected_keys": ordered_keys,
            "step_indices": full_step_indices,
            "window_indices": full_window_indices,
        }
    if bank_dim == step_dim:
        print(
            "[OOD] Legacy latent bank detected: using last-step latent for current OOD "
            f"(bank_dim={bank_dim}, step_dim={step_dim}, window_dim={window_dim})."
        )
        return {
            "mode": "step",
            "selected_keys": ordered_keys,
            "step_indices": full_step_indices,
            "window_indices": full_window_indices,
        }

    candidates = _find_subset_candidates(
        feature_spec=feature_spec,
        bank_dim=bank_dim,
        lowdim_keys=lowdim_keys,
        n_obs_steps=n_obs_steps,
        active_obs_keys=active_obs_keys,
    )
    if len(candidates) == 0:
        raise ValueError(
            "OOD bank latent dimension mismatch. "
            f"bank_dim={bank_dim}, step_dim={step_dim}, window_dim={window_dim}. "
            "Re-export the bank with the current policy or use a matching bank."
        )

    selected = candidates[0]
    if len(candidates) > 1:
        print(
            "[OOD] Multiple obs-key subsets match the bank latent dimension. "
            f"Using {selected['selected_keys']} for current OOD."
        )
    else:
        print(
            "[OOD] Obs-key subset bank detected. "
            f"Using {selected['selected_keys']} for current OOD."
        )
    return selected


def _get_obs_encoder_feature_spec(policy) -> dict:
    obs_encoder = policy.obs_encoder
    key_slices = {}
    start = 0

    if hasattr(obs_encoder, "obs_shapes") and hasattr(obs_encoder, "obs_nets"):
        ordered_keys = list(obs_encoder.obs_shapes.keys())
        for key in ordered_keys:
            feat_shape = obs_encoder.obs_shapes[key]
            randomizer = obs_encoder.obs_randomizers[key]
            net = obs_encoder.obs_nets[key]
            if randomizer is not None:
                feat_shape = randomizer.output_shape_in(feat_shape)
            if net is not None:
                feat_shape = net.output_shape(feat_shape)
            if randomizer is not None:
                feat_shape = randomizer.output_shape_out(feat_shape)
            dim = int(np.prod(feat_shape))
            key_slices[key] = slice(start, start + dim)
            start += dim
    else:
        ordered_keys = list(obs_encoder.rgb_keys) + list(obs_encoder.low_dim_keys)
        with torch.no_grad():
            for key in obs_encoder.rgb_keys:
                shape = tuple(obs_encoder.key_shape_map[key])
                dummy = torch.zeros((1,) + shape, dtype=obs_encoder.dtype, device=obs_encoder.device)
                dummy = obs_encoder.key_transform_map[key](dummy)
                if obs_encoder.share_rgb_model:
                    feature = obs_encoder.key_model_map["rgb"](dummy)
                else:
                    feature = obs_encoder.key_model_map[key](dummy)
                dim = int(np.prod(feature.shape[1:]))
                key_slices[key] = slice(start, start + dim)
                start += dim

        for key in obs_encoder.low_dim_keys:
            shape = tuple(obs_encoder.key_shape_map[key])
            dim = int(np.prod(shape))
            key_slices[key] = slice(start, start + dim)
            start += dim

    return {
        "ordered_keys": ordered_keys,
        "key_slices": key_slices,
        "step_dim": start,
    }


def _repeat_step_indices(step_indices: torch.Tensor, step_dim: int, n_obs_steps: int) -> torch.Tensor:
    all_indices = []
    for step_idx in range(n_obs_steps):
        all_indices.append(step_indices + step_idx * step_dim)
    return torch.cat(all_indices, dim=0)


def _find_subset_candidates(feature_spec, bank_dim, lowdim_keys, n_obs_steps, active_obs_keys=None):
    ordered_keys = feature_spec["ordered_keys"]
    key_slices = feature_spec["key_slices"]
    step_dim = feature_spec["step_dim"]
    lowdim_key_set = set(lowdim_keys)
    active_obs_keys = set(active_obs_keys) if active_obs_keys is not None else None

    if active_obs_keys is not None:
        subsets = [[key for key in ordered_keys if key in active_obs_keys]]
    else:
        subsets = []
        for r in range(1, len(ordered_keys) + 1):
            for subset in itertools.combinations(ordered_keys, r):
                subsets.append(list(subset))

    candidates = []
    for selected_subset in subsets:
        selected_keys = [key for key in ordered_keys if key in set(selected_subset)]
        if len(selected_keys) == 0:
            continue

        step_indices = []
        for key in selected_keys:
            slc = key_slices[key]
            step_indices.append(torch.arange(slc.start, slc.stop, dtype=torch.long))
        step_indices = torch.cat(step_indices, dim=0)
        subset_step_dim = int(step_indices.numel())
        contains_all_lowdim = lowdim_key_set.issubset(set(selected_keys))
        visual_count = sum(key not in lowdim_key_set for key in selected_keys)
        order_rank = tuple(ordered_keys.index(key) for key in selected_keys)

        if bank_dim == subset_step_dim:
            candidates.append({
                "mode": "step",
                "selected_keys": selected_keys,
                "step_indices": step_indices,
                "window_indices": _repeat_step_indices(step_indices, step_dim, n_obs_steps),
                "rank": (
                    0 if active_obs_keys is not None else 1,
                    0 if contains_all_lowdim else 1,
                    visual_count,
                    order_rank,
                ),
            })
        if bank_dim == subset_step_dim * n_obs_steps:
            candidates.append({
                "mode": "window",
                "selected_keys": selected_keys,
                "step_indices": step_indices,
                "window_indices": _repeat_step_indices(step_indices, step_dim, n_obs_steps),
                "rank": (
                    0 if active_obs_keys is not None else 1,
                    0 if contains_all_lowdim else 1,
                    visual_count,
                    order_rank,
                ),
            })

    if len(candidates) == 0:
        return candidates

    candidates.sort(key=lambda x: x["rank"])
    return candidates
