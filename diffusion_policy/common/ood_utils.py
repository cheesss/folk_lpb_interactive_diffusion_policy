"""OOD 계산에서 공통으로 쓰는 관측 인코딩, 거리 계산, 점수 정규화 유틸리티."""

from typing import Dict, List, Tuple

import torch

from diffusion_policy.common.pytorch_util import dict_apply


def get_lowdim_keys(shape_meta: dict) -> List[str]:
    keys = []
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "low_dim":
            keys.append(key)
    return keys


def add_time_dim(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return dict_apply(obs_dict, lambda x: x.unsqueeze(1))


def stack_lowdim(obs_dict: Dict[str, torch.Tensor], lowdim_keys: List[str]) -> torch.Tensor:
    return torch.cat([obs_dict[key] for key in lowdim_keys], dim=-1)


def encode_policy_obs(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    lowdim_keys: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Dynamics 모델 학습/예측용 단일 step latent 인코딩.
    obs_seq = add_time_dim(obs_dict)
    nobs = policy.normalizer.normalize(obs_seq)
    encoder_in = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    latent = policy.obs_encoder(encoder_in)
    proprio = stack_lowdim({k: nobs[k][:, 0] for k in lowdim_keys}, lowdim_keys)
    return latent, proprio


def encode_policy_obs_window(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    lowdim_keys: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """정책이 실제로 보는 obs window 전체를 같은 방식으로 latent로 인코딩한다."""
    nobs = policy.normalizer.normalize(obs_dict)
    batch_size = next(iter(nobs.values())).shape[0]
    encoder_in = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    step_latent = policy.obs_encoder(encoder_in)
    window_latent = step_latent.reshape(batch_size, -1)
    proprio = torch.cat(
        [nobs[key].reshape(batch_size, -1) for key in lowdim_keys],
        dim=-1,
    )
    return window_latent, proprio


def normalize_action(policy, action: torch.Tensor) -> torch.Tensor:
    return policy.normalizer["action"].normalize(action)


def chunked_min_l2(
    query: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    mins = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start:start + chunk_size]
        best = None
        for bank_start in range(0, bank.shape[0], chunk_size):
            b = bank[bank_start:bank_start + chunk_size]
            dist = torch.cdist(q, b, p=2)
            min_dist = dist.min(dim=-1).values
            if best is None:
                best = min_dist
            else:
                best = torch.minimum(best, min_dist)
        mins.append(best)
    return torch.cat(mins, dim=0)


def compute_reference_stats(
    bank: torch.Tensor,
    chunk_size: int = 2048,
) -> Dict[str, float]:
    if bank.shape[0] < 2:
        return {
            "median": 0.0,
            "mean": 0.0,
            "std": 1.0,
            "p95": 1.0,
            "p99": 1.0,
        }

    distances = []
    for start in range(0, bank.shape[0], chunk_size):
        q = bank[start:start + chunk_size]
        best = None
        for bank_start in range(0, bank.shape[0], chunk_size):
            b = bank[bank_start:bank_start + chunk_size]
            dist = torch.cdist(q, b, p=2)
            if start == bank_start:
                diag_len = min(dist.shape[0], dist.shape[1])
                idx = torch.arange(diag_len, device=dist.device)
                dist[idx, idx] = float("inf")
            min_dist = dist.min(dim=-1).values
            if best is None:
                best = min_dist
            else:
                best = torch.minimum(best, min_dist)
        distances.append(best)

    distances = torch.cat(distances, dim=0)
    return {
        "median": float(torch.quantile(distances, 0.5).item()),
        "mean": float(distances.mean().item()),
        "std": float(distances.std(unbiased=False).item() + 1e-6),
        "p95": float(torch.quantile(distances, 0.95).item()),
        "p99": float(torch.quantile(distances, 0.99).item()),
    }


def normalize_ood_score(score: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    median = stats.get("median", 0.0)
    p95 = stats.get("p95", 1.0)
    denom = max(p95 - median, 1e-6)
    return torch.clamp((score - median) / denom, min=0.0)


def compute_reference_distance_percentiles(
    bank: torch.Tensor,
    chunk_size: int = 2048,
    sample_size: int = 2048,
) -> torch.Tensor:
    if bank.shape[0] < 2:
        return torch.zeros((1,), device=bank.device, dtype=bank.dtype)

    total = bank.shape[0]
    if sample_size is None or sample_size <= 0 or sample_size >= total:
        sample_indices = torch.arange(total, device=bank.device)
    else:
        sample_indices = torch.linspace(
            0, total - 1, steps=sample_size, device=bank.device
        ).round().long().unique()

    distances = []
    for start in range(0, sample_indices.shape[0], chunk_size):
        query_indices = sample_indices[start:start + chunk_size]
        q = bank[query_indices]
        best = None
        for bank_start in range(0, total, chunk_size):
            b = bank[bank_start:bank_start + chunk_size]
            dist = torch.cdist(q, b, p=2)

            relative_indices = query_indices - bank_start
            valid_self = (relative_indices >= 0) & (relative_indices < b.shape[0])
            if torch.any(valid_self):
                row_idx = torch.nonzero(valid_self, as_tuple=False).squeeze(-1)
                col_idx = relative_indices[valid_self]
                dist[row_idx, col_idx] = float("inf")

            min_dist = dist.min(dim=-1).values
            if best is None:
                best = min_dist
            else:
                best = torch.minimum(best, min_dist)
        distances.append(best)

    return torch.sort(torch.cat(distances, dim=0)).values


def percentile_rank(score: torch.Tensor, reference_distances: torch.Tensor) -> torch.Tensor:
    if reference_distances.numel() == 0:
        return torch.zeros_like(score)

    flattened = score.reshape(-1)
    ranks = torch.searchsorted(reference_distances, flattened, right=True)
    percentile = 100.0 * ranks.to(dtype=flattened.dtype) / reference_distances.numel()
    percentile = torch.clamp(percentile, 0.0, 100.0)
    return percentile.reshape(score.shape)
