import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ActionNormStats:
    action_min: np.ndarray
    action_max: np.ndarray

    @property
    def action_range(self) -> np.ndarray:
        return np.maximum(self.action_max - self.action_min, 1e-6)

    def to_dict(self) -> Dict[str, list]:
        return {
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
        }

    @staticmethod
    def from_dict(stats: Dict[str, Sequence[float]]) -> "ActionNormStats":
        return ActionNormStats(
            action_min=np.asarray(stats["action_min"], dtype=np.float32),
            action_max=np.asarray(stats["action_max"], dtype=np.float32),
        )


class DualArmSequenceDataset(Dataset):
    """
    Windowed dataset for diffusion policy:
    condition: observation history [H, D_obs]
    target: action chunk [K, D_act]
    """

    def __init__(
        self,
        obs_path: Path,
        action_path: Path,
        history: int,
        horizon: int,
        stride: int = 1,
        episode_indices: Optional[np.ndarray] = None,
        norm_stats: Optional[ActionNormStats] = None,
        mmap_mode: str = "r",
    ) -> None:
        super().__init__()
        self.obs = np.load(obs_path, mmap_mode=mmap_mode)
        self.actions = np.load(action_path, mmap_mode=mmap_mode)
        if self.obs.shape[:2] != self.actions.shape[:2]:
            raise ValueError(
                f"obs shape {self.obs.shape} and action shape {self.actions.shape} mismatch"
            )
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        if self.history < 1 or self.horizon < 1:
            raise ValueError("history and horizon must be >= 1")

        self.num_episodes, self.ep_len, self.obs_dim = self.obs.shape
        _, _, self.act_dim = self.actions.shape

        if episode_indices is None:
            self.episode_indices = np.arange(self.num_episodes, dtype=np.int32)
        else:
            self.episode_indices = np.asarray(episode_indices, dtype=np.int32)
        if len(self.episode_indices) == 0:
            raise ValueError("episode_indices is empty")

        if norm_stats is None:
            selected_actions = self.actions[self.episode_indices]
            action_min = np.min(selected_actions, axis=(0, 1)).astype(np.float32)
            action_max = np.max(selected_actions, axis=(0, 1)).astype(np.float32)
            self.norm_stats = ActionNormStats(action_min=action_min, action_max=action_max)
        else:
            self.norm_stats = norm_stats

        self.index = self._build_index()
        if len(self.index) == 0:
            raise ValueError(
                "No valid windows. Check history/horizon values against episode length."
            )

    def _build_index(self) -> np.ndarray:
        # valid chunk starts at t, needs obs[t-history+1:t+1] and act[t:t+horizon]
        t_start = self.history - 1
        t_end = self.ep_len - self.horizon
        if t_start > t_end:
            return np.empty((0, 2), dtype=np.int32)

        pairs = []
        for ep in self.episode_indices:
            for t in range(t_start, t_end + 1, self.stride):
                pairs.append((int(ep), int(t)))
        return np.asarray(pairs, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.index)

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        s = self.norm_stats
        return (2.0 * (actions - s.action_min) / s.action_range) - 1.0

    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        s = self.norm_stats
        return ((actions + 1.0) * 0.5) * s.action_range + s.action_min

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep, t = self.index[idx]
        obs_hist = self.obs[ep, t - self.history + 1 : t + 1]  # [H, D_obs]
        action_chunk = self.actions[ep, t : t + self.horizon]  # [K, D_act]

        action_chunk = self.normalize_actions(action_chunk.astype(np.float32))
        obs_hist = obs_hist.astype(np.float32)

        return {
            "obs": torch.from_numpy(obs_hist),
            "action": torch.from_numpy(action_chunk),
            "episode": torch.tensor(ep, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }


def _load_stats_from_json(stats_path: Optional[Path]) -> Optional[ActionNormStats]:
    if stats_path is None:
        return None
    if not stats_path.exists():
        return None
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if "action_min" in stats and "action_max" in stats:
        return ActionNormStats.from_dict(stats)
    return None


def create_train_val_datasets(
    obs_path: Path,
    action_path: Path,
    history: int,
    horizon: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    stride: int = 1,
    stats_path: Optional[Path] = None,
    train_episode_limit: int = 0,
) -> Tuple[DualArmSequenceDataset, DualArmSequenceDataset]:
    obs_mm = np.load(obs_path, mmap_mode="r")
    num_eps = obs_mm.shape[0]
    if num_eps < 2:
        raise ValueError("Need at least 2 episodes for train/val split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_eps)
    n_val = max(1, int(round(num_eps * val_ratio)))
    n_val = min(n_val, num_eps - 1)

    val_eps = perm[:n_val]
    train_eps = perm[n_val:]
    if train_episode_limit and train_episode_limit > 0:
        limit = min(int(train_episode_limit), len(train_eps))
        train_eps = train_eps[:limit]

    base_stats = _load_stats_from_json(stats_path)
    if base_stats is None:
        act_mm = np.load(action_path, mmap_mode="r")
        train_actions = act_mm[train_eps]
        action_min = np.min(train_actions, axis=(0, 1)).astype(np.float32)
        action_max = np.max(train_actions, axis=(0, 1)).astype(np.float32)
        base_stats = ActionNormStats(action_min=action_min, action_max=action_max)

    train_ds = DualArmSequenceDataset(
        obs_path=obs_path,
        action_path=action_path,
        history=history,
        horizon=horizon,
        stride=stride,
        episode_indices=train_eps,
        norm_stats=base_stats,
    )
    val_ds = DualArmSequenceDataset(
        obs_path=obs_path,
        action_path=action_path,
        history=history,
        horizon=horizon,
        stride=stride,
        episode_indices=val_eps,
        norm_stats=base_stats,
    )
    return train_ds, val_ds
