from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import DiffusionScheduler


@dataclass
class PolicyConfig:
    history: int
    horizon: int
    action_dim: int
    obs_dim: int


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        scheduler: DiffusionScheduler,
        history: int,
        horizon: int,
        action_dim: int,
        obs_dim: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.config = PolicyConfig(
            history=history,
            horizon=horizon,
            action_dim=action_dim,
            obs_dim=obs_dim,
        )

    def compute_loss(self, obs_hist: torch.Tensor, action_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = action_chunk.shape[0]
        device = action_chunk.device
        t = self.scheduler.sample_timesteps(bsz, device)
        noisy_action, noise = self.scheduler.add_noise(action_chunk, t)
        pred_noise = self.model(noisy_action, obs_hist, t)
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss}

    @torch.no_grad()
    def sample_action_chunk(self, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.scheduler.sample(
            model=self.model,
            obs_hist=obs_hist,
            horizon=self.config.horizon,
            action_dim=self.config.action_dim,
        )
