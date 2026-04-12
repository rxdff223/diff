from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DiffusionHyperParams:
    num_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionScheduler:
    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_steps = int(num_steps)
        betas = torch.linspace(beta_start, beta_end, self.num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-12)
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        bsz = t.shape[0]
        out = a.gather(0, t).reshape(bsz, *([1] * (len(x_shape) - 1)))
        return out

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_ac * x_start + sqrt_om * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        obs_hist: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        eps_theta = model(x_t, obs_hist, t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t.clamp(min=1e-12)
        )

        posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *([1] * (x_t.dim() - 1)))
        noise = torch.randn_like(x_t)
        return model_mean + nonzero_mask * torch.sqrt(posterior_var_t.clamp(min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        obs_hist: torch.Tensor,
        horizon: int,
        action_dim: int,
    ) -> torch.Tensor:
        bsz = obs_hist.shape[0]
        device = obs_hist.device
        x_t = torch.randn((bsz, horizon, action_dim), device=device)
        for i in reversed(range(self.num_steps)):
            t = torch.full((bsz,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model=model, x_t=x_t, obs_hist=obs_hist, t=t)
        return x_t

    def to(self, device: torch.device) -> "DiffusionScheduler":
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self
