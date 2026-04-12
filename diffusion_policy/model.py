import math

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard diffusion sinusoidal embedding.
    timesteps: [B]
    returns: [B, dim]
    """
    half = dim // 2
    device = timesteps.device
    exponent = -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    freqs = torch.exp(exponent)  # [half]
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ActionDiffusionTransformer(nn.Module):
    """
    Predicts noise epsilon for action chunks conditioned on observation history.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        history: int,
        horizon: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history = history
        self.horizon = horizon
        self.d_model = d_model

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        self.obs_pos = nn.Parameter(torch.randn(1, history, d_model) * 0.02)
        self.action_pos = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        obs_hist: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy_action: [B, K, D_act]
        obs_hist: [B, H, D_obs]
        timesteps: [B]
        """
        bsz = noisy_action.shape[0]
        if obs_hist.shape[1] != self.history:
            raise ValueError(f"Expected history={self.history}, got {obs_hist.shape[1]}")
        if noisy_action.shape[1] != self.horizon:
            raise ValueError(f"Expected horizon={self.horizon}, got {noisy_action.shape[1]}")

        obs_tok = self.obs_proj(obs_hist) + self.obs_pos
        act_tok = self.action_proj(noisy_action) + self.action_pos

        t_emb = sinusoidal_timestep_embedding(timesteps, self.d_model)
        t_emb = self.time_mlp(t_emb).view(bsz, 1, self.d_model)
        act_tok = act_tok + t_emb

        x = torch.cat([obs_tok, act_tok], dim=1)  # [B, H+K, D]
        x = self.encoder(x)
        x = self.norm(x)
        act_out = x[:, self.history :, :]  # action-token positions
        eps = self.out(act_out)
        return eps
