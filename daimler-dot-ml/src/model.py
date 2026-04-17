"""Deep Sets: пермутационно-инвариантное кодирование рецепта + глобальные условия DOT."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsMT(nn.Module):
    """Два выхода: ΔKV100 и Oxidation EOT (многозадачная голова)."""

    def __init__(
        self,
        elem_dim: int,
        cond_dim: int,
        phi_hidden: int = 128,
        rho_hidden: int = 128,
        latent: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(elem_dim, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden, latent),
            nn.ReLU(),
        )
        in_rho = latent + cond_dim
        self.rho = nn.Sequential(
            nn.Linear(in_rho, rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden, rho_hidden // 2),
            nn.ReLU(),
        )
        self.head_visc = nn.Linear(rho_hidden // 2, 1)
        self.head_eot = nn.Linear(rho_hidden // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D_elem), mask: (B, N), cond: (B, D_cond)
        """
        h = self.phi(x)
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / (m.sum(dim=1).clamp(min=1.0))
        z = torch.cat([pooled, cond], dim=-1)
        u = self.rho(z)
        return self.head_visc(u).squeeze(-1), self.head_eot(u).squeeze(-1)
