import torch
import torch.nn as nn

class OrderedSensorMask(nn.Module):
    """
    Disables 0-3 sensors with uniform probability.
      • order  : list with the deactivation order (0-based indices)
      • min_keep: minimum number of active sensors at the end
    """
    def __init__(self,
                 order: list[int] = [0, 2, 4],   # sensores 1,3,5
                 min_keep: int = 3, enabled=True):
        super().__init__()
        self.order = order
        self.min_keep = min_keep
        self.enabled = enabled

    def forward(
        self,
        x: torch.Tensor,
        *,                       # solo por clave
        mask: torch.Tensor | None = None,
        return_mask: bool = False
    ):
        D = x.shape[-1]
        if not self.enabled:
            mask_out = torch.ones(D, dtype=torch.bool, device=x.device)
            return (x, mask_out) if return_mask else x
        # 1.  Si traen máscara externa ------------------------------------
        if mask is not None:
            x_masked = x.masked_fill(~mask, 0.0)
            return (x_masked, mask) if return_mask else x_masked

        # 2.  Generar nueva máscara ---------------------------------------
        if (not self.training):
            mask = torch.ones(D, dtype=torch.bool, device=x.device)
            return (x, mask) if return_mask else x

        # ----- ELEGIR CUÁNTOS SENSORES APAGAR, CON DISTRIBUCIÓN UNIFORME -----
        k_max = min(len(self.order), max(0, D - self.min_keep))
        #   k_max = 3 con D=6 y min_keep=3
        n_remove = torch.randint(0, k_max + 1, ()).item()   # 0,1,2,3 equiprobables

        # ----- Construir máscara --------------------------------------------
        mask = torch.ones(D, dtype=torch.bool, device=x.device)
        if n_remove > 0:
            idx_to_zero = self.order[:n_remove]             # 0 → [0], 1 → [0,2], 2 → [0,2,4]
            mask[idx_to_zero] = False

        x_masked = x.masked_fill(~mask, 0.0)
        return (x_masked, mask) if return_mask else x_masked