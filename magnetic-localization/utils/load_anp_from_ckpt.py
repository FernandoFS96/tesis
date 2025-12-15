"""load_anp_from_ckpt.py
========================
Utilidad para cargar un modelo ANP entrenado a partir de su checkpoint
`anp_best_epXXXX.pt` y devolverlo listo para evaluación o inferencia.

El script puede usarse de dos formas:

1. **Como ejecutable de línea de comandos**::

       python load_anp_from_ckpt.py \
           --ckpt checkpoints/anp_best_ep012.pt \
           --data-root ANP/data/trajectories_10_sensors \
           --device cuda --use-meta

   Imprime un resumen del modelo cargado y guarda una copia serializada
   con TorchScript opcional ( `--export-ts` ).

2. **Como módulo importable**::

       from load_anp_from_ckpt import load_anp
       model = load_anp("checkpoints/anp_best_ep012.pt", \
                        data_root="ANP/data/trajectories_10_sensors", \
                        device="cuda", use_meta=True)
       mu, sigma = model.predict(x_c, y_c, x_t)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # ANP root
import torch

from utils.data_loading import MagneticTrajectoryDataset
from model.anp_improved import ANP, ANPConfig

# -----------------------------------------------------------------------------
#  Función principal de carga
# -----------------------------------------------------------------------------

def load_anp(
    ckpt_path: str | Path,
    *,
    data_root: str | Path,
    device: str = "cpu",
    use_meta: bool = False,
    hidden_dim: int = 128,
    latent_dim: int = 128,
    n_heads: int = 8,
) -> ANP:
    """Carga un ANP desde `ckpt_path` y lo pone en modo `eval()`.

    Parameters
    ----------
    ckpt_path : str o Path
        Ruta al archivo `.pt` guardado por el script de entrenamiento.
    data_root : str o Path
        Carpeta de los datasets (solo para deducir `x_dim`).
    device : "cpu" o "cuda", default "cpu"
        Dónde colocar el modelo.
    use_meta : bool, default False
        Indica si se concatenaron [depth, length, width] a los sensores;
        debe coincidir con el entrenamiento para que `x_dim` sea correcto.
    hidden_dim, latent_dim, n_heads : int
        Deben coincidir también con los valores usados durante entrenamiento.

    Returns
    -------
    model : ANP
        El modelo con pesos cargados y `model.eval()`.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    # 1. Inferir x_dim leyendo un solo archivo CSV
    ds = MagneticTrajectoryDataset(data_root, use_meta=use_meta, verbose=False, cache=False)
    x_dim = ds.x_dim

    # 2. Construir la misma configuración
    cfg = ANPConfig(x_dim=x_dim, y_dim=2, hidden_dim=hidden_dim,
                    latent_dim=latent_dim, n_heads=n_heads)

    # 3. Instanciar y cargar pesos
    model = ANP(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Carga un modelo ANP desde checkpoint")
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint .pt")
    ap.add_argument("--data-root", required=True, help="Raíz de los CSV (para x_dim)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--use-meta", action="store_true", help="Usar meta features depth/size")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--export-ts", action="store_true", help="Exportar a TorchScript")
    args = ap.parse_args()

    model = load_anp(
        args.ckpt,
        data_root=args.data_root,
        device=args.device,
        use_meta=args.use_meta,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
    )
    print(model)

    if args.export_ts:
        ts_path = Path(args.ckpt).with_suffix(".torchscript.pt")
        scripted = torch.jit.script(model)
        scripted.save(ts_path)
        print(f"TorchScript guardado en {ts_path}")


if __name__ == "__main__":
    main()
