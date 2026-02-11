#!/usr/bin/env python3
"""
Distributed deployment (node = sensor) evaluation for ANP (deployment-time only, no retraining).

Key change vs evaluate_distributed_context.py:
- Nodes are now sensors (S nodes). Each node sees only its sensor features.
- I DO NOT change input dimension. I mask all other sensors to zero so the ANP checkpoint remains compatible.
- Context selection still happens over time indices (first/random), but is shared across sensors unless it is changed.

Assumptions:
- X is flattened with shape (T, num_time_points*num_sensors) where sensor features are interleaved:
  for sensor s in [0..S-1], its slice is X[:, s::S] of length num_time_points.

Usage example:
python evaluate_distributed_sensors.py \
  --data_dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp_result_dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance/ctx_first \
  --output_dir /home/fernando/tesis/underwater-localization-topologies/results/eval_distributed_sensors \
  --topologies aligned,ellipsoidal,random \
  --ctx_sample_mode first \
  --context_percent 30 \
  --num_time_points 201 \
  --num_sensors 10 \
  --methods centralized,single_node,ci_fusion,poe_fusion,gpoe_fusion,moe_fusion,consensus_poe \
  --single_sensor_idx 0 \
  --consensus_rounds 5 \
  --consensus_graph ring \
  --mc_samples 1

python evaluate_distributed_sensors.py \
  --data_dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp_result_dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance/ctx_random \
  --output_dir /home/fernando/tesis/underwater-localization-topologies/results/eval_distributed_sensors \
  --topologies aligned,ellipsoidal,random \
  --ctx_sample_mode random \
  --context_percent 30 \
  --num_time_points 201 \
  --num_sensors 10 \
  --methods centralized,single_node,ci_fusion,poe_fusion,gpoe_fusion,moe_fusion,consensus_poe \
  --single_sensor_idx 0 \
  --consensus_rounds 5 \
  --consensus_graph ring \
  --mc_samples 1
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.anp import LatentModel


# ----------------------------
# Utility: timing
# ----------------------------
class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.t0 = None
        self.dt = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.t0 is not None:
            self.dt = time.perf_counter() - self.t0
        else:
            self.dt = 0.0


# ----------------------------
# Core evaluator
# ----------------------------
class DistributedSensorEvaluator:
    def __init__(
        self,
        data_dir: Path,
        anp_result_dir: Path,
        output_dir: Path,
        device: Optional[str] = None,
        seed: int = 18,
    ):
        self.data_dir = Path(data_dir)
        self.anp_result_dir = Path(anp_result_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._y_stats_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._x_stats_cache: Dict[str, torch.Tensor] = {}

    # ---- data i/o ----
    def load_topology_data(self, topology: str) -> Tuple[Dict[float, List], List[float]]:
        topo_dir = self.data_dir / f"topology_{topology}"
        test_path = topo_dir / "test_data.pkl"
        meta_path = topo_dir / "metadata.pkl"
        if not test_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing test/metadata in {topo_dir}")

        with open(test_path, "rb") as f:
            test_data = pickle.load(f)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        theta_groups: Dict[float, List] = {}
        for sample, theta in zip(test_data, meta["test_thetas"]):
            theta_groups.setdefault(theta, []).append(sample)

        theta_values = sorted(theta_groups.keys())
        return theta_groups, theta_values

    def get_y_stats(self, topology: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if topology in self._y_stats_cache:
            return self._y_stats_cache[topology]

        topo_dir = self.data_dir / f"topology_{topology}"
        train_path = topo_dir / "train_data.pkl"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train_data.pkl in {topo_dir}")

        with open(train_path, "rb") as f:
            train_data = pickle.load(f)

        Y = np.concatenate([y for _, y in train_data], axis=0)  # (N*T, out_dim)
        y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=self.device)
        y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device)

        self._y_stats_cache[topology] = (y_mean, y_std)
        return y_mean, y_std
    
    def get_x_mean(self, topology: str) -> torch.Tensor:
        """
        Compute per-feature mean of X from train_data.pkl for the given topology.
        Returns: x_mean (Dx,) on self.device
        """
        if topology in self._x_stats_cache:
            return self._x_stats_cache[topology]

        topo_dir = self.data_dir / f"topology_{topology}"
        train_path = topo_dir / "train_data.pkl"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train_data.pkl in {topo_dir}")

        with open(train_path, "rb") as f:
            train_data = pickle.load(f)

        # train_data: list of (X, Y) where X is (T, Dx)
        X = np.concatenate([x for x, _ in train_data], axis=0)  # (N*T, Dx)
        x_mean = torch.tensor(X.mean(axis=0), dtype=torch.float32, device=self.device)
        self._x_stats_cache[topology] = x_mean
        return x_mean


    # ---- normalization ----
    @staticmethod
    def normalize_y(y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)

    @staticmethod
    def denormalize_y(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)

    # ---- model loading ----
    def load_anp_model(self, topology: str, input_dim: int, output_dim: int) -> torch.nn.Module:
        ckpt_path = self.anp_result_dir / f"ANP_{topology}" / "best_checkpoint.pth.tar"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ANP checkpoint not found: {ckpt_path}")

        model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(self.device).eval()
        return model

    # ---- context sampling ----
    def sample_context_indices(
        self,
        total_points: int,
        n_context: int,
        g: torch.Generator,
        mode: str = "first",
    ) -> torch.Tensor:
        if mode == "first":
            return torch.arange(n_context, device=self.device)
        if mode == "random":
            perm = torch.randperm(total_points, generator=g, device=self.device)
            return perm[:n_context].sort().values
        raise ValueError(f"Unknown context sampling mode: {mode}")

    # ---- sensor masking ----
    def mask_to_single_sensor(
        self,
        x: torch.Tensor, # (1,T,Dx)
        sensor_idx: int,
        num_sensors: int,
        x_fill: torch.Tensor, # (Dx,)
    ) -> torch.Tensor:
        """
        Keep only sensor_idx features, fill all other features with x_fill (mean-imputation).
        Assumes interleaved layout: sensor s features are x[..., s::num_sensors].
        """
        # start from mean background
        x_masked = x_fill.view(1, 1, -1).expand_as(x).clone()
        # overwrite the selected sensor slice with real data
        x_masked[..., sensor_idx::num_sensors] = x[..., sensor_idx::num_sensors]
        return x_masked

    # ---- ANP prediction with MC averaging ----
    def anp_predict_mc(
        self,
        model: torch.nn.Module,
        cx: torch.Tensor,
        cy: torch.Tensor,
        tx: torch.Tensor,
        mc_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Returns:
          mean_norm: (1,T,Dy)
          var_norm : (1,T,Dy)
          t_serial : float (seconds)
          t_ideal  : float (seconds)  (here equal to serial per call)
        """
        means = []
        vars_ = []
        with Timer(self.device) as tmr:
            for _ in range(max(1, mc_samples)):
                out = model(cx, cy, tx)

                # Case 1: model returns tuple/list (often includes extra outputs like KL)
                if isinstance(out, (tuple, list)):
                    # If it's (mean, var, *extras)
                    if len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                        m, v = out[0], out[1]
                    # If it's ((mean, var), *extras)
                    elif len(out) >= 1 and isinstance(out[0], (tuple, list)) and len(out[0]) >= 2:
                        m, v = out[0][0], out[0][1]
                    else:
                        raise TypeError(f"Unexpected tuple output structure from model: type={type(out)}, len={len(out)}")

                # Case 2: model returns dict-like
                elif isinstance(out, dict):
                    m = out.get("mean", None)
                    v = out.get("var", None)
                    if m is None or v is None:
                        raise KeyError(f"Model returned dict without 'mean'/'var' keys: {out.keys()}")

                else:
                    raise TypeError(f"Unexpected model output type: {type(out)}")

                # Validate shapes
                means.append(m)
                vars_.append(v)
        mean = torch.stack(means, dim=0).mean(dim=0)
        var = torch.stack(vars_, dim=0).mean(dim=0)
        return mean, var, tmr.dt, tmr.dt

    # ---- fuse gaussians (diagonal) ----
    @staticmethod
    def fuse_gaussians(
        means: torch.Tensor,  # (K,1,T,Dy)
        vars_: torch.Tensor,  # (K,1,T,Dy)
        method: str = "poe",
        gpoe_beta: Optional[float] = None,
        ci_alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        K = means.shape[0]

        if method == "mean":
            m = means.mean(dim=0)
            v = vars_.mean(dim=0)
            return m, v

        if method == "poe":
            prec = (1.0 / (vars_ + eps)).sum(dim=0)
            m = (means / (vars_ + eps)).sum(dim=0) / (prec + eps)
            v = 1.0 / (prec + eps)
            return m, v

        if method == "gpoe":
            beta = (1.0 / K) if (gpoe_beta is None) else float(gpoe_beta)
            prec = (beta / (vars_ + eps)).sum(dim=0)
            m = (beta * means / (vars_ + eps)).sum(dim=0) / (prec + eps)
            v = 1.0 / (prec + eps)
            return m, v

        if method == "moe":
            m = means.mean(dim=0)
            second = (vars_ + means**2).mean(dim=0)
            v = torch.clamp(second - m**2, min=1e-6)
            return m, v

        if method == "ci":
            a = float(ci_alpha)
            m = means[0]
            v = vars_[0]
            for i in range(1, K):
                v2 = vars_[i]
                m2 = means[i]
                P1 = 1.0 / (v + eps)
                P2 = 1.0 / (v2 + eps)
                P = a * P1 + (1 - a) * P2
                m = (a * P1 * m + (1 - a) * P2 * m2) / (P + eps)
                v = 1.0 / (P + eps)
            return m, v

        raise ValueError(f"Unknown fusion method: {method}")

    # ---- neighbor consensus on natural parameters (ring/line) ----
    @staticmethod
    def build_graph(n_nodes: int, graph: str) -> List[List[int]]:
        if n_nodes <= 1:
            return [[]]
        if graph == "ring":
            return [[(i - 1) % n_nodes, (i + 1) % n_nodes] for i in range(n_nodes)]
        if graph == "line":
            neigh = []
            for i in range(n_nodes):
                nb = []
                if i - 1 >= 0:
                    nb.append(i - 1)
                if i + 1 < n_nodes:
                    nb.append(i + 1)
                neigh.append(nb)
            return neigh
        raise ValueError(f"Unknown graph: {graph}")

    def consensus_fuse_poe(
        self,
        means: torch.Tensor,  # (K,1,T,Dy)
        vars_: torch.Tensor,  # (K,1,T,Dy)
        rounds: int = 5,
        graph: str = "ring",
        step: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        eps = 1e-8
        K = means.shape[0]
        if K == 1 or rounds <= 0:
            m, v = self.fuse_gaussians(means, vars_, method="poe")
            return m, v, 0

        neigh = self.build_graph(K, graph)

        eta2 = 1.0 / (vars_ + eps)
        eta1 = means * eta2

        for _ in range(rounds):
            new_eta1 = eta1.clone()
            new_eta2 = eta2.clone()
            for i in range(K):
                deg = len(neigh[i])
                w_self = 1.0 - step * deg
                acc1 = w_self * eta1[i]
                acc2 = w_self * eta2[i]
                for j in neigh[i]:
                    acc1 = acc1 + step * eta1[j]
                    acc2 = acc2 + step * eta2[j]
                new_eta1[i] = acc1
                new_eta2[i] = acc2
            eta1, eta2 = new_eta1, new_eta2

        # average consensus -> approximate sum (PoE) by scaling with K
        eta1_global = eta1[0] * K
        eta2_global = eta2[0] * K
        m = eta1_global / (eta2_global + eps)
        v = 1.0 / (eta2_global + eps)

        T = means.shape[2]
        Dy = means.shape[3]
        avg_deg = sum(len(n) for n in neigh) / K
        floats_per_msg = 2 * T * Dy
        floats_per_node_per_round = int(avg_deg * floats_per_msg)
        return m, v, floats_per_node_per_round * rounds

    # ---- main evaluation loop ----
    def evaluate(
        self,
        topology: str,
        context_percent: int,
        ctx_sample_mode: str,
        num_time_points: int,
        num_sensors: int,
        methods: List[str],
        single_sensor_idx: int,
        consensus_rounds: int,
        consensus_graph: str,
        consensus_step: float,
        mc_samples: int,
        max_traj_per_theta: int,
        seed_eval: int,
        gpoe_beta: Optional[float],
        ci_alpha: float,
        mask_fill: str,
    ) -> Dict[str, Any]:
        theta_groups, theta_values = self.load_topology_data(topology)

        # infer dims from first sample
        any_theta = theta_values[0]
        x0, y0 = theta_groups[any_theta][0]
        T = x0.shape[0]
        input_dim = x0.shape[1]
        output_dim = y0.shape[1]

        expected_dim = num_time_points * num_sensors
        if input_dim != expected_dim:
            raise ValueError(
                f"[{topology}] input_dim={input_dim} but expected num_time_points*num_sensors={expected_dim} "
                f"(num_time_points={num_time_points}, num_sensors={num_sensors})."
            )

        model = self.load_anp_model(topology, input_dim=input_dim, output_dim=output_dim)
        y_mean, y_std = self.get_y_stats(topology)

        x_mean = self.get_x_mean(topology)   # (Dx,)
        if mask_fill == "zero":
            x_fill = torch.zeros_like(x_mean)
        else:
            x_fill = x_mean

        g = torch.Generator(device=self.device)
        g.manual_seed(seed_eval)

        n_ctx = max(1, int((context_percent / 100.0) * T))
        n_ctx = min(n_ctx, T)

        results: Dict[str, Any] = {}
        for m in methods:
            results[m] = {
                "mae_by_theta": {},
                "errors": [],
                "serial_time_s": 0.0,
                "ideal_parallel_time_s": 0.0,
                "comm_bytes": 0,
            }

        K = num_sensors  # node = sensor

        for theta in theta_values:
            samples = theta_groups[theta]
            if max_traj_per_theta > 0:
                samples = samples[:max_traj_per_theta]

            theta_errs = {m: [] for m in methods}

            for (x_np, y_np) in tqdm(samples, desc=f"theta={theta:.3f}", leave=False):
                x_full = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,Dx)
                y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)       # (1,T,Dy)
                y_norm = self.normalize_y(y, y_mean, y_std)

                ctx_idx = self.sample_context_indices(T, n_ctx, g, mode=ctx_sample_mode)
                non_ctx_mask = torch.ones(T, dtype=torch.bool, device=self.device)
                non_ctx_mask[ctx_idx] = False

                # --- centralized baseline: full x
                cx_full = x_full[:, ctx_idx, :]
                cy_full = y_norm[:, ctx_idx, :]
                mean_full, var_full, t_central, _ = self.anp_predict_mc(model, cx_full, cy_full, x_full, mc_samples=mc_samples)

                # --- sensor nodes: each node sees only its sensor
                node_means = []
                node_vars = []
                node_serial_times = []

                if any(m != "centralized" for m in methods):
                    for s in range(num_sensors):
                        x_s = self.mask_to_single_sensor(x_full, sensor_idx=s, num_sensors=num_sensors, x_fill=x_fill)
                        cx_s = x_s[:, ctx_idx, :]
                        cy_s = y_norm[:, ctx_idx, :]
                        m_s, v_s, t_s, _ = self.anp_predict_mc(model, cx_s, cy_s, x_s, mc_samples=mc_samples)
                        node_means.append(m_s)
                        node_vars.append(v_s)
                        node_serial_times.append(t_s)

                    node_means = torch.stack(node_means, dim=0)  # (K,1,T,Dy)
                    node_vars = torch.stack(node_vars, dim=0)

                # --- evaluate methods
                for method in methods:
                    if method == "centralized":
                        pred = self.denormalize_y(mean_full, y_mean, y_std)
                        mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                        theta_errs[method].append(mae)
                        results[method]["serial_time_s"] += t_central
                        results[method]["ideal_parallel_time_s"] += t_central
                        continue

                    if method == "single_node":
                        s0 = int(single_sensor_idx) % num_sensors
                        fused_mean = node_means[s0]
                        fused_var = node_vars[s0]
                        comm_floats = 0

                    elif method in ("mean_fusion", "poe_fusion", "gpoe_fusion", "moe_fusion", "ci_fusion"):
                        fuse_key = {
                            "mean_fusion": "mean",
                            "poe_fusion": "poe",
                            "gpoe_fusion": "gpoe",
                            "moe_fusion": "moe",
                            "ci_fusion": "ci",
                        }[method]
                        with Timer(self.device) as tfuse:
                            fused_mean, fused_var = self.fuse_gaussians(
                                node_means, node_vars,
                                method=fuse_key,
                                gpoe_beta=gpoe_beta,
                                ci_alpha=ci_alpha,
                            )
                        comm_floats = int((K - 1) * 2 * T * output_dim)
                        serial_fuse = tfuse.dt

                    elif method == "consensus_poe":
                        with Timer(self.device) as tfuse:
                            fused_mean, fused_var, comm_floats = self.consensus_fuse_poe(
                                node_means, node_vars,
                                rounds=consensus_rounds,
                                graph=consensus_graph,
                                step=consensus_step,
                            )
                        serial_fuse = tfuse.dt

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # MAE (mean only)
                    pred = self.denormalize_y(fused_mean, y_mean, y_std)
                    mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                    theta_errs[method].append(mae)

                    # Timing model (compute-only):
                    serial_nodes = float(sum(node_serial_times))
                    parallel_nodes = float(max(node_serial_times))
                    fuse_dt = serial_fuse if method in ("mean_fusion", "poe_fusion", "gpoe_fusion", "moe_fusion", "ci_fusion", "consensus_poe") else 0.0
                    results[method]["serial_time_s"] += serial_nodes + fuse_dt
                    results[method]["ideal_parallel_time_s"] += parallel_nodes + fuse_dt

                    # Comm estimate in bytes (float32)
                    results[method]["comm_bytes"] += int(comm_floats * 4)

            # per-theta aggregation
            for method in methods:
                if len(theta_errs[method]) > 0:
                    results[method]["mae_by_theta"][theta] = float(np.mean(theta_errs[method]))
                    results[method]["errors"].extend(theta_errs[method])
                else:
                    results[method]["mae_by_theta"][theta] = float("nan")

        for method in methods:
            errs = results[method]["errors"]
            results[method]["mae_overall"] = float(np.mean(errs)) if len(errs) else float("nan")

        meta = {
            "topology": topology,
            "context_percent": context_percent,
            "ctx_sample_mode": ctx_sample_mode,
            "node_mode": "sensor",
            "num_time_points": num_time_points,
            "num_sensors": num_sensors,
            "methods": methods,
            "single_sensor_idx": single_sensor_idx,
            "consensus_rounds": consensus_rounds,
            "consensus_graph": consensus_graph,
            "consensus_step": consensus_step,
            "mc_samples": mc_samples,
            "seed_eval": seed_eval,
            "gpoe_beta": gpoe_beta,
            "ci_alpha": ci_alpha,
            "theta_values": sorted(results[methods[0]]["mae_by_theta"].keys()),
        }
        return {"meta": meta, "results": results}

    # ---- save to CSV ----
    def save_results_csv(self, out: Dict[str, Any], csv_path: Path):
        meta = out["meta"]
        res = out["results"]
        theta_values = meta["theta_values"]
        rows = []
        for method, r in res.items():
            for theta in theta_values:
                rows.append({
                    "topology": meta["topology"],
                    "method": method,
                    "theta": float(theta),
                    "mae": r["mae_by_theta"].get(theta, float("nan")),
                    "context_percent": meta["context_percent"],
                    "ctx_sample_mode": meta["ctx_sample_mode"],
                    "node_mode": meta["node_mode"],
                    "num_time_points": meta["num_time_points"],
                    "num_sensors": meta["num_sensors"],
                    "single_sensor_idx": meta["single_sensor_idx"] if method == "single_node" else -1,
                    "consensus_rounds": meta["consensus_rounds"] if method == "consensus_poe" else 0,
                    "consensus_graph": meta["consensus_graph"] if method == "consensus_poe" else "",
                    "consensus_step": meta["consensus_step"] if method == "consensus_poe" else 0.0,
                    "mc_samples": meta["mc_samples"],
                    "seed_eval": meta["seed_eval"],
                    "serial_time_s_total": r["serial_time_s"],
                    "ideal_parallel_time_s_total": r["ideal_parallel_time_s"],
                    "comm_bytes_total": r["comm_bytes"],
                    "mae_overall": r["mae_overall"],
                })

        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mask_fill", type=str, default="mean", choices=["mean", "zero"], help="Method to fill masked values (mean or zero)")

    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--anp_result_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)

    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")
    p.add_argument("--context_percent", type=int, default=30)
    p.add_argument("--ctx_sample_mode", type=str, default="random", choices=["random", "first"])

    p.add_argument("--num_time_points", type=int, default=201)
    p.add_argument("--num_sensors", type=int, default=10)

    p.add_argument(
        "--methods",
        type=str,
        default="centralized,single_node,poe_fusion,gpoe_fusion,moe_fusion,consensus_poe",
        help="Comma-separated: centralized, single_node, mean_fusion, poe_fusion, gpoe_fusion, moe_fusion, ci_fusion, consensus_poe",
    )
    p.add_argument("--single_sensor_idx", type=int, default=0)

    p.add_argument("--consensus_rounds", type=int, default=5)
    p.add_argument("--consensus_graph", type=str, default="ring", choices=["ring", "line"])
    p.add_argument("--consensus_step", type=float, default=0.25)

    p.add_argument("--mc_samples", type=int, default=1)
    p.add_argument("--seed_eval", type=int, default=0)
    p.add_argument("--max_traj_per_theta", type=int, default=-1)

    p.add_argument("--gpoe_beta", type=float, default=None)
    p.add_argument("--ci_alpha", type=float, default=0.5, help="alpha parameter for CI fusion (0.0 = only use neighbor, 1.0 = only use self, 0.5 = average)")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    evaluator = DistributedSensorEvaluator(
        data_dir=args.data_dir,
        anp_result_dir=args.anp_result_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    for topo in topologies:
        out = evaluator.evaluate(
            topology=topo,
            context_percent=args.context_percent,
            ctx_sample_mode=args.ctx_sample_mode,
            num_time_points=args.num_time_points,
            num_sensors=args.num_sensors,
            methods=methods,
            single_sensor_idx=args.single_sensor_idx,
            consensus_rounds=args.consensus_rounds,
            consensus_graph=args.consensus_graph,
            consensus_step=args.consensus_step,
            mc_samples=args.mc_samples,
            max_traj_per_theta=args.max_traj_per_theta,
            seed_eval=args.seed_eval,
            gpoe_beta=args.gpoe_beta,
            ci_alpha=args.ci_alpha,
            mask_fill=args.mask_fill,
        )

        csv_path = evaluator.output_dir / f"distributed_sensors_topology_{topo}.csv"
        df = evaluator.save_results_csv(out, csv_path)

        print("\n" + "=" * 95)
        print(
            f"Topology {topo} | ctx={args.context_percent}% ({args.ctx_sample_mode}) | "
            f"nodes=sensors (S={args.num_sensors})"
        )
        for m in methods:
            d = df[df["method"] == m]
            mae_overall = d["mae_overall"].iloc[0] if len(d) else float("nan")
            serial = d["serial_time_s_total"].iloc[0] if len(d) else float("nan")
            ideal = d["ideal_parallel_time_s_total"].iloc[0] if len(d) else float("nan")
            comm = int(d["comm_bytes_total"].iloc[0]) if len(d) else 0
            print(
                f"  {m:<14}  MAE={mae_overall:8.4f}  "
                f"time(serial)={serial:7.3f}s  time(ideal)={ideal:7.3f}s  comm={comm/1024:.1f} KiB"
            )
        print(f"Saved: {csv_path}")
        print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
