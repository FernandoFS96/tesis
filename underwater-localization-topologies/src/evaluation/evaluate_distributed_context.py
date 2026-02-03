#!/usr/bin/env python3
"""
Distributed-context evaluation for ANP (deployment-time only, no retraining).

Idea:
- Keep the ANP checkpoint fixed.
- For each trajectory, choose a context subset (same as normal evaluation).
- Split that context across N "nodes" (context experts).
- Each node runs ANP with its local context and predicts all target points.
- Fuse node predictive Gaussians to form a global prediction, then compute MAE on non-context points.

Use:
    python evaluate_distributed_context.py \
  --data_dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp_result_dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance \
  --output_dir /home/fernando/tesis/underwater-localization-topologies/results/eval_distributed_context \
  --topologies aligned,ellipsoidal,random \
  --context_percent 40 \
  --n_nodes 4 \
  --ctx_split_mode round_robin \
  --methods centralized,single_node,poe_fusion,gpoe_fusion,moe_fusion,consensus_poe \
  --consensus_rounds 5 \
  --consensus_graph ring \
  --mc_samples 1

"""

import argparse
import os
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
class DistributedContextEvaluator:
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

        # Cache y stats per topology (computed from train_data.pkl)
        self._y_stats_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

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
        # Compatible with your training checkpoints:
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(self.device).eval()
        return model

    # ---- context sampling ----
    def sample_context_indices(self, total_points: int, n_context: int, g: torch.Generator) -> torch.Tensor:
        perm = torch.randperm(total_points, generator=g, device=self.device)
        return perm[:n_context].sort().values

    # ---- context partitioning ----
    def split_context(self, ctx_idx: torch.Tensor, n_nodes: int, mode: str = "round_robin") -> List[torch.Tensor]:
        """
        Split context indices across nodes.

        mode:
          - round_robin: distribute indices i%N -> node i
          - contiguous: split sorted ctx_idx into N contiguous chunks
        """
        if n_nodes <= 1:
            return [ctx_idx]

        if mode == "round_robin":
            chunks = [ctx_idx[i::n_nodes] for i in range(n_nodes)]
        elif mode == "contiguous":
            chunks = torch.chunk(ctx_idx, n_nodes)
            chunks = list(chunks)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

        # Ensure no empty nodes (if context too small)
        chunks = [c for c in chunks if c.numel() > 0]
        return chunks

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
          var_norm:  (1,T,Dy)
          serial_time_s: wall-clock time spent here
          parallel_time_s: same as serial (caller can aggregate max across nodes)
        """
        means = []
        vars_ = []
        with Timer(self.device) as t:
            with torch.no_grad():
                for _ in range(mc_samples):
                    m, v, *_ = model(cx, cy, tx)
                    means.append(m)
                    vars_.append(v)
        mean = torch.stack(means, dim=0).mean(dim=0)
        var = torch.stack(vars_, dim=0).mean(dim=0)
        return mean, var, t.dt, t.dt

    # ---- Gaussian fusion rules (diagonal covariance only) ----
    @staticmethod
    def fuse_gaussians(
        means: torch.Tensor,  # (K,1,T,Dy)
        vars_: torch.Tensor,  # (K,1,T,Dy)
        method: str = "poe",
        gpoe_beta: Optional[float] = None,
        ci_alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fusion in diagonal Gaussian space (per target point and output dim).

        Supported methods:
          - mean: arithmetic mean of means (variance = mean(vars))
          - poe: product of experts (precision sum) -> can get overconfident if experts correlated
          - gpoe: generalized PoE with exponent beta (default beta=1/K)
          - moe: mixture-of-experts moment match (higher variance, robust)
          - ci: covariance intersection (pairwise fold with fixed alpha; conservative)
        """
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
            # Moment match for equal-weight mixture:
            # E[y] = mean(m_i)
            # Var[y] = E[var_i + m_i^2] - (E[m_i])^2
            m = means.mean(dim=0)
            second = (vars_ + means**2).mean(dim=0)
            v = second - m**2
            v = torch.clamp(v, min=1e-6)
            return m, v

        if method == "ci":
            # Pairwise fold: combine sequentially with fixed alpha (conservative for unknown correlations)
            # CI for diagonal: P = a*P1 + (1-a)*P2, m = P^{-1}(a*P1*m1 + (1-a)*P2*m2)
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
        """
        Simulate neighbor-only sharing with average consensus over natural params (eta1, eta2),
        then scale by K to approximate sum (PoE).

        Returns fused mean/var and estimated comm float count per node per round.
        """
        eps = 1e-8
        K = means.shape[0]
        if K == 1 or rounds <= 0:
            m, v = self.fuse_gaussians(means, vars_, method="poe")
            return m, v, 0

        neigh = self.build_graph(K, graph)

        # natural parameters
        eta2 = 1.0 / (vars_ + eps)          # precision
        eta1 = means * eta2                 # precision*mean

        # Consensus: eta <- W eta, where W is local averaging (self + neighbors)
        # Use simple weights: self = 1- step*deg, each neighbor = step
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

        # After average consensus, eta ~ average(eta_init). To approximate PoE sum: multiply by K.
        eta1_global = eta1[0] * K
        eta2_global = eta2[0] * K
        m = eta1_global / (eta2_global + eps)
        v = 1.0 / (eta2_global + eps)

        # communication: each node sends (eta1, eta2) to each neighbor per round
        # float_count per message = 2*(T*Dy) ; messages per node per round = deg
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
        n_nodes: int,
        ctx_split_mode: str,
        methods: List[str],
        consensus_rounds: int,
        consensus_graph: str,
        mc_samples: int,
        max_traj_per_theta: int = -1,
        seed_eval: int = 0,
        gpoe_beta: Optional[float] = None,
        ci_alpha: float = 0.5,
        consensus_step: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Returns a nested dict with per-method results:
          results[method]["mae_by_theta"][theta] = float
          results[method]["serial_time_s"] = float (total)
          results[method]["ideal_parallel_time_s"] = float (total)
          results[method]["comm_bytes"] = int (estimate)
        """
        theta_groups, theta_values = self.load_topology_data(topology)
        # infer dims from first sample
        any_theta = theta_values[0]
        x0, y0 = theta_groups[any_theta][0]
        T = x0.shape[0]
        input_dim = x0.shape[1]
        output_dim = y0.shape[1]

        model = self.load_anp_model(topology, input_dim=input_dim, output_dim=output_dim)
        y_mean, y_std = self.get_y_stats(topology)

        # deterministic context sampling across runs
        g = torch.Generator(device=self.device)
        g.manual_seed(seed_eval)

        context_frac = context_percent / 100.0
        n_ctx = max(1, int(context_frac * T))
        n_ctx = min(n_ctx, T)

        # init result holders
        results: Dict[str, Any] = {}
        for m in methods:
            results[m] = {
                "mae_by_theta": {},
                "errors": [],
                "serial_time_s": 0.0,
                "ideal_parallel_time_s": 0.0,
                "comm_bytes": 0,
            }

        for theta in theta_values:
            samples = theta_groups[theta]
            if max_traj_per_theta > 0:
                samples = samples[:max_traj_per_theta]

            # accumulate errors for theta
            theta_errs = {m: [] for m in methods}

            for sample_idx, (x_np, y_np) in enumerate(tqdm(samples, desc=f"theta={theta:.3f}", leave=False)):
                x = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,Dx)
                y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,Dy)
                y_norm = self.normalize_y(y, y_mean, y_std)

                # context selection (deterministic across methods)
                ctx_idx = self.sample_context_indices(T, n_ctx, g)
                non_ctx_mask = torch.ones(T, dtype=torch.bool, device=self.device)
                non_ctx_mask[ctx_idx] = False

                # centralized context (all points)
                cx_full = x[:, ctx_idx, :]
                cy_full = y_norm[:, ctx_idx, :]
                mean_full, var_full, t_serial, _ = self.anp_predict_mc(model, cx_full, cy_full, x, mc_samples=mc_samples)

                # for methods requiring node predictions
                node_ctx_list = self.split_context(ctx_idx, n_nodes=n_nodes, mode=ctx_split_mode)
                node_means = []
                node_vars = []
                node_serial_times = []

                if n_nodes > 1 and any(m != "centralized" for m in methods):
                    for node_ctx in node_ctx_list:
                        cx_i = x[:, node_ctx, :]
                        cy_i = y_norm[:, node_ctx, :]
                        mi, vi, ti, _ = self.anp_predict_mc(model, cx_i, cy_i, x, mc_samples=mc_samples)
                        node_means.append(mi)
                        node_vars.append(vi)
                        node_serial_times.append(ti)
                    node_means = torch.stack(node_means, dim=0)  # (K,1,T,Dy)
                    node_vars = torch.stack(node_vars, dim=0)
                else:
                    node_means = None
                    node_vars = None
                    node_serial_times = []

                # Evaluate each method
                for method in methods:
                    if method == "centralized":
                        pred = self.denormalize_y(mean_full, y_mean, y_std)
                        mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                        theta_errs[method].append(mae)
                        results[method]["serial_time_s"] += t_serial
                        results[method]["ideal_parallel_time_s"] += t_serial
                        continue

                    # Node-based methods (need node_means/node_vars)
                    assert node_means is not None and node_vars is not None, "Node predictions not computed"

                    # Fuse
                    with Timer(self.device) as tfuse:
                        if method in ("mean_fusion", "poe_fusion", "gpoe_fusion", "moe_fusion", "ci_fusion"):
                            fuse_key = {
                                "mean_fusion": "mean",
                                "poe_fusion": "poe",
                                "gpoe_fusion": "gpoe",
                                "moe_fusion": "moe",
                                "ci_fusion": "ci",
                            }[method]
                            fused_mean, fused_var = self.fuse_gaussians(
                                node_means, node_vars,
                                method=fuse_key,
                                gpoe_beta=gpoe_beta,
                                ci_alpha=ci_alpha,
                            )
                            comm_floats = int((node_means.shape[0] - 1) * 2 * T * output_dim)  # send (mean,var) to fusion
                        elif method == "consensus_poe":
                            fused_mean, fused_var, comm_floats = self.consensus_fuse_poe(
                                node_means, node_vars,
                                rounds=consensus_rounds,
                                graph=consensus_graph,
                                step=consensus_step,
                            )
                        elif method == "single_node":
                            fused_mean = node_means[0]
                            fused_var = node_vars[0]
                            comm_floats = 0
                        else:
                            raise ValueError(f"Unknown method: {method}")

                    # MAE (mean only)
                    pred = self.denormalize_y(fused_mean, y_mean, y_std)
                    mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                    theta_errs[method].append(mae)

                    # Timing model:
                    # - serial: sum of node times + fusion time
                    # - ideal parallel: max(node time) + fusion time
                    serial_nodes = float(sum(node_serial_times)) if node_serial_times else 0.0
                    parallel_nodes = float(max(node_serial_times)) if node_serial_times else 0.0
                    results[method]["serial_time_s"] += serial_nodes + tfuse.dt
                    results[method]["ideal_parallel_time_s"] += parallel_nodes + tfuse.dt

                    # Communication estimate (bytes, float32)
                    results[method]["comm_bytes"] += int(comm_floats * 4)

            # finalize theta stats
            for method in methods:
                if len(theta_errs[method]) > 0:
                    results[method]["mae_by_theta"][theta] = float(np.mean(theta_errs[method]))
                    results[method]["errors"].extend(theta_errs[method])
                else:
                    results[method]["mae_by_theta"][theta] = float("nan")

        # summarize
        for method in methods:
            errs = results[method]["errors"]
            results[method]["mae_overall"] = float(np.mean(errs)) if len(errs) else float("nan")

        meta = {
            "topology": topology,
            "context_percent": context_percent,
            "n_nodes": n_nodes,
            "ctx_split_mode": ctx_split_mode,
            "methods": methods,
            "consensus_rounds": consensus_rounds,
            "consensus_graph": consensus_graph,
            "mc_samples": mc_samples,
            "seed_eval": seed_eval,
            "gpoe_beta": gpoe_beta,
            "ci_alpha": ci_alpha,
            "consensus_step": consensus_step,
            "theta_values": theta_values,
        }
        return {"meta": meta, "results": results}

    # ---- save to CSV ----
    def save_results_csv(self, out: Dict[str, Any], csv_path: Path):
        meta = out["meta"]
        res = out["results"]
        rows = []
        theta_values = meta["theta_values"]
        for method, r in res.items():
            for theta in theta_values:
                rows.append({
                    "topology": meta["topology"],
                    "method": method,
                    "theta": float(theta),
                    "mae": r["mae_by_theta"].get(theta, float("nan")),
                    "context_percent": meta["context_percent"],
                    "n_nodes": meta["n_nodes"],
                    "ctx_split_mode": meta["ctx_split_mode"],
                    "consensus_rounds": meta["consensus_rounds"] if method == "consensus_poe" else 0,
                    "consensus_graph": meta["consensus_graph"] if method == "consensus_poe" else "",
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
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--anp_result_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)

    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random", help="Comma-separated, e.g. A,B,C")
    p.add_argument("--context_percent", type=int, default=40)
    p.add_argument("--n_nodes", type=int, default=4)
    p.add_argument("--ctx_split_mode", type=str, default="round_robin", choices=["round_robin", "contiguous"])

    p.add_argument(
        "--methods",
        type=str,
        default="centralized,single_node,mean_fusion,poe_fusion,moe_fusion,consensus_poe",
        help="Comma-separated: centralized, single_node, mean_fusion, poe_fusion, gpoe_fusion, moe_fusion, ci_fusion, consensus_poe",
    )

    p.add_argument("--consensus_rounds", type=int, default=5)
    p.add_argument("--consensus_graph", type=str, default="ring", choices=["ring", "line"])
    p.add_argument("--consensus_step", type=float, default=0.25)

    p.add_argument("--mc_samples", type=int, default=1)
    p.add_argument("--seed_eval", type=int, default=0)
    p.add_argument("--max_traj_per_theta", type=int, default=-1, help="For quick tests; -1 means all")

    p.add_argument("--gpoe_beta", type=float, default=None, help="If set, overrides default beta=1/K for gpoe_fusion")
    p.add_argument("--ci_alpha", type=float, default=0.5, help="CI alpha for ci_fusion (0..1)")

    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--seed", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    evaluator = DistributedContextEvaluator(
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
            n_nodes=args.n_nodes,
            ctx_split_mode=args.ctx_split_mode,
            methods=methods,
            consensus_rounds=args.consensus_rounds,
            consensus_graph=args.consensus_graph,
            mc_samples=args.mc_samples,
            max_traj_per_theta=args.max_traj_per_theta,
            seed_eval=args.seed_eval,
            gpoe_beta=args.gpoe_beta,
            ci_alpha=args.ci_alpha,
            consensus_step=args.consensus_step,
        )

        csv_path = evaluator.output_dir / f"distributed_context_topology_{topo}.csv"
        df = evaluator.save_results_csv(out, csv_path)

        # Print compact summary
        print("\n" + "=" * 90)
        print(f"Topology {topo} | ctx={args.context_percent}% | nodes={args.n_nodes} | split={args.ctx_split_mode}")
        for m in methods:
            d = df[df["method"] == m]
            mae_overall = d["mae_overall"].iloc[0] if len(d) else float("nan")
            serial = d["serial_time_s_total"].iloc[0] if len(d) else float("nan")
            ideal = d["ideal_parallel_time_s_total"].iloc[0] if len(d) else float("nan")
            comm = int(d["comm_bytes_total"].iloc[0]) if len(d) else 0
            print(f"  {m:<14}  MAE={mae_overall:8.4f}  time(serial)={serial:7.3f}s  time(ideal)={ideal:7.3f}s  comm={comm/1024:.1f} KiB")
        print(f"Saved: {csv_path}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
