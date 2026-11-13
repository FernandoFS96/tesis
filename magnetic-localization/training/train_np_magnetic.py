"""train_np_magnetic.py -- retrain the magnetic-anomaly localisation NP of the
JMSE 2025 paper with its latent bug fixed, and with the newest NP implementations
from sensor-agnostic-underwater-localization, all under the paper's protocol.

Model arms (--arch)
-------------------
  anp_orig        model/anp_improved.ANP exactly as published. Its latent encoder
                  averages only the padded rows (ANPConfig.fix_latent_mask), so z
                  carries no information about the data.
  anp_orig_fixed  the same model with fix_latent_mask=True. This is the only
                  difference from anp_orig, so the pair isolates the bug.
  cnp_orig        the published model with the latent path removed (ANPNoLatent).
                  Against anp_orig_fixed, the latent is the only difference.
  anp             sensor-agnostic src/models/anp.LatentModel (ANP)
  cnp             sensor-agnostic src/models/anp.DeterministicModel (attentive CNP)
  reg             no-context sensor-set regressor (SetRegressor): the AOR study's
                  regressor (sensor-agnostic scripts/eval/acoustic_regressor.py) on
                  one reading at a time, Gaussian output; it never sees the context
  mlp             no-context MLP, the paper's 6-128-128 GELU template with a
                  Gaussian output (PointMLP)

Latent vs no latent: compare cnp_orig with anp_orig_fixed (published
architecture) and cnp with anp (sensor-agnostic implementation). anp_orig shows
where the published model sits. reg and mlp score the same target points with no
context, so against the NPs they measure what the context adds.

Protocol (shared by every arm; the same as training/train_anp_magnetic.py)
-------------------------------------------------------------------------
  * the paper's in-distribution corpus (Section 3.2, Table 2): depths
    {7.5, 10, 20} m x hulls {2x1, 4x2, 8x4} m x {6, 5, 4} sensors = 27 CSVs,
    2700 trajectories, split 80/20 at trajectory level with generator seed 18.
    The run aborts if an OOD probe (30 m, 20x10 m, 3 sensors) is among the
    training files. Note: utils.data_loading.MagneticTrajectoryDataset would
    load only the 9 six-sensor CSVs from this tree, because its glob prefers
    dataset.csv over dataset*.csv; --train-set selects either;
  * episodes from utils.data_loading.episodic_collate: random 15-55 context
    points (interpolation) or, with --extrapolation, a sequential prefix;
  * batch 10, Adam lr 9e-5, gradient clipping 1.0, fp16 AMP on CUDA, up to 10000
    epochs, early stopping after 1000 epochs without a better val MAE;
  * MAE = mean over target points of |dx| + |dy| (the paper's metric).

Deliberate differences from the original script
------------------------------------------------
  * the validation episodes are drawn once, with a fixed seed, and reused every
    epoch and by every arm, so checkpoint selection is not noisy and all arms
    are selected on the same episodes;
  * --seed seeds model init and the training episodes (the original did not);
  * the sensor-agnostic arms use their own recommended settings: dropout 0.2,
    beta 1 with a 100-epoch linear KL warmup, prediction from the prior mean
    at evaluation.

After training, best.pt is evaluated the way the paper tests, in
interpolation mode (random context) and extrapolation mode (first --extrap-ctx
points as context), with the same episodes for every arm:
  * the 12 per-task sets of experiments/eval_all_models.py (depth, size and
    sensor tasks with their OOD probes; Tables 5, 7, 9 and A2). DEPTH_20m,
    SIZE_8x4m and SENS_4s are one and the same file;
  * the 9 in-distribution depth x hull scenarios of Table 3, each pooling its
    6-, 5- and 4-sensor files.
The in-distribution test CSVs are the same files as the training CSVs, so the
evaluation runs twice: test_results.csv scores only their held-out (validation)
trajectories, as the paper's main tables do; test_results_full.csv scores all
of them, training trajectories included, as eval_all_models.py (Table A2) does.
Reported per set:
  mae_l1_single  one forward pass, as in the paper (anp_orig draws one z from
                 the prior; anp predicts from the prior mean)
  mae_l1         mean of the predictive mixture over --n-z latent samples
  eucl           Euclidean error of that mean
  nll            -log of the predictive mixture density, per target point
  cov1 / cov2    share of coordinates within 1 / 2 predictive std
                 (a calibrated model gives 0.683 / 0.954)

Usage
-----
  # training data in data/trajectories/ (the paper's ANP/data/trajectories)
  python training/train_np_magnetic.py --arch anp_orig_fixed --seed 0
  # re-evaluate a finished run on the test sets only:
  python training/train_np_magnetic.py --eval-only runs/.../best.pt

Data loading does not need pandas: the CSV reader below reproduces
utils.data_loading.MagneticTrajectoryDataset (same trajectory order within a CSV,
so a given file list splits identically).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import random
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parent.parent                      # magnetic-localization/
SA_ROOT = ROOT.parent / "sensor-agnostic-underwater-localization"
sys.path.insert(0, str(ROOT))
sys.path.append(str(SA_ROOT))

from model.anp_improved import ANP, ANPConfig, MLP  # noqa: E402
from utils.data_loading import SENSOR_COLS, Y_COLS, episodic_collate, parse_meta  # noqa: E402
from utils.sensor_dropout import OrderedSensorMask  # noqa: E402
from src.models import anp as sa_anp  # noqa: E402

ARCHS = ("anp_orig", "anp_orig_fixed", "cnp_orig", "anp", "cnp", "reg", "mlp")
NO_CONTEXT = ("reg", "mlp")
SPLIT_SEED = 18          # the paper's train/val split
VAL_EPISODE_SEED = 12345

# The paper's in-distribution training corpus (Section 3.2 and Table 2: 27 CSVs,
# 2700 trajectories): depths x hulls x sensor sets. The OOD probes (30 m depth,
# 20 x 10 m hull, 3 sensors) must never be in it.
ID_DEPTHS = ("7-5m", "10m", "20m")
ID_HULLS = ("2mx1m", "4mx2m", "8mx4m")
ID_SENSOR_FILES = ("dataset.csv",                       # 6 sensors
                   "dataset_without_sensor_1.csv",      # 5 sensors
                   "dataset_without_sensor_1_3.csv")    # 4 sensors
OOD_FOLDER_TOKENS = ("30m-", "20mx10m")
OOD_FILE = "dataset_without_sensor_1_3_5.csv"            # 3 sensors

# The 12 test sets of experiments/eval_all_models.py (label -> path under --tasks-root).
TEST_SETS = OrderedDict([
    ("DEPTH_7-5m",      "3x3x3/7-5m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("DEPTH_10m",       "3x3x3/10m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("DEPTH_20m",       "3x3x3/20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("DEPTH_30m_OOD",   "1x1x1/30m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("SIZE_2x1m",       "3x3x3/20m-64T_2mx1m/dataset_without_sensor_1_3.csv"),
    ("SIZE_4x2m",       "3x3x3/20m-64T_4mx2m/dataset_without_sensor_1_3.csv"),
    ("SIZE_8x4m",       "3x3x3/20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("SIZE_20x10m_OOD", "1x1x1/20m-64T_20mx10m/dataset_without_sensor_1_3.csv"),
    ("SENS_6s",         "3x3x3/20m-64T_8mx4m/dataset.csv"),
    ("SENS_5s",         "3x3x3/20m-64T_8mx4m/dataset_without_sensor_1.csv"),
    ("SENS_4s",         "3x3x3/20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ("SENS_3s_OOD",     "1x1x1/20m-64T_8mx4m/dataset_without_sensor_1_3_5.csv"),
])
# The paper's Table 3: the 9 in-distribution depth x hull scenarios, each pooling
# its 6-, 5- and 4-sensor files (the 2700 ID trajectories).
SCENARIOS = OrderedDict(
    (f"ID_{d}_{h}", [f"3x3x3/{d}-64T_{h}/{n}" for n in ID_SENSOR_FILES])
    for d in ID_DEPTHS for h in ID_HULLS)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_csvs(root):
    """Same file discovery (and precedence) as MagneticTrajectoryDataset."""
    root = str(root)
    files = (glob.glob(os.path.join(root, "*", "dataset.csv"))
             or glob.glob(os.path.join(root, "dataset.csv"))
             or glob.glob(os.path.join(root, "*", "dataset*.csv")))
    return sorted(files)


def csv_key(path):
    """'<configuration folder>/<file>': identifies a dataset across data/trajectories/
    and data/tasks/, which hold byte-identical copies of the same files."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def training_files(root, train_set):
    """id    the paper's in-distribution corpus: 9 depth x hull configurations x
             {6, 5, 4} sensors = 27 CSVs.
       id6s  the same 9 configurations, 6-sensor files only. This is what
             MagneticTrajectoryDataset's glob picks from a folder of that tree,
             because dataset.csv takes precedence over dataset*.csv.
       glob  MagneticTrajectoryDataset's discovery on whatever is under root."""
    if train_set == "glob":
        return find_csvs(root)
    names = ID_SENSOR_FILES if train_set == "id" else ID_SENSOR_FILES[:1]
    files = [os.path.join(str(root), f"{d}-64T_{h}", n) for d in ID_DEPTHS for h in ID_HULLS for n in names]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"--train-set {train_set}: {len(missing)} file(s) missing under {root}, "
                         f"e.g. {missing[:3]}")
    return sorted(files)


def ood_in_training(files):
    """Training CSVs that belong to an OOD probe configuration."""
    return [f for f in files if any(tok in Path(f).parent.name for tok in OOD_FOLDER_TOKENS)
            or Path(f).name == OOD_FILE]


def read_trajectories(csv_path, use_meta=False):
    """One CSV -> list of (sensors (N, 6[+3]), coords (N, 2)) float32 tensors.
    Missing sensor columns are zero-filled; trajectories come in sorted traj_id
    order with their rows in file order (what DataFrame.groupby yields)."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.asarray([[float(v) for v in row] for row in reader if row], dtype=np.float64)
    col = {c: i for i, c in enumerate(header)}
    n = data.shape[0]
    sensors = np.stack([data[:, col[c]] if c in col else np.zeros(n) for c in SENSOR_COLS], axis=1)
    coords = np.stack([data[:, col[c]] for c in Y_COLS], axis=1)
    tids = data[:, col["traj_id"]]
    meta = torch.tensor(parse_meta(Path(csv_path).parent.name), dtype=torch.float32)
    out = []
    for tid in np.unique(tids):
        idx = np.nonzero(tids == tid)[0]
        s = torch.as_tensor(sensors[idx], dtype=torch.float32)
        if use_meta:
            s = torch.cat([s, meta.repeat(s.size(0), 1)], dim=-1)
        out.append((s, torch.as_tensor(coords[idx], dtype=torch.float32)))
    return out


class Trajectories(Dataset):
    def __init__(self, files, use_meta=False):
        self.samples, self.index = [], []      # index[i] = (csv_key, position in that CSV)
        for f in files:
            trajs = read_trajectories(f, use_meta)
            self.samples += trajs
            self.index += [(csv_key(f), j) for j in range(len(trajs))]
        if not self.samples:
            raise RuntimeError(f"no trajectories in {files}")
        self.x_dim = self.samples[0][0].shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(extrapolation, fixed_ctx=None):
    return partial(episodic_collate, ctx_mode="sequential" if extrapolation else "random",
                   fixed_ctx_size=fixed_ctx)


def norm_stats(subset):
    xs = torch.cat([subset[i][0] for i in range(len(subset))])
    ys = torch.cat([subset[i][1] for i in range(len(subset))])
    return {"x_mean": xs.mean(0), "x_std": xs.std(0).clamp(min=1e-6),
            "y_mean": ys.mean(0), "y_std": ys.std(0).clamp(min=1e-6)}


def apply_norm(batch, norm):
    """Normalize a collated batch (x_c, y_c, x_t, y_t, ctx_mask, tgt_mask)."""
    if norm is None:
        return batch
    x_c, y_c, x_t, y_t, cm, tm = batch
    dev = x_c.device
    nx = lambda x: (x - norm["x_mean"].to(dev)) / norm["x_std"].to(dev)
    ny = lambda y: (y - norm["y_mean"].to(dev)) / norm["y_std"].to(dev)
    return nx(x_c), ny(y_c), nx(x_t), ny(y_t), cm, tm


def to_physical(mean, var, norm):
    if norm is None:
        return mean, var
    ys = norm["y_std"].to(mean.device)
    return mean * ys + norm["y_mean"].to(mean.device), var * ys ** 2


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ANPNoLatent(ANP):
    """The published ANP with its latent path removed: same sensor masking,
    cross-attention stack, decoder MLP and loss, but the decoder sees [r, x_t]
    instead of [r, z, x_t]. Against anp_orig_fixed, the latent is the only
    difference."""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.latent_enc
        self.decoder = MLP(cfg.hidden_dim + cfg.x_dim, cfg.hidden_dim, cfg.y_dim * 2, cfg.dec_layers)

    def forward(self, x_c, y_c, x_t, y_t=None, *, ctx_mask=None, tgt_mask=None):
        x_c, sens_mask = self.mask(x_c, return_mask=True)
        x_t = self.mask(x_t, mask=sens_mask)
        r = self._r(x_c, y_c, x_t, ctx_mask)
        mu, log_s = self.decoder(torch.cat([r, x_t], -1)).chunk(2, -1)
        return torch.distributions.Normal(mu, self._pos_std(log_s)), torch.zeros((), device=x_c.device)


class SetRegressor(torch.nn.Module):
    """No-context baseline: each present magnetometer is a token ([reading,
    one-hot slot in the fixed array] through a 2-layer MLP, as the AOR regressor
    embeds [position, features]); missing sensors are masked out of the attention.
    Pre-norm transformer over the sensors, readout token -> diagonal Gaussian."""

    def __init__(self, n_sensors, d=256, layers=6, heads=8):
        super().__init__()
        nn = torch.nn
        self.register_buffer("slot", torch.eye(n_sensors), persistent=False)
        self.inp = nn.Sequential(nn.Linear(1 + n_sensors, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.register_buffer("var_floor", torch.tensor(sa_anp.DEFAULT_VAR_FLOOR), persistent=False)

    def forward(self, x, present):                     # (B, N, S), (B, N, S) True = present
        B, N, S = x.shape
        h = torch.cat([x.reshape(B * N, S, 1), self.slot.expand(B * N, -1, -1)], -1)
        h = torch.cat([self.cls.expand(B * N, -1, -1), self.inp(h)], 1)
        kpm = torch.cat([torch.zeros(B * N, 1, dtype=torch.bool, device=x.device),
                         ~present.reshape(B * N, S)], 1)
        o = self.out(self.enc(h, src_key_padding_mask=kpm)[:, 0]).float().reshape(B, N, 4)
        return o[..., :2], self.var_floor + torch.nn.functional.softplus(o[..., 2:])


class PointMLP(torch.nn.Module):
    """No-context baseline: the paper's MLP template (6-128-128 GELU, Figure 1) on
    the zero-filled readings, with a diagonal Gaussian output instead of (x, y)."""

    def __init__(self, x_dim, hidden=128):
        super().__init__()
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(x_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 4))
        self.register_buffer("var_floor", torch.tensor(sa_anp.DEFAULT_VAR_FLOOR), persistent=False)

    def forward(self, x, present=None):
        o = self.net(x).float()
        return o[..., :2], self.var_floor + torch.nn.functional.softplus(o[..., 2:])


def apply_var_floor(model, arch, a, norm):
    """Sensor-agnostic arms: put the predictive-variance floor in PHYSICAL units
    (--var-floor, m^2). Their decoders add a floor in model units, so with
    --normalize the built-in 1e-3 would be 1e-3 * y_std^2 m^2 (a 1.7 m std floor
    here). Checkpoints from before --var-floor existed keep the built-in floor."""
    vf = getattr(a, "var_floor", None)
    if is_orig(arch) or vf is None:
        return
    sa_anp.set_var_floor(model, sa_anp.physical_var_floor(vf, norm["y_std"] if norm is not None else None))


def build_model(arch, x_dim, a):
    if arch in ("anp_orig", "anp_orig_fixed", "cnp_orig"):
        cfg = ANPConfig(x_dim=x_dim, y_dim=2, hidden_dim=a.hidden_dim,
                        latent_dim=a.latent_dim, n_heads=a.n_heads,
                        sensor_mask=a.sensor_mask,
                        fix_latent_mask=(arch == "anp_orig_fixed"))
        return ANPNoLatent(cfg) if arch == "cnp_orig" else ANP(cfg)
    if arch == "anp":
        return sa_anp.LatentModel(a.hidden_dim, x_dim, 2, dropout=a.dropout, attn_ffn=a.attn_ffn)
    if arch == "cnp":
        return sa_anp.DeterministicModel(a.hidden_dim, x_dim, 2, dropout=a.dropout, attn_ffn=a.attn_ffn)
    if arch in NO_CONTEXT and (x_dim != len(SENSOR_COLS) or a.sensor_mask):
        raise ValueError(f"{arch} takes the {len(SENSOR_COLS)} sensor columns only "
                         f"(no --use-meta, no --sensor-mask)")
    if arch == "reg":
        return SetRegressor(x_dim, a.hidden_dim, getattr(a, "reg_layers", 6))
    if arch == "mlp":
        return PointMLP(x_dim, a.hidden_dim)
    raise ValueError(f"unknown arch {arch} (use {'|'.join(ARCHS)})")


def is_orig(arch):
    """Built on the published model (its forward, sensor masking and loss)."""
    return arch in ("anp_orig", "anp_orig_fixed", "cnp_orig")


def is_deterministic(arch):
    return arch in ("cnp", "cnp_orig") + NO_CONTEXT


class Runner:
    """Arch-specific calls behind one interface. Masks follow the magnetic
    collate (True = PADDING); the sensor-agnostic models take the inverse."""

    def __init__(self, model, arch, sensor_mask, norm=None):
        self.model, self.arch = model, arch
        # Missing sensors are zero-filled columns; --normalize maps them to -x_mean / x_std.
        self.x_zero = None if norm is None else -norm["x_mean"] / norm["x_std"]
        # The published ANP applies OrderedSensorMask internally (cfg.sensor_mask);
        # the sensor-agnostic models get the same augmentation from here.
        self.smask = None if is_orig(arch) else OrderedSensorMask(
            order=[0, 2, 4], min_keep=3, enabled=sensor_mask)

    def train(self, mode):
        self.model.train(mode)
        if self.smask is not None:
            self.smask.train(mode)

    def _sensor_mask(self, x_c, x_t):
        if self.smask is None:
            return x_c, x_t
        x_c, m = self.smask(x_c, return_mask=True)
        return x_c, self.smask(x_t, mask=m)

    def _present(self, x):
        """(..., S) True where the magnetometer reads (the zero-filled columns are absent)."""
        zero = 0.0 if self.x_zero is None else self.x_zero.to(x.device)
        return (x - zero).abs() > 1e-4

    def loss(self, batch, beta):
        """Training objective -> (loss, nll, kl, pred_mean)."""
        x_c, y_c, x_t, y_t, cm, tm = batch
        if self.arch in NO_CONTEXT:
            # No context: every labelled point of the episode is a training example.
            x, y, pad = torch.cat([x_c, x_t], 1), torch.cat([y_c, y_t], 1), torch.cat([cm, tm], 1)
            mean, var = self.model(x, self._present(x))
            nll = -torch.distributions.Normal(mean, var.sqrt()).log_prob(y.float()).sum(-1)
            nll = nll.masked_fill(pad, 0.0).sum() / ((~pad).sum() + 1e-6)
            return nll, nll, torch.zeros((), device=nll.device), mean[:, x_c.size(1):]
        if is_orig(self.arch):   # exactly training/train_anp_magnetic.loss_step
            dist, kl = self.model.forward(x_c, y_c, x_t, y_t, ctx_mask=cm, tgt_mask=tm)
            log_p = dist.log_prob(y_t).sum(-1)
            log_p = log_p.masked_fill(tm, 0.0)
            log_p = log_p.sum() / ((~tm).sum() + 1e-6)
            return -(log_p - kl), -log_p, kl, dist.mean
        x_c, x_t = self._sensor_mask(x_c, x_t)
        mean, _, loss, kl, nll = self.model(x_c, y_c, x_t, y_t, beta,
                                            context_mask=~cm, target_mask=~tm)
        return loss, nll, (kl if kl is not None else torch.zeros((), device=loss.device)), mean

    @torch.no_grad()
    def predict(self, batch):
        """One prior-path forward pass -> (mean, var). anp_orig: one z drawn from
        the prior (the paper's protocol); anp: the prior mean of z."""
        x_c, y_c, x_t, _, cm, tm = batch
        if self.arch in NO_CONTEXT:
            return self.model(x_t, self._present(x_t))
        if is_orig(self.arch):
            dist, _ = self.model.forward(x_c, y_c, x_t, ctx_mask=cm, tgt_mask=tm)
            return dist.mean, dist.variance
        x_c, x_t = self._sensor_mask(x_c, x_t)
        mean, var, *_ = self.model(x_c, y_c, x_t, None, context_mask=~cm, target_mask=~tm)
        return mean, var

    @torch.no_grad()
    def predict_samples(self, batch, n_z):
        """(S, B, N, 2) means and variances, one slice per latent sample from the
        prior; S = 1 for the deterministic models."""
        if is_deterministic(self.arch):
            m, v = self.predict(batch)
            return m[None], v[None]
        if is_orig(self.arch):
            outs = [self.predict(batch) for _ in range(n_z)]
            return torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs])
        x_c, y_c, x_t, _, cm, _ = batch
        x_c, x_t = self._sensor_mask(x_c, x_t)
        m = self.model
        mu, log_var, _ = m.latent_encoder(x_c, y_c, mask=~cm)
        r = m.deterministic_encoder(x_c, y_c, x_t, context_mask=~cm)
        means, vars_ = [], []
        for _ in range(n_z):
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
            mean, var, _ = m.decoder(r, z.unsqueeze(1).expand(-1, x_t.size(1), -1), x_t)
            means.append(mean); vars_.append(var)
        return torch.stack(means), torch.stack(vars_)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mae_l1(mean, y, tm):
    """Paper metric: mean over valid target points of |dx| + |dy|. -> (sum, count)."""
    err = (mean - y).abs().sum(-1).masked_fill(tm, 0.0)
    return err.sum().item(), (~tm).sum().item()


def mixture_metrics(means, vars_, y, tm):
    """Sums over valid points of the predictive-mixture metrics."""
    valid = ~tm
    mix_mean = means.mean(0)
    mix_var = vars_.mean(0) + means.var(0, unbiased=False)
    # log of the mixture density of the joint (x, y) position, per point
    logp = torch.distributions.Normal(means, vars_.sqrt()).log_prob(y.unsqueeze(0)).sum(-1)  # (S,B,N)
    logp = torch.logsumexp(logp, 0) - np.log(means.size(0))
    z = (y - mix_mean).abs() / mix_var.sqrt()
    vm = valid.unsqueeze(-1).float()
    return {
        "mae_l1": (mix_mean - y).abs().sum(-1)[valid].sum().item(),
        "eucl": (mix_mean - y).norm(dim=-1)[valid].sum().item(),
        "nll": (-logp)[valid].sum().item(),
        "cov1": ((z <= 1).float() * vm).sum().item() / 2,
        "cov2": ((z <= 2).float() * vm).sum().item() / 2,
    }


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
def autocast_ctx(amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()


def validate(runner, val_batches, device, norm):
    runner.train(False)
    err = cnt = nll_sum = 0.0
    for b in val_batches:
        b = [t.to(device) for t in b]
        mean, var = runner.predict(apply_norm(b, norm))
        mean, var = to_physical(mean.float(), var.float(), norm)
        e, c = mae_l1(mean, b[3], b[5]); err += e; cnt += c
        nll = torch.distributions.Normal(mean, var.sqrt()).log_prob(b[3]).sum(-1)
        nll_sum += (-nll).masked_fill(b[5], 0.0).sum().item()
    return err / max(cnt, 1), nll_sum / max(cnt, 1)


def train(a, out_dir, device):
    files = training_files(a.data_root, a.train_set)
    if not files:
        raise SystemExit(f"no dataset*.csv found under {a.data_root}")
    leak = ood_in_training(files)
    if leak and not a.allow_ood_in_train:
        raise SystemExit(f"{len(leak)} training CSV(s) belong to an OOD probe configuration "
                         f"(e.g. {[csv_key(f) for f in leak[:3]]}), so the OOD test sets would not be "
                         f"out of distribution. Use --train-set id, or pass --allow-ood-in-train.")
    full = Trajectories(files, use_meta=a.use_meta)
    val_len = int(0.2 * len(full))
    train_ds, val_ds = random_split(full, [len(full) - val_len, val_len],
                                    generator=torch.Generator().manual_seed(SPLIT_SEED))
    print(f"[data] --train-set {a.train_set}: {len(files)} CSVs, {len(full)} trajectories -> "
          f"train {len(train_ds)} / val {len(val_ds)} (split seed {SPLIT_SEED}); x_dim={full.x_dim}")
    # Which trajectories of each training CSV are held out, so the test evaluation
    # can score in-distribution sets on trajectories the model never trained on.
    val_index = {}
    for i in val_ds.indices:
        key, pos = full.index[i]
        val_index.setdefault(key, []).append(pos)
    val_index = {k: sorted(v) for k, v in val_index.items()}
    norm = norm_stats(train_ds) if a.normalize else None

    # Validation episodes: drawn once with a fixed seed, reused every epoch.
    random.seed(VAL_EPISODE_SEED); torch.manual_seed(VAL_EPISODE_SEED)
    val_batches = list(DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                                  collate_fn=make_collate(a.extrapolation)))

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              collate_fn=make_collate(a.extrapolation),
                              generator=torch.Generator().manual_seed(a.seed))
    model = build_model(a.arch, full.x_dim, a).to(device)
    apply_var_floor(model, a.arch, a, norm)
    runner = Runner(model, a.arch, a.sensor_mask, norm)
    print(f"[{a.arch}] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, device={device}, "
          f"amp={a.amp and device == 'cuda'}, normalize={a.normalize}")

    amp = a.amp and device == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=amp)
    latent_sa = a.arch == "anp"
    beta_at = lambda ep: (a.beta if (not latent_sa or a.kl_warmup_epochs <= 0)
                          else a.beta * min(1.0, ep / float(a.kl_warmup_epochs)))

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,beta,train_loss,train_nll,train_kl,train_mae_l1,val_mae_l1,val_nll,sec\n")
    ckpt_meta = {"arch": a.arch, "args": vars(a), "x_dim": full.x_dim, "norm": norm,
                 "fix_latent_mask": a.arch == "anp_orig_fixed",
                 "train_keys": sorted({csv_key(f) for f in files}), "val_index": val_index}
    best, best_ep, wait = float("inf"), 0, 0
    try:
        for ep in range(1, a.epochs + 1):
            t0 = time.time(); beta = beta_at(ep)
            runner.train(True)
            s_loss = s_nll = s_kl = s_err = n_b = 0.0; n_pts = 0
            for b in train_loader:
                b = [t.to(device) for t in b]
                with autocast_ctx(amp):
                    loss, nll, kl, mean = runner.loss(apply_norm(b, norm), beta)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    pm, _ = to_physical(mean.float(), mean.float(), norm)
                    e, c = mae_l1(pm, b[3], b[5])
                s_loss += loss.item(); s_nll += nll.item(); s_kl += kl.item(); n_b += 1
                s_err += e; n_pts += c
            val_mae, val_nll = validate(runner, val_batches, device, norm)
            with open(log_path, "a") as f:
                f.write(f"{ep},{beta:.4f},{s_loss / n_b:.6f},{s_nll / n_b:.6f},{s_kl / n_b:.6e},"
                        f"{s_err / max(n_pts, 1):.6f},{val_mae:.6f},{val_nll:.6f},{time.time() - t0:.1f}\n")
            if val_mae < best - 1e-6:
                best, best_ep, wait = val_mae, ep, 0
                torch.save({**ckpt_meta, "model_state": model.state_dict(), "epoch": ep,
                            "best_val_mae": best}, out_dir / "best.pt")
            else:
                wait += 1
            if ep == 1 or ep % a.print_every == 0:
                print(f"[{a.arch}] ep {ep:5d}  train {s_err / max(n_pts, 1):.4f}  val {val_mae:.4f}  "
                      f"best {best:.4f}@{best_ep}  kl {s_kl / n_b:.3e}  ({time.time() - t0:.1f}s)")
            if wait >= a.patience:
                print(f"[{a.arch}] early stop: {a.patience} epochs without improvement")
                break
    except KeyboardInterrupt:
        print(f"\n[{a.arch}] interrupted at epoch {ep}; evaluating the best checkpoint")
    torch.save({**ckpt_meta, "model_state": model.state_dict(), "epoch": ep}, out_dir / "last.pt")
    print(f"[{a.arch}] best val MAE {best:.4f} at epoch {best_ep} -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


# --------------------------------------------------------------------------- #
# Test evaluation
# --------------------------------------------------------------------------- #
def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model = build_model(ck["arch"], ck["x_dim"], a)
    model.load_state_dict(ck["model_state"])
    apply_var_floor(model, ck["arch"], a, ck.get("norm"))
    return model.to(device).eval(), ck, a


def load_test_group(paths, use_meta, train_keys, val_index, population):
    """Trajectories of one test group (one or more CSVs) -> (list of trajectories,
    population label); the list serves directly as a map-style dataset.
    A CSV that was in the training corpus contributes only its validation
    trajectories under population='heldout', all of them under 'full'; a CSV never
    trained on always contributes all of its trajectories."""
    samples, pops = [], set()
    for p in paths:
        trajs = read_trajectories(p, use_meta)
        if csv_key(p) in train_keys:
            if population == "heldout":
                trajs = [trajs[j] for j in val_index.get(csv_key(p), [])]
            pops.add(population)
        else:
            pops.add("unseen")
        samples += trajs
    return samples, "+".join(sorted(pops))


@torch.no_grad()
def evaluate(ckpt_path, tasks_root, device, eval_seeds, n_z, extrap_ctx, out_csv, population="heldout"):
    """Scores best.pt on the paper's test groups: the 12 per-task sets of
    experiments/eval_all_models.py (Tables 5, 7, 9 and A2) and the 9 in-distribution
    scenarios of Table 3. population: for CSVs that were in the training corpus,
    'heldout' scores only their validation-split trajectories (as the paper's
    main tables do); 'full' scores all of them, including training trajectories
    (as eval_all_models.py / Table A2 do)."""
    model, ck, a = load_checkpoint(ckpt_path, device)
    arch, norm = ck["arch"], ck.get("norm")
    runner = Runner(model, arch, a.sensor_mask, norm); runner.train(False)
    train_keys, val_index = set(ck.get("train_keys", [])), ck.get("val_index", {})
    groups = [("task", label, [rel]) for label, rel in TEST_SETS.items()] + \
             [("scenario", label, rels) for label, rels in SCENARIOS.items()]
    rows = []
    for mode in ("interp", "extrap"):
        collate = make_collate(mode == "extrap", fixed_ctx=extrap_ctx if mode == "extrap" else None)
        for kind, label, rels in groups:
            paths = [Path(tasks_root) / r for r in rels]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"[eval] {label}: missing {missing[0]}, skipped"); continue
            ds, pop = load_test_group(paths, a.use_meta, train_keys, val_index, population)
            if not ds:
                print(f"[eval] {label}: no held-out trajectories, skipped"); continue
            for es in eval_seeds:
                # Episodes are drawn up front from a fixed seed, so every arm is
                # scored on the same episodes whatever its own RNG use.
                random.seed(es); torch.manual_seed(es)
                batches = list(DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate))
                torch.manual_seed(10_000 + es)                   # latent sampling
                acc = dict.fromkeys(("mae_l1_single", "mae_l1", "eucl", "nll", "cov1", "cov2"), 0.0)
                n = 0
                for b in batches:
                    b = [t.to(device) for t in b]
                    bn = apply_norm(b, norm)
                    m1, v1 = runner.predict(bn)
                    m1, _ = to_physical(m1.float(), v1.float(), norm)
                    acc["mae_l1_single"] += mae_l1(m1, b[3], b[5])[0]
                    ms, vs = runner.predict_samples(bn, n_z)
                    ms, vs = to_physical(ms.float(), vs.float(), norm)
                    for k, v in mixture_metrics(ms, vs, b[3], b[5]).items():
                        acc[k] += v
                    n += (~b[5]).sum().item()
                rows.append({"arch": arch, "seed": a.seed, "mode": mode, "group": kind, "test_set": label,
                             "population": pop, "n_traj": len(ds), "eval_seed": es, "n_points": n,
                             **{k: v / max(n, 1) for k, v in acc.items()}})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    lines = [f"{arch} seed {a.seed}  ({ckpt_path})",
             f"mean over eval seeds {list(eval_seeds)}; mae = |dx|+|dy| in metres, n_z = {n_z}",
             "population: heldout = validation trajectories of a training CSV; "
             "full = all of them (incl. training); unseen = CSV never trained on"]
    for mode in ("interp", "extrap"):
        lines.append(f"\n[{mode}]  {'set':18s} {'population':>10s} {'n_traj':>6s} {'mae_single':>10s} "
                     f"{'mae_mix':>8s} {'eucl':>8s} {'nll':>8s} {'cov1':>6s} {'cov2':>6s}")
        for kind, label, _ in groups:
            rs = [r for r in rows if r["mode"] == mode and r["test_set"] == label]
            if not rs:
                continue
            if kind == "scenario" and label == next(iter(SCENARIOS)):
                lines.append("         -- Table 3 scenarios (6/5/4-sensor files pooled) --")
            m = {k: np.mean([r[k] for r in rs]) for k in ("mae_l1_single", "mae_l1", "eucl", "nll", "cov1", "cov2")}
            lines.append(f"         {label:18s} {rs[0]['population']:>10s} {rs[0]['n_traj']:6d} "
                         f"{m['mae_l1_single']:10.3f} {m['mae_l1']:8.3f} {m['eucl']:8.3f} "
                         f"{m['nll']:8.3f} {m['cov1']:6.3f} {m['cov2']:6.3f}")
    text = "\n".join(lines)
    print(text)
    out_csv = Path(out_csv)
    out_csv.with_name(out_csv.stem.replace("results", "summary") + ".txt").write_text(text + "\n")
    return rows


def run_evals(ckpt, a, eval_seeds):
    pops = ("heldout", "full") if a.eval_population == "both" else (a.eval_population,)
    for pop in pops:
        evaluate(ckpt, a.tasks_root, a.device, eval_seeds, a.n_z, a.extrap_ctx,
                 ckpt.with_name("test_results.csv" if pop == "heldout" else "test_results_full.csv"), pop)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "trajectories"),
                    help="folder of <depth>-64T_<hull>/ configuration folders with dataset*.csv, i.e. the "
                         "paper's ANP/data/trajectories (default: data/trajectories/, git-ignored)")
    ap.add_argument("--train-set", choices=("id", "id6s", "glob"), default="id",
                    help="id: the paper's 27-CSV in-distribution corpus (default); id6s: its 9 six-sensor "
                         "CSVs only; glob: MagneticTrajectoryDataset's file discovery")
    ap.add_argument("--allow-ood-in-train", action="store_true",
                    help="do not abort when an OOD probe configuration is in the training files")
    ap.add_argument("--tasks-root", default=str(ROOT / "data" / "tasks"),
                    help="test sets (3x3x3/ and 1x1x1/), as in eval_all_models.py")
    ap.add_argument("--out-dir", default=None,
                    help="default: training/np_magnetic_runs/<interpolation|extrapolation>/<arch>/seed_<seed>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extrapolation", action="store_true",
                    help="train with sequential (prefix) contexts, as the paper's extrapolation model")
    ap.add_argument("--use-meta", action="store_true", help="append [depth, length, width] to the inputs")
    ap.add_argument("--sensor-mask", action="store_true", help="random sensor switch-off augmentation")
    ap.add_argument("--normalize", action="store_true",
                    help="z-score inputs and positions with train statistics (off = the paper's raw data)")
    # optimisation (the paper's values)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=9e-5)
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="disable fp16 AMP on CUDA")
    # model sizes
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=128, help="anp_orig* only (the sensor-agnostic ANP uses hidden-dim)")
    ap.add_argument("--n-heads", type=int, default=8, help="anp_orig* only")
    ap.add_argument("--reg-layers", type=int, default=6, help="reg only: transformer layers over the sensors")
    # sensor-agnostic arms
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--var-floor", type=float, default=1e-3,
                    help="predictive-variance floor in physical units (m^2), sensor-agnostic arms only; "
                         "1e-3 = what their built-in floor gives on raw data")
    ap.add_argument("--attn-ffn", action="store_true")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kl-warmup-epochs", type=int, default=100)
    # evaluation
    ap.add_argument("--eval-only", metavar="CKPT", help="skip training; evaluate this best.pt")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--eval-population", choices=("heldout", "full", "both"), default="both",
                    help="in-distribution test sets: held-out trajectories only (test_results.csv), all "
                         "of them incl. training trajectories as eval_all_models.py does "
                         "(test_results_full.csv), or both (default)")
    ap.add_argument("--n-z", type=int, default=16, help="latent samples for the predictive mixture")
    ap.add_argument("--extrap-ctx", type=int, default=20, help="context size in extrapolation mode")
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    eval_seeds = [int(s) for s in a.eval_seeds.split(",") if s.strip()]

    if a.eval_only:
        run_evals(Path(a.eval_only), a, eval_seeds)
        return
    if not a.arch:
        ap.error("--arch is required (unless --eval-only)")
    out_dir = Path(a.out_dir) if a.out_dir else (
        ROOT / "training" / "np_magnetic_runs" / ("extrapolation" if a.extrapolation else "interpolation")
        / a.arch / f"seed_{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    run_evals(train(a, out_dir, a.device), a, eval_seeds)


if __name__ == "__main__":
    main()
