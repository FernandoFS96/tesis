"""Generate the random-layout datasets: many deployments that differ in the
positions of the hydrophones and, optionally, in the propagation environment.

For each channel option theta and each position set p:
  1. a trajectory ensemble is generated, once per theta and shared by every
     position set (mode shared) or separately for each set (mode distinct);
  2. a random sensor layout is drawn: a compact array whose center is shifted
     by a random offset (random_task.layout);
  3. the channel is simulated and the filtered acoustic features, trajectories
     and sensor positions are saved.

Output, under <out_dir>/<method>_<mode>/:
    position_set_<p>/channel_option_<theta>/random/
        channel_info/   trajs_<theta>.npy, sensor_positions_<theta>.npy,
                        channel_h_<theta>.npy (optional)
        filtered_data/  filtered_data.npy   (tau, ppt, n_traj, n_sensors)
        trajectory/     trajectories.npy    (3, n_traj, ppt)
    _manifest.pkl       seeds, positions and environments

All randomness derives from the master seed: set p uses master_seed + 1000 + p
for its layout, master_seed + traj_seed_offset + p (default 2000) for its
trajectories and master_seed + 3000 + p for its environment. A run split into
shards (random_task.start_set / end_set) therefore reproduces a serial run.
Called by data/generate.py.
"""

import os
import argparse
import pickle

import numpy as np
from tqdm import tqdm

# Channel physics (channel class, generate_params, trajectory helpers).
import acoustic_data_generator as base


# --------------------------------------------------------------------------- #
# Reproducible per-position-set layout seeds
# --------------------------------------------------------------------------- #
def layout_seed_for(master_seed: int, position_set_idx: int) -> int:
    """Seed for the sensor layout of a position set, in its own numeric band."""
    return int(master_seed) + 1000 + int(position_set_idx)


def env_seed_for(master_seed: int, position_set_idx: int, offset: int = 3000) -> int:
    """Seed for the environment of a position set, in its own numeric band, so
    an environment is reproducible from (master_seed, p) alone."""
    return int(master_seed) + int(offset) + int(position_set_idx)


# --------------------------------------------------------------------------- #
# Environment sampling
# --------------------------------------------------------------------------- #
# Parameters used by mpgeometry():
#   h0   water depth [m]
#   ht0  transmitter height [m] (equal to h0 by default: source at the surface)
#   hr0  receiver height above the bottom [m] (also the sensors' z)
#   c    sound speed in water [m/s]
#   c2   sound speed in the bottom [m/s], drawn as a fraction of c
#   k    spreading factor (path-loss exponent)
# c2 must stay below c, or reflcoeff() returns NaN.
ENV_KEYS = ("h0", "ht0", "hr0", "c", "c2", "k")


def sample_environment(seed: int, env_cfg) -> dict:
    """Draw one environment. Ranges are [lo, hi] pairs; lo == hi pins a value.

    Returns a dict of channel_info overrides. Deterministic in `seed`, so the
    same position-set index yields the same environment in any run."""
    rng = np.random.default_rng(int(seed))

    def u(key, default):
        r = None if env_cfg is None else env_cfg.get(key, None)
        if r is None:
            return float(default)
        lo, hi = float(r[0]), float(r[1])
        return lo if hi <= lo else float(rng.uniform(lo, hi))

    h0 = u("h0", 50.0)
    c = u("c", 1500.0)
    c2_frac = u("c2_frac", 0.8)
    # Clamp defensively: a bottom at or above the water speed makes the
    # reflection coefficient complex and the whole channel NaN.
    c2_frac = float(min(max(c2_frac, 0.30), 0.98))
    env = {
        "h0": h0,
        # The source stays at the surface unless ht_frac is given.
        "ht0": h0 * u("ht_frac", 1.0),
        "hr0": u("hr0", 1.0),
        "c": c,
        "c2": c * c2_frac,
        "k": u("k", 1.7),
    }
    return env


def apply_environment(params: dict, env) -> dict:
    """Overlay an environment onto params from generate_params(), after the
    theta scaling: theta sets the small-scale variability and the environment
    the propagation geometry. env=None leaves params unchanged."""
    if not env:
        return params
    ci = params['ci']
    for k in ENV_KEYS:
        if k in env:
            ci[k] = float(env[k])
    return params


def traj_seed_for(master_seed: int, position_set_idx: int, offset: int) -> int:
    """Seed for the trajectory ensemble of a position set (distinct mode), in
    its own numeric band."""
    return int(master_seed) + int(offset) + int(position_set_idx)


# --------------------------------------------------------------------------- #
# Random sensor placement (one independent layout per position-set)
# --------------------------------------------------------------------------- #
def random_sensor_positions(traj, n_sensors, hr0, layout_seed,
                            aperture_frac=0.5, offset_frac=0.3,
                            scale_jitter=0.0, min_span=20.0):
    """Random hydrophone layout for one position set: a compact array whose
    center is shifted by a random offset from the center of the trajectory
    field. Sizes are fractions of the field's extent along each axis.

    traj: trajectory ensemble, (3, n_traj, ppt+1).
    n_sensors: number of hydrophones.
    hr0: receiver height [m] (z of every sensor).
    layout_seed: RNG seed of this layout.
    aperture_frac: array span as a fraction of the field extent.
    offset_frac: maximum shift of the array center, as a fraction of the field
        extent (0 keeps it centered).
    scale_jitter: optional relative jitter of the aperture.
    min_span: minimum array span [m].

    Returns the sensor positions, (3, n_sensors).
    """
    xs = traj[0].ravel()
    ys = traj[1].ravel()
    if xs.size == 0:
        raise ValueError("Trajectories array is empty")

    # Robust bounding box of the trajectory ensemble
    x_lo, x_hi = np.percentile(xs, [2.0, 98.0])
    y_lo, y_hi = np.percentile(ys, [2.0, 98.0])

    field_cx = 0.5 * (x_lo + x_hi)
    field_cy = 0.5 * (y_lo + y_hi)
    extent_x = x_hi - x_lo
    extent_y = y_hi - y_lo

    rng = np.random.default_rng(layout_seed)

    # 1) Array center: field center plus a random shift
    array_cx = field_cx + rng.uniform(-offset_frac, offset_frac) * extent_x
    array_cy = field_cy + rng.uniform(-offset_frac, offset_frac) * extent_y

    # 2) Array aperture, with optional jitter
    jitter = rng.uniform(-scale_jitter, scale_jitter) if scale_jitter else 0.0
    span_x = max(extent_x * aperture_frac * (1.0 + jitter), min_span)
    span_y = max(extent_y * aperture_frac * (1.0 + jitter), min_span)
    max_x = 0.5 * span_x
    max_y = max(0.5 * span_y, min_span / 2.0)

    # 3) Scatter the sensors inside the translated box.
    x = rng.uniform(array_cx - max_x, array_cx + max_x, n_sensors)
    y = rng.uniform(array_cy - max_y, array_cy + max_y, n_sensors)

    r_posicion = np.zeros((3, n_sensors))
    r_posicion[0, :] = x
    r_posicion[1, :] = y
    r_posicion[2, :] = hr0
    return r_posicion


# --------------------------------------------------------------------------- #
# Single (theta, position-set) generation
# --------------------------------------------------------------------------- #
def generate_one(option, position_set_idx, trajectories, layout_seed,
                 out_dir, snr, rep, nop, signal_n=1024, traj_config=None,
                 layout_params=None, signal_type='sinusoid',
                 save_channel_h=True, env=None, fixed_positions=None):
    """Simulate the channel and the filtered features for one channel option
    (theta) and one sensor layout (position set), and save them.

    `trajectories` is the ensemble to use (shared or per set). The channel is
    built with it as precomputed_trajectories and with sensor placement patched
    to return our layout, so obtain_h() runs once. Returns the (3, n_sensors)
    sensor positions used.
    """
    # Fresh params for this theta (df / n_traj / ppt set by override_base_params).
    params = base.generate_params(options=option)
    # Per-set environment, applied after the theta scaling (None: unchanged).
    params = apply_environment(params, env)

    # 1) Sensor layout. A pinned layout (fixed_positions) is drawn once by the
    #    caller and reused; z follows hr0, so it still tracks the environment.
    if fixed_positions is not None:
        r_posicion = np.array(fixed_positions, dtype=float, copy=True)
        r_posicion[2, :] = float(params['ci']['hr0'])
    else:
        r_posicion = random_sensor_positions(
            traj=trajectories,
            n_sensors=params['n_sensors'],
            hr0=params['ci']['hr0'],
            layout_seed=layout_seed,
            **(layout_params or {}),
        )

    # 2) Patch sensor placement on the class, so that the constructor's call to
    #    obtain_h() uses our layout.
    original_gsp = base.channel.generate_sensor_positions

    def _fixed_layout(self, *args, **kwargs):
        return r_posicion

    base.channel.generate_sensor_positions = _fixed_layout
    try:
        # 3) Build the channel; obtain_h() runs once.
        c = base.channel(
            load=False,
            params=params,
            number_of_processes=nop,
            name=str(option),
            topology='random',
            precomputed_trajectories=trajectories,
            traj_config=traj_config,
        )
    finally:
        base.channel.generate_sensor_positions = original_gsp  # always restore

    # ---- check that the channel used the given trajectories (ppt+1 points each)
    if not np.array_equal(np.asarray(c.traj), np.asarray(trajectories)):
        raise RuntimeError(
            f"[set {position_set_idx}, theta {option}] channel did NOT reuse the "
            f"supplied trajectories (max|diff|="
            f"{np.abs(np.asarray(c.traj) - np.asarray(trajectories)).max():.3g})."
            " Precomputed-trajectory invariant violated, aborting before writing."
        )
    if not np.array_equal(np.asarray(c.r_posicion), r_posicion):
        raise RuntimeError(
            f"[set {position_set_idx}, theta {option}] channel did NOT use the "
            "requested sensor layout. Aborting before writing."
        )

    # Filtered features and target coordinates. A fixed trajectory order
    # (`specific=`) keeps trajectory i the same in every position set;
    # filter() would otherwise shuffle them.
    n_traj = c.params['n_traj'] #type: ignore
    canonical = list(range(n_traj))
    data, trjs = c.filter(
        signal_n, snr=snr, nt=n_traj, signal_type=signal_type, rep=rep,
        specific=canonical,
    )

    # ---- write the outputs ----
    set_dir = os.path.join(out_dir, f'position_set_{position_set_idx:02d}')
    topology_dir = os.path.join(set_dir, f'channel_option_{option}', 'random')
    info_dir = os.path.join(topology_dir, 'channel_info')
    os.makedirs(os.path.join(topology_dir, 'trajectory'), exist_ok=True)
    os.makedirs(os.path.join(topology_dir, 'filtered_data'), exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    # Check that the saved trajectories are the given ones, in order.
    ppt = c.params['ppt'] #type: ignore
    expected_trjs = np.asarray(trajectories)[:, :n_traj, 0:ppt]
    if not np.array_equal(np.asarray(trjs), expected_trjs):
        raise RuntimeError(
            f"[set {position_set_idx}, theta {option}] saved target trajectories "
            f"are not the supplied trajectories in canonical order "
            f"(max|diff|={np.abs(np.asarray(trjs) - expected_trjs).max():.3g}). "
            "filter() may have reintroduced random trajectory selection."
        )
    np.save(os.path.join(topology_dir, 'trajectory', 'trajectories.npy'), trjs)
    np.save(os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy'), data)
    # The raw channel response is not used for training; save_channel_h=False
    # skips it.
    if save_channel_h:
        np.save(os.path.join(info_dir, f'channel_h_{option}.npy'), c.h) #type: ignore
    np.save(os.path.join(info_dir, f'trajs_{option}.npy'), c.traj) #type: ignore
    np.save(os.path.join(info_dir, f'sensor_positions_{option}.npy'), r_posicion)

    return r_posicion


# --------------------------------------------------------------------------- #
# Base-params override (df / n_traj / ppt) applied to the imported module
# --------------------------------------------------------------------------- #
def override_base_params(df, n_traj, ppt, B=None, absolute_delay=False,
                         chirp_band=None, matched_filter=False):
    """Wrap base.generate_params so that every call picks up df, n_traj, ppt
    and the optional B, absolute_delay and chirp_band settings from the config.
    With the defaults the physics is unchanged."""
    _orig_generate_params = base.generate_params

    def _patched(options=None):
        params = _orig_generate_params(options=options)
        if B is not None:
            params['ci']['B'] = float(B)
        params['ci']['df'] = float(df)
        params['ci']['fmax'] = params['ci']['fmin'] + params['ci']['B']
        params['n_traj'] = int(n_traj)
        params['ppt'] = int(ppt)
        params['absolute_delay'] = bool(absolute_delay)
        params['matched_filter'] = bool(matched_filter)
        if chirp_band is not None:
            params['chirp_band'] = (float(chirp_band[0]), float(chirp_band[1]))
        return params

    base.generate_params = _patched  # type: ignore


# --------------------------------------------------------------------------- #
# Shared trajectory generation (once per theta, reused across all sets)
# --------------------------------------------------------------------------- #
def make_trajectories(option, nop, physics_seed, traj_config=None):
    """Trajectory ensemble for one theta, seeded with physics_seed (once per
    theta in shared mode, once per set and theta in distinct mode). The global
    NumPy RNG is seeded because trajectory generation draws from it."""
    np.random.seed(physics_seed)
    params = base.generate_params(options=option)
    temp = base.channel(
        load=False,
        params=params,
        number_of_processes=nop,
        name=str(option),
        topology='random',  # topology irrelevant for the trajectories themselves
        traj_config=traj_config,
    )
    return temp.traj


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(channel_options, n_position_sets, out_dir, snr, rep, nop,
        master_seed, start_set, end_set, signal_n=1024,
        distinct_trajectories=False, traj_seed_offset=2000, traj_config=None,
        traj_method=None, variant_tag=None, layout_params=None,
        signal_type='sinusoid', save_channel_h=True,
        env_cfg=None, env_seed_offset=3000, layout_fixed_seed=None):
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        'channel_options': channel_options,
        'n_position_sets': n_position_sets,
        'master_seed': master_seed,
        'snr': snr,
        'rep': rep,
        'distinct_trajectories': bool(distinct_trajectories),
        'traj_seed_offset': int(traj_seed_offset),
        'traj_method': traj_method,
        'variant_tag': variant_tag,
        'layout_params': dict(layout_params or {}),  # sensor-layout distribution
        'signal_type': signal_type,
        'positions': {},  # (position_set_idx, theta) -> (3, n_sensors)
        'layout_seeds': {},
        'traj_seeds': {},  # only populated in distinct mode
        'layout_fixed_seed': layout_fixed_seed,
        'env_cfg': dict(env_cfg or {}),   # the sampling ranges actually used
        'environments': {},               # position_set_idx -> drawn environment
    }

    # Shared mode: one trajectory ensemble per theta, reused by every set.
    shared_traj = {}
    if not distinct_trajectories:
        print("Generating shared trajectories (once per theta)...")
        for option in channel_options:
            shared_traj[option] = make_trajectories(
                option, nop=nop, physics_seed=master_seed, traj_config=traj_config
            )
            print(f"  theta={option}: trajectories {shared_traj[option].shape}")
    else:
        print("Distinct mode: trajectories generated per (position-set, theta).")

    # For each position set: its layout (and, in distinct mode, trajectories),
    # then every theta. A pinned layout is drawn once (see generate_one).
    _fixed_pos = None
    set_range = range(start_set, end_set)
    for p in tqdm(set_range, desc="Position sets"):
        # layout_fixed_seed pins one array for every position set, so that only
        # the environment varies between deployments.
        l_seed = (int(layout_fixed_seed) if layout_fixed_seed is not None
                  else layout_seed_for(master_seed, p))
        manifest['layout_seeds'][p] = l_seed
        # One environment per position set, seeded from the global set index.
        env_p = (sample_environment(env_seed_for(master_seed, p, env_seed_offset),
                                    env_cfg) if env_cfg else None)
        if env_p is not None:
            manifest['environments'][p] = dict(env_p)
        if distinct_trajectories:
            manifest['traj_seeds'][p] = traj_seed_for(master_seed, p, traj_seed_offset)
        for option in channel_options:
            if distinct_trajectories:
                trajectories = make_trajectories(
                    option, nop=nop,
                    physics_seed=traj_seed_for(master_seed, p, traj_seed_offset),
                    traj_config=traj_config,
                )
            else:
                trajectories = shared_traj[option]
            if layout_fixed_seed is not None and _fixed_pos is None:
                # The pinned array is derived from the trajectories of set 0,
                # so every shard of a split run pins the same array.
                _ref_traj = trajectories
                if distinct_trajectories:
                    _ref_traj = make_trajectories(
                        option, nop=nop,
                        physics_seed=traj_seed_for(master_seed, 0, traj_seed_offset),
                        traj_config=traj_config)
                _p0 = apply_environment(
                    base.generate_params(options=option),
                    sample_environment(env_seed_for(master_seed, 0, env_seed_offset),
                                       env_cfg) if env_cfg else None)
                _fixed_pos = random_sensor_positions(
                    traj=_ref_traj, n_sensors=_p0['n_sensors'],
                    hr0=_p0['ci']['hr0'], layout_seed=int(layout_fixed_seed),
                    **(layout_params or {}))
            r_pos = generate_one(
                option=option,
                position_set_idx=p,
                trajectories=trajectories,
                layout_seed=l_seed,
                out_dir=out_dir,
                snr=snr,
                rep=rep,
                nop=nop,
                signal_n=signal_n,
                traj_config=traj_config,
                layout_params=layout_params,
                signal_type=signal_type,
                save_channel_h=save_channel_h,
                env=env_p,
                fixed_positions=_fixed_pos,
            )
            manifest['positions'][(p, option)] = r_pos

    # Manifest. A shard writes its own range-tagged file so parallel shards do
    # not overwrite each other; a full run writes _manifest.pkl.
    sharded = (start_set, end_set) != (0, n_position_sets)
    m_name = (f'_manifest_{start_set:06d}_{end_set:06d}.pkl' if sharded
              else '_manifest.pkl')
    with open(os.path.join(out_dir, m_name), 'wb') as f:
        pickle.dump(manifest, f)
    print(f"\nDone. Wrote position-sets [{start_set}, {end_set}) to: {out_dir}")
    print(f"Manifest: {os.path.join(out_dir, m_name)}")


def parse_float_list(s):
    if s is None:
        return None
    parts = [p for p in s.replace(',', ' ').split() if p != '']
    return [float(p) for p in parts]


def _env_cfg(rt):
    """The random_task.environment block as a dict, or None when absent or
    disabled."""
    try:
        env = rt['environment']
    except Exception:
        return None
    if env is None or not bool(env.get('enabled', False)):
        return None
    out = {}
    for k in ('h0', 'c', 'c2_frac', 'k', 'hr0', 'ht_frac'):
        try:
            v = env[k]
        except Exception:
            continue
        if v is not None:
            out[k] = [float(v[0]), float(v[1])]
    return out or None


def _opt(block, key, default):
    """Optional config key: None when absent or null."""
    try:
        v = block[key]
    except Exception:
        return default
    return default if v is None else int(v)


def run_random_task(cfg):
    """Generate the random-layout datasets from config/data_pipeline.yaml (the
    channel and random_task blocks and method) under
    <random_task.out_dir>/<method>_<mode>/. Called by data/generate.py."""
    ch = cfg['channel']
    rt = cfg['random_task']
    method = cfg.get('method', 'spiral')

    channel_options = [float(x) for x in ch['channel_options']]
    n_traj = int(ch['n_traj']); ppt = int(ch['ppt']); df = float(ch['df'])
    snr = float(ch['snr']); rep = int(ch['rep'])
    master_seed = int(ch.get('master_seed', 11)); nop = int(ch.get('nop', -1))
    # --- optional signal settings ---
    signal_type = str(ch.get('signal_type', 'sinusoid'))
    absolute_delay = bool(ch.get('absolute_delay', False))
    matched_filter = bool(ch.get('matched_filter', False))
    save_channel_h = bool(ch.get('save_channel_h', True))
    # Probe length; filter() needs signal_n >= 2 * Lf.
    signal_n = int(ch.get('signal_n', 1024))
    B = ch.get('B', None)
    cb = ch.get('chirp_band', None)
    chirp_band = (float(cb[0]), float(cb[1])) if cb else None

    n_position_sets = int(rt['n_position_sets'])
    distinct_trajectories = bool(rt.get('distinct_trajectories', False))
    traj_seed_offset = int(rt.get('traj_seed_offset', 2000))

    # Sensor-layout distribution (random_task.layout).
    lt = rt.get('layout', {}) or {}
    layout_params = {
        'aperture_frac': float(lt.get('aperture_frac', 0.5)),
        'offset_frac':   float(lt.get('offset_frac', 0.3)),
        'scale_jitter':  float(lt.get('scale_jitter', 0.0)),
        'min_span':      float(lt.get('min_span', 20.0)),
    }

    # Apply df / n_traj / ppt to the physics module and disable its side-effect
    # write to ./data.
    override_base_params(df=df, n_traj=n_traj, ppt=ppt, B=B,
                         absolute_delay=absolute_delay, chirp_band=chirp_band,
                         matched_filter=matched_filter)
    base.channel.save_channel_info = lambda self, name: None  # type: ignore

    # Variant tag keeps each (method, mode) dataset in its own folder.
    variant_tag = f"{method}_{'distinct' if distinct_trajectories else 'shared'}"
    out_dir = os.path.join(str(rt['out_dir']), variant_tag)

    _fmin = 10000.0
    _B = float(B) if B is not None else 10000.0
    Lf = len(base.range_m(_fmin, _fmin + _B, df))
    _c = 1500.0
    # filter() needs signal_n >= 2 * Lf; check it here.
    _need = 2 * Lf
    if signal_n < _need:
        raise SystemExit(
            f"channel.signal_n={signal_n} is too short for Lf={Lf}: "
            f"channel.filter() requires signal_n >= 2*Lf = {_need}. "
            f"Set channel.signal_n={1 << (_need - 1).bit_length()} or larger.")
    print("=" * 64)
    print("RANDOM task -- dataset generation")
    print("=" * 64)
    print(f"  thetas             : {channel_options}")
    print(f"  position sets      : {n_position_sets}")
    print(f"  n_traj / ppt       : {n_traj} / {ppt}")
    print(f"  band / df          : {_fmin/1e3:.1f}-{(_fmin+_B)/1e3:.1f} kHz (B={_B:.0f} Hz)"
          f" / df={df} Hz  ->  Lf={Lf} time-points")
    print(f"  signal_type        : {signal_type}"
          f"{f' (band {chirp_band})' if chirp_band else ''}")
    print(f"  matched_filter     : {matched_filter}")
    print(f"  save channel_h     : {save_channel_h}")
    print(f"  probe length       : {signal_n} samples (needs >= 2*Lf = {2*Lf})")
    print(f"  absolute_delay     : {absolute_delay}"
          f"   [unambiguous range < c/df = {_c/df:.0f} m;"
          f" range resolution ~ c/2B = {_c/(2*_B):.2f} m]")
    print(f"  feature dim / point: Lf*n_sensors = {Lf}*10 = {Lf*10}")
    print(f"  snr / rep          : {snr} / {rep}")
    print(f"  master seed        : {master_seed}")
    print(f"  traj method        : {method}")
    print(f"  trajectory mode    : "
          f"{'DISTINCT (own trajectories per set; seed offset ' + str(traj_seed_offset) + ')' if distinct_trajectories else 'SHARED (one ensemble reused across sets)'}")
    print(f"  sensor layout      : compact array, aperture_frac="
          f"{layout_params['aperture_frac']}, offset_frac={layout_params['offset_frac']} "
          f"(translation = the displacement OOD axis), scale_jitter={layout_params['scale_jitter']}")
    print(f"  out dir            : {out_dir}")
    print("=" * 64)

    run(
        channel_options=channel_options,
        n_position_sets=n_position_sets,
        out_dir=out_dir,
        snr=snr,
        rep=rep,
        nop=nop,
        master_seed=master_seed,
        start_set=_opt(rt, 'start_set', 0),
        end_set=_opt(rt, 'end_set', n_position_sets),
        env_cfg=_env_cfg(rt),
        env_seed_offset=_opt(rt, 'env_seed_offset', 3000),
        layout_fixed_seed=_opt(rt.get('layout', {}), 'fixed_seed', None),
        distinct_trajectories=distinct_trajectories,
        traj_seed_offset=traj_seed_offset,
        traj_config=cfg,
        traj_method=method,
        variant_tag=variant_tag,
        layout_params=layout_params,
        signal_type=signal_type,
        save_channel_h=save_channel_h,
        signal_n=signal_n,
    )


if __name__ == '__main__':
    print("random_position_generator.py is now a library used by the unified "
          "generator. Run:\n  python data/generate.py task=random")