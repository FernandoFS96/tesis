"""
random_position_generator.py
==================================

Revised acoustic-channel data generator for the *sensor-displacement* robustness study.

WHAT THIS SCRIPT DOES
---------------------
It produces ``n_position_sets`` (default 80) datasets that differ **only in the
(x, y) positions of the 10 hydrophones**. For every channel-variability value
``theta`` (the ``channel_option``) the script:

  1. Generates a trajectory ensemble (default 50 trajectories, 50 points each).
     In ``shared`` mode this is generated ONCE per theta and reused by every
     position-set, so sensor geometry is the only varying factor; in ``distinct``
     mode each set gets its own ensemble.
  2. For each position-set, draws a *different* random sensor layout
     (RANDOM topology only) via ``random_sensor_positions``, a COMPACT array
     whose CENTRE is translated by a per-set random offset (the sensor-
     displacement OOD axis; see ``random_task.layout``), NOT a re-jitter of one
     fixed box. It then simulates the channel and filtered acoustic features.
  3. Saves, per (theta, position-set), the filtered acoustic data, the
     trajectories, and the sensor positions themselves (for the spatial-encoder
     experiments).

Design notes vs the original ``acoustic_data_generator.py``:

  * Only the RANDOM topology is generated.
  * Each position-set ``p`` uses its own reproducible layout seed
    (``master_seed + 1000 + p``), and the layout is a translated compact array
    parameterised by ``random_task.layout`` (``aperture_frac``, ``offset_frac``,
    ``scale_jitter``), so the 80 layouts genuinely span sensor displacement.
  * All feature/physics knobs (``df``, ``n_traj``, ``ppt``, ...) come from the
    shared ``channel`` block of ``config/data_pipeline.yaml``.
  * Sensor positions and the layout params are always saved next to the data
    (``sensor_positions_<theta>.npy`` and ``_manifest.pkl``).

OUTPUT LAYOUT
-------------
By default the data is nested under a per-variant tag ``<method>_<mode>`` (e.g.
``hermite_shared``, ``spiral_distinct``) so the four trajectory-method x
cross-set-mode combinations never overwrite each other. Pass
``--no-variant-subdir`` to write straight into ``<out-dir>`` (legacy flat).
::

    <out-dir>/
      <method>_<mode>/                     # e.g. hermite_shared (omit with --no-variant-subdir)
        position_set_00/
          channel_option_0.0/
            random/
              channel_info/
                channel_h_0.0.npy          # raw impulse responses (tau, t, traj, sensor)
                trajs_0.0.npy              # full trajectories (3, n_traj, ppt+1)
                sensor_positions_0.0.npy   # (3, n_sensors) for THIS position-set
              filtered_data/
                filtered_data.npy          # (tau, ppt, n_traj*rep, n_sensors)
              trajectory/
                trajectories.npy           # (3, n_traj*rep, ppt) target coords
            ...
          channel_option_0.1/ ...
        position_set_01/ ...
        ...
        position_set_19/ ...
        _manifest.pkl                      # bookkeeping (seeds, positions, method, variant_tag)

The per-(theta) sub-structure inside each ``position_set_XX`` directory is
deliberately IDENTICAL to what the original ``data_process_topology.py`` expects
(``channel_option_<opt>/random/filtered_data/filtered_data.npy`` and
``.../trajectory/trajectories.npy``), so we can post-process each position-set
folder with the existing processing script by pointing ``--data-dir`` at it.

The original physics (``generate_params``, ``obtain_h``, ``filter``, the
multipath / Doppler model) is reused UNCHANGED and imported from the original
module, so results stay consistent with the published acoustic pipeline.

USAGE
-----
Basic (20 position-sets, default thetas, collaboration spec)::

    python random_position_generator.py

Explicit / typical run::

    python random_position_generator.py \
        --channel_options "0.0,0.1,0.2,0.3,0.4,0.5" \
        --n_position_sets 20 \
        --n_traj 100 \
        --ppt 50 \
        --df 100 \
        --snr 10 \
        --rep 1 \
        --out-dir ./data_random_positions \
        --master-seed 11

Then post-process each position-set with the existing splitter, e.g.::

    for p in $(seq -w 0 19); do
        python random_position_generator.py \
            --data-dir ./data_random_positions/position_set_$p \
            --mode separate
    done

NOTES
-----
* ``--df`` overrides the frequency resolution. The per-trajectory-point feature
  dimension after reshaping is ``Lf * n_sensors`` where
  ``Lf = len(range(fmin, fmax, df))``: 1010 for df=100 / 10 sensors.
  Remember to set the model ``input_dim`` accordingly when training.
* All randomness is derived from ``--master-seed`` so an entire 20-set run is
  bit-for-bit reproducible. Position-set ``p`` uses layout seed
  ``master_seed + 1000 + p`` (see ``layout_seed_for``).
* This is compute-heavy: cost scales as
  ``n_position_sets * n_thetas`` channel simulations. Use ``--nop`` to control
  the joblib parallelism and consider running subsets of position-sets in
  parallel jobs (``--start-set`` / ``--end-set``).
"""

import os
import argparse
import pickle

import numpy as np
from tqdm import tqdm

# Reuse the original, unmodified physics / channel implementation.
# (channel class, generate_params, generate_batch_of_trajs, helpers.)
import acoustic_data_generator as base


# --------------------------------------------------------------------------- #
# Reproducible per-position-set layout seeds
# --------------------------------------------------------------------------- #
def layout_seed_for(master_seed: int, position_set_idx: int) -> int:
    """Deterministic, well-separated RNG seed for the sensor layout of a given
    position-set. Keeping layout seeds in a separate numeric band from the
    physics seed avoids accidental correlation between the two."""
    return int(master_seed) + 1000 + int(position_set_idx)


def env_seed_for(master_seed: int, position_set_idx: int, offset: int = 3000) -> int:
    """Deterministic RNG seed for a position-set's ENVIRONMENT draw.

    Third numeric band, alongside layout (``master_seed + 1000 + p``) and
    trajectory (``master_seed + traj_seed_offset + p``), so the three streams
    never collide and an environment is reproducible from (master_seed, p)
    alone -- which is what lets a sharded run reproduce a serial one."""
    return int(master_seed) + int(offset) + int(position_set_idx)


# --------------------------------------------------------------------------- #
# Environment sampling (Path D: the environment as an inferable latent)
# --------------------------------------------------------------------------- #
# WHY THIS EXISTS. `theta` (channel_options) is NOT an environment: it is a
# single scalar `aux` that scales sig2s / sig2b / mu_p / nu_p / Sp and the
# height-and-range bounds together, i.e. one "how choppy" dimension. Everything
# that physically defines the channel -- depth, sound speeds, spreading -- is
# hard-coded in generate_params and identical in every dataset. Measured on
# pfn10kd, theta=0.1 vs 0.2 differ LESS than two trajectories within one theta
# (separability ratio 0.28), so theta cannot carry an environment-inference task.
#
# These are the parameters mpgeometry() actually consumes:
#   h0   water depth [m]                 -> surface/bottom bounce geometry
#   ht0  transmitter height [m]          -> kept equal to h0 by default, which
#        reproduces the current source-at-surface setup (h - ht = 0)
#   hr0  receiver height above bottom [m]-> also sets sensor z in the layout
#   c    sound speed in water [m/s]
#   c2   sound speed in bottom [m/s]     -> sampled as a FRACTION of c
#   k    spreading factor                -> path-loss exponent
#
# HARD CONSTRAINT: reflcoeff() evaluates sqrt(1 - (c2/c1)^2 cos^2(theta)), which
# goes negative -> NaN whenever c2 > c. c2 is therefore always drawn as
# c2_frac * c with c2_frac < 1 (a slower, "soft" bottom), never sampled freely.
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
        # Source stays at the surface (h - ht = 0), matching the shipped setup,
        # unless ht_frac is given as a fraction of depth.
        "ht0": h0 * u("ht_frac", 1.0),
        "hr0": u("hr0", 1.0),
        "c": c,
        "c2": c * c2_frac,
        "k": u("k", 1.7),
    }
    return env


def apply_environment(params: dict, env) -> dict:
    """Overlay an environment onto a params dict from generate_params().

    Applied AFTER the theta scaling so the two axes compose: theta still sets
    the small-scale variability, the environment sets the propagation geometry.
    ``env=None`` leaves params untouched, byte for byte."""
    if not env:
        return params
    ci = params['ci']
    for k in ENV_KEYS:
        if k in env:
            ci[k] = float(env[k])
    return params


def traj_seed_for(master_seed: int, position_set_idx: int, offset: int) -> int:
    """Deterministic RNG seed for a position-set's OWN trajectory ensemble
    (distinct-trajectories mode only). Lives in a separate numeric band from the
    layout seeds (``master_seed + 1000 + p``) so the trajectory and layout RNG
    streams never collide."""
    return int(master_seed) + int(offset) + int(position_set_idx)


# --------------------------------------------------------------------------- #
# Random sensor placement (one independent layout per position-set)
# --------------------------------------------------------------------------- #
def random_sensor_positions(traj, n_sensors, hr0, layout_seed,
                            aperture_frac=0.5, offset_frac=0.3,
                            scale_jitter=0.0, min_span=20.0):
    """
    Draw a single RANDOM hydrophone layout for one position-set.

    Sensor-displacement study design: instead of re-jittering 10 points inside
    ONE fixed box centred on the trajectory field (the old behaviour, which made
    all layouts nearly identical), this places a COMPACT array whose CENTRE is
    translated by a per-set random offset. Translation is the controlled OOD axis
   , "where the sensors sit changes between runs", while the aperture stays
    roughly fixed so the localization conditioning (the difficulty floor) is
    preserved. All knobs are fractions of the trajectory field's per-axis extent,
    so the same config works for any trajectory family (spiral, hermite, ...).

    Parameters
    ----------
    traj : np.ndarray
        Trajectory ensemble this layout is fitted to, shape (3, n_traj, ppt+1).
    n_sensors : int
        Number of hydrophones (10).
    hr0 : float
        Receiver height [m] (z-coordinate for every sensor).
    layout_seed : int
        RNG seed for THIS position-set's layout (varies across sets).
    aperture_frac : float
        Array span as a fraction of the field extent (compactness). 0.5 keeps a
        large-ish, well-conditioned array.
    offset_frac : float
        Max array-centre translation, +/- this fraction of the field extent (the
        displacement axis). 0.3 gives genuinely different layouts while staying
        climbable. Set 0.0 to recover a single fixed-centre distribution.
    scale_jitter : float
        Optional +/- fractional aperture jitter per set (0.0 = fixed aperture).
    min_span : float
        Floor on the array span [m] so tiny fields still get a usable aperture.

    Returns
    -------
    r_posicion : np.ndarray, shape (3, n_sensors)
    """
    xs = traj[0].ravel()
    ys = traj[1].ravel()
    if xs.size == 0:
        raise ValueError("Trajectories array is empty")

    # Robust bounding box of the trajectory ensemble (same percentiles as orig).
    x_lo, x_hi = np.percentile(xs, [2.0, 98.0])
    y_lo, y_hi = np.percentile(ys, [2.0, 98.0])

    field_cx = 0.5 * (x_lo + x_hi)
    field_cy = 0.5 * (y_lo + y_hi)
    extent_x = x_hi - x_lo
    extent_y = y_hi - y_lo

    rng = np.random.default_rng(layout_seed)

    # 1) Per-set array CENTRE: field centre + a random translation (the OOD axis).
    array_cx = field_cx + rng.uniform(-offset_frac, offset_frac) * extent_x
    array_cy = field_cy + rng.uniform(-offset_frac, offset_frac) * extent_y

    # 2) Array APERTURE: a compact, roughly-fixed span (optional per-set jitter).
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
    """
    Simulate the channel + filtered features for ONE channel option (theta) and
    ONE sensor layout (position-set), writing the result in the
    original-compatible directory layout.

    ``trajectories`` is the trajectory ensemble this (set, theta) should use. In
    shared mode it is the same array for every position-set; in distinct mode it
    is the per-set ensemble. Either way the channel is forced to reuse it
    verbatim (via ``precomputed_trajectories``), and the guards below assert it.

    Design (single-pass, no double obtain_h):
      * We compute our seeded sensor layout FIRST.
      * We patch ``channel.generate_sensor_positions`` to return that layout,
        and pass ``precomputed_trajectories=shared_trajectories``, BEFORE the
        channel is constructed.
      * ``channel.__init__`` then calls ``obtain_h()`` exactly once. Inside it,
        ``generate_trajectories()`` returns the shared array (because
        ``precomputed_trajectories`` is set) and ``generate_sensor_positions()``
        returns our layout. The heavy multipath/Doppler physics runs once,
        unmodified.

    This removes the fragile re-run of ``obtain_h`` and the lambda-signature
    hazard that previously crashed on the ``scale=`` keyword.

    Returns the (3, n_sensors) sensor positions used, for the manifest.
    """
    # Fresh params for this theta. df / n_traj / ppt were already injected into
    # the base.generate_params defaults via override_base_params() in main().
    params = base.generate_params(options=option)
    # Per-set ENVIRONMENT (depth / sound speeds / spreading), applied after the
    # theta scaling. env=None -> untouched, so every existing dataset is
    # reproduced exactly.
    params = apply_environment(params, env)

    # 1) Our per-set random layout (varies across position-sets, reproducible).
    # fixed_positions pins the array across sets. It cannot be reproduced by
    # simply reusing layout_seed: random_sensor_positions fits the array to THIS
    # set's trajectory field extent, so with distinct trajectories the same seed
    # still yields different positions. The caller therefore draws it once and
    # passes it in. Sensor z is re-set from hr0 so depth still tracks the
    # environment while x/y stay fixed.
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

    # 2) Patch placement on the CLASS so the constructor's single obtain_h()
    #    call uses our layout. The patched function accepts and ignores the
    #    positional ``traj`` and any keywords (e.g. scale=, min_span=).
    original_gsp = base.channel.generate_sensor_positions

    def _fixed_layout(self, *args, **kwargs):
        return r_posicion

    base.channel.generate_sensor_positions = _fixed_layout
    try:
        # 3) Construct ONCE; obtain_h runs once with our traj + our layout.
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

    # ---- Invariant guard: the channel MUST have used the trajectories we gave it.
    #      ppt = params['ppt']; the input traj has ppt+1 points, c.traj likewise.
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

    # Filtered acoustic features + target trajectory coordinates.
    # IMPORTANT: channel.filter() otherwise draws a RANDOM trajectory ordering
    # (np.random.choice over n_traj) and returns self.traj reindexed by it. That
    # would shuffle the trajectories into a DIFFERENT row order for every
    # position-set, breaking row-alignment across geometries (trajectory i in set
    # 0 would not correspond to trajectory i in set 1). In shared mode that also
    # breaks the byte-identity invariant. We pass an explicit, canonical ordering
    # via `specific=` so every set stores trajectories (and their matched
    # features) in the SAME row order.
    n_traj = c.params['n_traj'] #type: ignore
    canonical = list(range(n_traj))
    data, trjs = c.filter(
        signal_n, snr=snr, nt=n_traj, signal_type=signal_type, rep=rep,
        specific=canonical,
    )

    # ---- Write in the original-compatible structure ----
    set_dir = os.path.join(out_dir, f'position_set_{position_set_idx:02d}')
    topology_dir = os.path.join(set_dir, f'channel_option_{option}', 'random')
    info_dir = os.path.join(topology_dir, 'channel_info')
    os.makedirs(os.path.join(topology_dir, 'trajectory'), exist_ok=True)
    os.makedirs(os.path.join(topology_dir, 'filtered_data'), exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    # Guard: the saved target trajectories must be the supplied trajectories in
    # canonical order (first ppt points). Aborts before writing if filter()
    # reordered or altered them.
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
    # The raw impulse response is a diagnostic artefact, not a training input,
    # and it doubles the on-disk size. Skipped when save_channel_h=False.
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
    """
    The physics lives in ``base.generate_params``. We wrap it so every call made
    inside this script (and inside the base ``channel``) picks up the
    collaboration spec (df=100, n_traj=100, ppt=50) without editing the original
    file. This keeps the original module pristine and importable.

    ``B`` (bandwidth, Hz), ``absolute_delay`` and ``chirp_band`` are opt-in; the
    defaults (B untouched, absolute_delay False) reproduce the original physics
    exactly. See config/data_pipeline.yaml for what they are for.
    """
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
    """
    Generate a trajectory ensemble for a given theta.

    In shared mode this is called ONCE per theta (with a fixed ``physics_seed``)
    and the result is reused for every position-set, so sensor geometry is the
    only thing that changes between datasets. In distinct mode it is called once
    per (set, theta) with a per-set ``physics_seed`` so each position-set gets
    its own trajectories.

    We seed the global NumPy RNG immediately before constructing the temporary
    channel because trajectory generation (``generate_trajectories``) draws from
    the global RNG.
    """
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

    # In SHARED mode, generate the trajectories ONCE per theta (fixed seed ==
    # master_seed) and reuse them for every position-set. In DISTINCT mode we
    # generate per-(set, theta) inside the loop instead, so leave this empty.
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

    # For each position-set, draw a layout (and, in distinct mode, its own
    # trajectories) and simulate every theta.
    # A pinned layout is drawn ONCE, from the first set's trajectory ensemble,
    # and reused verbatim by every set (see generate_one).
    _fixed_pos = None
    set_range = range(start_set, end_set)
    for p in tqdm(set_range, desc="Position sets"):
        # layout_fixed_seed pins ONE array geometry across every position-set, so
        # the ENVIRONMENT becomes the only thing that varies between tasks. Needed
        # for the environments-only split: otherwise a held-out set differs in
        # both layout and environment and the two axes cannot be separated.
        # (Sensor z still follows hr0, which is part of the environment.)
        l_seed = (int(layout_fixed_seed) if layout_fixed_seed is not None
                  else layout_seed_for(master_seed, p))
        manifest['layout_seeds'][p] = l_seed
        # One environment per position-set, seeded from the GLOBAL set index so
        # shards reproduce a serial run set-for-set.
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
                # Derive the pinned array from the trajectories of a FIXED
                # REFERENCE set (index 0), never from whichever set this process
                # happens to reach first. Sharded runs otherwise each pin a
                # different array -- measured: 4 shards produced 4 distinct
                # layouts, switching exactly at the shard boundaries -- because
                # random_sensor_positions fits the array to the trajectory field
                # extent it is handed. Set 0's ensemble is reproducible from
                # (master_seed, 0) in every process, so all shards now agree.
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

    # Manifest for bookkeeping / reproducibility. Parallel shards write disjoint
    # set ranges into ONE out_dir, so a shard must not clobber its siblings'
    # bookkeeping with its own partial view; it gets a range-tagged file instead.
    # A whole-range run is unaffected and still writes plain _manifest.pkl.
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
    """The random_task.environment block as a plain dict, or None when absent or
    disabled. None means every dataset generated so far is reproduced exactly."""
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
    """Read an OPTIONAL knob that older configs are not required to declare.

    Configs written before a knob existed simply lack the key, and OmegaConf in
    struct mode raises on a missing attribute rather than returning None -- so
    the lookup is guarded and an absent or explicitly-null key both fall back to
    the previous behaviour.
    """
    try:
        v = block[key]
    except Exception:
        return default
    return default if v is None else int(v)


def run_random_task(cfg):
    """Generate the RANDOM-task datasets from a unified config object
    (config/data_pipeline.yaml). Reads the shared ``channel`` block, the
    ``random_task`` block and ``method``; writes ``n_position_sets`` datasets of
    the 'random' topology under ``<random_task.out_dir>/<method>_<mode>/``.
    Invoked by data/generate.py (task=random)."""
    ch = cfg['channel']
    rt = cfg['random_task']
    method = cfg.get('method', 'spiral')

    channel_options = [float(x) for x in ch['channel_options']]
    n_traj = int(ch['n_traj']); ppt = int(ch['ppt']); df = float(ch['df'])
    snr = float(ch['snr']); rep = int(ch['rep'])
    master_seed = int(ch.get('master_seed', 11)); nop = int(ch.get('nop', -1))
    # --- broadband / timing options (all default to the ORIGINAL physics) ---
    signal_type = str(ch.get('signal_type', 'sinusoid'))
    absolute_delay = bool(ch.get('absolute_delay', False))
    matched_filter = bool(ch.get('matched_filter', False))
    save_channel_h = bool(ch.get('save_channel_h', True))
    # Probe length. channel.filter() requires signal_n >= 2*Lf, and Lf grows as
    # B/df -- the fine df needed for unambiguous delay makes the legacy 1024
    # too short (Lf=601 needs >=1202). Default keeps the legacy value.
    signal_n = int(ch.get('signal_n', 1024))
    B = ch.get('B', None)
    cb = ch.get('chirp_band', None)
    chirp_band = (float(cb[0]), float(cb[1])) if cb else None

    n_position_sets = int(rt['n_position_sets'])
    distinct_trajectories = bool(rt.get('distinct_trajectories', False))
    traj_seed_offset = int(rt.get('traj_seed_offset', 2000))

    # Sensor-layout distribution (translated compact array). Defaults match the
    # gentle sensor-displacement spec; override in random_task.layout.
    lt = rt.get('layout', {}) or {}
    layout_params = {
        'aperture_frac': float(lt.get('aperture_frac', 0.5)),
        'offset_frac':   float(lt.get('offset_frac', 0.3)),
        'scale_jitter':  float(lt.get('scale_jitter', 0.0)),
        'min_span':      float(lt.get('min_span', 20.0)),
    }

    # Inject df / n_traj / ppt into the imported physics, and suppress the
    # hardcoded ./data side-effect write in channel.__init__.
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
    # channel.filter() rejects a probe shorter than 2*Lf. Lf grows as B/df, and
    # the fine df needed for unambiguous delay makes the legacy 1024 too short
    # (Lf=601 needs >=1202) -- fail here with a fix, not deep inside joblib.
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