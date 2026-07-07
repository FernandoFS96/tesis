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
     (RANDOM topology only) via ``random_sensor_positions`` -- a COMPACT array
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
    ``scale_jitter``) -- so the 80 layouts genuinely span sensor displacement.
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
    -- "where the sensors sit changes between runs" -- while the aperture stays
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
                 layout_params=None):
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

    # 1) Our per-set random layout (varies across position-sets, reproducible).
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
            " Precomputed-trajectory invariant violated -- aborting before writing."
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
        signal_n, snr=snr, nt=n_traj, signal_type='sinusoid', rep=rep,
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
    np.save(os.path.join(info_dir, f'channel_h_{option}.npy'), c.h) #type: ignore
    np.save(os.path.join(info_dir, f'trajs_{option}.npy'), c.traj) #type: ignore
    np.save(os.path.join(info_dir, f'sensor_positions_{option}.npy'), r_posicion)

    return r_posicion


# --------------------------------------------------------------------------- #
# Base-params override (df / n_traj / ppt) applied to the imported module
# --------------------------------------------------------------------------- #
def override_base_params(df, n_traj, ppt):
    """
    The physics lives in ``base.generate_params``. We wrap it so every call made
    inside this script (and inside the base ``channel``) picks up the
    collaboration spec (df=100, n_traj=100, ppt=50) without editing the original
    file. This keeps the original module pristine and importable.
    """
    _orig_generate_params = base.generate_params

    def _patched(options=None):
        params = _orig_generate_params(options=options)
        params['ci']['df'] = float(df)
        params['ci']['fmax'] = params['ci']['fmin'] + params['ci']['B']
        params['n_traj'] = int(n_traj)
        params['ppt'] = int(ppt)
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
        traj_method=None, variant_tag=None, layout_params=None):
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
        'positions': {},  # (position_set_idx, theta) -> (3, n_sensors)
        'layout_seeds': {},
        'traj_seeds': {},  # only populated in distinct mode
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
    set_range = range(start_set, end_set)
    for p in tqdm(set_range, desc="Position sets"):
        l_seed = layout_seed_for(master_seed, p)
        manifest['layout_seeds'][p] = l_seed
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
            )
            manifest['positions'][(p, option)] = r_pos

    # Manifest for bookkeeping / reproducibility.
    with open(os.path.join(out_dir, '_manifest.pkl'), 'wb') as f:
        pickle.dump(manifest, f)
    print(f"\nDone. Wrote position-sets [{start_set}, {end_set}) to: {out_dir}")
    print(f"Manifest: {os.path.join(out_dir, '_manifest.pkl')}")


def parse_float_list(s):
    if s is None:
        return None
    parts = [p for p in s.replace(',', ' ').split() if p != '']
    return [float(p) for p in parts]


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
    override_base_params(df=df, n_traj=n_traj, ppt=ppt)
    base.channel.save_channel_info = lambda self, name: None  # type: ignore

    # Variant tag keeps each (method, mode) dataset in its own folder.
    variant_tag = f"{method}_{'distinct' if distinct_trajectories else 'shared'}"
    out_dir = os.path.join(str(rt['out_dir']), variant_tag)

    Lf = len(base.range_m(10000.0, 20000.0, df))
    print("=" * 64)
    print("RANDOM task -- dataset generation")
    print("=" * 64)
    print(f"  thetas             : {channel_options}")
    print(f"  position sets      : {n_position_sets}")
    print(f"  n_traj / ppt       : {n_traj} / {ppt}")
    print(f"  df                 : {df} Hz  ->  Lf={Lf} time-points")
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
        start_set=0,
        end_set=n_position_sets,
        distinct_trajectories=distinct_trajectories,
        traj_seed_offset=traj_seed_offset,
        traj_config=cfg,
        traj_method=method,
        variant_tag=variant_tag,
        layout_params=layout_params,
    )


if __name__ == '__main__':
    print("random_position_generator.py is now a library used by the unified "
          "generator. Run:\n  python data/generate.py task=random")