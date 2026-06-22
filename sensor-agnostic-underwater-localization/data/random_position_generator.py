"""
data_generator_random_positions.py
==================================

Revised acoustic-channel data generator for the *sensor-displacement* robustness
study (collaboration with KIT-ALR).

WHAT THIS SCRIPT DOES
---------------------
It produces ``N_POSITION_SETS`` (default 20) datasets that are **identical in
every respect except the (x, y) positions of the 10 hydrophones**. Concretely,
for every channel-variability value ``theta`` (the ``channel_option``) the script:

  1. Generates ONE fixed set of trajectories (100 trajectories, 50 points each).
     These trajectories are shared across *all* position-sets and *all* theta
     values handling is identical to the original pipeline.
  2. For each of the 20 position-sets, draws a *different* random placement of
     the 10 sensors (RANDOM topology only), then simulates the channel impulse
     response and the filtered acoustic features for those positions.
  3. Saves, per (theta, position-set), the filtered acoustic data, the
     trajectories, and crucially the sensor positions themselves (needed later
     for the spatial-encoder experiments).

Compared with the original ``data_generator_topology.py`` the important changes
are:

  * Only the RANDOM topology is generated (per the collaboration plan).
  * The random sensor layout is NO LONGER hardcoded to a single seed. Each
    position-set ``p`` uses its own reproducible seed, so the 20 layouts are
    genuinely different from one another while remaining fully reproducible.
  * Trajectories are generated ONCE per theta and reused for every position-set,
    so sensor geometry is the only varying factor between the 20 datasets.
  * Defaults updated to the collaboration spec: ``n_traj=100``, ``ppt=50``,
    ``df=100`` (this reduces the per-point acoustic feature dimension from
    4010 at df=25 / 2010 at df=50 down to 1010 at df=100).
  * Sensor positions are always saved next to the data.

OUTPUT LAYOUT
-------------
::

    <out-dir>/
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
      _manifest.pkl                      # bookkeeping (seeds, positions, args)

The per-(theta) sub-structure inside each ``position_set_XX`` directory is
deliberately IDENTICAL to what the original ``data_process_topology.py`` expects
(``channel_option_<opt>/random/filtered_data/filtered_data.npy`` and
``.../trajectory/trajectories.npy``), so you can post-process each position-set
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
        python data_process_topology.py \
            --data-dir ./data_random_positions/position_set_$p \
            --mode separate
    done

NOTES
-----
* ``--df`` overrides the frequency resolution. The per-trajectory-point feature
  dimension after reshaping is ``Lf * n_sensors`` where
  ``Lf = len(range(fmin, fmax, df))``: 1010 for df=100 / 10 sensors.
  Remember to set the model ``input_dim`` accordingly when you train.
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


# --------------------------------------------------------------------------- #
# Random sensor placement (one independent layout per position-set)
# --------------------------------------------------------------------------- #
def random_sensor_positions(traj, n_sensors, hr0, layout_seed,
                            scale=0.6, min_span=20.0):
    """
    Draw a single RANDOM hydrophone layout adapted to the spatial extent of the
    (shared) trajectories. This mirrors the 'random' branch of the original
    ``channel.generate_sensor_positions`` but takes an explicit ``layout_seed``
    so that every position-set gets a genuinely different layout.

    Parameters
    ----------
    traj : np.ndarray
        Shared trajectories, shape (3, n_traj, ppt+1).
    n_sensors : int
        Number of hydrophones (10).
    hr0 : float
        Receiver height [m] (z-coordinate for every sensor).
    layout_seed : int
        RNG seed for THIS position-set's layout (varies across the 20 sets).
    scale, min_span : float
        Same geometry knobs as the original generator. ``scale=0.6`` and
        ``min_span=20.0`` reproduce the values used in the published 'random'
        topology so the layouts live in the same operating area.

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

    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    span_x = max((x_hi - x_lo) * scale, min_span)
    span_y = max((y_hi - y_lo) * scale, min_span)

    max_x = 0.5 * span_x
    max_y = max(0.5 * span_y, min_span / 2.0)

    # The ONLY substantive change vs. the original: a per-set seed instead of
    # the hardcoded default_rng(10).
    rng = np.random.default_rng(layout_seed)
    x = rng.uniform(cx - max_x, cx + max_x, n_sensors)
    y = rng.uniform(cy - max_y, cy + max_y, n_sensors)

    r_posicion = np.zeros((3, n_sensors))
    r_posicion[0, :] = x
    r_posicion[1, :] = y
    r_posicion[2, :] = hr0
    return r_posicion


# --------------------------------------------------------------------------- #
# Single (theta, position-set) generation
# --------------------------------------------------------------------------- #
def generate_one(option, position_set_idx, shared_trajectories, layout_seed,
                 out_dir, snr, rep, nop, signal_n=1024):
    """
    Simulate the channel + filtered features for ONE channel option (theta) and
    ONE sensor layout (position-set), writing the result in the
    original-compatible directory layout.

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
        traj=shared_trajectories,
        n_sensors=params['n_sensors'],
        hr0=params['ci']['hr0'],
        layout_seed=layout_seed,
    )

    # 2) Patch placement on the CLASS so the constructor's single obtain_h()
    #    call uses our layout. The patched function accepts and ignores the
    #    positional ``traj`` and any keywords (e.g. scale=, min_span=).
    original_gsp = base.channel.generate_sensor_positions

    def _fixed_layout(self, *args, **kwargs):
        return r_posicion

    base.channel.generate_sensor_positions = _fixed_layout
    try:
        # 3) Construct ONCE; obtain_h runs once with shared traj + our layout.
        c = base.channel(
            load=False,
            params=params,
            number_of_processes=nop,
            name=str(option),
            topology='random',
            precomputed_trajectories=shared_trajectories,
        )
    finally:
        base.channel.generate_sensor_positions = original_gsp  # always restore

    # ---- Invariant guard: the channel MUST have used the shared trajectories.
    #      ppt = params['ppt']; shared traj has ppt+1 points, c.traj likewise.
    if not np.array_equal(np.asarray(c.traj), np.asarray(shared_trajectories)):
        raise RuntimeError(
            f"[set {position_set_idx}, theta {option}] channel did NOT reuse the "
            f"shared trajectories (max|diff|="
            f"{np.abs(np.asarray(c.traj) - np.asarray(shared_trajectories)).max():.3g})."
            " Shared-trajectory invariant violated -- aborting before writing."
        )
    if not np.array_equal(np.asarray(c.r_posicion), r_posicion):
        raise RuntimeError(
            f"[set {position_set_idx}, theta {option}] channel did NOT use the "
            "requested sensor layout. Aborting before writing."
        )

    # Filtered acoustic features + target trajectory coordinates.
    data, trjs = base.generate_batch_of_trajs(
        c, 'sinusoid', n=signal_n, snr=snr, rep=rep
    )

    # ---- Write in the original-compatible structure ----
    set_dir = os.path.join(out_dir, f'position_set_{position_set_idx:02d}')
    topology_dir = os.path.join(set_dir, f'channel_option_{option}', 'random')
    info_dir = os.path.join(topology_dir, 'channel_info')
    os.makedirs(os.path.join(topology_dir, 'trajectory'), exist_ok=True)
    os.makedirs(os.path.join(topology_dir, 'filtered_data'), exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    np.save(os.path.join(topology_dir, 'trajectory', 'trajectories.npy'), trjs)
    np.save(os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy'), data)
    np.save(os.path.join(info_dir, f'channel_h_{option}.npy'), c.h)
    np.save(os.path.join(info_dir, f'trajs_{option}.npy'), c.traj)
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
def make_shared_trajectories(option, nop, physics_seed):
    """
    Generate the trajectories for a given theta ONCE. These are reused for every
    position-set so that sensor geometry is the only thing that changes between
    the 20 datasets.

    We seed the global NumPy RNG immediately before constructing the temporary
    channel because trajectory generation (``generate_trajectories``) draws from
    the global RNG. Using the same physics_seed for every theta means the
    trajectory *shapes* are consistent with the original pipeline.
    """
    np.random.seed(physics_seed)
    params = base.generate_params(options=option)
    temp = base.channel(
        load=False,
        params=params,
        number_of_processes=nop,
        name=str(option),
        topology='random',  # topology irrelevant for the trajectories themselves
    )
    return temp.traj


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(channel_options, n_position_sets, out_dir, snr, rep, nop,
        master_seed, start_set, end_set, signal_n=1024):
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        'channel_options': channel_options,
        'n_position_sets': n_position_sets,
        'master_seed': master_seed,
        'snr': snr,
        'rep': rep,
        'positions': {},  # (position_set_idx, theta) -> (3, n_sensors)
        'layout_seeds': {},
    }

    # 1) Shared trajectories: generated ONCE per theta, reused for all sets.
    #    physics_seed fixed (== master_seed) so trajectory shapes are stable.
    print("Generating shared trajectories (once per theta)...")
    shared_traj = {}
    for option in channel_options:
        shared_traj[option] = make_shared_trajectories(
            option, nop=nop, physics_seed=master_seed
        )
        print(f"  theta={option}: trajectories {shared_traj[option].shape}")

    # 2) For each position-set, draw a layout and simulate every theta.
    set_range = range(start_set, end_set)
    for p in tqdm(set_range, desc="Position sets"):
        l_seed = layout_seed_for(master_seed, p)
        manifest['layout_seeds'][p] = l_seed
        for option in channel_options:
            r_pos = generate_one(
                option=option,
                position_set_idx=p,
                shared_trajectories=shared_traj[option],
                layout_seed=l_seed,
                out_dir=out_dir,
                snr=snr,
                rep=rep,
                nop=nop,
                signal_n=signal_n,
            )
            manifest['positions'][(p, option)] = r_pos

    # 3) Manifest for bookkeeping / reproducibility.
    with open(os.path.join(out_dir, '_manifest.pkl'), 'wb') as f:
        pickle.dump(manifest, f)
    print(f"\nDone. Wrote position-sets [{start_set}, {end_set}) to: {out_dir}")
    print(f"Manifest: {os.path.join(out_dir, '_manifest.pkl')}")


def parse_float_list(s):
    if s is None:
        return None
    parts = [p for p in s.replace(',', ' ').split() if p != '']
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Generate N random-position datasets sharing identical "
                    "trajectories (sensor-displacement robustness study)."
    )
    parser.add_argument('--channel_options', type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5",
                        help="Comma/space separated theta values "
                             "(default: 0.0..0.5).")
    parser.add_argument('--n_position_sets', type=int, default=20,
                        help="Number of distinct random sensor layouts (default 20).")
    parser.add_argument('--n_traj', type=int, default=100,
                        help="Trajectories per dataset (default 100).")
    parser.add_argument('--ppt', type=int, default=50,
                        help="Points per trajectory (default 50).")
    parser.add_argument('--df', type=float, default=100.0,
                        help="Frequency resolution [Hz] (default 100 -> "
                             "feature dim 1010 for 10 sensors).")
    parser.add_argument('--snr', type=float, default=10.0,
                        help="SNR [dB] for the filtered features (default 10).")
    parser.add_argument('--rep', type=int, default=1,
                        help="Filtering repetitions (default 1).")
    parser.add_argument('--nop', type=int, default=-1,
                        help="joblib processes (-1 == all cores).")
    parser.add_argument('--out-dir', type=str, default='./data_random_positions',
                        help="Output root directory.")
    parser.add_argument('--master-seed', type=int, default=11,
                        help="Master seed; controls trajectories and all layouts.")
    parser.add_argument('--start-set', type=int, default=0,
                        help="First position-set index to generate (inclusive). "
                             "Use with --end-set to shard a run across jobs.")
    parser.add_argument('--end-set', type=int, default=None,
                        help="Last position-set index (exclusive). "
                             "Defaults to --n_position_sets.")
    args = parser.parse_args()

    channel_options = parse_float_list(args.channel_options) or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    end_set = args.end_set if args.end_set is not None else args.n_position_sets
    start_set = max(0, args.start_set)
    end_set = min(end_set, args.n_position_sets)
    assert start_set < end_set, "Empty position-set range."

    # Inject collaboration spec (df / n_traj / ppt) into the imported physics.
    override_base_params(df=args.df, n_traj=args.n_traj, ppt=args.ppt)

    # Report the resulting feature dimension so the user can set model input_dim.
    Lf = len(base.range_m(10000.0, 20000.0, args.df))
    print("=" * 64)
    print("Random-position dataset generation")
    print("=" * 64)
    print(f"  thetas             : {channel_options}")
    print(f"  position sets      : [{start_set}, {end_set}) of {args.n_position_sets}")
    print(f"  n_traj / ppt       : {args.n_traj} / {args.ppt}")
    print(f"  df                 : {args.df} Hz  ->  Lf={Lf} time-points")
    print(f"  feature dim / point: Lf*n_sensors = {Lf}*10 = {Lf*10}")
    print(f"  snr / rep          : {args.snr} / {args.rep}")
    print(f"  master seed        : {args.master_seed}")
    print(f"  out dir            : {args.out_dir}")
    print("=" * 64)

    run(
        channel_options=channel_options,
        n_position_sets=args.n_position_sets,
        out_dir=args.out_dir,
        snr=args.snr,
        rep=args.rep,
        nop=args.nop,
        master_seed=args.master_seed,
        start_set=start_set,
        end_set=end_set,
    )


if __name__ == '__main__':
    main()