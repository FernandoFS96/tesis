"""
visualize_trajectories.py
=========================

Generate a handful of trajectories from config/data_pipeline.yaml and plot
them, to eyeball the trajectory-generation method before running the (slow)
full data generation.

Uses the same `channel.generate_trajectories` dispatch as the real data
generation, so whatever `method` is set in the YAML (spiral, hermite, ...) is
what gets visualised.

Use:
    python data/visualize_trajectories.py
    python data/visualize_trajectories.py --n_traj 10 --seed 11
    python data/visualize_trajectories.py --traj_config some/other.yaml --out my.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; we save to a file
import matplotlib.pyplot as plt

from acoustic_data_generator import channel, load_traj_config


def generate_trajectories(traj_config, n_traj, ppt):
    """Generate trajectories using the channel's generate_trajectories dispatch.

    We build a bare `channel` instance (bypassing __init__, which would also
    run the expensive channel-matrix computation) and set only the attributes
    that generate_trajectories needs: params, traj_config and
    precomputed_trajectories.
    """
    c = channel.__new__(channel)
    c.precomputed_trajectories = None
    c.params = {'n_traj': n_traj, 'ppt': ppt}
    c.traj_config = traj_config
    return c.generate_trajectories()  # (3, n_traj, ppt + 1)


def plot_trajectories(traj, method, n_context, out_path):
    """Plot each trajectory with start / context / end markers."""
    n_traj = traj.shape[1]
    ctx_idx = min(n_context - 1, traj.shape[2] - 1)  # index of last context point

    fig, ax = plt.subplots(figsize=(8, 8))
    for it in range(n_traj):
        x, y = traj[0, it], traj[1, it]
        line, = ax.plot(x, y, '-', lw=1.2)
        # context portion (first n_context points) drawn thicker, same colour
        ax.plot(x[:n_context], y[:n_context], '-', lw=3.0, color=line.get_color(), alpha=0.6)
        ax.plot(x[0], y[0], 'go', ms=7)             # start
        ax.plot(x[ctx_idx], y[ctx_idx], 'b^', ms=9)  # last context point
        ax.plot(x[-1], y[-1], 'ks', ms=7)            # end

    # legend (proxy handles so it does not repeat per trajectory)
    ax.plot([], [], 'go', ms=7, label='start')
    ax.plot([], [], 'b^', ms=9, label=f'context #{n_context}')
    ax.plot([], [], 'ks', ms=7, label='end')
    ax.legend(loc='upper left', fontsize=11)

    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"'{method}' trajectories (n={n_traj})")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise generated trajectories.")
    parser.add_argument('--traj_config', type=str, default=None,
                        help="Path to the trajectory-generation YAML (default: config/data_pipeline.yaml).")
    parser.add_argument('--n_traj', type=int, default=None,
                        help="Number of trajectories to generate "
                             "(default: generation.n_traj from the config, else 10).")
    parser.add_argument('--ppt', type=int, default=None,
                        help="Points per trajectory, -> ppt + 1 samples "
                             "(default: generation.ppt from the config, else 50).")
    parser.add_argument('--n_context', type=int, default=5, help="Number of context points to highlight.")
    parser.add_argument('--seed', type=int, default=11, help="Random seed.")
    parser.add_argument('--out', type=str, default='data/validation/trajectories_preview.png',
                        help="Output image path.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    traj_config = load_traj_config(args.traj_config)
    # Default n_traj / ppt to the shared `channel:` block so the preview matches
    # what generation will actually produce.
    ch = traj_config.get('channel', {}) or {} #type: ignore
    n_traj = args.n_traj if args.n_traj is not None else int(ch.get('n_traj', 10))
    ppt = args.ppt if args.ppt is not None else int(ch.get('ppt', 50))
    traj = generate_trajectories(traj_config, n_traj=n_traj, ppt=ppt)
    plot_trajectories(traj, method=traj_config['method'], n_context=args.n_context, out_path=args.out) #type: ignore
    print(f"Saved {n_traj} '{traj_config['method']}' trajectories to: {args.out}") #type: ignore
