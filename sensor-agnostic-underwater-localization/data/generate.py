"""
generate.py -- unified data-generation entry point.

Two problems, one door. The base config is config/data_pipeline.yaml (shared
`channel:` block + `topology_task:` / `random_task:` blocks + spiral/hermite
params). The RECOMMENDED way to select what to build is a dataset identity from
config/dataset/ (the same file training composes, so the two never disagree):

    python data/generate.py dataset=topology_spiral
    python data/generate.py dataset=random_spiral_shared

A dataset file just sets `task`, `method`, and (random only) `mode` -> which is
translated to random_task.distinct_trajectories. You can still drive it directly
and override anything dotlist-style:

    python data/generate.py task=random method=hermite channel.df=100 \
        random_task.n_position_sets=5 channel.channel_options=[0.1,0.2,0.3]

    python data/generate.py --config path/to/other.yaml dataset=topology_spiral

Run from the repository ROOT so the `./data/...` output roots resolve to
<repo>/data/ (the training data configs expect that location).
"""
import os
import sys

from omegaconf import OmegaConf

# Make the sibling generator modules importable regardless of launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import acoustic_data_generator as topo       # topology task + shared physics/lib
import random_position_generator as rand      # random (sensor-displacement) task

_DATASET_DIR = os.path.join(os.path.dirname(_HERE), "config", "dataset")


def parse_cli(argv):
    """Split argv into an optional --config path, an optional dataset name, and
    OmegaConf dotlist overrides (everything else of the form key=value)."""
    config_path = None
    dataset = None
    overrides = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            config_path = argv[i + 1]; i += 2; continue
        if a.startswith("--config="):
            config_path = a.split("=", 1)[1]; i += 1; continue
        if a.startswith("dataset="):          # identity selector (config/dataset/<name>.yaml)
            dataset = a.split("=", 1)[1]; i += 1; continue
        overrides.append(a); i += 1
    return config_path, dataset, overrides


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, dataset, overrides = parse_cli(argv)

    cfg = topo.load_traj_config(config_path)   # OmegaConf DictConfig (base pipeline)

    # Layer the dataset identity (task/method/mode + any overrides) onto the base,
    # then apply explicit CLI overrides last.
    if dataset is not None:
        ds_path = os.path.join(_DATASET_DIR, f"{dataset}.yaml")
        if not os.path.exists(ds_path):
            raise SystemExit(f"Unknown dataset '{dataset}' ({ds_path} not found).")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ds_path))
    if overrides:
        cfg.merge_with_dotlist(overrides)

    # `mode` (shared|distinct) is the canonical identity; translate it to the
    # generator's random_task.distinct_trajectories flag.
    mode = cfg.get("mode", None)
    if mode in ("shared", "distinct"):
        cfg.random_task.distinct_trajectories = (mode == "distinct")

    task = cfg.get("task", None)
    if task == "topology":
        topo.run_topology_task(cfg)
    elif task == "random":
        rand.run_random_task(cfg)
    else:
        raise SystemExit(
            f"Unknown or missing task={task!r}. Select a dataset "
            "(dataset=topology_spiral | dataset=random_spiral_shared | ...) "
            "or pass task=topology|random.")


if __name__ == "__main__":
    main()
