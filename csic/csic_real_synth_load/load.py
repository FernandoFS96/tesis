# Author: Juan Parras and Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 04 11 2025


# Packages to import
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from csic.csic_real_synth_load.data import load_real_and_synth_data
else:
    from .data import load_real_and_synth_data


def main():
    base_folder = Path(__file__).resolve().parent
    config_params = {"base_folder": str(base_folder),
                     "output_folder": os.path.join(os.getcwd(), "prepared_data"),
                     "real_data_version": "v2",
                     "synth_data_version": "v3",
                     "targets": ["SoC (%)", "Cycle"] # antes: ["SoC (%)", "Cycle"]
                     }

    # Load data
    data = load_real_and_synth_data(config_params)

    # Print data stats
    print(f"[shared_raw] Number of synthetic datasets: {len(data['normalized_synth_datasets'])}")
    print(f"[shared_raw] Number of real samples: {len(data['normalized_real_dataset'][0])}")
    print(f"[shared_raw] Saved prepared data to: {data['saved_data_info']['output_dir']}")

if __name__ == "__main__":
    main()