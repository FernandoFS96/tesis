# Author: Juan Parras and Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 04 11 2025


# Packages to import
import os

from csic.csic_real_synth_load.data import load_real_and_synth_data


def main():
    config_params = {"base_folder": os.getcwd(),
                     "real_data_version": "V2",
                     "synth_data_version": "V3",
                     "targets": ["SoC (%)", "Cycle"]}

    # Load data
    data = load_real_and_synth_data(config_params)

    # Print data stats
    print(f"[shared_raw] Number of synthetic datasets: {len(data['normalized_synth_datasets'])}")
    print(f"[shared_raw] Number of real samples: {len(data['normalized_real_dataset'][0])}")

if __name__ == "__main__":
    main()