# Author: Juan Parras and Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 04 11 2025


# Packages to import
import os
import re
import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler

V3_FILENAME_PATTERN = re.compile(r"^run_T(?P<temperature>-?\d+(?:p\d+)?)C_"
                                 r"(?P<cd_token>CD[^_]+)_"
                                 r"(?P<cc_token>CC[^_]+)_"
                                 r"(?P<run_id>[A-Za-z0-9]+)"
                                 r"(?:_EIS_vectors)?\.csv$")
SHARED_RAW_EIS_PREFIXES = ("Zre", "Zim", "Zmag", "Phase")


def parse_v3_filename_metadata(file_name):
    base_name = os.path.basename(file_name)
    match = V3_FILENAME_PATTERN.match(base_name)
    if not match:
        raise ValueError(f"Could not parse V3 metadata from file name '{base_name}'. "
                         "Expected format like 'run_T40C_CD2_CC2_e446046776_EIS_vectors.csv'.")

    temperature_token = match.group("temperature").replace("p", ".")
    return {"run_temperature_c": float(temperature_token),
            "run_cd_token": match.group("cd_token"),
            "run_cc_token": match.group("cc_token"),
            "run_id": match.group("run_id"),
            "file_name": base_name}


def add_domain_normalized_cycle_target(data):
    data = data.copy()
    if "Cycle" not in data.columns:
        return data

    cycle_max = data["Cycle"].max()
    if pd.isna(cycle_max) or cycle_max == 0:
        data["Cycle_progress"] = 0.0
    else:
        data["Cycle_progress"] = data["Cycle"] / cycle_max
    return data


def normalize_data(x_train, x_test, y_train, y_test):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(x_scaler.fit_transform(x_train),
                                  columns=x_train.columns,
                                  index=x_train.index)
    x_test_scaled = pd.DataFrame(x_scaler.transform(x_test),
                                 columns=x_train.columns,
                                 index=x_test.index)

    y_train_scaled = pd.DataFrame(y_scaler.fit_transform(y_train),
                                  columns=y_train.columns,
                                  index=y_train.index)
    y_test_scaled = pd.DataFrame(y_scaler.transform(y_test),
                                 columns=y_train.columns,
                                 index=y_test.index)

    norm_values = {"X_train": x_train_scaled,
                   "X_test": x_test_scaled,
                   "y_train": y_train_scaled,
                   "y_test": y_test_scaled}

    denorm_values = {"X_mean": x_train.mean(),
                     "X_std": x_train.std(),
                     "y_mean": y_train.mean(),
                     "y_std": y_train.std()}

    return norm_values, denorm_values


def build_shared_raw_dataset(data, target_names):
    y = data[target_names].copy()
    feature_columns = ["Potential"]
    for prefix in SHARED_RAW_EIS_PREFIXES:
        feature_columns.extend([f"{prefix}_{i}" for i in range(50)])
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required shared_raw columns: {missing_columns}")
    x = data[feature_columns].copy()
    return x, y, feature_columns


def load_real_and_synth_data(exp_params):
    # Load real data
    real_data_path = exp_params["base_folder"] + os.sep + "data" + os.sep + exp_params[
        "real_data_version"] + os.sep + "EIS_Cycle_Feature_Vectors.csv"
    real_data = pd.read_csv(real_data_path)
    real_data = add_domain_normalized_cycle_target(real_data)

    # Load synthetic data
    synth_data_path = exp_params["base_folder"] + os.sep + "data" + os.sep + exp_params["synth_data_version"] + os.sep
    file_names = sorted(
        file_name for file_name in os.listdir(synth_data_path) if file_name.endswith("_EIS_vectors.csv"))
    synth_data_frames = []
    for file_name in file_names:
        file_data = pd.read_csv(synth_data_path + os.sep + file_name)
        file_data = add_domain_normalized_cycle_target(file_data)
        metadata = parse_v3_filename_metadata(file_name)
        synth_data_frames.append((file_data, metadata))

    # First, save number of samples per frame and concatenate them
    synth_lengths = []
    for synth_frame, metadata in synth_data_frames:  # TODO: ignore metadata at the moment!
        synth_lengths.append(len(synth_frame))
    concat_synth_frames = pd.concat([frame for frame, _ in synth_data_frames], axis=0)

    # Second, we need to fix same features for every frame and divide into features and targets
    x_train, y_train, train_feature_columns = build_shared_raw_dataset(concat_synth_frames, exp_params["targets"])
    x_test, y_test, test_feature_columns = build_shared_raw_dataset(real_data, exp_params["targets"])
    print(f"[shared_raw] Final number of features: {len(train_feature_columns)}")
    print(f"[shared_raw] Feature columns: {train_feature_columns}")

    # Then, normalize
    norm_values, denorm_values = normalize_data(x_train, x_test, y_train, y_test)

    # Finally, get back to individual synth frames
    normalized_synth_datasets = []
    start_idx = 0
    for synth_len in synth_lengths:
        end_idx = start_idx + synth_len
        normalized_synth_datasets.append((norm_values["X_train"].iloc[start_idx:end_idx].copy(),
                                          norm_values["y_train"].iloc[start_idx:end_idx].copy()))
        start_idx = end_idx

    normalized_real_dataset = (norm_values["X_test"].copy(), norm_values["y_test"].copy())

    # Save data in a dictionary
    data = {"normalized_synth_datasets": normalized_synth_datasets,
            "normalized_real_dataset": normalized_real_dataset,
            "denorm_values": denorm_values}

    return data
