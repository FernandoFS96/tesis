# Author: Juan Parras and Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 04 11 2025


# Packages to import
import json
import os
import re
import pickle
import warnings
from pathlib import Path

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


def _series_to_dict(series):
    return {key: value.item() if hasattr(value, "item") else value for key, value in series.items()}


def save_prepared_data(data, output_dir, exp_params, feature_columns, synth_lengths, synth_metadata):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_synth_datasets = data["normalized_synth_datasets"]
    normalized_real_dataset = data["normalized_real_dataset"]

    manifest = {
        "base_folder": exp_params["base_folder"],
        "real_data_version": exp_params["real_data_version"],
        "synth_data_version": exp_params["synth_data_version"],
        "targets": list(exp_params["targets"]),
        "feature_columns": list(feature_columns),
        "num_features": len(feature_columns),
        "num_synthetic_datasets": len(normalized_synth_datasets),
        "synth_lengths": list(synth_lengths),
        "synth_metadata": synth_metadata,
    }

    real_x_path = output_path / "real_X_normalized.csv"
    real_y_path = output_path / "real_y_normalized.csv"
    normalized_real_dataset[0].to_csv(real_x_path, index=True)
    normalized_real_dataset[1].to_csv(real_y_path, index=True)

    synth_entries = []
    for index, (x_frame, y_frame) in enumerate(normalized_synth_datasets, start=1):
        x_path = output_path / f"synthetic_{index:03d}_X_normalized.csv"
        y_path = output_path / f"synthetic_{index:03d}_y_normalized.csv"
        x_frame.to_csv(x_path, index=True)
        y_frame.to_csv(y_path, index=True)
        synth_entries.append({"index": index,
                              "x_path": x_path.name,
                              "y_path": y_path.name,
                              "rows": len(x_frame)})

    manifest["real_paths"] = {"X": real_x_path.name, "y": real_y_path.name}
    manifest["synthetic_paths"] = synth_entries
    manifest["denorm_values"] = {
        "X_mean": _series_to_dict(data["denorm_values"]["X_mean"]),
        "X_std": _series_to_dict(data["denorm_values"]["X_std"]),
        "y_mean": _series_to_dict(data["denorm_values"]["y_mean"]),
        "y_std": _series_to_dict(data["denorm_values"]["y_std"]),
    }

    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    pickle_path = output_path / "prepared_data.pkl"
    with pickle_path.open("wb") as pickle_file:
        pickle.dump(data, pickle_file)

    return {"output_dir": str(output_path),
            "manifest_path": str(manifest_path),
            "pickle_path": str(pickle_path),
            "real_x_path": str(real_x_path),
            "real_y_path": str(real_y_path),
            "synthetic_entries": synth_entries}


def load_prepared_data(output_dir):
    output_path = Path(output_dir)
    pickle_path = output_path / "prepared_data.pkl"
    with pickle_path.open("rb") as pickle_file:
        return pickle.load(pickle_file)


def build_shared_raw_dataset(data, target_names):
    y = data[target_names].copy()
    feature_columns = ["Potential", "Cycle_progress"] #["Potential"]  # Removed "Cycle" to avoid redundancy with "Cycle_progress"
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
    synth_metadata = []
    for file_name in file_names:
        file_data = pd.read_csv(synth_data_path + os.sep + file_name)
        file_data = add_domain_normalized_cycle_target(file_data)
        metadata = parse_v3_filename_metadata(file_name)
        synth_metadata.append(metadata)
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

    output_dir = exp_params.get("output_folder", os.getcwd())
    data["saved_data_info"] = save_prepared_data(data, output_dir, exp_params, train_feature_columns,
                                                  synth_lengths, synth_metadata)

    return data
