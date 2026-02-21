#!/usr/bin/env python3
# ============================================================
# prepare_benchmark_sets_edgeiiotset.py
# Generates standardized benchmark datasets for Edge-IIoTset IDS.
#
# Outputs (saved to arduino_code/edgeiiotset/):
#   Balanced (seed 42, 43, 44):
#     bench_balanced_X_raw_1000_seed{s}.csv
#     bench_balanced_y_1000_seed{s}.csv
#   Natural (seed 99):
#     bench_natural_X_raw_1000_seed99.csv
#     bench_natural_y_1000_seed99.csv
#
# Methodology (identik dengan NSL-KDD):
#   - Balanced: stratified sampling → exactly 500 normal + 500 attack
#   - Natural: random sampling preserving original class distribution
#   - Three balanced seeds → enables mean ± std reporting
#   - All data inverse-transformed to RAW values so device
#     normalizes exactly once (avoids double-normalization bug)
#
# Reproducibility:
#   - Fixed seeds, deterministic sampling, saved as CSV
#   - Run this script once; never regenerate during experiment
# ============================================================

import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

N = 1000                          # Total samples per set
BALANCED_SEEDS = [42, 43, 44]    # For mean ± std across seeds
NATURAL_SEED = 99                 # Single seed for natural dist.

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'edgeiiotset')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'edgeiiotset')


# ============================================================
# FUNCTIONS
# ============================================================

def load_data():
    """Load preprocessed top-10 test data for Edge-IIoTset."""
    data_path = os.path.join(DATA_DIR, 'preprocessed_top_10.npz')
    
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: {data_path} not found!\n"
                 f"Run notebook 10_edgeiiotset_preprocess.ipynb first.")
    
    data = np.load(data_path)
    X_scaled = data['X_test']
    y = data['y_test']
    
    print(f"Loaded {len(X_scaled)} test samples from preprocessed_top_10.npz")
    print(f"  Class distribution: Normal={np.sum(y == 0)}, "
          f"Attack={np.sum(y == 1)}")
    return X_scaled, y


def load_scaler_params():
    """Load scaler params from JSON exported by preprocessing notebook."""
    params_path = os.path.join(MODELS_DIR, 'scaler_params_edgeiiotset.json')
    
    if not os.path.exists(params_path):
        sys.exit(f"ERROR: {params_path} not found!\n"
                 f"Run notebook 10_edgeiiotset_preprocess.ipynb first.")
    
    with open(params_path) as f:
        params = json.load(f)
    
    return params


def load_feature_names():
    """Load top-10 feature names."""
    feat_path = os.path.join(MODELS_DIR, 'top10_features.json')
    
    if not os.path.exists(feat_path):
        sys.exit(f"ERROR: {feat_path} not found!\n"
                 f"Run notebook 10_edgeiiotset_preprocess.ipynb first.")
    
    with open(feat_path) as f:
        return json.load(f)


def inverse_transform(X_scaled, params):
    """Inverse MinMaxScaler: X_raw = X_scaled * range + min."""
    feature_min = np.array(params['min'], dtype=np.float64)
    feature_range = np.array(params['range'], dtype=np.float64)
    
    X_raw = X_scaled * feature_range + feature_min
    
    # Clip to valid ranges (safety)
    feature_max = np.array(params['max'], dtype=np.float64)
    for i in range(X_raw.shape[1]):
        X_raw[:, i] = np.clip(X_raw[:, i], feature_min[i], feature_max[i])
    
    return X_raw


def sample_balanced(X, y, n, seed):
    """Stratified sampling: exactly n/2 normal + n/2 attack."""
    rng = np.random.RandomState(seed)
    half = n // 2
    
    idx_normal = np.where(y == 0)[0]
    idx_attack = np.where(y == 1)[0]
    
    if len(idx_normal) < half:
        sys.exit(f"ERROR: Not enough normal samples ({len(idx_normal)}) "
                 f"for balanced set of {n}")
    if len(idx_attack) < half:
        sys.exit(f"ERROR: Not enough attack samples ({len(idx_attack)}) "
                 f"for balanced set of {n}")
    
    chosen_normal = rng.choice(idx_normal, size=half, replace=False)
    chosen_attack = rng.choice(idx_attack, size=half, replace=False)
    
    chosen = np.concatenate([chosen_normal, chosen_attack])
    rng.shuffle(chosen)
    
    return X[chosen], y[chosen]


def sample_natural(X, y, n, seed):
    """Random sampling preserving natural class distribution."""
    rng = np.random.RandomState(seed)
    
    if len(X) < n:
        sys.exit(f"ERROR: Not enough samples ({len(X)}) for N={n}")
    
    chosen = rng.choice(len(X), size=n, replace=False)
    return X[chosen], y[chosen]


def save_dataset(X_raw, y, protocol, seed, feature_names, output_dir):
    """Save X_raw and y to CSV files with standardized naming."""
    x_file = f"bench_{protocol}_X_raw_{len(y)}_seed{seed}.csv"
    y_file = f"bench_{protocol}_y_{len(y)}_seed{seed}.csv"
    
    x_path = os.path.join(output_dir, x_file)
    y_path = os.path.join(output_dir, y_file)
    
    pd.DataFrame(X_raw, columns=feature_names).to_csv(x_path, index=False)
    pd.DataFrame(y, columns=['label']).to_csv(y_path, index=False)
    
    return x_path, y_path


def validate_dataset(X_raw, y, protocol, seed, scaler_params, feature_names):
    """Run QC checks on generated dataset."""
    errors = []
    
    if len(X_raw) != N:
        errors.append(f"Expected {N} samples, got {len(X_raw)}")
    if len(y) != N:
        errors.append(f"Expected {N} labels, got {len(y)}")
    
    if np.any(np.isnan(X_raw)):
        errors.append("NaN found in features!")
    if np.any(np.isnan(y)):
        errors.append("NaN found in labels!")
    
    n_normal = np.sum(y == 0)
    n_attack = np.sum(y == 1)
    
    if protocol == 'balanced':
        if n_normal != N // 2:
            errors.append(f"Balanced: expected {N//2} normal, got {n_normal}")
        if n_attack != N // 2:
            errors.append(f"Balanced: expected {N//2} attack, got {n_attack}")
    
    # Check feature ranges
    feature_min = np.array(scaler_params['min'])
    feature_max = np.array(scaler_params['max'])
    for i in range(X_raw.shape[1]):
        if np.any(X_raw[:, i] < feature_min[i] - 1e-6):
            errors.append(f"Feature {feature_names[i]} below min!")
        if np.any(X_raw[:, i] > feature_max[i] + 1e-6):
            errors.append(f"Feature {feature_names[i]} above max!")
    
    tag = f"{protocol}_seed{seed}"
    if errors:
        print(f"  FAIL {tag}:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"  PASS {tag}: N={len(y)}, "
              f"Normal={n_normal}, Attack={n_attack}")
        return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  PREPARE BENCHMARK DATASETS — Edge-IIoTset")
    print("  N = {}, Balanced seeds = {}, Natural seed = {}".format(
        N, BALANCED_SEEDS, NATURAL_SEED))
    print("=" * 60)
    
    # Load data
    X_scaled, y = load_data()
    scaler_params = load_scaler_params()
    feature_names = load_feature_names()
    
    print(f"  Features: {feature_names}")
    
    # Inverse transform to raw values
    print("\nInverse-transforming to raw values...")
    X_raw_all = inverse_transform(X_scaled, scaler_params)
    print(f"  Sample 0 raw:    {X_raw_all[0][:4]}...")
    print(f"  Sample 0 scaled: {X_scaled[0][:4]}...")
    
    # Output directory (same as this script → arduino_code/edgeiiotset/)
    output_dir = SCRIPT_DIR
    
    print("\n--- Generating Balanced Datasets ---")
    all_ok = True
    
    for seed in BALANCED_SEEDS:
        X_s, y_s = sample_balanced(X_raw_all, y, N, seed)
        x_path, y_path = save_dataset(
            X_s, y_s, 'balanced', seed, feature_names, output_dir)
        ok = validate_dataset(X_s, y_s, 'balanced', seed,
                              scaler_params, feature_names)
        all_ok = all_ok and ok
        print(f"  -> {os.path.basename(x_path)}")
        print(f"  -> {os.path.basename(y_path)}")
    
    print("\n--- Generating Natural Dataset ---")
    X_n, y_n = sample_natural(X_raw_all, y, N, NATURAL_SEED)
    x_path, y_path = save_dataset(
        X_n, y_n, 'natural', NATURAL_SEED, feature_names, output_dir)
    ok = validate_dataset(X_n, y_n, 'natural', NATURAL_SEED,
                          scaler_params, feature_names)
    all_ok = all_ok and ok
    print(f"  -> {os.path.basename(x_path)}")
    print(f"  -> {os.path.basename(y_path)}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL DATASETS VALID")
    else:
        print("  SOME DATASETS FAILED VALIDATION — CHECK ABOVE")
    print("=" * 60)
    
    # Print file listing
    print("\nGenerated files:")
    for f in sorted(Path(output_dir).glob("bench_*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:.1f} KB)")
    
    # Print usage hint
    print(f"\n--- USAGE HINT ---")
    print(f"Untuk menjalankan benchmark Edge-IIoTset:")
    print(f"  cd arduino_code")
    print(f"  python run_benchmark.py --platform ESP32 --model all "
          f"--data-dir edgeiiotset "
          f"--output-dir ../results/edgeiiotset/benchmark_runs")


if __name__ == '__main__':
    main()
