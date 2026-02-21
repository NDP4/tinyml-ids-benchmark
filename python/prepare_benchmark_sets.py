#!/usr/bin/env python3
# ============================================================
# prepare_benchmark_sets.py
# Generates standardized benchmark datasets for IDS research.
#
# Outputs (saved to arduino_code/):
#   Balanced (seed 42, 43, 44):
#     bench_balanced_X_raw_1000_seed{s}.csv
#     bench_balanced_y_1000_seed{s}.csv
#   Natural (seed 99):
#     bench_natural_X_raw_1000_seed99.csv
#     bench_natural_y_1000_seed99.csv
#
# Methodology note (Creswell & Field):
#   - Balanced: stratified sampling → exactly 500 normal + 500 attack
#   - Natural: random sampling preserving original class distribution
#   - Three balanced seeds → enables mean ± std reporting
#   - All data inverse-transformed to RAW values so device
#     normalizes exactly once (avoids double-normalization bug)
#
# Reproducibility (Popper):
#   - Fixed seeds, deterministic sampling, saved as CSV
#   - Run this script once; never regenerate during experiment
# ============================================================

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

N = 1000                          # Total samples per set
BALANCED_SEEDS = [42, 43, 44]    # For mean ± std across seeds
NATURAL_SEED = 99                 # Single seed for natural dist.

# Feature metadata (top 10 by MI score)
FEATURE_NAMES = [
    'src_bytes',              # [0] continuous
    'service',                # [1] categorical (LabelEncoded 0-69)
    'dst_bytes',              # [2] continuous
    'flag',                   # [3] categorical (LabelEncoded 0-10)
    'same_srv_rate',          # [4] rate 0-1
    'diff_srv_rate',          # [5] rate 0-1
    'dst_host_srv_count',     # [6] integer 0-255
    'dst_host_same_srv_rate', # [7] rate 0-1
    'logged_in',              # [8] binary 0/1
    'dst_host_serror_rate',   # [9] rate 0-1
]

# Indices of categorical / integer features that need rounding
# after inverse MinMaxScaler transform
CATEGORICAL_INDICES = [1, 3, 8]   # service, flag, logged_in
INTEGER_INDICES = [6]              # dst_host_srv_count

# MinMaxScaler params (must match scaler_params.h exactly)
FEATURE_MIN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
FEATURE_MAX = np.array([
    1379963888.0,  # src_bytes
    69.0,          # service
    1309937401.0,  # dst_bytes
    10.0,          # flag
    1.0,           # same_srv_rate
    1.0,           # diff_srv_rate
    255.0,         # dst_host_srv_count
    1.0,           # dst_host_same_srv_rate
    1.0,           # logged_in
    1.0,           # dst_host_serror_rate
], dtype=np.float64)
FEATURE_RANGE = FEATURE_MAX - FEATURE_MIN


# ============================================================
# FUNCTIONS
# ============================================================

def load_data():
    """Load preprocessed top-10 test data."""
    data_path = os.path.join(os.path.dirname(__file__),
                             '..', 'data', 'preprocessed_top_10.npz')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: {data_path} not found!")
    
    data = np.load(data_path)
    X_scaled = data['X_test']
    y = data['y_test']
    
    print(f"Loaded {len(X_scaled)} test samples from preprocessed_top_10.npz")
    print(f"  Class distribution: Normal={np.sum(y == 0)}, "
          f"Attack={np.sum(y == 1)}")
    return X_scaled, y


def inverse_transform(X_scaled):
    """Inverse MinMaxScaler: X_raw = X_scaled * range + min."""
    X_raw = X_scaled * FEATURE_RANGE + FEATURE_MIN
    
    # Round categorical / integer features
    for idx in CATEGORICAL_INDICES + INTEGER_INDICES:
        X_raw[:, idx] = np.round(X_raw[:, idx]).astype(int)
    
    # Clip to valid ranges (safety)
    for i in range(X_raw.shape[1]):
        X_raw[:, i] = np.clip(X_raw[:, i], FEATURE_MIN[i], FEATURE_MAX[i])
    
    return X_raw


def sample_balanced(X, y, n, seed):
    """
    Stratified sampling: exactly n/2 normal + n/2 attack.
    Raises error if not enough samples of either class.
    """
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
    rng.shuffle(chosen)  # Shuffle so normal/attack are interleaved
    
    return X[chosen], y[chosen]


def sample_natural(X, y, n, seed):
    """
    Random sampling preserving natural class distribution.
    Records the resulting distribution for documentation.
    """
    rng = np.random.RandomState(seed)
    
    if len(X) < n:
        sys.exit(f"ERROR: Not enough samples ({len(X)}) for N={n}")
    
    chosen = rng.choice(len(X), size=n, replace=False)
    return X[chosen], y[chosen]


def save_dataset(X_raw, y, protocol, seed, output_dir):
    """Save X_raw and y to CSV files with standardized naming."""
    x_file = f"bench_{protocol}_X_raw_{len(y)}_seed{seed}.csv"
    y_file = f"bench_{protocol}_y_{len(y)}_seed{seed}.csv"
    
    x_path = os.path.join(output_dir, x_file)
    y_path = os.path.join(output_dir, y_file)
    
    # Save X with feature names as header
    pd.DataFrame(X_raw, columns=FEATURE_NAMES).to_csv(x_path, index=False)
    
    # Save y with header
    pd.DataFrame(y, columns=['label']).to_csv(y_path, index=False)
    
    return x_path, y_path


def validate_dataset(X_raw, y, protocol, seed):
    """Run QC checks on generated dataset."""
    errors = []
    
    # Check dimensions
    if len(X_raw) != N:
        errors.append(f"Expected {N} samples, got {len(X_raw)}")
    if len(y) != N:
        errors.append(f"Expected {N} labels, got {len(y)}")
    
    # Check for NaN
    if np.any(np.isnan(X_raw)):
        errors.append("NaN found in features!")
    if np.any(np.isnan(y)):
        errors.append("NaN found in labels!")
    
    # Check balanced constraint
    n_normal = np.sum(y == 0)
    n_attack = np.sum(y == 1)
    
    if protocol == 'balanced':
        if n_normal != N // 2:
            errors.append(f"Balanced: expected {N//2} normal, got {n_normal}")
        if n_attack != N // 2:
            errors.append(f"Balanced: expected {N//2} attack, got {n_attack}")
    
    # Check feature ranges
    for i in range(X_raw.shape[1]):
        if np.any(X_raw[:, i] < FEATURE_MIN[i]):
            errors.append(f"Feature {FEATURE_NAMES[i]} below min!")
        if np.any(X_raw[:, i] > FEATURE_MAX[i]):
            errors.append(f"Feature {FEATURE_NAMES[i]} above max!")
    
    # Check categorical features are integers
    for idx in CATEGORICAL_INDICES + INTEGER_INDICES:
        vals = X_raw[:, idx]
        if not np.allclose(vals, np.round(vals)):
            errors.append(f"Feature {FEATURE_NAMES[idx]} has non-integer values!")
    
    # Report
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
    print("  PREPARE BENCHMARK DATASETS")
    print("  N = {}, Balanced seeds = {}, Natural seed = {}".format(
        N, BALANCED_SEEDS, NATURAL_SEED))
    print("=" * 60)
    
    # Load data
    X_scaled, y = load_data()
    
    # Inverse transform to raw values
    print("\nInverse-transforming to raw values...")
    X_raw_all = inverse_transform(X_scaled)
    print(f"  Sample 0 raw: {X_raw_all[0][:4]}...")
    print(f"  Sample 0 scaled: {X_scaled[0][:4]}...")
    
    # Output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n--- Generating Balanced Datasets ---")
    all_ok = True
    
    for seed in BALANCED_SEEDS:
        X_s, y_s = sample_balanced(X_raw_all, y, N, seed)
        x_path, y_path = save_dataset(X_s, y_s, 'balanced', seed, output_dir)
        ok = validate_dataset(X_s, y_s, 'balanced', seed)
        all_ok = all_ok and ok
        print(f"  -> {os.path.basename(x_path)}")
        print(f"  -> {os.path.basename(y_path)}")
    
    print("\n--- Generating Natural Dataset ---")
    X_n, y_n = sample_natural(X_raw_all, y, N, NATURAL_SEED)
    x_path, y_path = save_dataset(X_n, y_n, 'natural', NATURAL_SEED, output_dir)
    ok = validate_dataset(X_n, y_n, 'natural', NATURAL_SEED)
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


if __name__ == '__main__':
    main()
