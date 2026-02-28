#!/usr/bin/env python3
"""
aggregate_v2.py — Aggregate Edge-IIoTset v2 benchmark raw_runs → summary tables.

Output format IDENTIK dengan NSL-KDD summary agar NB20 kompatibel:
  - table_balanced_mean_std.csv  (mean ± std dari seed 42/43/44)
  - table_natural_seed99.csv     (seed 99, single run)

Usage:
  python aggregate_v2.py            # aggregate both full_v2 and sanitized_v2
  python aggregate_v2.py --variant full_v2
  python aggregate_v2.py --variant sanitized_v2

Author: IDS Research Pipeline
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# VARIANT CONFIG
# ============================================================
# raw_dir  = tempat file CSV benchmark individual
# summary_dir = tempat output summary (NB20 baca dari sini)
VARIANTS = {
    'full_v2': {
        'raw_dir': os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'v2', 'full', 'raw_runs'),
        'summary_dir': os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'summary_full_v2'),
    },
    'sanitized_v2': {
        'raw_dir': os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'v2', 'sanitized', 'raw_runs'),
        'summary_dir': os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'summary_sanitized_v2'),
    },
}

BALANCED_SEEDS = [42, 43, 44]
NATURAL_SEED = 99
PLATFORMS = ['ESP32', 'ArduinoUno', 'ArduinoNano']
MODELS_BY_PLATFORM = {
    'ESP32': ['DT', 'RF', 'KNN'],
    'ArduinoUno': ['DT', 'RF'],
    'ArduinoNano': ['DT', 'RF'],
}

METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'avg_inference_us', 'std_inference_us',
    'min_inference_us', 'max_inference_us',
]

CONFUSION = ['tp', 'fp', 'tn', 'fn']


# ============================================================
# LOAD RAW RUNS
# ============================================================
def load_raw_runs(raw_dir):
    """Load semua CSV benchmark results dari folder raw_runs."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df.iloc[[0]])  # Ambil baris pertama (summary row)
        except Exception as e:
            print(f"  WARNING: gagal baca {os.path.basename(f)}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ============================================================
# COMPLETION CHECK
# ============================================================
def check_completion(df, variant_name):
    """Cek apakah semua 28 run sudah ada."""
    expected = []
    for platform in PLATFORMS:
        for model in MODELS_BY_PLATFORM[platform]:
            for seed in BALANCED_SEEDS:
                expected.append((platform, model, 'balanced', seed))
            expected.append((platform, model, 'natural', NATURAL_SEED))

    present = set()
    for _, row in df.iterrows():
        present.add((row['platform'], row['model'].upper(),
                     row['protocol'], int(row['seed'])))

    missing = [e for e in expected if e not in present]

    print(f"\n  [{variant_name}] Completion: {len(present)}/{len(expected)} runs")
    if missing:
        print(f"  MISSING {len(missing)} runs:")
        for m in missing:
            print(f"    - {m[0]} / {m[1]} / {m[2]} seed{m[3]}")
    else:
        print(f"  ✅ Semua {len(expected)} run lengkap!")

    return missing


# ============================================================
# AGGREGATE BALANCED → mean ± std (FORMAT IDENTIK NSL-KDD)
# ============================================================
def aggregate_balanced(df):
    """
    Aggregate balanced runs (seed 42/43/44) → mean ± std per platform×model.
    Output format IDENTIK dengan results/benchmarks/summary/table_balanced_mean_std.csv.
    """
    bal = df[df['protocol'] == 'balanced'].copy()
    if len(bal) == 0:
        return pd.DataFrame()

    # Normalize model names
    bal['model'] = bal['model'].str.upper()

    grouped = bal.groupby(['platform', 'model'], sort=False)

    rows = []
    for (platform, model), group in grouped:
        row = {
            'platform': platform,
            'model': model,
            'n_seeds': len(group),
            'seeds': str(sorted(group['seed'].astype(int).tolist())),
        }

        # Mean ± std untuk semua metrik
        for m in METRICS:
            if m in group.columns:
                vals = group[m].dropna().values.astype(float)
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    row[f'{m}_mean'] = round(mean_val, 4)
                    row[f'{m}_std'] = round(std_val, 4)
                    # Paper format: "mean ± std"
                    row[f'{m}_paper'] = f"{mean_val:.2f} ± {std_val:.2f}"
                else:
                    row[f'{m}_mean'] = None
                    row[f'{m}_std'] = None
                    row[f'{m}_paper'] = 'N/A'

        # Confusion matrix means
        for c in CONFUSION:
            if c in group.columns:
                vals = group[c].dropna().values.astype(float)
                row[f'{c}_mean'] = round(np.mean(vals), 1) if len(vals) > 0 else None

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort: Platform order → Model order
    pl_order = {'ArduinoNano': 0, 'ArduinoUno': 1, 'ESP32': 2}
    md_order = {'DT': 0, 'RF': 1, 'KNN': 2}
    result['_p'] = result['platform'].map(pl_order).fillna(9)
    result['_m'] = result['model'].map(md_order).fillna(9)
    result = result.sort_values(['_p', '_m']).drop(columns=['_p', '_m']).reset_index(drop=True)

    return result


# ============================================================
# AGGREGATE NATURAL → single run (FORMAT IDENTIK NSL-KDD)
# ============================================================
def aggregate_natural(df):
    """Extract natural runs (seed 99) as-is."""
    nat = df[df['protocol'] == 'natural'].copy()
    if len(nat) == 0:
        return pd.DataFrame()

    nat['model'] = nat['model'].str.upper()

    cols_keep = ['platform', 'model', 'seed'] + METRICS + CONFUSION + ['n_parsed', 'n_missed']
    cols_available = [c for c in cols_keep if c in nat.columns]
    result = nat[cols_available].copy().reset_index(drop=True)

    # Sort
    pl_order = {'ArduinoNano': 0, 'ArduinoUno': 1, 'ESP32': 2}
    md_order = {'DT': 0, 'RF': 1, 'KNN': 2}
    result['_p'] = result['platform'].map(pl_order).fillna(9)
    result['_m'] = result['model'].map(md_order).fillna(9)
    result = result.sort_values(['_p', '_m']).drop(columns=['_p', '_m']).reset_index(drop=True)

    return result


# ============================================================
# PRINT & SAVE
# ============================================================
def process_variant(variant_name):
    """Process one variant: load → check → aggregate → save."""
    cfg = VARIANTS[variant_name]
    raw_dir = cfg['raw_dir']
    summary_dir = cfg['summary_dir']

    print(f"\n{'='*70}")
    print(f"  AGGREGATE: {variant_name.upper()}")
    print(f"{'='*70}")
    print(f"  Raw dir    : {raw_dir}")
    print(f"  Summary dir: {summary_dir}")

    # 1. Load
    df = load_raw_runs(raw_dir)
    if len(df) == 0:
        print(f"\n  ❌ Tidak ada data di {raw_dir}")
        return False

    print(f"  Loaded: {len(df)} runs")

    # 2. Check completion
    missing = check_completion(df, variant_name)

    # 3. Aggregate
    bal = aggregate_balanced(df)
    nat = aggregate_natural(df)

    # 4. Print summary
    if len(bal) > 0:
        print(f"\n  BALANCED (mean ± std):")
        print(f"  {'Platform':<15} {'Model':<6} {'Acc':>12} {'F1':>12} {'Latency(μs)':>18}")
        print(f"  {'-'*63}")
        for _, row in bal.iterrows():
            print(f"  {row['platform']:<15} {row['model']:<6} "
                  f"{row.get('accuracy_paper','N/A'):>12} "
                  f"{row.get('f1_score_paper','N/A'):>12} "
                  f"{row.get('avg_inference_us_paper','N/A'):>18}")

    if len(nat) > 0:
        print(f"\n  NATURAL (seed 99):")
        print(f"  {'Platform':<15} {'Model':<6} {'Acc':>8} {'F1':>8} {'Latency(μs)':>12}")
        print(f"  {'-'*49}")
        for _, row in nat.iterrows():
            print(f"  {row['platform']:<15} {row['model']:<6} "
                  f"{row.get('accuracy',0):>8.1f} "
                  f"{row.get('f1_score',0):>8.1f} "
                  f"{row.get('avg_inference_us',0):>12.2f}")

    # 5. Save
    os.makedirs(summary_dir, exist_ok=True)

    if len(bal) > 0:
        bal_path = os.path.join(summary_dir, 'table_balanced_mean_std.csv')
        bal.to_csv(bal_path, index=False)
        print(f"\n  Saved: {bal_path}")

    if len(nat) > 0:
        nat_path = os.path.join(summary_dir, 'table_natural_seed99.csv')
        nat.to_csv(nat_path, index=False)
        print(f"  Saved: {nat_path}")

    # Save all runs flat (untuk referensi)
    all_path = os.path.join(summary_dir, 'all_runs_flat.csv')
    df.to_csv(all_path, index=False)
    print(f"  Saved: {all_path}")

    # Completion report
    report_path = os.path.join(summary_dir, 'completion_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Variant: {variant_name}\n")
        f.write(f"Total runs: {len(df)}/28\n")
        f.write(f"Missing: {len(missing)}\n")
        if missing:
            for m in missing:
                f.write(f"  - {m[0]} / {m[1]} / {m[2]} seed{m[3]}\n")
        else:
            f.write("All 28 runs complete!\n")
    print(f"  Saved: {report_path}")

    print(f"\n  {'✅' if not missing else '⚠️'} "
          f"{variant_name}: {len(df)}/28 runs aggregated")

    return True


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Aggregate Edge-IIoTset v2 benchmark results → NB20-compatible summaries')
    parser.add_argument('--variant',
                        choices=['full_v2', 'sanitized_v2', 'all'],
                        default='all',
                        help='Which variant to aggregate (default: all)')
    args = parser.parse_args()

    if args.variant == 'all':
        targets = list(VARIANTS.keys())
    else:
        targets = [args.variant]

    for variant in targets:
        process_variant(variant)

    print(f"\n{'='*70}")
    print(f"  AGGREGATE SELESAI")
    print(f"  Summary dirs siap untuk NB20 (20_compare_datasets.ipynb)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
