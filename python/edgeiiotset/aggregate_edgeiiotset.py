#!/usr/bin/env python3
# ============================================================
# aggregate_edgeiiotset.py
# Aggregates all 28 Edge-IIoTset benchmark runs into
# publication-ready tables and generates trade-off visualizations.
#
# Outputs (in results/edgeiiotset/summary/):
#   table_balanced_mean_std.csv   — Mean ± std across seeds 42/43/44
#   table_natural_seed99.csv      — Single-seed robustness check
#   figure_f1_vs_latency.png      — F1 vs inference latency scatter
#   figure_accuracy_grouped_bar.png — Platform × model accuracy
#   figure_resource_tradeoff.png  — Multi-axis trade-off chart
#   completion_report.txt         — Human-readable run status report
#   all_runs_flat.csv             — All runs in flat CSV
#
# Identical methodology as NSL-KDD aggregate_results.py:
#   - Balanced results: mean ± std across 3 seeds (Field, 2018)
#   - Natural results: single observation for robustness (Eco, 1977)
# ============================================================

import json
import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Skipping visualizations.")


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

RAW_RUNS_DIR = os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'benchmark_runs')
SUMMARY_DIR = os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'summary')

# Expected runs (28 total)
PLATFORMS = ['ESP32', 'ArduinoUno', 'ArduinoNano']
MODELS_BY_PLATFORM = {
    'ESP32': ['DT', 'RF', 'KNN'],
    'ArduinoUno': ['DT', 'RF'],
    'ArduinoNano': ['DT', 'RF'],
}
BALANCED_SEEDS = [42, 43, 44]
NATURAL_SEED = 99
N = 1000
DATASET_NAME = 'Edge-IIoTset'

METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'avg_inference_us', 'std_inference_us',
    'min_inference_us', 'max_inference_us',
]


# ============================================================
# LOADING
# ============================================================

def load_all_runs(raw_dir=None):
    """Load all JSON result files from raw_runs directory."""
    if raw_dir is None:
        raw_dir = RAW_RUNS_DIR
    
    raw_dir = os.path.abspath(raw_dir)
    
    if not os.path.exists(raw_dir):
        print(f"WARNING: Raw runs directory not found: {raw_dir}")
        print(f"  Membuat directory kosong...")
        os.makedirs(raw_dir, exist_ok=True)
        return pd.DataFrame()
    
    json_files = sorted(glob.glob(os.path.join(raw_dir, '*.json')))
    
    if not json_files:
        print(f"WARNING: No JSON files found in {raw_dir}")
        print(f"  Jalankan benchmark dulu dengan run_benchmark.py")
        return pd.DataFrame()
    
    runs = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        data.pop('per_sample', None)
        data['source_file'] = os.path.basename(jf)
        data['dataset'] = DATASET_NAME
        runs.append(data)
    
    print(f"Loaded {len(runs)} raw runs from {raw_dir}")
    return pd.DataFrame(runs)


# ============================================================
# COMPLETION CHECK
# ============================================================

def check_completion(df):
    """Check which of the 28 expected runs are present/missing."""
    expected = []
    for platform in PLATFORMS:
        for model in MODELS_BY_PLATFORM[platform]:
            for seed in BALANCED_SEEDS:
                expected.append((platform, model, 'balanced', seed))
            expected.append((platform, model, 'natural', NATURAL_SEED))
    
    found = set()
    if len(df) > 0:
        for _, row in df.iterrows():
            key = (row['platform'], row['model'], row['protocol'],
                   int(row['seed']))
            found.add(key)
    
    missing = [e for e in expected if e not in found]
    
    print(f"\n--- COMPLETION STATUS ({DATASET_NAME}) ---")
    print(f"  Expected : {len(expected)} runs")
    print(f"  Found    : {len(found)} runs")
    print(f"  Missing  : {len(missing)} runs")
    
    if missing:
        print(f"\n  Missing runs:")
        for m in missing:
            print(f"    {m[0]:15s} {m[1]:4s} {m[2]:10s} seed{m[3]}")
    
    return expected, found, missing


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_balanced(df):
    """Compute mean ± std across balanced seeds (42, 43, 44)."""
    balanced = df[df['protocol'] == 'balanced'].copy()
    
    if len(balanced) == 0:
        print("  WARNING: No balanced runs found!")
        return pd.DataFrame()
    
    groups = balanced.groupby(['platform', 'model'])
    
    rows = []
    for (platform, model), group in groups:
        row = {
            'dataset': DATASET_NAME,
            'platform': platform,
            'model': model,
            'n_seeds': len(group),
            'seeds': sorted(group['seed'].unique().tolist()),
        }
        
        for metric in METRICS:
            if metric in group.columns:
                vals = group[metric].values
                row[f'{metric}_mean'] = round(float(np.mean(vals)), 4)
                row[f'{metric}_std'] = round(float(np.std(vals, ddof=1)), 4) \
                    if len(vals) > 1 else 0.0
                row[f'{metric}_paper'] = (
                    f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
                    if len(vals) > 1
                    else f"{np.mean(vals):.2f}")
        
        for cm in ['tp', 'fp', 'tn', 'fn']:
            if cm in group.columns:
                row[f'{cm}_mean'] = round(float(group[cm].mean()), 1)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def aggregate_natural(df):
    """Extract natural (seed99) results as-is."""
    natural = df[df['protocol'] == 'natural'].copy()
    
    if len(natural) == 0:
        print("  WARNING: No natural runs found!")
        return pd.DataFrame()
    
    cols = ['platform', 'model', 'seed'] + \
           [m for m in METRICS if m in natural.columns]
    for cm in ['tp', 'fp', 'tn', 'fn', 'n_parsed', 'n_missed']:
        if cm in natural.columns:
            cols.append(cm)
    
    result = natural[cols].copy()
    result.insert(0, 'dataset', DATASET_NAME)
    result = result.reset_index(drop=True)
    return result


# ============================================================
# PAPER-READY TABLES
# ============================================================

def format_paper_table_balanced(agg_df):
    if len(agg_df) == 0:
        return ""
    
    lines = []
    lines.append("=" * 95)
    lines.append(f"TABLE: {DATASET_NAME} Balanced Benchmark Results "
                 f"(Mean ± Std, N=1000, Seeds: 42/43/44)")
    lines.append("=" * 95)
    lines.append(f"{'Platform':15s} {'Model':5s} "
                 f"{'Accuracy':16s} {'Precision':16s} "
                 f"{'Recall':16s} {'F1':16s} "
                 f"{'Latency (us)':16s}")
    lines.append("-" * 95)
    
    for _, row in agg_df.iterrows():
        lines.append(
            f"{row['platform']:15s} {row['model']:5s} "
            f"{row.get('accuracy_paper', 'N/A'):16s} "
            f"{row.get('precision_paper', 'N/A'):16s} "
            f"{row.get('recall_paper', 'N/A'):16s} "
            f"{row.get('f1_score_paper', 'N/A'):16s} "
            f"{row.get('avg_inference_us_paper', 'N/A'):16s}")
    
    lines.append("=" * 95)
    return "\n".join(lines)


def format_paper_table_natural(nat_df):
    if len(nat_df) == 0:
        return ""
    
    lines = []
    lines.append("=" * 95)
    lines.append(f"TABLE: {DATASET_NAME} Natural Distribution Results "
                 f"(N=1000, Seed=99)")
    lines.append("=" * 95)
    lines.append(f"{'Platform':15s} {'Model':5s} "
                 f"{'Accuracy':10s} {'Precision':10s} "
                 f"{'Recall':10s} {'F1':10s} "
                 f"{'Latency':10s} {'TP':5s} {'FP':5s} "
                 f"{'TN':5s} {'FN':5s}")
    lines.append("-" * 95)
    
    for _, row in nat_df.iterrows():
        lines.append(
            f"{row['platform']:15s} {row['model']:5s} "
            f"{row.get('accuracy', 0):9.2f}% "
            f"{row.get('precision', 0):9.2f}% "
            f"{row.get('recall', 0):9.2f}% "
            f"{row.get('f1_score', 0):9.2f}% "
            f"{row.get('avg_inference_us', 0):8.1f}us "
            f"{int(row.get('tp', 0)):5d} "
            f"{int(row.get('fp', 0)):5d} "
            f"{int(row.get('tn', 0)):5d} "
            f"{int(row.get('fn', 0)):5d}")
    
    lines.append("=" * 95)
    return "\n".join(lines)


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_f1_vs_latency(agg_balanced, nat_df, output_dir):
    if not HAS_MPL or len(agg_balanced) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    platform_style = {
        'ESP32':       {'color': '#2196F3', 'marker': 'o'},
        'ArduinoUno':  {'color': '#4CAF50', 'marker': 's'},
        'ArduinoNano': {'color': '#FF9800', 'marker': '^'},
    }
    
    for _, row in agg_balanced.iterrows():
        p = row['platform']
        style = platform_style.get(p, {'color': 'gray', 'marker': 'x'})
        
        f1_mean = row.get('f1_score_mean', 0)
        lat_mean = row.get('avg_inference_us_mean', 0)
        f1_std = row.get('f1_score_std', 0)
        lat_std = row.get('avg_inference_us_std', 0)
        
        ax.errorbar(lat_mean, f1_mean,
                     xerr=lat_std, yerr=f1_std,
                     fmt=style['marker'], color=style['color'],
                     markersize=10, capsize=4, capthick=1.5,
                     label=f"{p} / {row['model']}")
    
    ax.set_xlabel('Average Inference Latency (μs)', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title(f'F1-Score vs Inference Latency — {DATASET_NAME}\n'
                 f'(Balanced, N=1000)', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    lat_vals = agg_balanced['avg_inference_us_mean'].dropna().values
    if len(lat_vals) > 1 and max(lat_vals) / max(min(lat_vals), 1) > 50:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    fig.tight_layout()
    path = os.path.join(output_dir, 'figure_f1_vs_latency.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_grouped_bar(agg_balanced, output_dir):
    if not HAS_MPL or len(agg_balanced) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    platforms = agg_balanced['platform'].unique()
    models = agg_balanced['model'].unique()
    
    x = np.arange(len(platforms))
    width = 0.25
    offsets = np.linspace(-width, width, len(models))
    
    colors = {'DT': '#2196F3', 'RF': '#4CAF50', 'KNN': '#FF9800'}
    
    for i, model in enumerate(models):
        subset = agg_balanced[agg_balanced['model'] == model]
        means = []
        stds = []
        for p in platforms:
            row = subset[subset['platform'] == p]
            if len(row) > 0:
                means.append(row.iloc[0].get('accuracy_mean', 0))
                stds.append(row.iloc[0].get('accuracy_std', 0))
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + offsets[i], means, width,
               yerr=stds, capsize=4,
               label=model, color=colors.get(model, 'gray'),
               alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Platform', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Classification Accuracy — {DATASET_NAME}\n'
                 f'(Balanced, N=1000, Mean ± Std)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.legend(title='Model')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 5))
    
    fig.tight_layout()
    path = os.path.join(output_dir, 'figure_accuracy_grouped_bar.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_resource_tradeoff(agg_balanced, output_dir):
    if not HAS_MPL or len(agg_balanced) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = [
        ('f1_score_mean', 'F1-Score (%)',
         f'F1-Score — {DATASET_NAME}'),
        ('avg_inference_us_mean', 'Avg Latency (μs)',
         f'Inference Latency — {DATASET_NAME}'),
        ('std_inference_us_mean', 'Std Latency (μs)',
         f'Latency Variability — {DATASET_NAME}'),
    ]
    
    colors = {'DT': '#2196F3', 'RF': '#4CAF50', 'KNN': '#FF9800'}
    
    for ax, (metric, ylabel, title) in zip(axes, metrics_to_plot):
        platforms = agg_balanced['platform'].unique()
        models = agg_balanced['model'].unique()
        
        x = np.arange(len(platforms))
        width = 0.25
        offsets = np.linspace(-width, width, len(models))
        
        for i, model in enumerate(models):
            subset = agg_balanced[agg_balanced['model'] == model]
            vals = []
            for p in platforms:
                row = subset[subset['platform'] == p]
                if len(row) > 0 and metric in row.columns:
                    vals.append(float(row.iloc[0][metric]))
                else:
                    vals.append(0)
            
            ax.bar(x + offsets[i], vals, width,
                   label=model, color=colors.get(model, 'gray'),
                   alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Platform', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms, fontsize=9)
        ax.legend(title='Model', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        
        if 'latency' in ylabel.lower() or 'inference' in ylabel.lower():
            vals_all = agg_balanced[metric].dropna().values
            if len(vals_all) > 1 and max(vals_all) / max(min(vals_all), 1) > 50:
                ax.set_yscale('log')
    
    fig.tight_layout()
    path = os.path.join(output_dir, 'figure_resource_tradeoff.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Aggregate Edge-IIoTset benchmark runs')
    parser.add_argument('--variant', 
                        choices=['full', 'sanitized', 'full_v2', 'sanitized_v2'],
                        default='full',
                        help='Feature variant: full, sanitized, full_v2, sanitized_v2')
    args = parser.parse_args()

    variant = args.variant

    # Adjust directories based on variant
    variant_dir_map = {
        'full': ('benchmark_runs', 'summary'),
        'sanitized': ('benchmark_runs_sanitized', 'summary_sanitized'),
        'full_v2': ('benchmark_runs_full_v2', 'summary_full_v2'),
        'sanitized_v2': ('benchmark_runs_sanitized_v2', 'summary_sanitized_v2'),
    }
    raw_subdir, sum_subdir = variant_dir_map[variant]
    raw_dir = os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', raw_subdir)
    summary_dir = os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', sum_subdir)

    os.makedirs(summary_dir, exist_ok=True)

    print("=" * 60)
    print(f"  AGGREGATE BENCHMARK — {DATASET_NAME} ({variant.upper()})")
    print("=" * 60)
    
    # Load all runs
    df = load_all_runs(raw_dir)
    
    if len(df) == 0:
        print("\nTidak ada data benchmark Edge-IIoTset yang ditemukan.")
        print("Langkah yang harus dilakukan:")
        print("  1. Jalankan prepare_benchmark_sets_edgeiiotset.py")
        print("  2. Upload sketch ke perangkat")
        print("  3. Jalankan run_benchmark.py --data-dir edgeiiotset "
              "--output-dir ../results/edgeiiotset/benchmark_runs")
        return
    
    # Completion check
    expected, found, missing = check_completion(df)
    
    # Aggregate balanced
    print("\n--- Aggregating Balanced Results ---")
    agg_balanced = aggregate_balanced(df)
    
    if len(agg_balanced) > 0:
        bal_path = os.path.join(summary_dir, 'table_balanced_mean_std.csv')
        agg_balanced.to_csv(bal_path, index=False)
        print(f"  Saved: {bal_path}")
        
        table_str = format_paper_table_balanced(agg_balanced)
        print(f"\n{table_str}")
    
    # Aggregate natural
    print("\n--- Extracting Natural Results ---")
    nat_df = aggregate_natural(df)
    
    if len(nat_df) > 0:
        nat_path = os.path.join(summary_dir, 'table_natural_seed99.csv')
        nat_df.to_csv(nat_path, index=False)
        print(f"  Saved: {nat_path}")
        
        table_str = format_paper_table_natural(nat_df)
        print(f"\n{table_str}")
    
    # Visualizations
    if HAS_MPL:
        print("\n--- Generating Visualizations ---")
        plot_f1_vs_latency(agg_balanced, nat_df, summary_dir)
        plot_accuracy_grouped_bar(agg_balanced, summary_dir)
        plot_resource_tradeoff(agg_balanced, summary_dir)
    
    # Completion report
    report_path = os.path.join(summary_dir, 'completion_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"BENCHMARK COMPLETION REPORT — {DATASET_NAME}\n")
        f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"Expected: {len(expected)} runs\n")
        f.write(f"Found:    {len(found)} runs\n")
        f.write(f"Missing:  {len(missing)} runs\n\n")
        
        if missing:
            f.write("Missing runs:\n")
            for m in missing:
                f.write(f"  {m[0]:15s} {m[1]:4s} {m[2]:10s} seed{m[3]}\n")
        
        f.write("\n\nFound runs:\n")
        for r in sorted(found):
            f.write(f"  {r[0]:15s} {r[1]:4s} {r[2]:10s} seed{r[3]}\n")
    
    print(f"\n  Saved: {report_path}")
    
    # Export flat CSV
    flat_path = os.path.join(summary_dir, 'all_runs_flat.csv')
    df_export = df.drop(columns=['per_sample'], errors='ignore')
    df_export.to_csv(flat_path, index=False)
    print(f"  Saved: {flat_path}")
    
    print(f"\n{'=' * 60}")
    print(f"  AGGREGATION COMPLETE — {DATASET_NAME} [{variant.upper()}]")
    print(f"  Output: {summary_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
