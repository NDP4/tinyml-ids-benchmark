#!/usr/bin/env python3
# ============================================================
# aggregate_all_models.py
# Aggregasi SEMUA benchmark: DT, RF, KNN + DNN, CNN
# Menghasilkan tabel komparasi menyeluruh dan visualisasi.
#
# Total 52 runs yang diharapkan:
#   DT/RF/KNN (28 runs existing) + DNN/CNN (24 runs baru)
#
# Output (results/benchmarks/summary/):
#   table_all_models_balanced.csv
#   table_all_models_natural.csv
#   figure_all_models_f1_vs_latency.png
#   figure_all_models_accuracy_grouped.png
#   figure_all_models_radar.png
#   table_latex_comparison.tex
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
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available.")


# ============================================================
# CONFIGURATION
# ============================================================

RAW_RUNS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results', 'benchmarks', 'raw_runs')

SUMMARY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results', 'benchmarks', 'summary')

PLATFORMS = ['ESP32', 'ArduinoUno', 'ArduinoNano']

# Updated: DNN dan CNN di semua platform
MODELS_BY_PLATFORM = {
    'ESP32': ['DT', 'RF', 'KNN', 'DNN', 'CNN'],
    'ArduinoUno': ['DT', 'RF', 'DNN', 'CNN'],
    'ArduinoNano': ['DT', 'RF', 'DNN', 'CNN'],
}

# Model categories
TRADITIONAL_MODELS = ['DT', 'RF', 'KNN']
NEURAL_MODELS = ['DNN', 'CNN']
ALL_MODELS = TRADITIONAL_MODELS + NEURAL_MODELS

BALANCED_SEEDS = [42, 43, 44]
NATURAL_SEED = 99
N = 1000

METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'avg_inference_us', 'std_inference_us',
    'min_inference_us', 'max_inference_us',
]


# ============================================================
# LOADING
# ============================================================

def load_all_runs(raw_dir=None):
    if raw_dir is None:
        raw_dir = os.path.abspath(RAW_RUNS_DIR)
    
    json_files = sorted(glob.glob(os.path.join(raw_dir, '*.json')))
    
    if not json_files:
        sys.exit(f"ERROR: No JSON files found in {raw_dir}")
    
    runs = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        data.pop('per_sample', None)
        data['source_file'] = os.path.basename(jf)
        
        # Tambahkan kolom kategori model
        model = data.get('model', '').upper()
        data['model_category'] = ('Neural' if model in NEURAL_MODELS 
                                  else 'Traditional')
        data['framework'] = data.get('framework', 
            'TFLite Micro' if model in NEURAL_MODELS else 'MicroMLGen')
        runs.append(data)
    
    df = pd.DataFrame(runs)
    print(f"Loaded {len(df)} runs ({len(df[df['model_category']=='Traditional'])} "
          f"traditional + {len(df[df['model_category']=='Neural'])} neural)")
    return df


# ============================================================
# COMPLETION CHECK
# ============================================================

def check_completion(df):
    expected = []
    for platform in PLATFORMS:
        for model in MODELS_BY_PLATFORM[platform]:
            for seed in BALANCED_SEEDS:
                expected.append((platform, model, 'balanced', seed))
            expected.append((platform, model, 'natural', NATURAL_SEED))
    
    found = set()
    for _, row in df.iterrows():
        key = (row['platform'], row['model'], row['protocol'], int(row['seed']))
        found.add(key)
    
    missing = [e for e in expected if e not in found]
    
    print(f"\n--- COMPLETION STATUS ---")
    print(f"  Expected: {len(expected)} runs")
    print(f"  Found:    {len(found)} runs")
    print(f"  Missing:  {len(missing)} runs")
    
    if missing:
        print(f"\n  Missing runs:")
        for m in missing:
            print(f"    {m[0]:15s} {m[1]:4s} {m[2]:10s} seed{m[3]}")
    
    return expected, found, missing


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_balanced(df):
    balanced = df[df['protocol'] == 'balanced'].copy()
    if len(balanced) == 0:
        return pd.DataFrame()
    
    groups = balanced.groupby(['platform', 'model'])
    rows = []
    
    for (platform, model), group in groups:
        row = {
            'platform': platform,
            'model': model,
            'model_category': group.iloc[0].get('model_category', 'Unknown'),
            'framework': group.iloc[0].get('framework', 'Unknown'),
            'n_seeds': len(group),
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
    natural = df[df['protocol'] == 'natural'].copy()
    if len(natural) == 0:
        return pd.DataFrame()
    
    result = natural.copy()
    result['model_category'] = result['model'].apply(
        lambda m: 'Neural' if m in NEURAL_MODELS else 'Traditional')
    return result


# ============================================================
# LATEX TABLE GENERATION
# ============================================================

def generate_latex_table(df_balanced, output_path):
    """Generate LaTeX table untuk paper."""
    lines = []
    lines.append(r'\begin{table*}[!t]')
    lines.append(r'\caption{Comprehensive Benchmark Results: Traditional ML vs Neural Network Models (Balanced Protocol, Mean $\pm$ Std)}')
    lines.append(r'\label{tab:all_models_comparison}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{llcccccc}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Platform} & \textbf{Model} & \textbf{Category} & '
                 r'\textbf{Accuracy (\%)} & \textbf{Precision (\%)} & '
                 r'\textbf{Recall (\%)} & \textbf{F1 (\%)} & '
                 r'\textbf{Latency ($\mu$s)} \\')
    lines.append(r'\hline')
    
    for platform in PLATFORMS:
        pdf = df_balanced[df_balanced['platform'] == platform]
        if len(pdf) == 0:
            continue
        
        first = True
        for _, row in pdf.iterrows():
            plat_str = platform if first else ''
            first = False
            cat = row.get('model_category', '—')
            lines.append(
                f"  {plat_str} & {row['model']} & {cat} & "
                f"{row.get('accuracy_paper', '—')} & "
                f"{row.get('precision_paper', '—')} & "
                f"{row.get('recall_paper', '—')} & "
                f"{row.get('f1_score_paper', '—')} & "
                f"{row.get('avg_inference_us_paper', '—')} \\\\"
            )
        lines.append(r'\hline')
    
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  LaTeX table: {output_path}")


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_f1_vs_latency(df_balanced, output_path):
    """F1 vs Latency scatter — semua 5 model."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    markers = {'DT': 'o', 'RF': 's', 'KNN': '^', 'DNN': 'D', 'CNN': 'P'}
    colors = {
        'ESP32': '#2196F3',
        'ArduinoUno': '#4CAF50',
        'ArduinoNano': '#FF9800'
    }
    
    for _, row in df_balanced.iterrows():
        model = row['model']
        platform = row['platform']
        f1 = row.get('f1_score_mean', 0)
        lat = row.get('avg_inference_us_mean', 0)
        
        ax.scatter(lat, f1, marker=markers.get(model, 'x'),
                   c=colors.get(platform, 'gray'), s=150, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(f"{model}", (lat, f1),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker=m, color='gray', linestyle='',
                            markersize=10, label=n) 
                     for n, m in markers.items()]
    plat_handles = [Line2D([0], [0], marker='o', color=c, linestyle='',
                           markersize=10, label=n)
                    for n, c in colors.items()]
    
    legend1 = ax.legend(handles=model_handles, loc='lower left', title='Model')
    ax.add_artist(legend1)
    ax.legend(handles=plat_handles, loc='lower right', title='Platform')
    
    ax.set_xlabel('Avg Inference Latency (μs)', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('F1-Score vs Inference Latency — All Models', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scatter plot: {output_path}")


def plot_accuracy_grouped_bar(df_balanced, output_path):
    """Grouped bar chart — accuracy per platform & model."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    models_order = [m for m in ALL_MODELS 
                    if m in df_balanced['model'].unique()]
    x = np.arange(len(PLATFORMS))
    width = 0.15
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    
    for i, model in enumerate(models_order):
        vals = []
        errs = []
        for platform in PLATFORMS:
            row = df_balanced[(df_balanced['platform'] == platform) & 
                              (df_balanced['model'] == model)]
            if len(row) > 0:
                vals.append(row.iloc[0].get('accuracy_mean', 0))
                errs.append(row.iloc[0].get('accuracy_std', 0))
            else:
                vals.append(0)
                errs.append(0)
        
        offset = (i - len(models_order)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      label=model, color=colors[i % len(colors)],
                      capsize=3, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Platform', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy — Traditional ML vs Neural Networks',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(PLATFORMS, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grouped bar: {output_path}")


def plot_latency_comparison(df_balanced, output_path):
    """Bar chart — inference latency per model per platform."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    models_order = [m for m in ALL_MODELS 
                    if m in df_balanced['model'].unique()]
    x = np.arange(len(PLATFORMS))
    width = 0.15
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    
    for i, model in enumerate(models_order):
        vals = []
        errs = []
        for platform in PLATFORMS:
            row = df_balanced[(df_balanced['platform'] == platform) & 
                              (df_balanced['model'] == model)]
            if len(row) > 0:
                vals.append(row.iloc[0].get('avg_inference_us_mean', 0))
                errs.append(row.iloc[0].get('avg_inference_us_std', 0))
            else:
                vals.append(0)
                errs.append(0)
        
        offset = (i - len(models_order)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs,
               label=model, color=colors[i % len(colors)],
               capsize=3, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Platform', fontsize=12)
    ax.set_ylabel('Avg Inference Latency (μs)', fontsize=12)
    ax.set_title('Inference Latency — Traditional ML vs Neural Networks', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(PLATFORMS, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Latency chart: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  AGGREGATE ALL MODELS: DT, RF, KNN + DNN, CNN")
    print("=" * 60)
    
    df = load_all_runs()
    expected, found, missing = check_completion(df)
    
    # Aggregate
    print("\n--- AGGREGATING BALANCED RESULTS ---")
    df_balanced = aggregate_balanced(df)
    balanced_path = os.path.join(SUMMARY_DIR, 'table_all_models_balanced.csv')
    df_balanced.to_csv(balanced_path, index=False)
    print(f"  Saved: {balanced_path}")
    
    print("\n--- AGGREGATING NATURAL RESULTS ---")
    df_natural = aggregate_natural(df)
    natural_path = os.path.join(SUMMARY_DIR, 'table_all_models_natural.csv')
    df_natural.to_csv(natural_path, index=False)
    print(f"  Saved: {natural_path}")
    
    # LaTeX table
    print("\n--- GENERATING LATEX TABLE ---")
    latex_path = os.path.join(SUMMARY_DIR, 'table_latex_all_models.tex')
    generate_latex_table(df_balanced, latex_path)
    
    # Visualizations
    print("\n--- GENERATING VISUALIZATIONS ---")
    plot_f1_vs_latency(
        df_balanced,
        os.path.join(SUMMARY_DIR, 'figure_all_models_f1_vs_latency.png'))
    plot_accuracy_grouped_bar(
        df_balanced,
        os.path.join(SUMMARY_DIR, 'figure_all_models_accuracy_grouped.png'))
    plot_latency_comparison(
        df_balanced,
        os.path.join(SUMMARY_DIR, 'figure_all_models_latency_comparison.png'))
    
    # Print paper-ready table
    print("\n" + "=" * 80)
    print("  PAPER-READY TABLE: Balanced Protocol (mean ± std)")
    print("=" * 80)
    print(f"{'Platform':<15} {'Model':<6} {'Category':<12} "
          f"{'Accuracy':<16} {'F1-Score':<16} {'Latency (μs)':<16}")
    print("-" * 80)
    
    for _, row in df_balanced.sort_values(['platform', 'model']).iterrows():
        print(f"{row['platform']:<15} {row['model']:<6} "
              f"{row.get('model_category','?'):<12} "
              f"{row.get('accuracy_paper','—'):<16} "
              f"{row.get('f1_score_paper','—'):<16} "
              f"{row.get('avg_inference_us_paper','—'):<16}")
    
    print("=" * 80)
    
    # Completion report
    report_path = os.path.join(SUMMARY_DIR, 'completion_report_all.txt')
    with open(report_path, 'w') as f:
        f.write("COMPLETION REPORT — ALL MODELS\n")
        f.write(f"Expected: {len(expected)} | Found: {len(found)} | "
                f"Missing: {len(missing)}\n\n")
        if missing:
            f.write("Missing runs:\n")
            for m in missing:
                f.write(f"  {m[0]} {m[1]} {m[2]} seed{m[3]}\n")
        else:
            f.write("ALL RUNS COMPLETE!\n")
    print(f"\n  Report: {report_path}")
    
    print(f"\n  ALL DONE! Check {SUMMARY_DIR}/")


if __name__ == '__main__':
    main()
