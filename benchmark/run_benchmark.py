#!/usr/bin/env python3
# ============================================================
# run_benchmark.py
# Standardized benchmark runner for IDS research (Week 3).
#
# Usage:
#   python run_benchmark.py --platform ESP32 --model dt \
#     --protocol balanced --seed 42 --port COM4 --samples 1000
#
# Supports all 28 run combinations:
#   ESP32:  DT/RF/KNN × (balanced 42/43/44 + natural 99) = 12
#   Uno:    DT/RF     × (balanced 42/43/44 + natural 99) =  8
#   Nano:   DT/RF     × (balanced 42/43/44 + natural 99) =  8
#
# Output:
#   results/benchmarks/raw_runs/{Platform}_{Model}_{protocol}_seed{seed}_N{n}.json
#   results/benchmarks/raw_runs/{Platform}_{Model}_{protocol}_seed{seed}_N{n}.csv
#
# Methodology (Field, 2018): each run is an independent
# replication with a fixed seed for reproducibility (Popper).
# ============================================================

import serial
import time
import numpy as np
import pandas as pd
import json
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BAUD_RATE = 115200
SERIAL_TIMEOUT = 10           # seconds
STARTUP_WAIT = 3              # seconds after serial connect
INTER_SAMPLE_DELAY = 0.05     # seconds between samples
PROGRESS_INTERVAL = 50        # print progress every N samples

# Platform-specific serial defaults
DEFAULT_PORTS = {
    'ESP32': 'COM4',
    'ArduinoUno': 'COM3',
    'ArduinoNano': 'COM9',
}

# Platform name normalization
PLATFORM_MAP = {
    'esp32': 'ESP32',
    'uno': 'ArduinoUno',
    'arduinouno': 'ArduinoUno',
    'arduino_uno': 'ArduinoUno',
    'nano': 'ArduinoNano',
    'arduinonano': 'ArduinoNano',
    'arduino_nano': 'ArduinoNano',
}


# ============================================================
# DATASET LOADING
# ============================================================

def load_benchmark_dataset(protocol, seed, samples, data_dir=None):
    """
    Load a pre-generated benchmark dataset.
    
    Returns:
        X_raw (np.array): Raw feature values (N x 10)
        y (np.array): True labels (N,)
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try two naming conventions:
    # Pattern A (NSL-KDD): bench_{protocol}_X_raw_{samples}_seed{seed}.csv
    # Pattern B (Edge-IIoTset): bench_{protocol}_{samples}_seed{seed}_X_raw.csv
    x_file_a = f"bench_{protocol}_X_raw_{samples}_seed{seed}.csv"
    y_file_a = f"bench_{protocol}_y_{samples}_seed{seed}.csv"
    x_file_b = f"bench_{protocol}_{samples}_seed{seed}_X_raw.csv"
    y_file_b = f"bench_{protocol}_{samples}_seed{seed}_y.csv"
    
    x_path_a = os.path.join(data_dir, x_file_a)
    x_path_b = os.path.join(data_dir, x_file_b)
    
    if os.path.exists(x_path_a):
        x_path = x_path_a
        y_path = os.path.join(data_dir, y_file_a)
    elif os.path.exists(x_path_b):
        x_path = x_path_b
        y_path = os.path.join(data_dir, y_file_b)
    else:
        sys.exit(f"ERROR: Dataset not found!\n"
                 f"  Tried: {x_path_a}\n"
                 f"  Tried: {x_path_b}\n"
                 f"Run prepare_benchmark_sets.py first!")
    
    if not os.path.exists(y_path):
        sys.exit(f"ERROR: Labels not found: {y_path}\n"
                 f"Run prepare_benchmark_sets.py first!")
    
    X = pd.read_csv(x_path).values
    y = pd.read_csv(y_path, header=None).values.ravel()
    
    assert len(X) == samples, \
        f"Expected {samples} samples, got {len(X)} in {x_file_a} or {x_file_b}"
    assert len(y) == samples, \
        f"Expected {samples} labels, got {len(y)} in {y_file_b} or {y_file_a}"
    
    return X, y


# ============================================================
# SERIAL COMMUNICATION
# ============================================================

class SerialBenchmark:
    """Handles serial communication with the target device."""
    
    def __init__(self, port, baud=BAUD_RATE, timeout=SERIAL_TIMEOUT):
        self.port = port
        self.baud = baud
        self.ser = None
        
        print(f"[SERIAL] Connecting to {port} at {baud} baud...")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(STARTUP_WAIT)
        
        # Read and discard startup messages
        startup = self._read_all()
        for line in startup:
            print(f"  DEVICE: {line}")
        print(f"[SERIAL] Connected!")
    
    def _read_all(self, timeout_extra=0.1):
        """Read all available lines from serial buffer."""
        lines = []
        time.sleep(timeout_extra)
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    lines.append(line)
            except Exception:
                break
        return lines
    
    def send(self, message):
        """Send a message over serial."""
        self.ser.write(f"{message}\n".encode())
    
    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Connection closed.")


# ============================================================
# PARSING
# ============================================================

def parse_response_esp32(responses):
    """
    Parse ESP32 benchmark response.
    Format: "[N] Pred: P | True: T | check | Nus"
    Returns: (prediction, inference_us) or (None, None)
    """
    for resp in responses:
        if 'Pred:' not in resp:
            continue
        
        pred = None
        inf_time = None
        
        # Parse prediction
        if 'Pred: 1' in resp or 'Pred:1' in resp:
            pred = 1
        elif 'Pred: 0' in resp or 'Pred:0' in resp:
            pred = 0
        
        # Parse inference time
        if 'μs' in resp or 'us' in resp.lower():
            parts = resp.split('|')
            for part in parts:
                if 'μs' in part or 'us' in part.lower():
                    time_str = ''.join(c for c in part if c.isdigit() or c == '.')
                    if time_str:
                        try:
                            inf_time = float(time_str)
                        except ValueError:
                            pass
                    break
        
        if pred is not None:
            return pred, inf_time
    
    return None, None


def parse_response_avr(responses):
    """
    Parse Arduino Uno/Nano response.
    Format: "RESULT | Prediction: ATTACK/NORMAL | Inference: N μs | ..."
    Returns: (prediction, inference_us) or (None, None)
    """
    for resp in responses:
        if 'RESULT' not in resp and 'Prediction' not in resp:
            continue
        
        pred = None
        inf_time = None
        
        # Parse prediction
        if 'ATTACK' in resp:
            pred = 1
        elif 'NORMAL' in resp:
            pred = 0
        elif 'Pred: 1' in resp:
            pred = 1
        elif 'Pred: 0' in resp:
            pred = 0
        
        # Parse inference time
        if 'Inference:' in resp:
            try:
                time_part = resp.split('Inference:')[1]
                time_str = ''.join(c for c in time_part.split('|')[0]
                                   if c.isdigit() or c == '.')
                if time_str:
                    inf_time = float(time_str)
            except (IndexError, ValueError):
                pass
        elif 'μs' in resp:
            parts = resp.split('|')
            for part in parts:
                if 'μs' in part:
                    time_str = ''.join(c for c in part if c.isdigit() or c == '.')
                    if time_str:
                        try:
                            inf_time = float(time_str)
                        except ValueError:
                            pass
                    break
        
        if pred is not None:
            return pred, inf_time
    
    return None, None


# ============================================================
# BENCHMARK ENGINE
# ============================================================

def run_single_benchmark(serial_conn, X_raw, y, model_name,
                         platform, protocol, seed):
    """
    Run a full benchmark: send all samples, collect predictions.
    
    Returns: dict with all metrics and per-sample data.
    """
    is_esp32 = (platform == 'ESP32')
    parse_fn = parse_response_esp32 if is_esp32 else parse_response_avr
    n_samples = len(X_raw)
    
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {platform} / {model_name.upper()} / "
          f"{protocol} / seed{seed}")
    print(f"  Samples: {n_samples}")
    print(f"{'=' * 60}")
    
    # For ESP32: send bench command
    if is_esp32:
        serial_conn.send(f"bench_{model_name}")
        time.sleep(1)
        startup = serial_conn._read_all()
        for line in startup:
            print(f"  DEVICE: {line}")
    
    # Tracking
    predictions = []
    inference_times = []
    per_sample = []       # Per-sample log for JSON
    tp, fp, tn, fn = 0, 0, 0, 0
    correct = 0
    missed = 0            # Samples with no parseable response
    
    t_start = time.time()
    
    for i in range(n_samples):
        features_str = ','.join([f'{v:.6f}' for v in X_raw[i]])
        true_label = int(y[i])
        
        if is_esp32:
            line = f"{features_str},{true_label}"
        else:
            line = features_str
        
        serial_conn.send(line)
        time.sleep(INTER_SAMPLE_DELAY)
        
        # Read response (with retry for slow devices)
        responses = serial_conn._read_all(timeout_extra=0.05)
        
        # For KNN on ESP32, inference is slower — give more time
        if not responses and model_name == 'knn':
            time.sleep(0.1)
            responses = serial_conn._read_all(timeout_extra=0.05)
        
        # Retry once more if no response
        if not responses:
            time.sleep(0.2)
            responses = serial_conn._read_all(timeout_extra=0.1)
        
        pred, inf_time = parse_fn(responses)
        
        if pred is not None:
            predictions.append(pred)
            
            if inf_time is not None:
                inference_times.append(inf_time)
            
            if pred == true_label:
                correct += 1
            if pred == 1 and true_label == 1: tp += 1
            if pred == 1 and true_label == 0: fp += 1
            if pred == 0 and true_label == 0: tn += 1
            if pred == 0 and true_label == 1: fn += 1
            
            per_sample.append({
                'idx': i,
                'pred': pred,
                'true': true_label,
                'correct': pred == true_label,
                'inference_us': inf_time
            })
        else:
            missed += 1
            per_sample.append({
                'idx': i,
                'pred': None,
                'true': true_label,
                'correct': None,
                'inference_us': None
            })
            if missed <= 5:
                print(f"  WARNING: No response for sample {i}")
            elif missed == 6:
                print(f"  WARNING: Suppressing further 'no response' warnings...")
        
        # Progress
        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t_start
            pct = (i + 1) / n_samples * 100
            print(f"  Progress: {i+1}/{n_samples} ({pct:.0f}%) "
                  f"| Elapsed: {elapsed:.0f}s | Missed: {missed}")
    
    # For ESP32: send END
    if is_esp32:
        serial_conn.send("END")
        time.sleep(2)
        summary = serial_conn._read_all()
        for line in summary:
            print(f"  DEVICE SUMMARY: {line}")
    
    t_total = time.time() - t_start
    
    # Compute metrics
    total_parsed = len(predictions)
    accuracy = correct / total_parsed * 100 if total_parsed > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    
    avg_inf = float(np.mean(inference_times)) if inference_times else 0
    std_inf = float(np.std(inference_times)) if inference_times else 0
    min_inf = float(np.min(inference_times)) if inference_times else 0
    max_inf = float(np.max(inference_times)) if inference_times else 0
    
    result = {
        'platform': platform,
        'model': model_name.upper(),
        'protocol': protocol,
        'seed': seed,
        'n_requested': n_samples,
        'n_parsed': total_parsed,
        'n_missed': missed,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'avg_inference_us': round(avg_inf, 2),
        'std_inference_us': round(std_inf, 2),
        'min_inference_us': round(min_inf, 2),
        'max_inference_us': round(max_inf, 2),
        'total_time_s': round(t_total, 2),
        'timestamp': datetime.now().isoformat(),
        'serial_port': serial_conn.port,
        'baud_rate': BAUD_RATE,
        'per_sample': per_sample,
    }
    
    return result


# ============================================================
# QUALITY CONTROL
# ============================================================

def qc_check(result):
    """Post-run quality control checks."""
    issues = []
    
    n = result['n_requested']
    parsed = result['n_parsed']
    
    # QC1: All samples parsed
    if parsed != n:
        issues.append(f"WARN: Only {parsed}/{n} samples parsed "
                      f"({result['n_missed']} missed)")
    
    # QC2: Confusion matrix sums to parsed
    cm_sum = result['tp'] + result['fp'] + result['tn'] + result['fn']
    if cm_sum != parsed:
        issues.append(f"FAIL: TP+FP+TN+FN={cm_sum} != parsed={parsed}")
    
    # QC3: Balanced check (if applicable)
    if result['protocol'] == 'balanced':
        n_true_attack = result['tp'] + result['fn']
        n_true_normal = result['tn'] + result['fp']
        if n_true_attack != n // 2 or n_true_normal != n // 2:
            issues.append(
                f"WARN: Balanced distribution off: "
                f"Normal={n_true_normal}, Attack={n_true_attack} "
                f"(expected {n//2}/{n//2})")
    
    # QC4: No zero-inference times en masse
    per_sample = result.get('per_sample', [])
    zero_inf = sum(1 for s in per_sample
                   if s.get('inference_us') is not None 
                   and s['inference_us'] == 0)
    if zero_inf > parsed * 0.1:
        issues.append(f"WARN: {zero_inf} samples with inference_us=0 "
                      f"(potential parsing error)")
    
    # Report
    print(f"\n--- QC CHECK ---")
    if not issues:
        print(f"  ALL CHECKS PASSED")
    else:
        for iss in issues:
            print(f"  {iss}")
    
    return len(issues) == 0, issues


# ============================================================
# OUTPUT
# ============================================================

def save_result(result, output_dir):
    """Save benchmark result to JSON and summary CSV."""
    platform = result['platform']
    model = result['model']
    protocol = result['protocol']
    seed = result['seed']
    n = result['n_requested']
    
    base = f"{platform}_{model}_{protocol}_seed{seed}_N{n}"
    json_path = os.path.join(output_dir, f"{base}.json")
    csv_path = os.path.join(output_dir, f"{base}.csv")
    
    # JSON (full data including per-sample)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # CSV (summary only, no per-sample)
    summary = {k: v for k, v in result.items() if k != 'per_sample'}
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    
    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")
    
    return json_path, csv_path


def print_result_summary(result):
    """Print a human-readable summary."""
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK RESULT")
    print(f"{'=' * 60}")
    print(f"  Platform  : {result['platform']}")
    print(f"  Model     : {result['model']}")
    print(f"  Protocol  : {result['protocol']}")
    print(f"  Seed      : {result['seed']}")
    print(f"  Samples   : {result['n_parsed']}/{result['n_requested']}")
    print(f"  Accuracy  : {result['accuracy']:.2f}%")
    print(f"  Precision : {result['precision']:.2f}%")
    print(f"  Recall    : {result['recall']:.2f}%")
    print(f"  F1-Score  : {result['f1_score']:.2f}%")
    print(f"  TP={result['tp']} FP={result['fp']} "
          f"TN={result['tn']} FN={result['fn']}")
    print(f"  Avg Inf.  : {result['avg_inference_us']:.1f} us")
    print(f"  Std Inf.  : {result['std_inference_us']:.1f} us")
    print(f"  Min Inf.  : {result['min_inference_us']:.1f} us")
    print(f"  Max Inf.  : {result['max_inference_us']:.1f} us")
    print(f"  Total Time: {result['total_time_s']:.1f} s")
    print(f"{'=' * 60}")


# ============================================================
# BATCH MODE
# ============================================================

def run_batch_for_platform(port, platform, models, protocols_seeds,
                           samples, output_dir, data_dir):
    """
    Run all combinations for one platform without re-uploading.
    
    For Uno/Nano: requires firmware re-upload between DT and RF.
    For ESP32: all models in one firmware, no re-upload needed.
    """
    conn = SerialBenchmark(port)
    
    try:
        for model in models:
            for protocol, seed in protocols_seeds:
                X, y = load_benchmark_dataset(
                    protocol, seed, samples, data_dir)
                
                result = run_single_benchmark(
                    conn, X, y, model, platform, protocol, seed)
                
                print_result_summary(result)
                qc_ok, issues = qc_check(result)
                save_result(result, output_dir)
                
                # Pause between runs
                print("\n  Pausing 3s before next run...")
                time.sleep(3)
    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='IDS Benchmark Runner (Week 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python run_benchmark.py --platform ESP32 --model dt --protocol balanced --seed 42 --port COM4

  # Batch: all 4 datasets for one model on ESP32
  python run_benchmark.py --platform ESP32 --model dt --protocol all --port COM4

  # Batch: all models × all datasets for ESP32 (12 runs)
  python run_benchmark.py --platform ESP32 --model all --protocol all --port COM4

  # Batch: DT × all datasets for Arduino Uno (4 runs)
  python run_benchmark.py --platform ArduinoUno --model dt --protocol all --port COM3
        """)
    
    parser.add_argument('--platform', required=True,
                        help='Target platform: ESP32, ArduinoUno, ArduinoNano')
    parser.add_argument('--model', required=True,
                        help='Model: dt, rf, knn, or all')
    parser.add_argument('--protocol', default='all',
                        help='Protocol: balanced, natural, or all')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed (42/43/44 for balanced, 99 for natural). '
                             'Omit for all seeds.')
    parser.add_argument('--port', default=None,
                        help='Serial port (e.g., COM3, COM4)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples (default: 1000)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')
    parser.add_argument('--data-dir', default=None,
                        help='Directory containing bench_*.csv files '
                             '(default: same as this script)')
    
    args = parser.parse_args()
    
    # Normalize platform name
    platform_key = args.platform.lower().replace(' ', '').replace('-', '')
    platform = PLATFORM_MAP.get(platform_key, args.platform)
    
    # Determine port
    port = args.port or DEFAULT_PORTS.get(platform, 'COM3')
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'results', 'benchmarks', 'raw_runs')
        output_dir = os.path.abspath(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Data directory
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build model list
    if args.model.lower() == 'all':
        if platform == 'ESP32':
            models = ['dt', 'rf', 'knn']
        else:
            models = ['dt', 'rf']
    else:
        models = [args.model.lower()]
    
    # Validate KNN only on ESP32
    if 'knn' in models and platform != 'ESP32':
        sys.exit("ERROR: KNN is only supported on ESP32!")
    
    # Build protocol/seed combinations
    if args.seed is not None:
        # Single seed specified
        protocol = args.protocol if args.protocol != 'all' else 'balanced'
        protocols_seeds = [(protocol, args.seed)]
    elif args.protocol == 'all':
        protocols_seeds = [
            ('balanced', 42),
            ('balanced', 43),
            ('balanced', 44),
            ('natural', 99),
        ]
    elif args.protocol == 'balanced':
        protocols_seeds = [
            ('balanced', 42),
            ('balanced', 43),
            ('balanced', 44),
        ]
    elif args.protocol == 'natural':
        protocols_seeds = [('natural', 99)]
    else:
        sys.exit(f"ERROR: Unknown protocol '{args.protocol}'")
    
    # Print plan
    total_runs = len(models) * len(protocols_seeds)
    print(f"\n{'=' * 60}")
    print(f"  IDS BENCHMARK PLAN")
    print(f"{'=' * 60}")
    print(f"  Platform : {platform}")
    print(f"  Port     : {port}")
    print(f"  Models   : {[m.upper() for m in models]}")
    print(f"  Datasets : {protocols_seeds}")
    print(f"  Samples  : {args.samples}")
    print(f"  Total    : {total_runs} runs")
    print(f"  Output   : {output_dir}")
    print(f"{'=' * 60}")
    
    if platform != 'ESP32':
        # For Uno/Nano: warn about firmware re-upload
        if len(models) > 1:
            print(f"\n  NOTE: For {platform}, you need to re-upload firmware")
            print(f"  between DT and RF models. This script will run all")
            print(f"  datasets for the FIRST model ({models[0].upper()}),")
            print(f"  then PAUSE for firmware re-upload before continuing.")
            
            for model in models:
                print(f"\n{'*' * 60}")
                print(f"  NOW RUNNING: {model.upper()} on {platform}")
                print(f"  Make sure firmware is uploaded with "
                      f"USE_{'DECISION_TREE' if model == 'dt' else 'RANDOM_FOREST'}")
                print(f"{'*' * 60}")
                input(f"  Press ENTER when {model.upper()} firmware is ready on {platform}...")
                
                run_batch_for_platform(
                    port, platform, [model], protocols_seeds,
                    args.samples, output_dir, data_dir)
        else:
            run_batch_for_platform(
                port, platform, models, protocols_seeds,
                args.samples, output_dir, data_dir)
    else:
        # ESP32: all models in one firmware
        run_batch_for_platform(
            port, platform, models, protocols_seeds,
            args.samples, output_dir, data_dir)
    
    # Final summary
    print(f"\n\n{'=' * 60}")
    print(f"  ALL RUNS COMPLETE ({total_runs} runs)")
    print(f"  Results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    # List output files
    for f in sorted(Path(output_dir).glob("*.json")):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
