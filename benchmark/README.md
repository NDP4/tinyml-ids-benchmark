# Benchmark Scripts

Scripts for running on-device benchmarks and aggregating results.

## Files

- `run_benchmark.py` — Main benchmark runner (serial communication with device)
- `send_test_data.py` — Send test samples to device for inference
- `aggregate_results.py` — Aggregate results across seeds (42, 43, 44) and compute mean ± std
