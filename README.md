# TinyML-IDS Benchmark: Classical and Neural Machine Learning for Intrusion Detection on Microcontrollers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Arduino](https://img.shields.io/badge/Platform-Arduino-blue.svg)](https://www.arduino.cc/)
[![ESP32](https://img.shields.io/badge/Platform-ESP32-green.svg)](https://www.espressif.com/)

Reproducible benchmark suite for deploying **five** machine-learning models—Decision Tree (DT), Random Forest (RF), K-Nearest Neighbors (KNN), Deep Neural Network (DNN), and Convolutional Neural Network (CNN)—as intrusion-detection systems on resource-constrained microcontrollers.

## Model Overview

| Model    | Framework           | Approach  | Flash (approx.) | Platforms        |
| -------- | ------------------- | --------- | --------------- | ---------------- |
| DT       | MicroMLGen          | Classical | ~2.5 KB         | ESP32, Uno, Nano |
| RF       | MicroMLGen          | Classical | ~7.0 KB         | ESP32, Uno, Nano |
| KNN      | MicroMLGen          | Classical | ~62 KB          | ESP32 only       |
| DNN ★NEW | TFLite Micro (INT8) | Neural    | ~4.2 KB         | ESP32, Uno, Nano |
| CNN ★NEW | TFLite Micro (INT8) | Neural    | ~3.8 KB         | ESP32, Uno, Nano |

## Repository Structure

```
tinyml-ids-benchmark/
├── benchmark/
│   ├── run_benchmark.py           # Classical model benchmark runner
│   ├── run_benchmark_dnn.py       # ★NEW Neural model benchmark runner
│   ├── aggregate_results.py       # Classical results aggregation
│   └── aggregate_all_models.py    # ★NEW All 5-model aggregation
├── firmware/
│   ├── esp32/                     # ESP32 classical (DT/RF/KNN)
│   │   └── ids_esp32.ino
│   ├── esp32_dnn/                 # ★NEW ESP32 neural (DNN/CNN)
│   │   └── ids_esp32_dnn.ino
│   ├── arduino_uno/               # Uno classical (DT/RF)
│   │   └── ids_arduino_uno.ino
│   ├── arduino_uno_dnn/           # ★NEW Uno neural (DNN/CNN)
│   │   └── ids_arduino_uno_dnn.ino
│   ├── arduino_nano/              # Nano classical (DT/RF)
│   │   └── ids_arduino_nano.ino
│   └── arduino_nano_dnn/          # ★NEW Nano neural (DNN/CNN)
│       └── ids_arduino_nano_dnn.ino
├── models/
│   └── nslkdd/
│       ├── classical/             # DT/RF/KNN C headers (MicroMLGen)
│       │   ├── dt_model.h
│       │   ├── rf_model.h
│       │   └── knn_data.h
│       └── neural/                # ★NEW DNN/CNN TFLite models
│           ├── dnn_model.h        # INT8 quantized DNN
│           └── cnn_model.h        # INT8 quantized CNN
├── python/
│   ├── 01_preprocessing.py
│   ├── 02_model_training.py
│   ├── 03_model_export.py
│   ├── 04_feature_importance.py
│   └── 05_dnn_training.py         # ★NEW DNN/CNN training & TFLite export
├── data/
│   └── nslkdd/                    # NSL-KDD preprocessed data
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Hardware Requirements

| Platform        | MCU        | Clock   | RAM    | Flash | Classical Models | Neural Models |
| --------------- | ---------- | ------- | ------ | ----- | ---------------- | ------------- |
| ESP32 DevKit V1 | Xtensa LX6 | 240 MHz | 520 KB | 4 MB  | DT, RF, KNN      | DNN, CNN      |
| Arduino Uno R3  | ATmega328P | 16 MHz  | 2 KB   | 32 KB | DT, RF           | DNN, CNN      |
| Arduino Nano    | ATmega328P | 16 MHz  | 2 KB   | 32 KB | DT, RF           | DNN, CNN      |

## Quick Start

### 1. Python Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Classical Models (DT, RF, KNN)

```bash
cd python
python 01_preprocessing.py
python 02_model_training.py
python 03_model_export.py
```

### 3. Train Neural Models (DNN, CNN) ★NEW

```bash
cd python
python 05_dnn_training.py
```

This script will:

- Train DNN (10→32→16→8→1) and CNN (Conv1D) architectures
- Apply INT8 post-training quantization via TFLite
- Export C header files (`dnn_model.h`, `cnn_model.h`) for embedded deployment
- Generate evaluation metrics and comparison plots

### 4. Arduino IDE Setup

#### Classical Models (MicroMLGen)

1. Install [MicroMLGen](https://github.com/eloquentarduino/micromlgen) library
2. Open firmware from `firmware/esp32/`, `firmware/arduino_uno/`, or `firmware/arduino_nano/`
3. Compile and upload

#### Neural Models (TFLite Micro) ★NEW

1. Install **TensorFlow Lite for Microcontrollers** library:
   - **ESP32**: Install via Arduino Library Manager → search "TensorFlowLite_ESP32"
   - **AVR (Uno/Nano)**: Install [EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML) or use the [Harvard TinyMLx](https://github.com/tinyMLx/arduino-library) library
2. Copy model headers (`dnn_model.h`, `cnn_model.h`) from `models/nslkdd/neural/` to the firmware directory
3. Copy `scaler_params.h` to the firmware directory
4. Open firmware from `firmware/esp32_dnn/`, `firmware/arduino_uno_dnn/`, or `firmware/arduino_nano_dnn/`
5. For AVR boards, toggle model with `#define USE_DNN` or `#define USE_CNN` (one at a time due to RAM constraints)
6. Compile and upload

### 5. Run Benchmarks

#### Classical Models

```bash
cd benchmark
python run_benchmark.py --port COM4 --platform esp32 --model dt --samples 1000
python run_benchmark.py --port COM3 --platform arduino_uno --model dt --samples 1000
```

#### Neural Models ★NEW

```bash
cd benchmark
python run_benchmark_dnn.py --port COM4 --platform esp32 --model dnn --samples 1000
python run_benchmark_dnn.py --port COM4 --platform esp32 --model cnn --samples 1000
python run_benchmark_dnn.py --port COM3 --platform arduino_uno --model dnn --samples 1000
```

### 6. Aggregate Results

#### All 5 Models (Comprehensive) ★NEW

```bash
cd benchmark
python aggregate_all_models.py
```

This generates unified comparison tables across DT, RF, KNN, DNN, and CNN for all platforms.

## Benchmark Protocol

- **Dataset**: NSL-KDD (binary classification: Normal vs. Attack)
- **Features**: Top 10 by Mutual Information, MinMaxScaler [0, 1]
- **Sample sizes**: N = 1000
- **Seeds**: Balanced sets (42, 43, 44), Natural distribution (99)
- **Metrics**: Accuracy, Precision, Recall, F1-score, Latency (μs), Memory (bytes)
- **Total runs**: 52 (28 classical + 24 neural)

## Key Results Summary

| Model | Platform | F1-Score | Latency (μs) | Flash (KB) |
| ----- | -------- | -------- | ------------ | ---------- |
| DT    | ESP32    | 0.976    | ~12          | ~2.5       |
| RF    | ESP32    | 0.975    | ~38          | ~7.0       |
| KNN   | ESP32    | 0.928    | ~1800        | ~62        |
| DNN   | ESP32    | 0.856    | ~85          | ~4.2       |
| CNN   | ESP32    | 0.848    | ~120         | ~3.8       |
| DT    | AVR      | 0.976    | ~180         | ~2.5       |
| RF    | AVR      | 0.975    | ~560         | ~7.0       |
| DNN   | AVR      | 0.856    | ~680         | ~4.2       |
| CNN   | AVR      | 0.848    | ~960         | ~3.8       |

## Framework Comparison

| Aspect       | MicroMLGen (Classical) | TFLite Micro (Neural)   |
| ------------ | ---------------------- | ----------------------- |
| Models       | DT, RF, KNN            | DNN, CNN                |
| Export       | Python → C header      | TFLite → INT8 → C array |
| Runtime      | Pure C arithmetic      | TFLite interpreter      |
| Quantization | Native (float/int)     | INT8 post-training      |
| RAM overhead | Minimal                | Tensor arena (3–24 KB)  |
| AVR support  | Direct                 | Requires adaptation     |

## Tensor Arena Configuration

| Platform     | DNN Arena | CNN Arena | Notes                |
| ------------ | --------- | --------- | -------------------- |
| ESP32        | 16 KB     | 24 KB     | Simultaneous DNN+CNN |
| Arduino Uno  | 3 KB      | 3 KB      | One model at a time  |
| Arduino Nano | 3 KB      | 3 KB      | One model at a time  |

## Known Issues

- **KNN**: Only feasible on ESP32 due to memory requirements (~62 KB for reference samples)
- **TFLite on AVR**: Tensor arena must be kept ≤3 KB; only one neural model can be loaded at a time
- **MicroMLGen leaf-node bug**: Resolved with `BYPASS_micromlgen.py` (see main project)
- **INT8 quantization accuracy**: Minor accuracy drop (~0.1–0.5%) compared to float32; within acceptable tolerance for edge deployment

## Citation

```bibtex
@article{ids_tinyml_benchmark_2025,
  title   = {Deployment-Oriented Benchmarking of Classical and Neural Machine Learning
             for Intrusion Detection on Resource-Constrained Microcontrollers},
  author  = {Glow, C.},
  year    = {2025},
  journal = {IEEE Access},
  note    = {Under review}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
