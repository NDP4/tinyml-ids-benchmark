# TinyML-IDS-Benchmark

**Deployment-Oriented Benchmarking of Decision Tree, Random Forest, and KNN for Intrusion Detection on Resource-Constrained IoT Platforms**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository provides the reproducible code, firmware, trained model headers, and benchmark scripts accompanying the paper:

> N. D. Priyambodo, "Deployment-Oriented Benchmarking of Decision Tree, Random Forest, and KNN for Intrusion Detection on Resource-Constrained IoT Platforms," _IEEE Access_, 2025.

We benchmark three classical machine learning models—Decision Tree (DT), Random Forest (RF), and _k_-Nearest Neighbors (KNN)—for binary intrusion detection deployed on **ESP32**, **Arduino Uno**, and **Arduino Nano** via [MicroMLGen](https://github.com/eloquentarduino/micromlgen).

### Key Findings

| Model  | F₁ (NSL-KDD)  | Latency (ESP32) | Latency (AVR) |  Flash  | Feasible on AVR? |
| ------ | :-----------: | :-------------: | :-----------: | :-----: | :--------------: |
| **DT** | 0.833 ± 0.005 |     2.65 μs     |   ~20.7 μs    |  ~2 KB  |        ✅        |
| RF     | 0.767 ± 0.011 |     8.94 μs     |    ~63 μs     |  ~8 KB  |        ✅        |
| KNN    | 0.828 ± 0.024 |    14,879 μs    |       —       | ~200 KB |        ❌        |

**Predictive rule:** AVR latency ≈ ESP32 latency × 8 for float-heavy classical ML.

## Repository Structure

```
tinyml-ids-benchmark/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── python/                        # Python training & preprocessing
│   ├── 01_preprocessing_eda.py    # Data loading, EDA, feature selection
│   ├── 02_model_training.py       # Model training with hyperparameter search
│   ├── 03_model_export.py         # Export to C headers via MicroMLGen
│   ├── 04_paper_visualizations.py # Generate paper figures
│   ├── prepare_benchmark_sets.py  # Create N=1000 benchmark CSVs
│   └── edgeiiotset/               # Edge-IIoTset v2 specific scripts
│       ├── preprocess.py
│       ├── train.py
│       └── export.py
│
├── firmware/                      # Arduino/ESP32 sketches
│   ├── esp32/
│   │   └── ids_esp32.ino
│   ├── arduino_uno/
│   │   └── ids_arduino_uno.ino
│   └── arduino_nano/
│       └── ids_arduino_nano.ino
│
├── models/                        # Trained model C headers
│   ├── nslkdd/
│   │   ├── dt_model.h
│   │   ├── rf_model.h
│   │   ├── knn_data.h
│   │   └── scaler_params.h
│   └── edgeiiotset/
│       ├── dt_model.h
│       ├── rf_model.h
│       ├── knn_data.h
│       └── scaler_params.h
│
├── benchmark/                     # Benchmark execution scripts
│   ├── run_benchmark.py           # Serial communication benchmark runner
│   ├── send_test_data.py          # Send test samples to device
│   └── aggregate_results.py       # Aggregate results across seeds
│
└── data/                          # Placeholder (datasets NOT included)
    └── README.md                  # Instructions to obtain datasets
```

## Datasets

Datasets are **NOT included** in this repository due to size and licensing.

| Dataset             | Source                                                                                                                      | Size   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------ |
| **NSL-KDD**         | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)                                            | ~25 MB |
| **Edge-IIoTset v2** | [Kaggle / Ferrag et al.](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) | ~3 GB  |

After downloading, place files in `data/`:

```
data/
├── KDDTrain+.csv
├── KDDTest+.csv
└── edgeiiotset/
    ├── DNN-EdgeIIoT-dataset.csv
    └── ...
```

## Quick Start

### 1. Setup Python Environment

```bash
git clone https://github.com/NDP4/tinyml-ids-benchmark.git
cd tinyml-ids-benchmark
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Preprocess & Train (NSL-KDD)

```bash
python python/01_preprocessing_eda.py
python python/02_model_training.py
python python/03_model_export.py
```

This generates model headers in `models/nslkdd/`.

### 3. Prepare Benchmark Data

```bash
python python/prepare_benchmark_sets.py
```

Creates balanced (seeds 42, 43, 44) and natural (seed 99) test sets of N=1,000 samples.

### 4. Flash Firmware

1. Open the appropriate `.ino` sketch in Arduino IDE
2. Copy model headers (`dt_model.h`, `rf_model.h`, `scaler_params.h`) into the sketch folder
3. Select the correct board and port
4. Upload

### 5. Run Benchmark

```bash
python benchmark/run_benchmark.py --port COM3 --model dt --dataset nslkdd
python benchmark/aggregate_results.py
```

## Hardware Requirements

| Platform                  | Architecture  |  Clock  |  SRAM  | Flash |   FPU    |
| ------------------------- | :-----------: | :-----: | :----: | :---: | :------: |
| ESP32 (DevKit v1)         | 32-bit Xtensa | 240 MHz | 520 KB | 4 MB  | Hardware |
| Arduino Uno (ATmega328P)  |   8-bit AVR   | 16 MHz  |  2 KB  | 32 KB | Software |
| Arduino Nano (ATmega328P) |   8-bit AVR   | 16 MHz  |  2 KB  | 32 KB | Software |

## Software Dependencies

- Python 3.11+
- scikit-learn 1.3.x
- MicroMLGen 1.3.6
- Arduino IDE 2.x
- ESP32 board package (for ESP32 sketches)

## Known Issues & Workarounds

1. **MicroMLGen leaf-node bug:** When `class_weight` is set in scikit-learn's DecisionTreeClassifier, MicroMLGen generates incorrect leaf-node class assignments. **Workaround:** Use the manual tree export script (`python/03_model_export.py` includes a verified manual export path).

2. **Double-normalization bug:** Benchmark samples must be sent in **raw** (unnormalized) form. The device applies MinMaxScaler normalization once. Sending pre-normalized data causes silent accuracy degradation.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{priyambodo2025tinyml,
  title={Deployment-Oriented Benchmarking of Decision Tree, Random Forest, and KNN for Intrusion Detection on Resource-Constrained IoT Platforms},
  author={Priyambodo, Nur Dwi},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- **Author:** Nur Dwi Priyambodo
- **Email:** nur.priyambodo@students.amikom.ac.id
- **Affiliation:** Department of Informatics Engineering, Universitas Amikom Yogyakarta
