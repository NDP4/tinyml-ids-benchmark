# Python Scripts

This directory contains the Python training and preprocessing pipeline.

## Pipeline Order

1. `01_preprocessing_eda.py` — Load raw datasets, EDA, feature selection (MI top-10), normalization
2. `02_model_training.py` — Train DT/RF/KNN with two-stage hyperparameter selection
3. `03_model_export.py` — Export trained models to C/C++ headers via MicroMLGen
4. `04_paper_visualizations.py` — Generate figures for the paper
5. `prepare_benchmark_sets.py` — Create benchmark CSV files (N=1000, balanced + natural)

## Edge-IIoTset v2 Scripts

Located in `edgeiiotset/` subdirectory with analogous pipeline for the IoT-specific dataset.
