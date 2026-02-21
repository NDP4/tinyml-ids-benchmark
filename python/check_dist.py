"""STEP 3.1 — Cek distribusi label + Attack_type dari data mentah."""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("STEP 3.1: DISTRIBUSI LABEL TRAIN/TEST (SPLIT SAAT INI)")
print("=" * 70)

ytr = pd.read_csv(os.path.join(ROOT, "data/edgeiiotset/processed/y_train_pool.csv"), header=None)[0]
yte = pd.read_csv(os.path.join(ROOT, "data/edgeiiotset/processed/y_test_pool.csv"), header=None)[0]

print(f"\nTrain shape: {len(ytr)}")
print("Train distribution:")
vc_tr = ytr.value_counts(normalize=True).sort_index()
for label, pct in vc_tr.items():
    cnt = ytr.value_counts()[label]
    print(f"  Label {label}: {cnt:>6d} ({pct*100:.2f}%)")

print(f"\nTest shape: {len(yte)}")
print("Test distribution:")
vc_te = yte.value_counts(normalize=True).sort_index()
for label, pct in vc_te.items():
    cnt = yte.value_counts()[label]
    print(f"  Label {label}: {cnt:>6d} ({pct*100:.2f}%)")

majority_pct = vc_te.max()
print(f"\nMajority class test: {majority_pct*100:.4f}%")
print(f"Permuted accuracy audit: 0.845976 (84.5976%)")
match = abs(majority_pct - 0.845976) < 0.01
print(f"Match? {match}")
if match:
    print("  >> KONFIRMASI: Permuted acc = majority baseline (BUKAN leakage)")
else:
    print("  >> Permuted acc TIDAK match majority baseline")

# --- Cek data mentah: Attack_type distribution ---
print("\n" + "=" * 70)
print("STEP 3.1b: DISTRIBUSI ATTACK_TYPE (DATA MENTAH)")
print("=" * 70)

raw_path = os.path.join(ROOT, "data/edgeiiotset/raw/ML-EdgeIIoT-dataset.csv")
if os.path.exists(raw_path):
    # Hanya baca kolom yang diperlukan
    df_raw = pd.read_csv(raw_path, usecols=["Attack_label", "Attack_type"], low_memory=False)
    print(f"\nTotal rows: {len(df_raw)}")
    
    print("\nAttack_type distribution:")
    vc = df_raw["Attack_type"].value_counts()
    for atype, cnt in vc.items():
        pct = cnt / len(df_raw) * 100
        label_val = df_raw[df_raw["Attack_type"] == atype]["Attack_label"].mode().values[0]
        print(f"  {atype:30s}: {cnt:>7d} ({pct:5.2f}%) [label={label_val}]")
    
    print(f"\nTotal attack types: {df_raw['Attack_type'].nunique()}")
    print(f"Normal rows: {(df_raw['Attack_label'] == 0).sum()}")
    print(f"Attack rows: {(df_raw['Attack_label'] == 1).sum()}")
    
    # Cek berapa banyak duplikat di data mentah
    print("\n" + "=" * 70)
    print("STEP 3.2: DUPLIKASI DI DATA MENTAH")
    print("=" * 70)
    
    # Load full raw untuk cek duplikasi
    df_full = pd.read_csv(raw_path, low_memory=False)
    print(f"Total rows: {len(df_full)}")
    n_dup = df_full.duplicated().sum()
    print(f"Duplicate rows: {n_dup} ({n_dup/len(df_full)*100:.2f}%)")
    print(f"Unique rows: {len(df_full) - n_dup}")
    
    # Per Attack_type
    print("\nDuplikasi per Attack_type:")
    for atype in df_full["Attack_type"].value_counts().index[:15]:
        subset = df_full[df_full["Attack_type"] == atype]
        n_dup_sub = subset.duplicated().sum()
        print(f"  {atype:30s}: {n_dup_sub:>6d}/{len(subset):>6d} "
              f"({n_dup_sub/max(len(subset),1)*100:5.1f}% dup)")
else:
    print(f"WARNING: {raw_path} not found")
    print("Mencoba cari file CSV di data/edgeiiotset/...")
    for f in os.listdir(os.path.join(ROOT, "data/edgeiiotset")):
        print(f"  {f}")
