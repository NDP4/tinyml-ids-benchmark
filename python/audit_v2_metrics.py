"""
AUDIT v2 METRICS — STEP A, B, C
Script independen untuk memverifikasi metrik v2 dari nol.

STEP A: Recompute metrik global (DT, RF, KNN) untuk FULL_v2 & SANITIZED_v2
STEP B: Per-attack-type evaluation dengan definisi Binary IDS yang benar
STEP C: Permutation test pada balanced test set (N=1000)
"""

import json, joblib, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_V2 = os.path.join(PROJECT_ROOT, 'data', 'edgeiiotset', 'processed_v2')
MODELS_V2 = os.path.join(PROJECT_ROOT, 'models', 'edgeiiotset', 'v2')
RESULTS_V2 = os.path.join(PROJECT_ROOT, 'results', 'edgeiiotset', 'v2')

print("=" * 80)
print("  AUDIT v2 METRICS — INDEPENDENT VERIFICATION")
print("=" * 80)

# ─── Load shared test data ───
print("\n[1] Loading test data...")
Xte = pd.read_parquet(os.path.join(DATA_V2, 'X_test_pool_v2.parquet'))
yte = pd.read_csv(os.path.join(DATA_V2, 'y_test_pool_v2.csv'), header=None).values.ravel()

types_path = os.path.join(DATA_V2, 'test_attack_types_v2.csv')
types = pd.read_csv(types_path, header=None)[0].astype(str).values

print(f"  X_test shape   : {Xte.shape}")
print(f"  y_test shape   : {yte.shape}")
print(f"  y_test dist    : {dict(zip(*np.unique(yte, return_counts=True)))}")
print(f"  attack types   : {sorted(set(types))}")
print(f"  types shape    : {types.shape}")

assert len(yte) == len(types), f"MISMATCH: y_test {len(yte)} != types {len(types)}"
assert len(yte) == len(Xte), f"MISMATCH: y_test {len(yte)} != X_test {len(Xte)}"

# ─── Verify types <-> labels consistency ───
print("\n[2] Verifying attack_type vs label consistency...")
for t in sorted(set(types)):
    mask = types == t
    labels = yte[mask]
    unique_labels = set(labels)
    expected = {0} if t == 'Normal' else {1}
    status = "OK" if unique_labels == expected else f"MISMATCH! got {unique_labels}"
    print(f"  {t:30s}: n={mask.sum():>6,}  labels={unique_labels}  {status}")


# ═══════════════════════════════════════════════════════════════
# STEP A: Recompute metrik global dari nol
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP A: RECOMPUTE GLOBAL METRICS")
print("=" * 80)

for VAR in ['full_v2', 'sanitized_v2']:
    print(f"\n{'─'*60}")
    print(f"  VARIANT: {VAR}")
    print(f"{'─'*60}")
    
    # Load feature list
    feat_path = os.path.join(MODELS_V2, f'top10_features_{VAR}.json')
    with open(feat_path) as f:
        top10 = json.load(f)
    print(f"  Features: {top10}")
    
    # Load scaler
    scaler = joblib.load(os.path.join(MODELS_V2, f'scaler_{VAR}.pkl'))
    
    # Transform
    Xte_s = scaler.transform(Xte[top10].values)
    
    # Load models
    dt = joblib.load(os.path.join(MODELS_V2, f'decision_tree_{VAR}.pkl'))
    rf = joblib.load(os.path.join(MODELS_V2, f'random_forest_{VAR}.pkl'))
    knn = joblib.load(os.path.join(MODELS_V2, f'knn_esp32_{VAR}.pkl'))
    
    for name, model in [('DT', dt), ('RF', rf), ('KNN', knn)]:
        yp = model.predict(Xte_s)
        
        print(f"\n  ══════ MODEL: {name} ({VAR}) ══════")
        print(f"  y_true dist: {dict(zip(*np.unique(yte, return_counts=True)))}")
        print(f"  y_pred dist: {dict(zip(*np.unique(yp, return_counts=True)))}")
        
        cm = confusion_matrix(yte, yp)
        tn, fp, fn, tp = cm.ravel()
        print(f"  Confusion Matrix:")
        print(f"    TN={tn:>6,}  FP={fp:>5,}")
        print(f"    FN={fn:>5,}  TP={tp:>6,}")
        
        acc = accuracy_score(yte, yp)
        prec = precision_score(yte, yp, zero_division=0)
        rec = recall_score(yte, yp, zero_division=0)
        f1_val = f1_score(yte, yp, zero_division=0)
        
        print(f"  Acc  = {acc:.6f}")
        print(f"  Prec = {prec:.6f}")
        print(f"  Rec  = {rec:.6f}")
        print(f"  F1   = {f1_val:.6f}")
        
        print(f"\n  Classification Report:")
        print(classification_report(yte, yp, digits=6, zero_division=0,
              target_names=['Normal(0)', 'Attack(1)']))

        # Sanity check: apakah ada prediksi 0?
        n_pred_0 = (yp == 0).sum()
        n_pred_1 = (yp == 1).sum()
        if n_pred_0 == 0:
            print(f"  ⚠️ WARNING: Model TIDAK PERNAH memprediksi Normal (0)!")
            print(f"    → Metrik accuracy bisa misleading (majority baseline)")


# ═══════════════════════════════════════════════════════════════
# STEP B: Per-Attack-Type Binary IDS Evaluation (CORRECT definition)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP B: PER-ATTACK-TYPE BINARY IDS EVAL (CORRECT DEFINITION)")
print("  Definisi: Untuk tiap type T, ambil subset (Normal + T),")
print("  lalu hitung metrik Binary IDS (Attack_label = target)")
print("=" * 80)

for VAR in ['full_v2', 'sanitized_v2']:
    print(f"\n{'─'*60}")
    print(f"  VARIANT: {VAR}")
    print(f"{'─'*60}")
    
    feat_path = os.path.join(MODELS_V2, f'top10_features_{VAR}.json')
    with open(feat_path) as f:
        top10 = json.load(f)
    scaler = joblib.load(os.path.join(MODELS_V2, f'scaler_{VAR}.pkl'))
    Xte_s = scaler.transform(Xte[top10].values)
    
    dt = joblib.load(os.path.join(MODELS_V2, f'decision_tree_{VAR}.pkl'))
    rf = joblib.load(os.path.join(MODELS_V2, f'random_forest_{VAR}.pkl'))
    knn = joblib.load(os.path.join(MODELS_V2, f'knn_esp32_{VAR}.pkl'))
    
    yp_dt = dt.predict(Xte_s)
    yp_rf = rf.predict(Xte_s)
    yp_knn = knn.predict(Xte_s)
    
    # ─── Metode LAMA (per-type isolation) — untuk referensi ───
    print(f"\n  [LAMA] Per-type isolation (single-class subset):")
    for t in sorted(set(types)):
        mask = types == t
        n = mask.sum()
        y_sub = yte[mask]
        f1_dt = f1_score(y_sub, yp_dt[mask], zero_division=0)
        f1_rf = f1_score(y_sub, yp_rf[mask], zero_division=0)
        f1_knn = f1_score(y_sub, yp_knn[mask], zero_division=0)
        print(f"    {t:30s} n={n:>6,}  true={y_sub[0]}  "
              f"DT_f1={f1_dt:.4f}  RF_f1={f1_rf:.4f}  KNN_f1={f1_knn:.4f}")
    
    # ─── Metode BARU (Normal + T, binary IDS) — CORRECT ───
    print(f"\n  [BARU] Binary IDS: subset (Normal + T), target=Attack_label")
    
    # Load split audit for seen/unseen info
    audit_path = os.path.join(RESULTS_V2, 'split_audit_v2.json')
    with open(audit_path) as f:
        audit = json.load(f)
    seen_set = set(audit['seen_attack_types'])
    unseen_set = set(audit['unseen_attack_types'])
    
    normal_mask = types == 'Normal'
    
    rows = []
    for t in sorted(set(types)):
        if t == 'Normal':
            continue  # Skip — Normal digunakan sebagai baseline di tiap subset
        
        # Subset: Normal + tipe T
        mask = (types == t) | normal_mask
        y_sub = yte[mask]
        n_total = mask.sum()
        n_normal = (y_sub == 0).sum()
        n_attack = (y_sub == 1).sum()
        pos_rate = n_attack / n_total
        
        status = 'SEEN' if t in seen_set else 'UNSEEN'
        
        row = {
            'attack_type': t,
            'n_total': n_total,
            'n_normal': n_normal,
            'n_attack': n_attack,
            'pos_rate': f'{pos_rate:.3f}',
            'status': status,
        }
        
        for name, yp in [('DT', yp_dt), ('RF', yp_rf), ('KNN', yp_knn)]:
            yp_sub = yp[mask]
            acc = accuracy_score(y_sub, yp_sub)
            prec = precision_score(y_sub, yp_sub, zero_division=0)
            rec = recall_score(y_sub, yp_sub, zero_division=0)
            f1_val = f1_score(y_sub, yp_sub, zero_division=0)
            row[f'{name}_acc'] = f'{acc:.4f}'
            row[f'{name}_prec'] = f'{prec:.4f}'
            row[f'{name}_rec'] = f'{rec:.4f}'
            row[f'{name}_f1'] = f'{f1_val:.4f}'
        
        rows.append(row)
        
    df_pertype = pd.DataFrame(rows)
    print(df_pertype.to_string(index=False))
    
    # Save
    out_path = os.path.join(RESULTS_V2, f'per_attack_type_binary_ids_{VAR}.csv')
    df_pertype.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    
    # Summary: SEEN vs UNSEEN
    print(f"\n  ─── Summary SEEN vs UNSEEN ({VAR}) ───")
    for status_label in ['SEEN', 'UNSEEN']:
        sub = df_pertype[df_pertype['status'] == status_label]
        if len(sub) > 0:
            for metric_col in ['DT_f1', 'RF_f1', 'KNN_f1']:
                vals = sub[metric_col].astype(float)
                print(f"    {status_label} {metric_col}: "
                      f"mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")


# ═══════════════════════════════════════════════════════════════
# STEP C: Permutation test pada BALANCED test set (N=1000)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STEP C: PERMUTATION TEST PADA BALANCED SET (N=1000)")
print("  Baseline random pada balanced = ~50%")
print("=" * 80)

# Cari benchmark balanced set
for VAR in ['full_v2', 'sanitized_v2']:
    bench_dir = os.path.join(PROJECT_ROOT, 'arduino_code', 'edgeiiotset', 
                             'v2', 'full' if 'full' in VAR else 'sanitized')
    
    bench_x_path = os.path.join(bench_dir, 'bench_balanced_X_raw_1000_seed42.csv')
    bench_y_path = os.path.join(bench_dir, 'bench_balanced_y_1000_seed42.csv')
    
    if not os.path.exists(bench_x_path):
        print(f"\n  [{VAR}] Benchmark files not found: {bench_x_path}")
        print(f"  → Jalankan NB10d Cell 8 dulu untuk generate benchmark sets.")
        continue
    
    print(f"\n{'─'*60}")
    print(f"  PERMUTATION TEST: {VAR}")
    print(f"{'─'*60}")
    
    X_bench = pd.read_csv(bench_x_path).values
    y_bench = pd.read_csv(bench_y_path, header=None).values.ravel()
    
    print(f"  X_bench shape: {X_bench.shape}")
    print(f"  y_bench dist : {dict(zip(*np.unique(y_bench, return_counts=True)))}")
    
    # Load feature list & scaler
    with open(os.path.join(MODELS_V2, f'top10_features_{VAR}.json')) as f:
        top10 = json.load(f)
    scaler = joblib.load(os.path.join(MODELS_V2, f'scaler_{VAR}.pkl'))
    
    # Scale benchmark data
    X_bench_s = scaler.transform(X_bench)
    
    # True predictions
    dt = joblib.load(os.path.join(MODELS_V2, f'decision_tree_{VAR}.pkl'))
    rf = joblib.load(os.path.join(MODELS_V2, f'random_forest_{VAR}.pkl'))
    knn = joblib.load(os.path.join(MODELS_V2, f'knn_esp32_{VAR}.pkl'))
    
    for name, model in [('DT', dt), ('RF', rf), ('KNN', knn)]:
        yp_true = model.predict(X_bench_s)
        acc_true = accuracy_score(y_bench, yp_true)
        f1_true = f1_score(y_bench, yp_true, zero_division=0)
        
        # Permutation: shuffle y_bench N_PERM times, train new model, predict
        # → Tapi karena model sudah trained, kita cukup evaluasi model vs shuffled labels
        # Ini mengukur: "kalau label di-random, berapa skor model?"
        N_PERM = 200
        perm_accs = []
        perm_f1s = []
        rng = np.random.RandomState(42)
        
        for _ in range(N_PERM):
            y_shuffled = rng.permutation(y_bench)
            perm_accs.append(accuracy_score(y_shuffled, yp_true))
            perm_f1s.append(f1_score(y_shuffled, yp_true, zero_division=0))
        
        perm_acc_mean = np.mean(perm_accs)
        perm_f1_mean = np.mean(perm_f1s)
        
        # p-value: berapa fraksi permuted yang >= true score
        p_acc = (np.array(perm_accs) >= acc_true).mean()
        p_f1 = (np.array(perm_f1s) >= f1_true).mean()
        
        print(f"\n  {name} ({VAR}):")
        print(f"    TRUE  Acc={acc_true:.4f}  F1={f1_true:.4f}")
        print(f"    PERM  Acc={perm_acc_mean:.4f} ±{np.std(perm_accs):.4f}  "
              f"F1={perm_f1_mean:.4f} ±{np.std(perm_f1s):.4f}")
        print(f"    p-value  Acc: {p_acc:.4f}  F1: {p_f1:.4f}")
        
        # Balanced → baseline ~50%
        baseline_expected = 0.5
        perm_near_baseline = abs(perm_acc_mean - baseline_expected) < 0.1
        
        if perm_near_baseline and acc_true > perm_acc_mean + 0.1:
            print(f"    ✓ LOLOS: Permuted ≈ {perm_acc_mean:.1%} (baseline ~50%), "
                  f"True = {acc_true:.1%} → Model benar-benar belajar")
        elif not perm_near_baseline:
            print(f"    ⚠️ WARNING: Permuted baseline = {perm_acc_mean:.1%}, "
                  f"expected ≈50% untuk balanced set")
        else:
            print(f"    ⚠️ Model tidak jauh dari baseline permuted")


print("\n" + "=" * 80)
print("  AUDIT SELESAI — Review output di atas")
print("=" * 80)
