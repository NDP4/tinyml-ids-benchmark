"""
STEP C (standalone): Permutation test pada balanced subsample dari test pool v2.
Membuat balanced N=1000 secara langsung dari test pool, lalu jalankan permutation test.
"""
import json, joblib, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_V2 = os.path.join(PROJECT_ROOT, 'data', 'edgeiiotset', 'processed_v2')
MODELS_V2 = os.path.join(PROJECT_ROOT, 'models', 'edgeiiotset', 'v2')

print("=" * 80)
print("  STEP C: PERMUTATION TEST (BALANCED SUBSAMPLE, N=1000)")
print("  Baseline random pada balanced 50/50 = ~50%")
print("=" * 80)

# Load test pool
Xte = pd.read_parquet(os.path.join(DATA_V2, 'X_test_pool_v2.parquet'))
yte = pd.read_csv(os.path.join(DATA_V2, 'y_test_pool_v2.csv'), header=None).values.ravel()

print(f"\nTest pool: {len(yte)} samples")
print(f"  Class 0 (Normal): {(yte==0).sum()}")
print(f"  Class 1 (Attack): {(yte==1).sum()}")

# Buat balanced subsample N=1000 (500 normal + 500 attack)
N = 1000
rng = np.random.RandomState(42)
idx_0 = np.where(yte == 0)[0]
idx_1 = np.where(yte == 1)[0]
sel_0 = rng.choice(idx_0, N // 2, replace=False)
sel_1 = rng.choice(idx_1, N // 2, replace=False)
sel = np.concatenate([sel_0, sel_1])
rng.shuffle(sel)

y_bal = yte[sel]
X_bal_full = Xte.iloc[sel]

print(f"\nBalanced subset: {len(y_bal)} samples")
print(f"  Class 0: {(y_bal==0).sum()}, Class 1: {(y_bal==1).sum()}")

for VAR in ['full_v2', 'sanitized_v2']:
    print(f"\n{'='*60}")
    print(f"  VARIANT: {VAR}")
    print(f"{'='*60}")
    
    with open(os.path.join(MODELS_V2, f'top10_features_{VAR}.json')) as f:
        top10 = json.load(f)
    scaler = joblib.load(os.path.join(MODELS_V2, f'scaler_{VAR}.pkl'))
    X_bal_s = scaler.transform(X_bal_full[top10].values)
    
    dt = joblib.load(os.path.join(MODELS_V2, f'decision_tree_{VAR}.pkl'))
    rf = joblib.load(os.path.join(MODELS_V2, f'random_forest_{VAR}.pkl'))
    knn = joblib.load(os.path.join(MODELS_V2, f'knn_esp32_{VAR}.pkl'))
    
    N_PERM = 500
    
    for name, model in [('DT', dt), ('RF', rf), ('KNN', knn)]:
        yp = model.predict(X_bal_s)
        acc_true = accuracy_score(y_bal, yp)
        f1_true = f1_score(y_bal, yp, zero_division=0)
        
        perm_accs = []
        perm_f1s = []
        perm_rng = np.random.RandomState(42)
        
        for _ in range(N_PERM):
            y_shuf = perm_rng.permutation(y_bal)
            perm_accs.append(accuracy_score(y_shuf, yp))
            perm_f1s.append(f1_score(y_shuf, yp, zero_division=0))
        
        perm_acc_mean = np.mean(perm_accs)
        perm_f1_mean = np.mean(perm_f1s)
        p_acc = (np.array(perm_accs) >= acc_true).mean()
        p_f1 = (np.array(perm_f1s) >= f1_true).mean()
        
        print(f"\n  {name} ({VAR}):")
        print(f"    y_pred dist : 0={int((yp==0).sum())}, 1={int((yp==1).sum())}")
        print(f"    TRUE   Acc={acc_true:.4f}  F1={f1_true:.4f}")
        print(f"    PERM   Acc={perm_acc_mean:.4f} +/-{np.std(perm_accs):.4f}  "
              f"F1={perm_f1_mean:.4f} +/-{np.std(perm_f1s):.4f}")
        print(f"    p-value  Acc={p_acc:.4f}  F1={p_f1:.4f}")
        
        # Interpretasi
        drop = acc_true - perm_acc_mean
        if abs(perm_acc_mean - 0.5) < 0.1 and acc_true > 0.9:
            verdict = "LOLOS"
            detail = (f"Permuted ~{perm_acc_mean:.1%} (baseline ~50%), "
                      f"True={acc_true:.1%}, drop={drop:.1%} -> Model belajar")
        elif abs(perm_acc_mean - 0.5) < 0.1:
            verdict = "MARGINAL"
            detail = f"Permuted ~{perm_acc_mean:.1%}, True={acc_true:.1%}"
        else:
            verdict = "PERIKSA"
            detail = f"Permuted baseline={perm_acc_mean:.1%} (expected ~50%)"
        
        print(f"    Verdict: {verdict} - {detail}")

print(f"\n{'='*80}")
print("  SELESAI")
print(f"{'='*80}")
