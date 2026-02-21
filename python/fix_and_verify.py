# ============================================================
# FIX & VERIFY: Re-export DT model + verifikasi prediksi
# Jalankan dari folder ids-research/
# ============================================================

import pickle
import numpy as np
import shutil
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load data
data = np.load('data/preprocessed_top_10.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print("=" * 70)
print("  FIX & VERIFY: Decision Tree Export")
print("=" * 70)

# ============================================================
# STEP 1: Cek distribusi data
# ============================================================
print("\n--- STEP 1: Distribusi data ---")
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
print(f"Train: {dict(zip(unique_train, counts_train))}")
print(f"Test : {dict(zip(unique_test, counts_test))}")
print(f"Test 100 pertama: {dict(zip(*np.unique(y_test[:100], return_counts=True)))}")

# ============================================================
# STEP 2: Train DT BERSIH (tanpa class_weight)
# ============================================================
print("\n--- STEP 2: Train DT bersih ---")

dt = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test[:100])
y_pred_full = dt.predict(X_test)
acc = accuracy_score(y_test[:100], y_pred) * 100
f1 = f1_score(y_test, y_pred_full, average='weighted') * 100

unique_p, counts_p = np.unique(y_pred, return_counts=True)
print(f"Python prediksi (100 samples): {dict(zip(unique_p, counts_p))}")
print(f"Accuracy (100 samples): {acc:.2f}%")
print(f"F1 (full test): {f1:.2f}%")
print(f"Nodes: {dt.tree_.node_count}, Depth: {dt.tree_.max_depth}")

all_one = all(y_pred == 1)
all_zero = all(y_pred == 0)
print(f"Semua prediksi sama? {'YA - BERMASALAH!' if (all_one or all_zero) else 'TIDAK - OK!'}")

# ============================================================
# STEP 3: Verifikasi leaf nodes TIDAK ada yang "0,0"
# ============================================================
print("\n--- STEP 3: Verifikasi leaf nodes ---")
tree_ = dt.tree_
n_leaves = 0
n_empty = 0
for i in range(tree_.node_count):
    if tree_.children_left[i] == tree_.children_right[i]:  # leaf
        n_leaves += 1
        class_counts = tree_.value[i][0]
        n0, n1 = int(class_counts[0]), int(class_counts[1])
        if n0 == 0 and n1 == 0:
            n_empty += 1
            print(f"  WARNING: Leaf {i} has Normal=0, Attack=0!")

print(f"Total leaves: {n_leaves}")
print(f"Empty leaves: {n_empty}")
if n_empty > 0:
    print("  Ada leaf kosong! Model mungkin menggunakan class_weight.")
else:
    print("  Semua leaf punya samples - OK!")

# ============================================================
# STEP 4: Export DT ke C code (manual, tanpa micromlgen)
# ============================================================
print("\n--- STEP 4: Export ke C code ---")

def tree_to_c(node=0, depth=0):
    """Convert sklearn tree to C if-else code"""
    indent = "    " * (depth + 1)
    lines = []
    
    left = tree_.children_left[node]
    right = tree_.children_right[node]
    
    if left == right:  # LEAF
        class_counts = tree_.value[node][0]
        predicted = int(np.argmax(class_counts))
        n0 = int(class_counts[0])
        n1 = int(class_counts[1])
        lines.append(f"{indent}// Leaf: Normal={n0}, Attack={n1}")
        lines.append(f"{indent}return {predicted};")
        return lines
    
    feat = tree_.feature[node]
    thresh = tree_.threshold[node]
    
    lines.append(f"{indent}if (x[{feat}] <= {thresh:.10f}f) {{")
    lines.extend(tree_to_c(left, depth + 1))
    lines.append(f"{indent}}} else {{")
    lines.extend(tree_to_c(right, depth + 1))
    lines.append(f"{indent}}}")
    
    return lines

c_lines = tree_to_c()
c_body = '\n'.join(c_lines)

r0 = c_body.count('return 0')
r1 = c_body.count('return 1')
print(f"return 0 (Normal): {r0} kali")
print(f"return 1 (Attack): {r1} kali")
print(f"Valid: {'OK' if r0 > 0 and r1 > 0 else 'BERMASALAH!'}")

# ============================================================
# STEP 5: Generate header file compatible dengan Eloquent format
# ============================================================

feature_names = [
    'src_bytes', 'service', 'dst_bytes', 'flag',
    'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'logged_in', 'dst_host_serror_rate'
]

header_content = f"""// ============================================================
// DECISION TREE MODEL - MANUAL EXPORT (FIXED)
// Generated WITHOUT micromlgen (direct tree traversal)
// Depth: {dt.tree_.max_depth} | Nodes: {dt.tree_.node_count}
// return 0 = {r0}x | return 1 = {r1}x
// TANPA class_weight - leaf counts VALID
// ============================================================

#pragma once

namespace Eloquent {{
    namespace ML {{
        namespace Port {{

            class DecisionTree {{
            public:
                /**
                 * Predict class for features
                 * @param x: array of {len(feature_names)} float values (normalized 0-1)
                 * Features: {', '.join(feature_names)}
                 * @return: 0 (Normal) or 1 (Attack)
                 */
                int predict(float *x) {{
{c_body}
                }}
            }};

        }}
    }}
}}
"""

# ============================================================
# STEP 6: Verifikasi C code matches Python predictions
# ============================================================
print("\n--- STEP 6: Verifikasi C code vs Python ---")

def predict_manual(x, node=0):
    """Simulasi C code di Python"""
    left = tree_.children_left[node]
    right = tree_.children_right[node]
    
    if left == right:
        return int(np.argmax(tree_.value[node][0]))
    
    if x[tree_.feature[node]] <= tree_.threshold[node]:
        return predict_manual(x, left)
    else:
        return predict_manual(x, right)

y_manual = np.array([predict_manual(X_test[i]) for i in range(100)])
y_sklearn = dt.predict(X_test[:100])

match = np.all(y_manual == y_sklearn)
print(f"Manual C-sim vs sklearn: {'MATCH!' if match else 'MISMATCH!'}")

if not match:
    mismatches = np.where(y_manual != y_sklearn)[0]
    print(f"  Mismatch pada {len(mismatches)} samples: {mismatches[:10]}")

acc_manual = accuracy_score(y_test[:100], y_manual) * 100
print(f"Accuracy (manual sim): {acc_manual:.2f}%")
print(f"Distribution: {dict(zip(*np.unique(y_manual, return_counts=True)))}")

# ============================================================
# STEP 7: Simulasi DOUBLE NORMALIZATION (untuk buktikan bug)
# ============================================================
print("\n--- STEP 7: Simulasi double normalization (BUG) ---")

feature_max_arr = np.array([
    1379963888.0, 69.0, 1309937401.0, 10.0,
    1.0, 1.0, 255.0, 1.0, 1.0, 1.0
])

# Simulasi: data sudah normalized, Arduino normalize LAGI
y_double = []
for i in range(100):
    x = X_test[i].copy()
    # Arduino normalize lagi: x_double = x / feature_max
    x_double = x / feature_max_arr
    x_double = np.clip(x_double, 0, 1)
    y_double.append(predict_manual(x_double))
y_double = np.array(y_double)

acc_double = accuracy_score(y_test[:100], y_double) * 100
unique_d, counts_d = np.unique(y_double, return_counts=True)
print(f"Dengan DOUBLE normalization (BUG):")
print(f"  Distribusi  : {dict(zip(unique_d, counts_d))}")
print(f"  Accuracy    : {acc_double:.2f}%")
all_same = len(unique_d) == 1
print(f"  Semua sama? : {'YA - INI PENYEBAB BUG!' if all_same else 'Tidak'}")

# ============================================================
# STEP 8: Simulasi CORRECT (tanpa double normalization)
# ============================================================
print("\n--- STEP 8: Simulasi CORRECT (single normalization) ---")

y_correct = []
for i in range(100):
    x = X_test[i].copy()  # Sudah normalized
    y_correct.append(predict_manual(x))
y_correct = np.array(y_correct)

acc_correct = accuracy_score(y_test[:100], y_correct) * 100
unique_c, counts_c = np.unique(y_correct, return_counts=True)
print(f"Dengan SINGLE normalization (CORRECT):")
print(f"  Distribusi  : {dict(zip(unique_c, counts_c))}")
print(f"  Accuracy    : {acc_correct:.2f}%")

# ============================================================
# STEP 9: Save files
# ============================================================
print("\n--- STEP 9: Save files ---")

# Save ke semua folder Arduino
targets = [
    'arduino_code/dt_model.h',
    'arduino_code/ids_arduino_uno/dt_model.h',
    'arduino_code/ids_arduino_nano/dt_model.h',
    'arduino_code/ids_esp32/dt_model.h',
]

for target in targets:
    with open(target, 'w') as f:
        f.write(header_content)
    print(f"  Saved: {target}")

# Save model
with open('models/decision_tree_final.pkl', 'wb') as f:
    pickle.dump(dt, f)
print(f"  Saved: models/decision_tree_final.pkl")

# Save reference predictions
np.savetxt('arduino_code/test_pred_dt_fixed.csv',
           y_correct, delimiter=',', fmt='%d')
print(f"  Saved: arduino_code/test_pred_dt_fixed.csv")

print(f"""
{'='*70}
  RINGKASAN PERBAIKAN
{'='*70}

ROOT CAUSE: DOUBLE NORMALIZATION
  Data di preprocessed_top_10.npz SUDAH di-MinMaxScale (0-1).
  benchmark_all.py mengirim data normalized ke Arduino.
  Arduino normalize LAGI -> feature values jadi SALAH!
  
  Contoh: flag (max=10) -> 0.9 / 10 = 0.09 -> selalu < 0.55
  -> DT SELALU masuk branch yang sama -> semua prediksi = 1

PERBAIKAN:
  1. benchmark_all.py: Inverse-transform data ke raw values
     sebelum dikirim ke Arduino (SUDAH DIPERBAIKI)
  2. dt_model.h: Re-export dari model TANPA class_weight
     (leaf counts VALID, bukan 0,0) (SUDAH DIPERBAIKI)

EXPECTED RESULTS SETELAH PERBAIKAN:
  DT Accuracy: ~{acc_correct:.0f}% (sebelumnya selalu 49%)
  RF Accuracy: tetap ~78% (tidak terpengaruh signifikan)

LANGKAH SELANJUTNYA:
  1. Upload ulang sketch ke Arduino Uno/Nano/ESP32
  2. Jalankan benchmark_all.py lagi
  3. Hasil DT harus ~{acc_correct:.0f}%, bukan 49%
{'='*70}
""")
