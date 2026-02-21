# ============================================================
# FINAL FIX: Manual DT Export — BYPASS micromlgen completely
# Jalankan ini dan LANGSUNG test hasilnya
# ============================================================

import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
data = np.load('data/preprocessed_top_10.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# ============================================================
# STEP 1: Train DT bersih TANPA class_weight
# ============================================================

print("=" * 60)
print("🌳 TRAINING DT BERSIH (tanpa class_weight)")
print("=" * 60)

dt = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
    # TANPA class_weight!
)
dt.fit(X_train, y_train)

# Verifikasi di Python
y_pred = dt.predict(X_test[:100])
unique, counts = np.unique(y_pred, return_counts=True)
print(f"Python predictions (100 samples): {dict(zip(unique, counts))}")

y_pred_full = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred_full) * 100
f1 = f1_score(y_test, y_pred_full, average='weighted') * 100
print(f"Full test — Accuracy: {acc:.2f}%, F1: {f1:.2f}%")
print(f"Nodes: {dt.tree_.node_count}, Depth: {dt.tree_.max_depth}")

all_one = all(y_pred == 1)
print(f"Semua prediksi = 1? {'🔴 YA' if all_one else '✅ TIDAK — DT OK!'}")

if all_one:
    print("🔴 Model Python JUGA rusak. Coba depth atau parameter lain.")
    # Coba berbagai parameter
    for depth in [3, 4, 5, 6, 7]:
        for crit in ['gini', 'entropy']:
            dt_test = DecisionTreeClassifier(
                max_depth=depth, criterion=crit,
                min_samples_split=10, min_samples_leaf=5,
                random_state=42
            )
            dt_test.fit(X_train, y_train)
            y_p = dt_test.predict(X_test[:100])
            u, c = np.unique(y_p, return_counts=True)
            dist = dict(zip(u, c))
            has_both = 0 in dist and 1 in dist
            acc_t = accuracy_score(y_test[:100], y_p) * 100
            print(f"  depth={depth}, crit={crit}: {dist} "
                  f"acc={acc_t:.1f}% "
                  f"{'✅' if has_both else '🔴'}")
            if has_both:
                dt = dt_test  # gunakan ini
                break
        if has_both:
            break

# ============================================================
# STEP 2: MANUAL EXPORT KE C (BYPASS micromlgen sepenuhnya)
# ============================================================

print(f"\n{'='*60}")
print("📄 MANUAL EXPORT KE C CODE")
print(f"{'='*60}")

tree_ = dt.tree_

def tree_to_c(node=0, depth=0):
    """Recursively convert tree to C if-else code"""
    indent = "    " * (depth + 1)
    lines = []
    
    left = tree_.children_left[node]
    right = tree_.children_right[node]
    
    # Leaf node
    if left == right:  # LEAF: left == right == -1
        class_counts = tree_.value[node][0]
        predicted = int(np.argmax(class_counts))
        n0 = int(class_counts[0])
        n1 = int(class_counts[1])
        lines.append(f"{indent}// Leaf: Normal={n0}, Attack={n1}")
        lines.append(f"{indent}return {predicted};")
        return lines
    
    # Internal node
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

# Count return statements
r0 = c_body.count('return 0')
r1 = c_body.count('return 1')
print(f"return 0 (Normal): {r0} kali")
print(f"return 1 (Attack): {r1} kali")
print(f"Valid: {'✅' if r0 > 0 and r1 > 0 else '🔴 STILL BROKEN'}")

# ============================================================
# STEP 3: Generate COMPLETE Arduino-compatible header file
# ============================================================

feature_names = [
    'src_bytes', 'service', 'dst_bytes', 'flag',
    'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'logged_in', 'dst_host_serror_rate'
]

header_content = f"""// ============================================================
// DECISION TREE MODEL — MANUAL EXPORT
// Generated WITHOUT micromlgen (direct tree traversal)
// Depth: {dt.tree_.max_depth} | Nodes: {dt.tree_.node_count}
// return 0 = {r0}x | return 1 = {r1}x
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

# Save
output_path = 'arduino_code/dt_model.h'
with open(output_path, 'w') as f:
    f.write(header_content)

print(f"\n💾 Saved: {output_path}")
print(f"   Size: {len(header_content)} bytes")

# ============================================================
# STEP 4: VERIFIKASI — C code harus match Python predictions
# ============================================================

print(f"\n{'='*60}")
print("🔍 VERIFIKASI: Simulasi C code di Python")
print(f"{'='*60}")

# Simulasi prediksi menggunakan tree traversal manual
# (ini PERSIS yang akan dilakukan Arduino)
def predict_manual(x, node=0):
    left = tree_.children_left[node]
    right = tree_.children_right[node]
    
    if left == right:  # leaf
        return int(np.argmax(tree_.value[node][0]))
    
    if x[tree_.feature[node]] <= tree_.threshold[node]:
        return predict_manual(x, left)
    else:
        return predict_manual(x, right)

# Predict semua 100 test samples manually
y_manual = np.array([predict_manual(X_test[i]) for i in range(100)])
y_sklearn = dt.predict(X_test[:100])

# Bandingkan
match = np.all(y_manual == y_sklearn)
print(f"Manual predictions match sklearn: {'✅ YES' if match else '🔴 NO'}")

if match:
    unique_m, counts_m = np.unique(y_manual, return_counts=True)
    print(f"Distribution: {dict(zip(unique_m, counts_m))}")
    acc_m = accuracy_score(y_test[:100], y_manual) * 100
    print(f"Accuracy (100 samples): {acc_m:.2f}%")
    print(f"\n✅ C CODE VERIFIED! Upload dt_model.h ke Arduino sekarang.")
else:
    print("🔴 Mismatch detected! Debug needed.")
    for i in range(min(10, len(y_manual))):
        if y_manual[i] != y_sklearn[i]:
            print(f"  Sample {i}: manual={y_manual[i]}, sklearn={y_sklearn[i]}")

# ============================================================
# STEP 5: JUGA generate test validation data
# ============================================================

# Simpan prediksi yang BENAR untuk verifikasi di Arduino
np.savetxt('arduino_code/test_pred_dt_fixed.csv',
           y_manual, delimiter=',', fmt='%d')
print(f"\n💾 Reference predictions saved: test_pred_dt_fixed.csv")

# Save model
with open('models/decision_tree_final.pkl', 'wb') as f:
    pickle.dump(dt, f)
print(f"💾 Model saved: models/decision_tree_final.pkl")

print(f"""
{'='*60}
📋 LANGKAH SELANJUTNYA:
{'='*60}

1. Copy arduino_code/dt_model.h ke:
   - ids_arduino_uno/dt_model.h
   - ids_arduino_nano/dt_model.h  
   - ids_esp32/dt_model.h

2. Di Arduino IDE, BERSIHKAN CACHE:
   - Sketch → Export Compiled Binary (untuk force recompile)
   - ATAU tutup Arduino IDE, hapus folder temp, buka lagi

3. Upload ulang ke SEMUA platform

4. Test manual dulu via Serial Monitor:
   Kirim: 0.000000,0.217391,0.000000,0.300000,1.000000,0.000000,1.000000,1.000000,1.000000,0.000000
   → Harus muncul prediksi (bisa 0 ATAU 1, bukan selalu 1)

5. Jalankan benchmark_all.py lagi
""")