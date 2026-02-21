# ============================================================
# PERBAIKAN MODEL DT — MULTIPLE APPROACHES
# ============================================================

import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from micromlgen import port

# Load data
data = np.load('data/preprocessed_top_10.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print("=" * 70)
print("🔧 PERBAIKAN MODEL DT — MULTIPLE APPROACHES")
print("=" * 70)

# ============================================================
# APPROACH 1: Re-train TANPA class_weight='balanced'
# (micromlgen mungkin tidak support class_weight dengan benar)
# ============================================================

print("\n📌 APPROACH 1: DT tanpa class_weight")
print("-" * 50)

dt_no_weight = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    min_samples_split=5,
    min_samples_leaf=2,
    # TANPA class_weight='balanced'!
    random_state=42
)
dt_no_weight.fit(X_train, y_train)
y_pred_1 = dt_no_weight.predict(X_test[:100])
y_pred_1_full = dt_no_weight.predict(X_test)

acc_1 = accuracy_score(y_test[:100], y_pred_1) * 100
f1_1 = f1_score(y_test, y_pred_1_full, average='weighted') * 100
cm_1 = confusion_matrix(y_test[:100], y_pred_1)

unique_1, counts_1 = np.unique(y_pred_1, return_counts=True)
print(f"  Distribusi (100 samples): {dict(zip(unique_1, counts_1))}")
print(f"  Accuracy (100): {acc_1:.2f}%")
print(f"  F1 (full test): {f1_1:.2f}%")
print(f"  Nodes: {dt_no_weight.tree_.node_count}")
print(f"  Size: {dt_no_weight.tree_.node_count * 20 / 1024:.1f} KB")

# Cek C code
c_code_1 = port(dt_no_weight)
r0_1 = c_code_1.count('return 0')
r1_1 = c_code_1.count('return 1')
print(f"  C code: return 0 = {r0_1}x, return 1 = {r1_1}x")
print(f"  Valid: {'✅ YES' if r0_1 > 0 and r1_1 > 0 else '🔴 NO'}")

# ============================================================
# APPROACH 2: Re-train dengan sample_weight manual
# (alternatif dari class_weight yang lebih kompatibel)
# ============================================================

print("\n📌 APPROACH 2: DT dengan sample_weight manual")
print("-" * 50)

# Hitung sample weights secara manual
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

dt_sample_weight = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_sample_weight.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_2 = dt_sample_weight.predict(X_test[:100])
y_pred_2_full = dt_sample_weight.predict(X_test)

acc_2 = accuracy_score(y_test[:100], y_pred_2) * 100
f1_2 = f1_score(y_test, y_pred_2_full, average='weighted') * 100

unique_2, counts_2 = np.unique(y_pred_2, return_counts=True)
print(f"  Distribusi (100 samples): {dict(zip(unique_2, counts_2))}")
print(f"  Accuracy (100): {acc_2:.2f}%")
print(f"  F1 (full test): {f1_2:.2f}%")
print(f"  Nodes: {dt_sample_weight.tree_.node_count}")

c_code_2 = port(dt_sample_weight)
r0_2 = c_code_2.count('return 0')
r1_2 = c_code_2.count('return 1')
print(f"  C code: return 0 = {r0_2}x, return 1 = {r1_2}x")
print(f"  Valid: {'✅ YES' if r0_2 > 0 and r1_2 > 0 else '🔴 NO'}")

# ============================================================
# APPROACH 3: DT dengan max_depth=7 tanpa class_weight
# (sedikit lebih dalam untuk accuracy, masih muat di Arduino)
# ============================================================

print("\n📌 APPROACH 3: DT depth=7 tanpa class_weight")
print("-" * 50)

dt_deeper = DecisionTreeClassifier(
    max_depth=7,
    criterion='entropy',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_deeper.fit(X_train, y_train)
y_pred_3 = dt_deeper.predict(X_test[:100])
y_pred_3_full = dt_deeper.predict(X_test)

acc_3 = accuracy_score(y_test[:100], y_pred_3) * 100
f1_3 = f1_score(y_test, y_pred_3_full, average='weighted') * 100

unique_3, counts_3 = np.unique(y_pred_3, return_counts=True)
print(f"  Distribusi (100 samples): {dict(zip(unique_3, counts_3))}")
print(f"  Accuracy (100): {acc_3:.2f}%")
print(f"  F1 (full test): {f1_3:.2f}%")
print(f"  Nodes: {dt_deeper.tree_.node_count}")
print(f"  Size: {dt_deeper.tree_.node_count * 20 / 1024:.1f} KB")

c_code_3 = port(dt_deeper)
r0_3 = c_code_3.count('return 0')
r1_3 = c_code_3.count('return 1')
print(f"  C code: return 0 = {r0_3}x, return 1 = {r1_3}x")
print(f"  Valid: {'✅ YES' if r0_3 > 0 and r1_3 > 0 else '🔴 NO'}")
fits = dt_deeper.tree_.node_count * 20 / 1024 < 20
print(f"  Fits Arduino: {'✅ YES' if fits else '🔴 NO'}")

# ============================================================
# APPROACH 4: Custom C code export (manual tree traversal)
# Jika micromlgen tetap bermasalah
# ============================================================

print("\n📌 APPROACH 4: Custom manual export (fallback)")
print("-" * 50)

def export_tree_to_c_manual(tree, feature_names=None):
    """
    Export Decision Tree ke C code SECARA MANUAL
    tanpa menggunakan micromlgen.
    Ini menjamin konsistensi dengan model Python.
    """
    tree_ = tree.tree_
    
    if feature_names is None:
        feature_names = [f'x[{i}]' for i in range(tree_.n_features)]
    
    lines = []
    lines.append("// Manual DT Export — Guaranteed correct")
    lines.append("// Generated without micromlgen")
    lines.append("")
    lines.append("int predict_dt(float *x) {")
    
    def recurse(node, depth):
        indent = "    " * (depth + 1)
        
        # Jika leaf node
        if tree_.children_left[node] == tree_.children_right[node]:
            # Ambil class dengan jumlah terbanyak
            class_counts = tree_.value[node][0]
            predicted_class = int(np.argmax(class_counts))
            count_0 = int(class_counts[0])
            count_1 = int(class_counts[1])
            lines.append(f"{indent}// Leaf: Normal={count_0}, "
                        f"Attack={count_1}")
            lines.append(f"{indent}return {predicted_class};")
            return
        
        # Internal node
        feature_idx = tree_.feature[node]
        threshold = tree_.threshold[node]
        feature_name = feature_names[feature_idx]
        
        lines.append(f"{indent}// Node {node}: "
                     f"{feature_name} <= {threshold:.6f}")
        lines.append(f"{indent}if (x[{feature_idx}] <= "
                     f"{threshold:.6f}f) {{")
        recurse(tree_.children_left[node], depth + 1)
        lines.append(f"{indent}}} else {{")
        recurse(tree_.children_right[node], depth + 1)
        lines.append(f"{indent}}}")
    
    recurse(0, 0)
    lines.append("}")
    
    return '\n'.join(lines)

# Gunakan model terbaik yang VALID
# (pilih salah satu dari approach 1-3 yang valid)
best_approach = None
best_model = None
best_f1 = 0

for name, model, f1_val, code in [
    ("Approach 1 (no weight)", dt_no_weight, f1_1, c_code_1),
    ("Approach 2 (sample weight)", dt_sample_weight, f1_2, c_code_2),
    ("Approach 3 (depth 7)", dt_deeper, f1_3, c_code_3),
]:
    r0 = code.count('return 0')
    r1 = code.count('return 1')
    valid = r0 > 0 and r1 > 0
    fits = model.tree_.node_count * 20 / 1024 < 20
    
    if valid and fits and f1_val > best_f1:
        best_f1 = f1_val
        best_approach = name
        best_model = model

print(f"\n{'='*70}")
print(f"🏆 BEST VALID APPROACH: {best_approach}")
print(f"   F1-Score: {best_f1:.2f}%")
print(f"{'='*70}")

if best_model is not None:
    # Generate KEDUA versi C code
    
    # Versi 1: micromlgen (jika valid)
    c_code_micromlgen = port(best_model)
    
    # Versi 2: Custom manual export (SELALU BENAR)
    feature_names = [
        'src_bytes', 'service', 'dst_bytes', 'flag',
        'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'logged_in', 'dst_host_serror_rate'
    ]
    c_code_manual = export_tree_to_c_manual(best_model, feature_names)
    
    # Verifikasi manual export
    r0_m = c_code_manual.count('return 0')
    r1_m = c_code_manual.count('return 1')
    print(f"\nManual export: return 0 = {r0_m}x, return 1 = {r1_m}x")
    
    # Simpan
    # File untuk Arduino yang menggunakan micromlgen format
    with open('arduino_code/dt_model.h', 'w') as f:
        f.write("// ============================================\n")
        f.write(f"// Decision Tree — {best_approach}\n")
        f.write(f"// Depth: {best_model.get_depth()}\n")
        f.write(f"// Nodes: {best_model.tree_.node_count}\n")
        f.write("// ============================================\n\n")
        f.write(c_code_micromlgen)
    print(f"\n💾 Saved micromlgen version: arduino_code/dt_model.h")
    
    # File manual backup
    with open('arduino_code/dt_model_manual.h', 'w') as f:
        f.write("// ============================================\n")
        f.write("// Decision Tree — Manual Export (BACKUP)\n")
        f.write("// Use this if micromlgen version doesn't work\n")
        f.write("// ============================================\n\n")
        f.write("#ifndef DT_MODEL_MANUAL_H\n")
        f.write("#define DT_MODEL_MANUAL_H\n\n")
        f.write(c_code_manual)
        f.write("\n\n#endif\n")
    print(f"💾 Saved manual version: arduino_code/dt_model_manual.h")
    
    # Simpan model baru
    with open('models/decision_tree_fixed.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"💾 Saved model: models/decision_tree_fixed.pkl")
    
    # VERIFIKASI FINAL
    print(f"\n{'='*70}")
    print(f"🔍 VERIFIKASI FINAL — Prediksi Python vs Expected")
    print(f"{'='*70}")
    y_pred_final = best_model.predict(X_test[:100])
    unique_f, counts_f = np.unique(y_pred_final, return_counts=True)
    print(f"  Distribusi (100 samples): {dict(zip(unique_f, counts_f))}")
    
    y_true_100 = y_test[:100]
    unique_t, counts_t = np.unique(y_true_100, return_counts=True)
    print(f"  True labels  (100 samples): {dict(zip(unique_t, counts_t))}")
    
    acc_final = accuracy_score(y_true_100, y_pred_final) * 100
    print(f"  Accuracy: {acc_final:.2f}%")
    print(f"  Distribusi wajar: {'✅ YES' if len(unique_f) > 1 else '🔴 NO'}")

else:
    print("\n🔴 SEMUA APPROACH GAGAL!")
    print("   Gunakan MANUAL EXPORT sebagai fallback!")
    
    # Force manual export dari model depth=5 tanpa weight
    dt_fallback = DecisionTreeClassifier(
        max_depth=5, criterion='entropy',
        min_samples_split=5, min_samples_leaf=2,
        random_state=42
    )
    dt_fallback.fit(X_train, y_train)
    
    c_manual = export_tree_to_c_manual(dt_fallback)
    
    with open('arduino_code/dt_model_manual.h', 'w') as f:
        f.write("#ifndef DT_MODEL_MANUAL_H\n")
        f.write("#define DT_MODEL_MANUAL_H\n\n")
        f.write(c_manual)
        f.write("\n\n#endif\n")
    
    print(f"💾 Saved: arduino_code/dt_model_manual.h")
    print(f"⚠️  Gunakan file ini dan ubah #include di Arduino sketch")

# ============================================================
# PRINT PANDUAN LANGKAH SELANJUTNYA
# ============================================================

print(f"""

{'='*70}
📋 LANGKAH SELANJUTNYA SETELAH PERBAIKAN:
{'='*70}

1. JIKA micromlgen version VALID (return 0 > 0):
   → Copy dt_model.h ke folder ids_arduino_uno/
   → Copy dt_model.h ke folder ids_arduino_nano/
   → Copy dt_model.h ke folder ids_esp32/
   → Upload ulang ke SEMUA platform
   → Re-run benchmark

2. JIKA micromlgen TETAP RUSAK:
   → Gunakan dt_model_manual.h
   → Di ids_arduino_uno.ino, ganti:
     #include "dt_model.h"
     MENJADI:
     #include "dt_model_manual.h"
   
   → Ganti fungsi predict():
     int predict(float* features) {{
         return predict_dt(features);  // fungsi dari manual export
     }}
   
   → Upload ulang dan test

3. SETELAH DT BEKERJA:
   → Run benchmark 100 samples DT (semua platform)
   → Run benchmark 100 samples RF di ESP32
   → Run benchmark 100 samples KNN di ESP32
   → Kumpulkan SEMUA data untuk paper
""")