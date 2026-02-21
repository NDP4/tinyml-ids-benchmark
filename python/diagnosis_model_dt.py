# ============================================================
# DIAGNOSIS: Apakah dt_model.h di-export dengan benar?
# ============================================================

import pickle
import numpy as np
from micromlgen import port

# 1. Load model DT yang di-optimasi
with open('../models/decision_tree_optimized.pkl', 'rb') as f:
    dt_model = pickle.load(f)

# 2. Verifikasi model di PYTHON dulu
data = np.load('../data/preprocessed_top_10.npz')
X_test = data['X_test']
y_test = data['y_test']

# Prediksi di Python
y_pred_python = dt_model.predict(X_test[:100])

print("=" * 60)
print("📊 DIAGNOSIS MODEL DT DI PYTHON")
print("=" * 60)

# Cek distribusi prediksi
unique, counts = np.unique(y_pred_python, return_counts=True)
print(f"\nDistribusi prediksi Python (100 samples):")
for u, c in zip(unique, counts):
    label = "Normal" if u == 0 else "Attack"
    print(f"  Class {u} ({label}): {c}")

# Jika Python JUGA selalu prediksi 1, masalah ada di model
# Jika Python mix 0 dan 1, masalah ada di export
all_one = all(y_pred_python == 1)
print(f"\nSemua prediksi = 1? {'🔴 YA — Model Python juga rusak!' if all_one else '✅ TIDAK — Model Python OK, masalah di export'}")

# 3. Cek C code yang di-generate
print("\n" + "=" * 60)
print("📄 ANALISIS C CODE YANG DI-GENERATE")
print("=" * 60)

c_code = port(dt_model)

# Hitung berapa kali "return 0" dan "return 1" muncul
return_0_count = c_code.count('return 0')
return_1_count = c_code.count('return 1')

print(f"\nDalam dt_model.h:")
print(f"  'return 0' (Normal) : {return_0_count} kali")
print(f"  'return 1' (Attack) : {return_1_count} kali")

if return_0_count == 0:
    print(f"\n🔴 MASALAH DITEMUKAN!")
    print(f"   TIDAK ADA 'return 0' di C code!")
    print(f"   Semua leaf nodes mengarah ke class 1!")
    print(f"   → Ini penyebab model selalu prediksi ATTACK")
elif return_0_count > 0 and return_1_count > 0:
    print(f"\n🟡 C code memiliki kedua class.")
    print(f"   Masalah mungkin di threshold/feature indexing.")

# 4. Tampilkan C code untuk inspeksi manual
print("\n" + "=" * 60)
print("📄 C CODE (FULL — untuk inspeksi)")
print("=" * 60)
print(c_code)

# 5. Simpan untuk referensi
with open('dt_model_debug.h', 'w') as f:
    f.write(c_code)
print(f"\n💾 Saved to: dt_model_debug.h")