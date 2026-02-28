# ============================================================
# benchmark_all.py
# Mengirim test data NSL-KDD ke Arduino/ESP32 via Serial
# dan mengumpulkan hasil benchmark
# ============================================================

import serial
import time
import numpy as np
import pandas as pd
import sys
import json
from datetime import datetime

class IDSBenchmark:
    """
    Kelas untuk menjalankan benchmark IDS pada Arduino/ESP32
    via koneksi Serial.
    
    Cara kerja:
    1. Hubungkan Arduino/ESP32 ke komputer via USB
    2. Jalankan script ini
    3. Script mengirim test samples satu per satu via Serial
    4. Arduino memproses dan mengirim kembali hasil prediksi
    5. Script mengumpulkan dan menganalisis semua hasil
    """
    
    def __init__(self, port, baud=115200, timeout=5):
        """
        Inisialisasi koneksi Serial
        
        Parameters:
        - port: Serial port (misal '/dev/ttyUSB0' atau 'COM3')
        - baud: Baud rate (harus sama dengan di Arduino)
        - timeout: Timeout Serial dalam detik
        """
        self.port = port
        self.baud = baud
        self.ser = None
        self.results = []
        
        print(f"📡 Connecting to {port} at {baud} baud...")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(3)  # Tunggu Arduino reset setelah koneksi
        
        # Baca pesan startup
        self._read_all()
        print("✅ Connected!")
    
    def _read_all(self):
        """Baca semua data yang tersedia di buffer Serial"""
        lines = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='replace').strip()
            if line:
                lines.append(line)
                print(f"   Arduino: {line}")
        return lines
    
    def _send_and_read(self, message, delay=0.1):
        """Kirim pesan dan baca respons"""
        self.ser.write(f"{message}\n".encode())
        time.sleep(delay)
        return self._read_all()
    
    def benchmark_model(self, X_test, y_test, model_name, 
                         platform_name, num_samples=100):
        """
        Jalankan benchmark untuk satu model
        
        Parameters:
        - X_test: array test features (normalized!)
        - y_test: array true labels
        - model_name: 'dt', 'rf', atau 'knn'
        - platform_name: 'Arduino Uno', 'Arduino Nano', 'ESP32'
        - num_samples: jumlah samples untuk benchmark
        
        Returns:
        - dict dengan hasil benchmark
        """
        
        print(f"\n{'='*60}")
        print(f"🧪 BENCHMARK: {model_name.upper()} on {platform_name}")
        print(f"{'='*60}")
        
        samples = min(num_samples, len(X_test))
        
        # Kirim command benchmark (untuk ESP32)
        if 'ESP32' in platform_name:
            self._send_and_read(f'bench_{model_name}', delay=1)
        
        # Kirim samples satu per satu
        predictions = []
        inference_times = []
        correct = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for i in range(samples):
            # Format: f0,f1,...,f9,true_label
            features_str = ','.join([f'{v:.6f}' for v in X_test[i]])
            
            if 'ESP32' in platform_name:
                # ESP32 benchmark mode expects features + label
                line = f'{features_str},{int(y_test[i])}'
            else:
                # Arduino Uno/Nano: kirim fitur saja
                line = features_str
            
            self.ser.write(f'{line}\n'.encode())
            time.sleep(0.05)  # Beri waktu Arduino memproses
            
            # Baca respons
            responses = self._read_all()
            
            # for resp in responses:
            #     if 'RESULT' in resp or 'Pred:' in resp:
            #         # Parse inference time
            #         if 'μs' in resp:
            #             try:
            #                 time_str = resp.split('Inference:')[1].split('μs')[0].strip()
            #                 inf_time = float(time_str)
            #                 inference_times.append(inf_time)
            #             except (IndexError, ValueError):
            #                 pass
                    
            #         # Parse prediction
            #         if 'ATTACK' in resp:
            #             pred = 1
            #         elif 'NORMAL' in resp:
            #             pred = 0
            #         elif 'Pred: 1' in resp:
            #             pred = 1
            #         elif 'Pred: 0' in resp:
            #             pred = 0
            #         else:
            #             continue
                    
            #         predictions.append(pred)

            for resp in responses:
                    # Parse ESP32 benchmark format
                    # Format: "[X] Pred: Y | True: Z | ✅/❌ | N μs"
                    if 'μs' in resp:
                        try:
                            # Cari angka sebelum "μs"
                            parts = resp.split('|')
                            for part in parts:
                                if 'μs' in part:
                                    time_str = part.replace('μs', '').strip()
                                    # Bersihkan karakter non-numeric
                                    time_str = ''.join(c for c in time_str 
                                                    if c.isdigit() or c == '.')
                                    if time_str:
                                        inf_time = float(time_str)
                                        inference_times.append(inf_time)
                                    break
                        except (ValueError, IndexError):
                            pass
                    
                    # Parse prediction
                    if 'Pred: 1' in resp or 'Pred:1' in resp:
                        pred = 1
                    elif 'Pred: 0' in resp or 'Pred:0' in resp:
                        pred = 0
                    elif 'ATTACK' in resp:
                        pred = 1
                    elif 'NORMAL' in resp:
                        pred = 0
                    else:
                        continue
                    
                    predictions.append(pred)
                    true = int(y_test[i])
                    
                    if pred == true:
                        correct += 1
                    if pred == 1 and true == 1: tp += 1
                    if pred == 1 and true == 0: fp += 1
                    if pred == 0 and true == 0: tn += 1
                    if pred == 0 and true == 1: fn += 1
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{samples} samples sent...")
        
        # Kirim END (untuk ESP32 benchmark mode)
        if 'ESP32' in platform_name:
            self._send_and_read('END', delay=2)
        
        # Hitung metrik
        total = len(predictions)
        accuracy = correct / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0)
        
        avg_inference = np.mean(inference_times) if inference_times else 0
        std_inference = np.std(inference_times) if inference_times else 0
        min_inference = np.min(inference_times) if inference_times else 0
        max_inference = np.max(inference_times) if inference_times else 0
        
        result = {
            'platform': platform_name,
            'model': model_name.upper(),
            'samples': total,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'avg_inference_us': avg_inference,
            'std_inference_us': std_inference,
            'min_inference_us': min_inference,
            'max_inference_us': max_inference,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Print hasil
        print(f"\n📊 HASIL BENCHMARK:")
        print(f"   Platform  : {platform_name}")
        print(f"   Model     : {model_name.upper()}")
        print(f"   Samples   : {total}")
        print(f"   Accuracy  : {accuracy:.2f}%")
        print(f"   Precision : {precision:.2f}%")
        print(f"   Recall    : {recall:.2f}%")
        print(f"   F1-Score  : {f1:.2f}%")
        print(f"   TP:{tp} | FP:{fp} | TN:{tn} | FN:{fn}")
        print(f"   Avg Inference: {avg_inference:.1f} μs")
        print(f"   Std Inference: {std_inference:.1f} μs")
        print(f"   Min Inference: {min_inference:.1f} μs")
        print(f"   Max Inference: {max_inference:.1f} μs")
        
        return result
    
    def save_results(self, filename='../results/benchmark_results.json'):
        """Simpan semua hasil benchmark ke JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
    
    def generate_comparison_table(self):
        """Generate tabel perbandingan untuk paper"""
        if not self.results:
            print("⚠️ No results to compare!")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("📊 COMPARISON TABLE — FOR PAPER")
        print("=" * 80)
        
        cols = ['platform', 'model', 'accuracy', 'f1_score', 
                'avg_inference_us', 'samples']
        print(df[cols].to_string(index=False, float_format='%.2f'))
        
        # Save sebagai CSV juga
        df.to_csv('../results/benchmark_comparison.csv', index=False)
        print("\n💾 Saved to ../results/benchmark_comparison.csv")
    
    def close(self):
        """Tutup koneksi Serial"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial connection closed.")


# ============================================================
# MAIN — JALANKAN BENCHMARK
# ============================================================

if __name__ == '__main__':
    
    # Load test data (SUDAH di-MinMaxScale ke 0-1)
    data = np.load('../data/preprocessed_top_10.npz')
    X_test_scaled = data['X_test']
    y_test = data['y_test']
    
    # ============================================================
    # FIX: INVERSE TRANSFORM — Arduino akan normalize sendiri!
    # Data di .npz SUDAH dinormalisasi. Jika dikirim langsung,
    # Arduino normalize LAGI → double normalization → DT GAGAL.
    # Solusi: kembalikan ke raw values, biarkan Arduino normalize.
    # ============================================================
    feature_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    feature_max = np.array([
        1379963888.0,  # src_bytes
        69.0,          # service
        1309937401.0,  # dst_bytes
        10.0,          # flag
        1.0,           # same_srv_rate
        1.0,           # diff_srv_rate
        255.0,         # dst_host_srv_count
        1.0,           # dst_host_same_srv_rate
        1.0,           # logged_in
        1.0            # dst_host_serror_rate
    ], dtype=np.float64)
    feature_range = feature_max - feature_min
    
    # Inverse MinMaxScaler: X_raw = X_scaled * range + min
    X_test = X_test_scaled * feature_range + feature_min
    
    print(f"📊 Loaded {len(X_test)} test samples")
    print(f"   ✅ Inverse-transformed ke raw values (Arduino akan normalize)")
    print(f"   Contoh sample 0 raw  : {X_test[0][:4]}...")
    print(f"   Contoh sample 0 scaled: {X_test_scaled[0][:4]}...")
    
    # ---- USAGE ----
    # Sesuaikan SERIAL_PORT dengan port Arduino/ESP32 Anda
    # Jalankan TERPISAH untuk setiap platform
    
    # Contoh 1: Benchmark Arduino Uno
    # ================================
    # 1. Upload ids_arduino_uno.ino (dengan USE_DECISION_TREE)
    # 2. Jalankan:
    
    # SERIAL_PORT = '/dev/ttyACM0'  # Sesuaikan!
    SERIAL_PORT = 'COM3'  # Sesuaikan dengan port Arduino Uno Anda
    # SERIAL_PORT = 'COM9'  # Sesuaikan dengan port Arduino nano Anda
    # SERIAL_PORT = 'COM4'  # Sesuaikan dengan port esp32 Anda
    NUM_SAMPLES = 100
    
    bench = IDSBenchmark(SERIAL_PORT)

    # ============================================================
    # BENCHMARK KETIGA MODEL BERURUTAN (TANPA UPLOAD ULANG!) khusus eps32
    # ============================================================
    
    # print("\n" + "🔥" * 30)
    # print("🔥 BENCHMARK ESP32 — ALL 3 MODELS")
    # print("🔥" * 30)
    
    # # 1. Decision Tree
    # print("\n\n" + "=" * 60)
    # print("📌 [1/3] DECISION TREE")
    # print("=" * 60)
    # bench.benchmark_model(
    #     X_test, y_test,
    #     model_name='dt',
    #     platform_name='ESP32',
    #     num_samples=NUM_SAMPLES
    # )
    
    # # Jeda antar benchmark agar ESP32 stabil
    # print("\n⏳ Waiting 3 seconds before next benchmark...")
    # time.sleep(3)
    
    # # 2. Random Forest
    # print("\n\n" + "=" * 60)
    # print("📌 [2/3] RANDOM FOREST")
    # print("=" * 60)
    # bench.benchmark_model(
    #     X_test, y_test,
    #     model_name='rf',
    #     platform_name='ESP32',
    #     num_samples=NUM_SAMPLES
    # )
    
    # # Jeda antar benchmark
    # print("\n⏳ Waiting 3 seconds before next benchmark...")
    # time.sleep(3)
    
    # # 3. KNN
    # print("\n\n" + "=" * 60)
    # print("📌 [3/3] KNN")
    # print("=" * 60)
    # bench.benchmark_model(
    #     X_test, y_test,
    #     model_name='knn',
    #     platform_name='ESP32',
    #     num_samples=NUM_SAMPLES
    # )
    
    # # ============================================================
    # # SIMPAN & TAMPILKAN PERBANDINGAN
    # # ============================================================
    
    # bench.save_results()
    # bench.generate_comparison_table()
    # bench.close()
    
    # # Print ringkasan akhir
    # print("\n\n" + "=" * 70)
    # print("📊 RINGKASAN FINAL — ESP32 ALL MODELS")
    # print("=" * 70)
    # print(f"{'Model':<8} {'Accuracy':>10} {'F1-Score':>10} "
    #       f"{'Avg Inf':>10} {'Min':>8} {'Max':>8}")
    # print("-" * 60)
    # for r in bench.results:
    #     print(f"{r['model']:<8} {r['accuracy']:>9.2f}% "
    #           f"{r['f1_score']:>9.2f}% "
    #           f"{r['avg_inference_us']:>8.1f}μs "
    #           f"{r['min_inference_us']:>6.1f}μs "
    #           f"{r['max_inference_us']:>6.1f}μs")
    # print("=" * 70)


    # Test DT
    # bench.benchmark_model(
    #     X_test, y_test, 
    #     model_name='dt', 
    #     platform_name='Arduino Uno',
    #     num_samples=NUM_SAMPLES
    # )
    # Test RF
    bench.benchmark_model(
        X_test, y_test, 
        model_name='rf', 
        platform_name='Arduino Uno',
        num_samples=NUM_SAMPLES
    )
    # Test DT di Arduino nano
    # bench.benchmark_model(
    #     X_test, y_test, 
    #     model_name='dt', 
    #     platform_name='Arduino nano',
    #     num_samples=NUM_SAMPLES
    # )
    # Test RF di Arduino nano
    # bench.benchmark_model(
    #     X_test, y_test, 
    #     model_name='rf',
    #     platform_name='Arduino nano',
    #     num_samples=NUM_SAMPLES
    # )
    # Test DT di ESP32
    # bench.benchmark_model(
    #     X_test, y_test, 
    #     model_name='dt', 
    #     platform_name='ESP32',
    #     num_samples=NUM_SAMPLES
    # )
    # Test RF di ESP32
    # bench.benchmark_model(
    #     X_test, y_test,
    #     model_name='rf',
    #     platform_name='ESP32',
    #     num_samples=NUM_SAMPLES
    # )
    
    # 3. Ganti sketch ke USE_RANDOM_FOREST, upload ulang
    # 4. Jalankan lagi:
    # bench.benchmark_model(X_test, y_test, 'rf', 'Arduino Uno', NUM_SAMPLES)
    
    bench.save_results()
    bench.generate_comparison_table()
    bench.close()