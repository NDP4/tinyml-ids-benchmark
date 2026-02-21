# ============================================================
# send_test_data.py
# Mengirim test data ke Arduino/ESP32 via Serial untuk benchmark
# ============================================================

import serial
import time
import numpy as np
import sys

# Configuration
# SERIAL_PORT = '/dev/ttyUSB0'  # Sesuaikan! (Linux)
# SERIAL_PORT = 'COM3'        # windows port arduino uno
SERIAL_PORT = 'COM4'        # windows port esp32
# SERIAL_PORT = 'COM9'        # windows port arduino nano
BAUD_RATE = 115200
NUM_SAMPLES = 100  # Jumlah test samples

def main():
    # Load test data
    X_test = np.loadtxt('test_samples.csv', delimiter=',')
    y_test = np.loadtxt('test_labels.csv', delimiter=',')

    print(f"📊 Loaded {len(X_test)} test samples")
    print(f"📡 Connecting to {SERIAL_PORT}...")

    # Connect to Arduino/ESP32
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)  # Wait for Arduino reset

    # Read startup messages
    while ser.in_waiting:
        print(ser.readline().decode('utf-8', errors='replace').strip())

    # Determine model to benchmark
    model = sys.argv[1] if len(sys.argv) > 1 else 'dt'
    model_cmd = f'bench_{model}'

    print(f"\n🧪 Starting benchmark: {model_cmd}")
    ser.write(f'{model_cmd}\n'.encode())
    time.sleep(1)

    # Read prompt
    while ser.in_waiting:
        print(ser.readline().decode('utf-8', errors='replace').strip())

    # Send test samples
    samples_to_send = min(NUM_SAMPLES, len(X_test))
    print(f"\n📤 Sending {samples_to_send} samples...")

    for i in range(samples_to_send):
        # Format: f0,f1,...,f9,true_label
        features_str = ','.join([f'{v:.6f}' for v in X_test[i]])
        line = f'{features_str},{int(y_test[i])}\n'
        ser.write(line.encode())
        time.sleep(0.05)  # Small delay between samples

        # Read response
        while ser.in_waiting:
            response = ser.readline().decode('utf-8', errors='replace').strip()
            print(response)

    # Send END command
    time.sleep(0.5)
    ser.write(b'END\n')
    time.sleep(1)

    # Read benchmark results
    print("\n📊 BENCHMARK RESULTS:")
    while ser.in_waiting:
        print(ser.readline().decode('utf-8', errors='replace').strip())

    time.sleep(1)
    while ser.in_waiting:
        print(ser.readline().decode('utf-8', errors='replace').strip())

    ser.close()
    print("\n✅ Benchmark complete!")

if __name__ == '__main__':
    main()