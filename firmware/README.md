# Firmware Sketches

Arduino/ESP32 firmware for on-device IDS benchmarking.

## Usage

1. Copy the appropriate model header files from `models/` into the sketch folder
2. Open the `.ino` file in Arduino IDE
3. Select correct board and port
4. Compile and upload

## Platforms

- `esp32/` — ESP32 DevKit v1 (supports DT, RF, KNN)
- `arduino_uno/` — Arduino Uno R3 (supports DT, RF only)
- `arduino_nano/` — Arduino Nano (supports DT, RF only)

## Protocol

Each sketch:

1. Receives 10 raw feature values via serial (115,200 baud)
2. Normalizes features using embedded MinMaxScaler parameters
3. Runs inference
4. Reports prediction and inference time (microseconds)
