// ============================================================
// IDS (Intrusion Detection System) — ESP32 + DNN/CNN
// Models: DNN (TFLite INT8) + CNN (TFLite INT8)
// Framework: TensorFlow Lite Micro
// Features: 10 | WiFi-enabled
// ============================================================

#include <WiFi.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dnn_model.h"      // DNN INT8 TFLite model
#include "cnn_model.h"      // CNN INT8 TFLite model
#include "scaler_params.h"  // Feature normalization (sama dgn DT/RF/KNN)

// ============================================================
// CONFIGURATION
// ============================================================

#define SERIAL_BAUD 115200
#define LED_ALERT 2   // ESP32 built-in LED

// TFLite Micro arena size (bytes)
// DNN: ~10KB arena, CNN: ~15KB arena (sesuaikan jika error)
constexpr int DNN_ARENA_SIZE = 16 * 1024;   // 16 KB
constexpr int CNN_ARENA_SIZE = 24 * 1024;   // 24 KB

// WiFi credentials
const char *ssid = "Tech 2.4G";
const char *password = "12345678";

// ============================================================
// TFLITE MICRO GLOBALS — DNN
// ============================================================

uint8_t dnn_tensor_arena[DNN_ARENA_SIZE];
const tflite::Model* dnn_tflite_model = nullptr;
tflite::MicroInterpreter* dnn_interpreter = nullptr;
TfLiteTensor* dnn_input = nullptr;
TfLiteTensor* dnn_output = nullptr;

// ============================================================
// TFLITE MICRO GLOBALS — CNN
// ============================================================

uint8_t cnn_tensor_arena[CNN_ARENA_SIZE];
const tflite::Model* cnn_tflite_model = nullptr;
tflite::MicroInterpreter* cnn_interpreter = nullptr;
TfLiteTensor* cnn_input = nullptr;
TfLiteTensor* cnn_output = nullptr;

// Ops resolver (shared)
tflite::AllOpsResolver resolver;

// ============================================================
// TFLITE MODEL INITIALIZATION
// ============================================================

bool init_dnn_model() {
    dnn_tflite_model = tflite::GetModel(dnn_model);
    if (dnn_tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(F("ERROR: DNN model schema version mismatch!"));
        return false;
    }

    static tflite::MicroInterpreter dnn_interp(
        dnn_tflite_model, resolver, dnn_tensor_arena, DNN_ARENA_SIZE);
    dnn_interpreter = &dnn_interp;

    if (dnn_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println(F("ERROR: DNN AllocateTensors failed!"));
        return false;
    }

    dnn_input = dnn_interpreter->input(0);
    dnn_output = dnn_interpreter->output(0);

    Serial.print(F("DNN Input:  type="));
    Serial.print(dnn_input->type);
    Serial.print(F(" shape=["));
    for (int i = 0; i < dnn_input->dims->size; i++) {
        Serial.print(dnn_input->dims->data[i]);
        if (i < dnn_input->dims->size - 1) Serial.print(",");
    }
    Serial.println(F("]"));

    Serial.print(F("DNN Arena used: "));
    Serial.print(dnn_interpreter->arena_used_bytes());
    Serial.println(F(" bytes"));

    return true;
}

bool init_cnn_model() {
    cnn_tflite_model = tflite::GetModel(cnn_model);
    if (cnn_tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(F("ERROR: CNN model schema version mismatch!"));
        return false;
    }

    static tflite::MicroInterpreter cnn_interp(
        cnn_tflite_model, resolver, cnn_tensor_arena, CNN_ARENA_SIZE);
    cnn_interpreter = &cnn_interp;

    if (cnn_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println(F("ERROR: CNN AllocateTensors failed!"));
        return false;
    }

    cnn_input = cnn_interpreter->input(0);
    cnn_output = cnn_interpreter->output(0);

    Serial.print(F("CNN Input:  type="));
    Serial.print(cnn_input->type);
    Serial.print(F(" shape=["));
    for (int i = 0; i < cnn_input->dims->size; i++) {
        Serial.print(cnn_input->dims->data[i]);
        if (i < cnn_input->dims->size - 1) Serial.print(",");
    }
    Serial.println(F("]"));

    Serial.print(F("CNN Arena used: "));
    Serial.print(cnn_interpreter->arena_used_bytes());
    Serial.println(F(" bytes"));

    return true;
}

// ============================================================
// INFERENCE FUNCTIONS
// ============================================================

int dnn_predict(float *normalized_features) {
    // Set input: quantize float → int8
    float input_scale = dnn_input->params.scale;
    int32_t input_zero_point = dnn_input->params.zero_point;

    for (int i = 0; i < NUM_FEATURES; i++) {
        int32_t quantized = (int32_t)(normalized_features[i] / input_scale) + input_zero_point;
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        dnn_input->data.int8[i] = (int8_t)quantized;
    }

    // Run inference
    if (dnn_interpreter->Invoke() != kTfLiteOk) {
        Serial.println(F("ERROR: DNN Invoke failed!"));
        return -1;
    }

    // Dequantize output
    float output_scale = dnn_output->params.scale;
    int32_t output_zero_point = dnn_output->params.zero_point;
    float output_val = (dnn_output->data.int8[0] - output_zero_point) * output_scale;

    return (output_val >= 0.5f) ? 1 : 0;
}

int cnn_predict(float *normalized_features) {
    // CNN input shape: [1, 10, 1] — reshape features
    float input_scale = cnn_input->params.scale;
    int32_t input_zero_point = cnn_input->params.zero_point;

    for (int i = 0; i < NUM_FEATURES; i++) {
        int32_t quantized = (int32_t)(normalized_features[i] / input_scale) + input_zero_point;
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        cnn_input->data.int8[i] = (int8_t)quantized;
    }

    // Run inference
    if (cnn_interpreter->Invoke() != kTfLiteOk) {
        Serial.println(F("ERROR: CNN Invoke failed!"));
        return -1;
    }

    // Dequantize output
    float output_scale = cnn_output->params.scale;
    int32_t output_zero_point = cnn_output->params.zero_point;
    float output_val = (cnn_output->data.int8[0] - output_zero_point) * output_scale;

    return (output_val >= 0.5f) ? 1 : 0;
}

// ============================================================
// BENCHMARK FUNCTION (kompatibel dgn run_benchmark.py)
// ============================================================

void runBenchmark(int model_type) {
    // model_type: 0=DNN, 1=CNN
    const char *model_names[] = {"DNN", "CNN"};

    Serial.println(F(""));
    Serial.print(F("🧪 BENCHMARKING: "));
    Serial.println(model_names[model_type]);
    Serial.println(F("═══════════════════════════════════════"));
    Serial.println(F("Send test data, then 'END' when done."));
    Serial.println(F("Format per line: f0,f1,...,f9,true_label"));

    int correct = 0;
    int total = 0;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    unsigned long total_inference_us = 0;

    while (true) {
        if (Serial.available() > 0) {
            String line = Serial.readStringUntil('\n');
            line.trim();

            if (line == "END" || line == "end") break;

            // Parse: 10 features + 1 true label
            float features[NUM_FEATURES];
            float normalized[NUM_FEATURES];
            int true_label = -1;

            int idx = 0;
            int startPos = 0;
            for (int i = 0; i <= (int)line.length(); i++) {
                if (i == (int)line.length() || line.charAt(i) == ',') {
                    if (idx < NUM_FEATURES) {
                        features[idx] = line.substring(startPos, i).toFloat();
                    } else if (idx == NUM_FEATURES) {
                        true_label = line.substring(startPos, i).toInt();
                    }
                    idx++;
                    startPos = i + 1;
                }
            }

            if (idx < NUM_FEATURES + 1) continue;

            // Normalize (sama dgn DT/RF/KNN)
            normalize_all(features, normalized);

            // Predict
            unsigned long start_us = micros();
            int prediction;

            switch (model_type) {
                case 0:
                    prediction = dnn_predict(normalized);
                    break;
                case 1:
                    prediction = cnn_predict(normalized);
                    break;
                default:
                    prediction = -1;
            }

            unsigned long inference_us = micros() - start_us;
            total_inference_us += inference_us;

            // Evaluate
            if (prediction == true_label) correct++;
            if (prediction == 1 && true_label == 1) tp++;
            if (prediction == 1 && true_label == 0) fp++;
            if (prediction == 0 && true_label == 0) tn++;
            if (prediction == 0 && true_label == 1) fn++;
            total++;

            // Print per-sample (format kompatibel dgn run_benchmark.py parser)
            Serial.print(F("["));
            Serial.print(total);
            Serial.print(F("] Pred: "));
            Serial.print(prediction);
            Serial.print(F(" | True: "));
            Serial.print(true_label);
            Serial.print(F(" | "));
            Serial.print(prediction == true_label ? "✅" : "❌");
            Serial.print(F(" | "));
            Serial.print(inference_us);
            Serial.println(F(" μs"));
        }
    }

    // Print summary
    float accuracy = (total > 0) ? (float)correct / total * 100.0 : 0.0;
    float avg_inference = (total > 0) ? (float)total_inference_us / total : 0.0;
    float precision_val = (tp + fp > 0) ? (float)tp / (tp + fp) * 100.0 : 0.0;
    float recall_val = (tp + fn > 0) ? (float)tp / (tp + fn) * 100.0 : 0.0;
    float f1_val = (precision_val + recall_val > 0)
        ? 2 * precision_val * recall_val / (precision_val + recall_val) : 0.0;

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════╗"));
    Serial.println(F("║       BENCHMARK RESULTS               ║"));
    Serial.println(F("╠═══════════════════════════════════════╣"));
    Serial.print(F("║ Model     : ")); Serial.println(model_names[model_type]);
    Serial.print(F("║ Samples   : ")); Serial.println(total);
    Serial.print(F("║ Accuracy  : ")); Serial.print(accuracy, 2); Serial.println(F("%"));
    Serial.print(F("║ Precision : ")); Serial.print(precision_val, 2); Serial.println(F("%"));
    Serial.print(F("║ Recall    : ")); Serial.print(recall_val, 2); Serial.println(F("%"));
    Serial.print(F("║ F1-Score  : ")); Serial.print(f1_val, 2); Serial.println(F("%"));
    Serial.print(F("║ Avg Inf.  : ")); Serial.print(avg_inference, 1); Serial.println(F(" μs"));
    Serial.print(F("║ TP: ")); Serial.print(tp);
    Serial.print(F(" | FP: ")); Serial.print(fp);
    Serial.print(F(" | TN: ")); Serial.print(tn);
    Serial.print(F(" | FN: ")); Serial.println(fn);
    Serial.println(F("╚═══════════════════════════════════════╝"));

    Serial.print(F("Free Heap: "));
    Serial.print(ESP.getFreeHeap());
    Serial.println(F(" bytes"));
}

// ============================================================
// SETUP
// ============================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_ALERT, OUTPUT);

    while (!Serial) { ; }

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════════╗"));
    Serial.println(F("║  IDS — Intrusion Detection System         ║"));
    Serial.println(F("║  Platform: ESP32 + TFLite Micro           ║"));
    Serial.println(F("║  Models: DNN (INT8) + CNN (INT8)          ║"));
    Serial.println(F("║  Features: 10 | WiFi-Enabled              ║"));
    Serial.println(F("╚═══════════════════════════════════════════╝"));

    // Memory info
    Serial.print(F("Free Heap: "));
    Serial.print(ESP.getFreeHeap());
    Serial.println(F(" bytes"));
    Serial.print(F("Flash Size: "));
    Serial.print(ESP.getFlashChipSize() / 1024);
    Serial.println(F(" KB"));

    // Initialize models
    Serial.println(F("\nInitializing DNN model..."));
    bool dnn_ok = init_dnn_model();
    Serial.println(dnn_ok ? F("  DNN: OK ✅") : F("  DNN: FAILED ❌"));

    Serial.println(F("Initializing CNN model..."));
    bool cnn_ok = init_cnn_model();
    Serial.println(cnn_ok ? F("  CNN: OK ✅") : F("  CNN: FAILED ❌"));

    Serial.print(F("\nFree Heap after init: "));
    Serial.print(ESP.getFreeHeap());
    Serial.println(F(" bytes"));

    Serial.println(F(""));
    Serial.println(F("COMMANDS:"));
    Serial.println(F("  'bench_dnn' — Benchmark DNN model"));
    Serial.println(F("  'bench_cnn' — Benchmark CNN model"));
    Serial.println(F("  'predict'   — Single prediction (both models)"));
    Serial.println(F("  'info'      — System info"));
    Serial.println(F(""));
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "bench_dnn") {
            runBenchmark(0);
        }
        else if (cmd == "bench_cnn") {
            runBenchmark(1);
        }
        else if (cmd == "info") {
            Serial.print(F("Free Heap: "));
            Serial.print(ESP.getFreeHeap());
            Serial.println(F(" bytes"));
            Serial.print(F("CPU Freq: "));
            Serial.print(ESP.getCpuFreqMHz());
            Serial.println(F(" MHz"));
            Serial.print(F("DNN model size: "));
            Serial.print(dnn_model_len);
            Serial.println(F(" bytes"));
            Serial.print(F("CNN model size: "));
            Serial.print(cnn_model_len);
            Serial.println(F(" bytes"));
        }
        else {
            // Single prediction: parse CSV features
            float features[NUM_FEATURES];
            float normalized[NUM_FEATURES];

            if (parseFeatures(cmd, features, NUM_FEATURES)) {
                normalize_all(features, normalized);

                Serial.println(F(""));
                unsigned long t;

                // DNN
                t = micros();
                int pred_dnn = dnn_predict(normalized);
                unsigned long dnn_time = micros() - t;

                // CNN
                t = micros();
                int pred_cnn = cnn_predict(normalized);
                unsigned long cnn_time = micros() - t;

                // Output
                Serial.println(F("┌─────────────┬────────┬──────────┐"));
                Serial.println(F("│ Model       │ Result │ Time     │"));
                Serial.println(F("├─────────────┼────────┼──────────┤"));

                Serial.print(F("│ DNN         │ "));
                Serial.print(pred_dnn == 1 ? "ATTACK" : "NORMAL");
                Serial.print(F(" │ "));
                Serial.print(dnn_time);
                Serial.println(F(" μs  │"));

                Serial.print(F("│ CNN         │ "));
                Serial.print(pred_cnn == 1 ? "ATTACK" : "NORMAL");
                Serial.print(F(" │ "));
                Serial.print(cnn_time);
                Serial.println(F(" μs  │"));

                Serial.println(F("└─────────────┴────────┴──────────┘"));

                // Ensemble
                int votes = pred_dnn + pred_cnn;
                Serial.print(F("ENSEMBLE (DNN+CNN): "));
                Serial.println(votes >= 2 ? "⚠️ ATTACK" : (votes == 1 ? "⚠️ UNCERTAIN" : "✅ NORMAL"));

                digitalWrite(LED_ALERT, votes >= 1 ? HIGH : LOW);
            }
        }
    }
}

// Parse features helper
bool parseFeatures(String input, float *features, int expected) {
    int idx = 0;
    int startPos = 0;
    for (int i = 0; i <= (int)input.length(); i++) {
        if (i == (int)input.length() || input.charAt(i) == ',') {
            if (idx < expected) {
                features[idx] = input.substring(startPos, i).toFloat();
                idx++;
            }
            startPos = i + 1;
        }
    }
    return (idx == expected);
}
