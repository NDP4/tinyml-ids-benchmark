// ============================================================
// IDS (Intrusion Detection System) — Arduino Nano + DNN
// Model: DNN (TFLite INT8) via TensorFlow Lite Micro
// Features: 10 (top_10 MI ranking)
// ============================================================
//
// CATATAN: Arduino Nano menggunakan ATmega328P (sama dgn Uno)
// sehingga kode identik dengan versi Uno.
// File terpisah untuk konsistensi benchmarking per-platform.
// ============================================================

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dnn_model.h"       // DNN INT8 TFLite model
#include "scaler_params.h"   // Feature normalization

// ============================================================
// CONFIGURATION
// ============================================================

#define USE_DNN
// #define USE_CNN

#ifdef USE_CNN
#include "cnn_model.h"
#endif

#define SERIAL_BAUD 115200
#define LED_ALERT 13

constexpr int TENSOR_ARENA_SIZE = 3 * 1024;  // 3 KB

// ============================================================
// TFLITE MICRO GLOBALS
// ============================================================

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model* tflite_model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
tflite::AllOpsResolver resolver;

// ============================================================
// GLOBAL VARIABLES
// ============================================================

float raw_features[NUM_FEATURES];
float normalized_features[NUM_FEATURES];
unsigned long inference_start;
unsigned long inference_time;
unsigned long total_inferences = 0;
unsigned long total_attacks_detected = 0;
unsigned long total_normal = 0;

// ============================================================
// TFLITE INITIALIZATION
// ============================================================

bool init_model() {
    #ifdef USE_DNN
    tflite_model_ptr = tflite::GetModel(dnn_model);
    #endif
    #ifdef USE_CNN
    tflite_model_ptr = tflite::GetModel(cnn_model);
    #endif

    if (tflite_model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(F("ERROR: Model schema mismatch!"));
        return false;
    }

    static tflite::MicroInterpreter interp(
        tflite_model_ptr, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println(F("ERROR: AllocateTensors failed!"));
        return false;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    Serial.print(F("Arena used: "));
    Serial.print(interpreter->arena_used_bytes());
    Serial.println(F(" bytes"));

    return true;
}

// ============================================================
// INFERENCE
// ============================================================

int predict(float* features) {
    float input_scale = model_input->params.scale;
    int32_t input_zero_point = model_input->params.zero_point;

    for (int i = 0; i < NUM_FEATURES; i++) {
        int32_t quantized = (int32_t)(features[i] / input_scale) + input_zero_point;
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        model_input->data.int8[i] = (int8_t)quantized;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println(F("ERROR: Invoke failed!"));
        return -1;
    }

    float output_scale = model_output->params.scale;
    int32_t output_zero_point = model_output->params.zero_point;
    float output_val = (model_output->data.int8[0] - output_zero_point) * output_scale;

    return (output_val >= 0.5f) ? 1 : 0;
}

// ============================================================
// SETUP
// ============================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_ALERT, OUTPUT);
    digitalWrite(LED_ALERT, LOW);

    while (!Serial) { ; }

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════╗"));
    Serial.println(F("║  IDS — Intrusion Detection System     ║"));
    Serial.println(F("║  Platform: Arduino Nano               ║"));

    #ifdef USE_DNN
    Serial.println(F("║  Model: DNN (TFLite INT8)             ║"));
    #endif
    #ifdef USE_CNN
    Serial.println(F("║  Model: CNN (TFLite INT8)             ║"));
    #endif

    Serial.println(F("║  Features: 10 | Classes: 2            ║"));
    Serial.println(F("║  0=Normal | 1=Attack                  ║"));
    Serial.println(F("╚═══════════════════════════════════════╝"));
    Serial.println(F(""));

    Serial.println(F("Initializing TFLite Micro..."));
    bool ok = init_model();
    Serial.println(ok ? F("Model: OK ✅") : F("Model: FAILED ❌"));

    Serial.print(F("Free RAM: "));
    Serial.print(freeRam());
    Serial.println(F(" bytes"));
    Serial.println(F(""));
    Serial.println(F("Waiting for data via Serial..."));
    Serial.println(F("Format: f0,f1,f2,f3,f4,f5,f6,f7,f8,f9"));
    Serial.println(F(""));
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input.length() > 0) {
            if (parseFeatures(input, raw_features)) {
                normalize_all(raw_features, normalized_features);

                inference_start = micros();
                int prediction = predict(normalized_features);
                inference_time = micros() - inference_start;

                total_inferences++;
                if (prediction == 1) {
                    total_attacks_detected++;
                    digitalWrite(LED_ALERT, HIGH);
                } else {
                    total_normal++;
                    digitalWrite(LED_ALERT, LOW);
                }

                Serial.print(F("RESULT | Prediction: "));
                if (prediction == 1) {
                    Serial.print(F("⚠️ ATTACK"));
                } else {
                    Serial.print(F("✅ NORMAL"));
                }
                Serial.print(F(" | Inference: "));
                Serial.print(inference_time);
                Serial.print(F(" μs | Total: "));
                Serial.print(total_inferences);
                Serial.print(F(" | Attacks: "));
                Serial.print(total_attacks_detected);
                Serial.print(F(" | Normal: "));
                Serial.println(total_normal);
            }
        }
    }
}

// ============================================================
// HELPERS
// ============================================================

bool parseFeatures(String input, float* features) {
    int featureIdx = 0;
    int startIdx = 0;
    for (int i = 0; i <= (int)input.length(); i++) {
        if (i == (int)input.length() || input.charAt(i) == ',') {
            if (featureIdx < NUM_FEATURES) {
                features[featureIdx] = input.substring(startIdx, i).toFloat();
                featureIdx++;
            }
            startIdx = i + 1;
        }
    }
    if (featureIdx != NUM_FEATURES) {
        Serial.print(F("ERROR: Expected "));
        Serial.print(NUM_FEATURES);
        Serial.print(F(" features, got "));
        Serial.println(featureIdx);
        return false;
    }
    return true;
}

int freeRam() {
    extern int __heap_start, *__brkval;
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}
