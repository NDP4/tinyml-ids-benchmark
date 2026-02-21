// ============================================================
// IDS (Intrusion Detection System) — Arduino Uno/Nano
// Model: Decision Tree (depth=5, 61 nodes, F1: 82.89%)
// Features: 10 (top_10 MI ranking)
// ============================================================

#include "dt_model.h"       // Decision Tree model
#include "rf_model.h"       // Random Forest model
#include "scaler_params.h"  // Feature normalization

// ============================================================
// CONFIGURATION
// ============================================================

// Pilih model yang aktif (uncomment salah satu)
// #define USE_DECISION_TREE
#define USE_RANDOM_FOREST

#define SERIAL_BAUD 115200
#define LED_ALERT 13          // Built-in LED untuk alert

// Feature indices (sesuai urutan di scaler_params.h)
// [0] src_bytes
// [1] service
// [2] dst_bytes
// [3] flag
// [4] same_srv_rate
// [5] diff_srv_rate
// [6] dst_host_srv_count
// [7] dst_host_same_srv_rate
// [8] logged_in
// [9] dst_host_serror_rate

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
// MODEL PREDICTION FUNCTION
// ============================================================

int predict(float* features) {
    #ifdef USE_DECISION_TREE
        // micromlgen generates a predict() function in dt_model.h
        Eloquent::ML::Port::DecisionTree dt_classifier;
        return dt_classifier.predict(features);
    #endif

    #ifdef USE_RANDOM_FOREST
        Eloquent::ML::Port::RandomForest rf_classifier;
        return rf_classifier.predict(features);
    #endif
}

// ============================================================
// SETUP
// ============================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_ALERT, OUTPUT);
    digitalWrite(LED_ALERT, LOW);

    // Tunggu serial connection
    while (!Serial) { ; }

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════╗"));
    Serial.println(F("║  IDS — Intrusion Detection System     ║"));
    Serial.println(F("║  Platform: Arduino Uno/Nano           ║"));

    #ifdef USE_DECISION_TREE
    Serial.println(F("║  Model: Decision Tree (depth=5)       ║"));
    #endif
    #ifdef USE_RANDOM_FOREST
    Serial.println(F("║  Model: Random Forest (3 trees)       ║"));
    #endif

    Serial.println(F("║  Features: 10 | Classes: 2            ║"));
    Serial.println(F("║  0=Normal | 1=Attack                  ║"));
    Serial.println(F("╚═══════════════════════════════════════╝"));
    Serial.println(F(""));

    // Print memory info
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
    // Mode 1: Receive data via Serial (untuk testing)
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input.length() > 0) {
            // Parse CSV input
            if (parseFeatures(input, raw_features)) {

                // Normalize features
                normalize_all(raw_features, normalized_features);

                // Run inference
                inference_start = micros();
                int prediction = predict(normalized_features);
                inference_time = micros() - inference_start;

                // Update counters
                total_inferences++;
                if (prediction == 1) {
                    total_attacks_detected++;
                    digitalWrite(LED_ALERT, HIGH);  // LED ON for attack
                } else {
                    total_normal++;
                    digitalWrite(LED_ALERT, LOW);   // LED OFF for normal
                }

                // Output result
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
// HELPER FUNCTIONS
// ============================================================

// Parse comma-separated features from Serial input
bool parseFeatures(String input, float* features) {
    int featureIdx = 0;
    int startIdx = 0;

    for (int i = 0; i <= input.length(); i++) {
        if (i == input.length() || input.charAt(i) == ',') {
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

// Get free RAM (Arduino AVR)
int freeRam() {
    extern int __heap_start, *__brkval;
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}