// ============================================================
// IDS (Intrusion Detection System) — ESP32
// Models: Decision Tree + Random Forest + KNN
// Features: 10 | WiFi-enabled
// ============================================================

#include <WiFi.h>
#include "dt_model.h"
#include "rf_model.h"
#include "knn_data.h"
#include "scaler_params.h"

// ============================================================
// CONFIGURATION
// ============================================================

#define SERIAL_BAUD 115200
#define LED_ALERT 2 // ESP32 built-in LED

// WiFi credentials (untuk monitoring via network)
const char *ssid = "Tech 2.4G";
const char *password = "12345678";

// ============================================================
// KNN IMPLEMENTATION (Custom for ESP32)
// ============================================================

// Manhattan distance
float manhattan_distance(const float *a, const float *b, int n)
{
    float dist = 0.0;
    for (int i = 0; i < n; i++)
    {
        dist += fabs(a[i] - b[i]);
    }
    return dist;
}

// KNN Predict
int knn_predict(float *features)
{
    // Array untuk menyimpan K nearest distances dan labels
    float nearest_dist[KNN_K];
    int nearest_labels[KNN_K];

    // Inisialisasi dengan jarak sangat besar
    for (int i = 0; i < KNN_K; i++)
    {
        nearest_dist[i] = 999999.0;
        nearest_labels[i] = -1;
    }

    // Cari K nearest neighbors
    for (int i = 0; i < KNN_SAMPLES; i++)
    {
        float dist = manhattan_distance(features, knn_train_X[i], KNN_FEATURES);

        // Cek apakah lebih dekat dari yang ada di nearest
        int max_idx = 0;
        for (int j = 1; j < KNN_K; j++)
        {
            if (nearest_dist[j] > nearest_dist[max_idx])
            {
                max_idx = j;
            }
        }

        if (dist < nearest_dist[max_idx])
        {
            nearest_dist[max_idx] = dist;
            nearest_labels[max_idx] = knn_train_y[i];
        }
    }

    // Weighted voting (distance-weighted)
    float weight_normal = 0.0;
    float weight_attack = 0.0;

    for (int i = 0; i < KNN_K; i++)
    {
        float weight = 1.0 / (nearest_dist[i] + 0.0001); // prevent div by 0
        if (nearest_labels[i] == 0)
        {
            weight_normal += weight;
        }
        else
        {
            weight_attack += weight;
        }
    }

    return (weight_attack > weight_normal) ? 1 : 0;
}

// ============================================================
// BENCHMARK FUNCTION
// ============================================================

struct BenchmarkResult
{
    float accuracy;
    float avg_inference_us;
    unsigned long total_time_ms;
    int correct;
    int total;
    int true_positives;
    int false_positives;
    int true_negatives;
    int false_negatives;
};

// Run benchmark on test samples received via Serial
void runBenchmark(int model_type)
{
    // model_type: 0=DT, 1=RF, 2=KNN

    const char *model_names[] = {"Decision Tree", "Random Forest", "KNN"};

    Serial.println(F(""));
    Serial.print(F("🧪 BENCHMARKING: "));
    Serial.println(model_names[model_type]);
    Serial.println(F("═══════════════════════════════════════"));

    Serial.println(F("Send test data (CSV format), "
                     "then send 'END' when done."));
    Serial.println(F("Format per line: f0,f1,...,f9,true_label"));

    int correct = 0;
    int total = 0;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    unsigned long total_inference_us = 0;

    while (true)
    {
        if (Serial.available() > 0)
        {
            String line = Serial.readStringUntil('\n');
            line.trim();

            if (line == "END" || line == "end")
                break;

            // Parse: 10 features + 1 true label
            float features[NUM_FEATURES];
            float normalized[NUM_FEATURES];
            int true_label = -1;

            int idx = 0;
            int startPos = 0;
            for (int i = 0; i <= line.length(); i++)
            {
                if (i == line.length() || line.charAt(i) == ',')
                {
                    if (idx < NUM_FEATURES)
                    {
                        features[idx] = line.substring(startPos, i).toFloat();
                    }
                    else if (idx == NUM_FEATURES)
                    {
                        true_label = line.substring(startPos, i).toInt();
                    }
                    idx++;
                    startPos = i + 1;
                }
            }

            if (idx < NUM_FEATURES + 1)
                continue;

            // Normalize
            normalize_all(features, normalized);

            // Predict
            unsigned long start_us = micros();
            int prediction;

            switch (model_type)
            {
            case 0:
            {
                Eloquent::ML::Port::DecisionTree clf;
                prediction = clf.predict(normalized);
                break;
            }
            case 1:
            {
                Eloquent::ML::Port::RandomForest clf;
                prediction = clf.predict(normalized);
                break;
            }
            case 2:
                prediction = knn_predict(normalized);
                break;
            }

            unsigned long inference_us = micros() - start_us;
            total_inference_us += inference_us;

            // Evaluate
            if (prediction == true_label)
                correct++;
            if (prediction == 1 && true_label == 1)
                tp++;
            if (prediction == 1 && true_label == 0)
                fp++;
            if (prediction == 0 && true_label == 0)
                tn++;
            if (prediction == 0 && true_label == 1)
                fn++;
            total++;

            // Print per-sample result
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
    float f1_val = (precision_val + recall_val > 0) ? 2 * precision_val * recall_val /
                                                          (precision_val + recall_val)
                                                    : 0.0;

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════╗"));
    Serial.println(F("║       BENCHMARK RESULTS               ║"));
    Serial.println(F("╠═══════════════════════════════════════╣"));
    Serial.print(F("║ Model     : "));
    Serial.println(model_names[model_type]);
    Serial.print(F("║ Samples   : "));
    Serial.println(total);
    Serial.print(F("║ Accuracy  : "));
    Serial.print(accuracy, 2);
    Serial.println(F("%"));
    Serial.print(F("║ Precision : "));
    Serial.print(precision_val, 2);
    Serial.println(F("%"));
    Serial.print(F("║ Recall    : "));
    Serial.print(recall_val, 2);
    Serial.println(F("%"));
    Serial.print(F("║ F1-Score  : "));
    Serial.print(f1_val, 2);
    Serial.println(F("%"));
    Serial.print(F("║ Avg Inf.  : "));
    Serial.print(avg_inference, 1);
    Serial.println(F(" μs"));
    Serial.print(F("║ TP: "));
    Serial.print(tp);
    Serial.print(F(" | FP: "));
    Serial.print(fp);
    Serial.print(F(" | TN: "));
    Serial.print(tn);
    Serial.print(F(" | FN: "));
    Serial.println(fn);
    Serial.println(F("╚═══════════════════════════════════════╝"));

    // Print free heap
    Serial.print(F("Free Heap: "));
    Serial.print(ESP.getFreeHeap());
    Serial.println(F(" bytes"));
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_ALERT, OUTPUT);

    while (!Serial)
    {
        ;
    }

    Serial.println(F(""));
    Serial.println(F("╔═══════════════════════════════════════════╗"));
    Serial.println(F("║  IDS — Intrusion Detection System         ║"));
    Serial.println(F("║  Platform: ESP32                          ║"));
    Serial.println(F("║  Models: DT + RF + KNN                    ║"));
    Serial.println(F("║  Features: 10 | WiFi-Enabled              ║"));
    Serial.println(F("╚═══════════════════════════════════════════╝"));

    // Print memory
    Serial.print(F("Free Heap: "));
    Serial.print(ESP.getFreeHeap());
    Serial.println(F(" bytes"));

    Serial.print(F("Flash Size: "));
    Serial.print(ESP.getFlashChipSize() / 1024);
    Serial.println(F(" KB"));

    Serial.println(F(""));
    Serial.println(F("COMMANDS:"));
    Serial.println(F("  'bench_dt'  — Benchmark Decision Tree"));
    Serial.println(F("  'bench_rf'  — Benchmark Random Forest"));
    Serial.println(F("  'bench_knn' — Benchmark KNN"));
    Serial.println(F("  'predict'   — Single prediction mode"));
    Serial.println(F("  'info'      — System info"));
    Serial.println(F(""));
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    if (Serial.available() > 0)
    {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "bench_dt")
        {
            runBenchmark(0);
        }
        else if (cmd == "bench_rf")
        {
            runBenchmark(1);
        }
        else if (cmd == "bench_knn")
        {
            runBenchmark(2);
        }
        else if (cmd == "info")
        {
            Serial.print(F("Free Heap: "));
            Serial.print(ESP.getFreeHeap());
            Serial.println(F(" bytes"));
            Serial.print(F("CPU Freq: "));
            Serial.print(ESP.getCpuFreqMHz());
            Serial.println(F(" MHz"));
        }
        else
        {
            // Single prediction: parse CSV features
            float features[NUM_FEATURES];
            float normalized[NUM_FEATURES];

            if (parseFeatures(cmd, features, NUM_FEATURES))
            {
                normalize_all(features, normalized);

                // Predict with all three models
                Serial.println(F(""));

                unsigned long t;

                // DT
                t = micros();
                Eloquent::ML::Port::DecisionTree dt_clf;
                int pred_dt = dt_clf.predict(normalized);
                unsigned long dt_time = micros() - t;

                // RF
                t = micros();
                Eloquent::ML::Port::RandomForest rf_clf;
                int pred_rf = rf_clf.predict(normalized);
                unsigned long rf_time = micros() - t;

                // KNN
                t = micros();
                int pred_knn = knn_predict(normalized);
                unsigned long knn_time = micros() - t;

                // Output
                Serial.println(F("┌─────────────┬────────┬──────────┐"));
                Serial.println(F("│ Model       │ Result │ Time     │"));
                Serial.println(F("├─────────────┼────────┼──────────┤"));

                Serial.print(F("│ DT          │ "));
                Serial.print(pred_dt == 1 ? "ATTACK" : "NORMAL");
                Serial.print(F(" │ "));
                Serial.print(dt_time);
                Serial.println(F(" μs  │"));

                Serial.print(F("│ RF          │ "));
                Serial.print(pred_rf == 1 ? "ATTACK" : "NORMAL");
                Serial.print(F(" │ "));
                Serial.print(rf_time);
                Serial.println(F(" μs  │"));

                Serial.print(F("│ KNN         │ "));
                Serial.print(pred_knn == 1 ? "ATTACK" : "NORMAL");
                Serial.print(F(" │ "));
                Serial.print(knn_time);
                Serial.println(F(" μs  │"));

                Serial.println(F("└─────────────┴────────┴──────────┘"));

                // Majority voting
                int votes = pred_dt + pred_rf + pred_knn;
                Serial.print(F("ENSEMBLE (majority): "));
                Serial.println(votes >= 2 ? "⚠️ ATTACK" : "✅ NORMAL");

                // LED alert
                digitalWrite(LED_ALERT, votes >= 2 ? HIGH : LOW);
            }
        }
    }
}

// Parse features helper
bool parseFeatures(String input, float *features, int expected)
{
    int idx = 0;
    int startPos = 0;
    for (int i = 0; i <= input.length(); i++)
    {
        if (i == input.length() || input.charAt(i) == ',')
        {
            if (idx < expected)
            {
                features[idx] = input.substring(startPos, i).toFloat();
                idx++;
            }
            startPos = i + 1;
        }
    }
    return (idx == expected);
}