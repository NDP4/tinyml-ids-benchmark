// ============================================
// MinMaxScaler Parameters — Edge-IIoTset
// Features: 10 (top_10 MI ranking)
// Formula: X_scaled = (X - min) / (max - min)
// ============================================

#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define NUM_FEATURES 10

// Feature order:
// [0] dns.qry.name.len
// [1] mqtt.conack.flags
// [2] tcp.dstport
// [3] tcp.seq
// [4] tcp.ack
// [5] tcp.len
// [6] tcp.flags
// [7] http.request.version
// [8] http.request.method
// [9] tcp.ack_raw

const float feature_min[NUM_FEATURES] = {
  0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2.000000, 0.000000, 0.000000
};

const float feature_max[NUM_FEATURES] = {
  7.000000, 2.000000, 65535.000000, 207964705.000000, 2147333291.000000, 65228.000000, 25.000000, 7.000000, 5.000000, 4294731662.000000
};

const float feature_range[NUM_FEATURES] = {
  7.000000, 2.000000, 65535.000000, 207964705.000000, 2147333291.000000, 65228.000000, 25.000000, 5.000000, 5.000000, 4294731662.000000
};

// Normalize a single feature value
float normalize_feature(float value, int feature_idx) {
  float scaled = (value - feature_min[feature_idx]) / feature_range[feature_idx];
  if (scaled < 0.0) scaled = 0.0;
  if (scaled > 1.0) scaled = 1.0;
  return scaled;
}

// Normalize entire feature vector
void normalize_all(float* features, float* normalized) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    normalized[i] = normalize_feature(features[i], i);
  }
}

#endif
