// ============================================
// MinMaxScaler Parameters
// Features: 10 (top_10 MI ranking)
// Formula: X_scaled = (X - min) / (max - min)
// ============================================

#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define NUM_FEATURES 10

// Feature order:
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

const float feature_min[NUM_FEATURES] = {
  0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
};

const float feature_max[NUM_FEATURES] = {
  1379963888.000000, 69.000000, 1309937401.000000, 10.000000, 1.000000, 1.000000, 255.000000, 1.000000, 1.000000, 1.000000
};

const float feature_range[NUM_FEATURES] = {
  1379963888.000000, 69.000000, 1309937401.000000, 10.000000, 1.000000, 1.000000, 255.000000, 1.000000, 1.000000, 1.000000
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
