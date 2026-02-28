// ============================================================
// DECISION TREE MODEL - MANUAL EXPORT (FIXED)
// Generated WITHOUT micromlgen (direct tree traversal)
// Depth: 5 | Nodes: 61
// return 0 = 16x | return 1 = 15x
// TANPA class_weight - leaf counts VALID
// ============================================================

#pragma once

namespace Eloquent {
    namespace ML {
        namespace Port {

            class DecisionTree {
            public:
                /**
                 * Predict class for features
                 * @param x: array of 10 float values (normalized 0-1)
                 * Features: src_bytes, service, dst_bytes, flag, same_srv_rate, diff_srv_rate, dst_host_srv_count, dst_host_same_srv_rate, logged_in, dst_host_serror_rate
                 * @return: 0 (Normal) or 1 (Attack)
                 */
                int predict(float *x) {
    if (x[3] <= 0.5500000119f) {
        if (x[7] <= 0.5749999881f) {
            if (x[4] <= 0.4950000048f) {
                if (x[2] <= 0.0000002088f) {
                    if (x[4] <= 0.2449999973f) {
                        // Leaf: Normal=0, Attack=1
                        return 1;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                } else {
                    if (x[4] <= 0.2900000066f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=1, Attack=0
                        return 0;
                    }
                }
            } else {
                if (x[2] <= 0.0000002878f) {
                    if (x[6] <= 0.0098039219f) {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                } else {
                    if (x[2] <= 0.0001281172f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                }
            }
        } else {
            if (x[3] <= 0.1500000022f) {
                if (x[1] <= 0.3188405782f) {
                    // Leaf: Normal=0, Attack=1
                    return 1;
                } else {
                    if (x[1] <= 0.6739130616f) {
                        // Leaf: Normal=1, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                }
            } else {
                if (x[6] <= 0.3862745166f) {
                    if (x[1] <= 0.1376811583f) {
                        // Leaf: Normal=1, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                } else {
                    if (x[0] <= 0.0000050835f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=1
                        return 1;
                    }
                }
            }
        }
    } else {
        if (x[8] <= 0.5000000000f) {
            if (x[1] <= 0.1811594218f) {
                if (x[7] <= 0.0349999992f) {
                    if (x[9] <= 0.0199999996f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=1
                        return 1;
                    }
                } else {
                    if (x[6] <= 0.0098039219f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=1, Attack=0
                        return 0;
                    }
                }
            } else {
                if (x[1] <= 0.2536231875f) {
                    if (x[5] <= 0.1649999991f) {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    }
                } else {
                    if (x[1] <= 0.6449275613f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                }
            }
        } else {
            if (x[0] <= 0.0000378419f) {
                if (x[1] <= 0.3260869533f) {
                    if (x[7] <= 0.9799999893f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                } else {
                    if (x[6] <= 0.0411764719f) {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    }
                }
            } else {
                if (x[7] <= 0.9900000095f) {
                    if (x[0] <= 0.0037062188f) {
                        // Leaf: Normal=1, Attack=0
                        return 0;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 0;
                    }
                } else {
                    if (x[0] <= 0.0000396416f) {
                        // Leaf: Normal=0, Attack=1
                        return 1;
                    } else {
                        // Leaf: Normal=0, Attack=0
                        return 1;
                    }
                }
            }
        }
    }
                }
            };

        }
    }
}
