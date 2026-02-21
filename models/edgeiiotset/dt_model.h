#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= 0.0714285746216774) {
                            return 0;
                        }

                        else {
                            if (x[1] <= 0.25) {
                                return 0;
                            }

                            else {
                                return 1;
                            }
                        }
                    }

                protected:
                };
            }
        }
    }