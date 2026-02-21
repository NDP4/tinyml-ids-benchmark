// ============================================
// Random Forest Model — Auto-generated
// Trees: 3 | Depth: 5 | Features: 10
// Optimized for Arduino Uno/Nano/ESP32
// ============================================

#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class RandomForest {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        uint8_t votes[2] = { 0 };
                        // tree #1
                        if (x[4] <= 0.4950000047683716) {
                            if (x[8] <= 0.5) {
                                if (x[7] <= 0.7350000143051147) {
                                    if (x[9] <= 0.014999999664723873) {
                                        if (x[5] <= 0.10500000044703484) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[4] <= 0.32500000298023224) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[6] <= 0.08627451211214066) {
                                        if (x[7] <= 0.9749999940395355) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[4] <= 0.044999999925494194) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[7] <= 0.004999999888241291) {
                                    if (x[9] <= 0.024999999441206455) {
                                        if (x[5] <= 0.9399999976158142) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[4] <= 0.004999999888241291) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[3] <= 0.30000000447034836) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        if (x[0] <= 3.438495781438178e-07) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        else {
                            if (x[9] <= 0.9050000011920929) {
                                if (x[6] <= 0.8411764800548553) {
                                    if (x[9] <= 0.36500000953674316) {
                                        if (x[8] <= 0.5) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[7] <= 0.03499999921768904) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[8] <= 0.5) {
                                        if (x[6] <= 0.9980392158031464) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[0] <= 1.115319037126028e-05) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 5.610955255974659e-08) {
                                    if (x[1] <= 0.36231884360313416) {
                                        if (x[8] <= 0.5) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[8] <= 0.5) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[6] <= 0.005882353289052844) {
                                        if (x[2] <= 1.7573358661593375e-06) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        votes[0] += 1;
                                    }
                                }
                            }
                        }

                        // tree #2
                        if (x[3] <= 0.550000011920929) {
                            if (x[4] <= 0.9799999892711639) {
                                if (x[4] <= 0.4950000047683716) {
                                    if (x[5] <= 0.019999999552965164) {
                                        if (x[7] <= 0.3700000047683716) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[8] <= 0.5) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[1] <= 0.021739130839705467) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        if (x[0] <= 1.9343983836961343e-05) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[3] <= 0.2500000074505806) {
                                    if (x[1] <= 0.6739130616188049) {
                                        if (x[9] <= 0.014999999664723873) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[1] <= 0.7681159377098083) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[6] <= 0.6058823764324188) {
                                        if (x[2] <= 1.3718212699131982e-06) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[0] <= 5.707395658305359e-06) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        else {
                            if (x[6] <= 0.8686274588108063) {
                                if (x[7] <= 0.9950000047683716) {
                                    if (x[1] <= 0.7753623127937317) {
                                        if (x[1] <= 0.6884058117866516) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[1] <= 0.9492753744125366) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[8] <= 0.5) {
                                        if (x[1] <= 0.2391304299235344) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[9] <= 0.024999999441206455) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[7] <= 0.35999999940395355) {
                                    votes[1] += 1;
                                }

                                else {
                                    if (x[1] <= 0.2826086953282356) {
                                        if (x[1] <= 0.18840579688549042) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[6] <= 0.9392156898975372) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        // tree #3
                        if (x[5] <= 0.03499999921768904) {
                            if (x[1] <= 0.25362318754196167) {
                                if (x[1] <= 0.18115942180156708) {
                                    if (x[3] <= 0.550000011920929) {
                                        if (x[7] <= 0.03499999921768904) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[6] <= 0.009803921915590763) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[6] <= 0.01764705963432789) {
                                        if (x[7] <= 0.16499999910593033) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[7] <= 0.9650000035762787) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[1] <= 0.36231884360313416) {
                                    if (x[7] <= 0.5649999976158142) {
                                        if (x[9] <= 0.9950000047683716) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[2] <= 0.0033957924460992217) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[3] <= 0.550000011920929) {
                                        if (x[1] <= 0.7753623127937317) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[1] <= 0.717391312122345) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        else {
                            if (x[7] <= 0.33500000834465027) {
                                if (x[3] <= 0.550000011920929) {
                                    if (x[0] <= 1.5942445941163896e-07) {
                                        if (x[4] <= 0.4950000047683716) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[0] <= 2.5131091661023675e-06) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[5] <= 0.32500000298023224) {
                                        if (x[8] <= 0.5) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[9] <= 0.024999999441206455) {
                                            votes[0] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[3] <= 0.550000011920929) {
                                    if (x[4] <= 0.39499999582767487) {
                                        if (x[7] <= 0.8449999988079071) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        if (x[7] <= 0.75) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[9] <= 0.07000000216066837) {
                                        if (x[5] <= 0.0950000025331974) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }

                                    else {
                                        if (x[2] <= 1.8665014067664742e-07) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[0] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 2; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                protected:
                };
            }
        }
    }