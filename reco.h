#ifndef RECO_H
#define RECO_H

/* 动作识别：静止 / 走路 / 跑步 / 跌倒。 */

#include "model.h"

#define RECO_WIN_SAMPLES   100        /* 1 秒窗口 @100Hz */
#define RECO_FEATURES      MODEL_N_FEATURES
#define RECO_CLASSES       MODEL_N_CLASSES

#define RECO_CLASS_STILL   0
#define RECO_CLASS_WALK    1
#define RECO_CLASS_RUN     2
#define RECO_CLASS_FALL    3

void reco_features(const float acc[][3], const float gyro[][3], int n, float *feat);

int reco_predict(const float *feat, float *proba);

#endif /* RECO_H */
