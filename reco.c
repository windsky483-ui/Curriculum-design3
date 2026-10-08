/* 53 维时域特征 + 随机森林推理，特征顺序需与训练脚本保持一致。 */

#include "reco.h"
#include <math.h>

#define EPS_STD   1e-9f
#define EPS_MAG   1e-6f

static float col_mean(const float x[][3], int n, int c)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        s += x[i][c];
    }
    return s / (float)n;
}

static float col_std(const float x[][3], int n, int c, float mean)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float d = x[i][c] - mean;
        s += d * d;
    }
    return sqrtf(s / (float)n);
}

static float col_min(const float x[][3], int n, int c)
{
    float v = x[0][c];
    int i;
    for (i = 1; i < n; i++) {
        if (x[i][c] < v) {
            v = x[i][c];
        }
    }
    return v;
}

static float col_max(const float x[][3], int n, int c)
{
    float v = x[0][c];
    int i;
    for (i = 1; i < n; i++) {
        if (x[i][c] > v) {
            v = x[i][c];
        }
    }
    return v;
}

static float col_diff_mean(const float x[][3], int n, int c)
{
    float s = 0.0f;
    int i;
    if (n < 2) {
        return 0.0f;
    }
    for (i = 0; i < n - 1; i++) {
        s += fabsf(x[i + 1][c] - x[i][c]);
    }
    return s / (float)(n - 1);
}

static float col_zcr(const float x[][3], int n, int c, float mean)
{
    int i, cnt = 0;
    int prev;
    if (n < 2) {
        return 0.0f;
    }
    prev = (x[0][c] - mean > 0.0f) ? 1 : ((x[0][c] - mean < 0.0f) ? -1 : 0);
    for (i = 1; i < n; i++) {
        float d = x[i][c] - mean;
        int cur = (d > 0.0f) ? 1 : ((d < 0.0f) ? -1 : 0);
        if (cur != prev) {
            cnt++;
        }
        prev = cur;
    }
    return (float)cnt / (float)(n - 1);
}

static float col_corr(const float x[][3], int n, int ca, int cb)
{
    float mx = 0.0f, my = 0.0f, sxy = 0.0f, sxx = 0.0f, syy = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        mx += x[i][ca];
        my += x[i][cb];
    }
    mx /= (float)n;
    my /= (float)n;
    for (i = 0; i < n; i++) {
        float dx = x[i][ca] - mx;
        float dy = x[i][cb] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if (sqrtf(sxx / (float)n) <= EPS_STD || sqrtf(syy / (float)n) <= EPS_STD) {
        return 0.0f;
    }
    return sxy / sqrtf(sxx * syy);
}

void reco_features(const float acc[][3], const float gyro[][3], int n, float *f)
{
    float mean[3], std[3], amag_mean, amag_std, gmag_mean, gmag_std;
    float amag_min, amag_max, gmag_max, sma = 0.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float asum = 0.0f, asum2 = 0.0f;
    int i, c, k = 0;

    /* 特征顺序与 训练随机森林.py 的 window_features() 一致。 */
    for (c = 0; c < 3; c++) {
        mean[c] = col_mean(acc, n, c);
        std[c] = col_std(acc, n, c, mean[c]);
        f[k++] = mean[c];
        f[k++] = std[c];
        f[k++] = col_min(acc, n, c);
        f[k++] = col_max(acc, n, c);
    }
    for (c = 0; c < 3; c++) {
        float m = col_mean(gyro, n, c);
        float s = col_std(gyro, n, c, m);
        f[k++] = m;
        f[k++] = s;
        f[k++] = col_min(gyro, n, c);
        f[k++] = col_max(gyro, n, c);
    }

    amag_min = 1e30f;
    amag_max = 0.0f;
    gmag_max = 0.0f;
    for (i = 0; i < n; i++) {
        float ax = acc[i][0], ay = acc[i][1], az = acc[i][2];
        float gx = gyro[i][0], gy = gyro[i][1], gz = gyro[i][2];
        float am = sqrtf(ax * ax + ay * ay + az * az);
        float gm = sqrtf(gx * gx + gy * gy + gz * gz);
        float inv;

        asum += am;
        asum2 += am * am;
        if (am < amag_min) {
            amag_min = am;
        }
        if (am > amag_max) {
            amag_max = am;
        }
        if (gm > gmag_max) {
            gmag_max = gm;
        }
        sma += fabsf(ax) + fabsf(ay) + fabsf(az);
        inv = (am > EPS_MAG) ? (1.0f / am) : 0.0f;
        ux += ax * inv;
        uy += ay * inv;
        uz += az * inv;
    }
    amag_mean = asum / (float)n;
    amag_std = sqrtf(asum2 / (float)n - amag_mean * amag_mean > 0.0f
                     ? asum2 / (float)n - amag_mean * amag_mean : 0.0f);
    f[k++] = amag_mean;
    f[k++] = amag_std;
    f[k++] = amag_min;
    f[k++] = amag_max;

    {
        float s = 0.0f, s2 = 0.0f;
        for (i = 0; i < n; i++) {
            float gx = gyro[i][0], gy = gyro[i][1], gz = gyro[i][2];
            float gm = sqrtf(gx * gx + gy * gy + gz * gz);
            s += gm;
            s2 += gm * gm;
        }
        gmag_mean = s / (float)n;
        {
            float var = s2 / (float)n - gmag_mean * gmag_mean;
            gmag_std = sqrtf(var > 0.0f ? var : 0.0f);
        }
    }
    f[k++] = gmag_mean;
    f[k++] = gmag_std;
    f[k++] = gmag_max;

    f[k++] = sma / (float)n;

    f[k++] = ux / (float)n;
    f[k++] = uy / (float)n;
    f[k++] = uz / (float)n;

    for (c = 0; c < 3; c++) {
        f[k++] = col_diff_mean(acc, n, c);
    }
    for (c = 0; c < 3; c++) {
        f[k++] = col_diff_mean(gyro, n, c);
    }

    for (c = 0; c < 3; c++) {
        f[k++] = col_zcr(acc, n, c, mean[c]);
    }
    for (c = 0; c < 3; c++) {
        f[k++] = col_zcr(gyro, n, c, col_mean(gyro, n, c));
    }

    f[k++] = col_corr(acc, n, 0, 1);
    f[k++] = col_corr(acc, n, 1, 2);
    f[k++] = col_corr(acc, n, 0, 2);
    f[k++] = col_corr(gyro, n, 0, 1);
    f[k++] = col_corr(gyro, n, 1, 2);
    f[k++] = col_corr(gyro, n, 0, 2);
}

int reco_predict(const float *feat, float *proba)
{
    float xs[RECO_FEATURES];
    float vote[RECO_CLASSES];
    int t, i, best = 0;

    for (i = 0; i < RECO_FEATURES; i++) {
        xs[i] = (feat[i] - FEAT_MEAN[i]) / FEAT_STD[i];
    }
    for (i = 0; i < RECO_CLASSES; i++) {
        vote[i] = 0.0f;
    }

    for (t = 0; t < MODEL_N_TREES; t++) {
        const rf_node_t *tree = RF_TREES[t];
        int node = 0;
        while (tree[node].feature >= 0) {
            node = (xs[tree[node].feature] <= tree[node].threshold)
                   ? tree[node].left : tree[node].right;
        }
        for (i = 0; i < RECO_CLASSES; i++) {
            vote[i] += (float)tree[node].proba[i];
        }
    }
    for (i = 1; i < RECO_CLASSES; i++) {
        if (vote[i] > vote[best]) {
            best = i;
        }
    }
    if (proba != 0) {
        float total = 0.0f;
        for (i = 0; i < RECO_CLASSES; i++) {
            total += vote[i];
        }
        if (total <= 0.0f) {
            total = 1.0f;
        }
        for (i = 0; i < RECO_CLASSES; i++) {
            proba[i] = vote[i] / total;
        }
    }
    return best;
}
