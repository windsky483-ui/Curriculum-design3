# -*- coding: utf-8 -*-
r"""读取 上位机\数据集 CSV，训练随机森林并导出 STM32 使用的 model.h。"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
from collections import Counter

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "上位机", "数据集")
OUT_DIR = APP_DIR

FS = 100.0            # 采样率（Hz）
WIN_SEC = 1.0         # 窗口长度 1 秒
HOP_SEC = 0.25        # 步长 0.25 秒 → 每 0.25 秒输出一次识别结果
WIN = int(WIN_SEC * FS)
HOP = int(HOP_SEC * FS)
BLOCK_SEC = 10        # 交叉验证分组：长录音按 10 秒分块，避免同段数据同时进训练和测试
CLASS_NAMES = ["still", "walk", "run", "fall"]
CLASS_ZH = ["静止", "走路", "跑步", "跌倒"]


def load_csv(path):
    """读取一个数据集 CSV，返回时间戳、六轴物理量和标签。"""
    t_ms = []
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []
    label = None
    with open(path, "r", encoding="utf-8-sig") as fp:
        for row in csv.reader(fp):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "pc_time":
                continue
            try:
                t_ms.append(int(row[2]))
                ax.append(float(row[3])); ay.append(float(row[4])); az.append(float(row[5]))
                gx.append(float(row[6])); gy.append(float(row[7])); gz.append(float(row[8]))
                label = int(row[10])
            except (ValueError, IndexError):
                continue
    return {
        "t_ms": np.asarray(t_ms, dtype=np.int64),
        "acc": np.stack([ax, ay, az], axis=1) if ax else np.zeros((0, 3)),
        "gyro": np.stack([gx, gy, gz], axis=1) if gx else np.zeros((0, 3)),
        "label": label,
        "path": path,
    }


def load_dataset(dataset_dir=None):
    """读取数据集目录下所有 CSV（默认 上位机\\数据集）。"""
    folder = dataset_dir or DATASET_DIR
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    data = []
    for path in files:
        d = load_csv(path)
        if len(d["t_ms"]) > 0 and d["label"] is not None:
            data.append(d)
    return data


# ------------------------------- 数据体检 ------------------------------- #
def qc():
    data = load_dataset()
    if not data:
        print("数据集为空：%s" % DATASET_DIR)
        return
    print("数据集目录：%s" % DATASET_DIR)
    print("共 %d 个文件\n" % len(data))
    per_class = {}
    for d in data:
        t = d["t_ms"]
        dur = (t[-1] - t[0]) / 1000.0
        fs = len(t) / dur if dur > 0 else 0.0
        dt = np.diff(t)
        a_mag = np.linalg.norm(d["acc"], axis=1)
        g_mag = np.linalg.norm(d["gyro"], axis=1)
        name = os.path.basename(d["path"])
        lab = d["label"]
        per_class[lab] = per_class.get(lab, 0.0) + dur
        print("%-34s label=%d  样本=%6d  时长=%6.1f s  采样率=%5.1f Hz"
              % (name, lab, len(t), dur, fs))
        print("     合加速度 |a|: 均值 %.2f  最小 %.2f  最大 %.2f g   | 角速度最大 %5.0f °/s"
              % (a_mag.mean(), a_mag.min(), a_mag.max(), g_mag.max()))
        print("     采样间隔: 平均 %.2f ms  最大 %.2f ms（>25ms 的丢帧 %d 次）"
              % (dt.mean(), dt.max(), int((dt > 25).sum())))
        step = int(10 * fs) or 1
        prof = []
        for i in range(0, len(t), step):
            seg_a = a_mag[i:i + step]; seg_g = g_mag[i:i + step]
            if len(seg_a) == 0:
                continue
            prof.append("%4.0fs:%.2f/%.0f" % (t[i] / 1000.0, seg_a.mean(), seg_g.max()))
        print("     每10秒 时间:平均|a|/最大|g| → " + "  ".join(prof[:8])
              + ("  ..." if len(prof) > 8 else ""))
        print()

    print("各类别总时长：")
    for lab in sorted(per_class):
        nm = CLASS_NAMES[lab] if lab < len(CLASS_NAMES) else str(lab)
        print("   %-6s %7.1f 秒" % (nm, per_class[lab]))
    total_win = sum(max(0, (len(d["t_ms"]) - WIN)) // HOP + 1 for d in data)
    print("\n预计可用窗口数：%d（每窗口 1 秒、步长 0.25 秒）" % total_win)


# ------------------------------- 特征提取 ------------------------------- #
def window_features(acc, gyro):
    """acc/gyro: (N,3)。返回特征向量（顺序固定，C 端按同样顺序、同样公式实现）。"""
    f = []
    # 1) 六通道各自：均值、标准差、最小、最大（24 维）
    for i in range(3):
        x = acc[:, i]
        f += [x.mean(), x.std(), x.min(), x.max()]
    for i in range(3):
        x = gyro[:, i]
        f += [x.mean(), x.std(), x.min(), x.max()]

    a_mag = np.linalg.norm(acc, axis=1)
    g_mag = np.linalg.norm(gyro, axis=1)

    # 2) 合加速度 / 合角速度统计（7 维）
    f += [a_mag.mean(), a_mag.std(), a_mag.min(), a_mag.max()]
    f += [g_mag.mean(), g_mag.std(), g_mag.max()]

    # 3) SMA：平均 |ax|+|ay|+|az|（1 维）
    f.append(float(np.abs(acc).sum(axis=1).mean()))

    # 4) 姿态余弦：重力方向单位向量分量（3 维）
    unit = acc / np.maximum(a_mag[:, None], 1e-6)
    f += [unit[:, 0].mean(), unit[:, 1].mean(), unit[:, 2].mean()]

    # 5) 一阶差分绝对值均值（抖动/冲击，6 维）
    for i in range(3):
        f.append(float(np.abs(np.diff(acc[:, i])).mean()) if len(acc) > 1 else 0.0)
    for i in range(3):
        f.append(float(np.abs(np.diff(gyro[:, i])).mean()) if len(gyro) > 1 else 0.0)

    # 6) 去均值后的过零率（6 维）
    for src in (acc, gyro):
        for i in range(3):
            x = src[:, i] - src[:, i].mean()
            f.append(float(np.mean(np.diff(np.sign(x)) != 0)) if len(x) > 1 else 0.0)

    # 7) 轴间相关系数（acc 三对 + gyro 三对，6 维）
    for src in (acc, gyro):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            x, y = src[:, a], src[:, b]
            if x.std() > 1e-9 and y.std() > 1e-9:
                f.append(float(np.corrcoef(x, y)[0, 1]))
            else:
                f.append(0.0)
    return np.asarray(f, dtype=np.float64)


FEATURE_NAMES = ["ax_mean", "ax_std", "ax_min", "ax_max",
                 "ay_mean", "ay_std", "ay_min", "ay_max",
                 "az_mean", "az_std", "az_min", "az_max",
                 "gx_mean", "gx_std", "gx_min", "gx_max",
                 "gy_mean", "gy_std", "gy_min", "gy_max",
                 "gz_mean", "gz_std", "gz_min", "gz_max",
                 "amag_mean", "amag_std", "amag_min", "amag_max",
                 "gmag_mean", "gmag_std", "gmag_max",
                 "sma",
                 "ux_mean", "uy_mean", "uz_mean",
                 "dax", "day", "daz", "dgx", "dgy", "dgz",
                 "zcr_ax", "zcr_ay", "zcr_az", "zcr_gx", "zcr_gy", "zcr_gz",
                 "corr_axay", "corr_ayaz", "corr_axaz",
                 "corr_gxgy", "corr_gygz", "corr_gxgz"]


def build_windows(data, win=WIN, hop=HOP, block_sec=BLOCK_SEC):
    """把每段录音切成窗口，返回 X, y, groups（groups 用于分组交叉验证）。"""
    X, y, groups = [], [], []
    block = max(1, int(block_sec * FS))
    for d in data:
        n = len(d["t_ms"])
        for start in range(0, n - win + 1, hop):
            seg_a = d["acc"][start:start + win]
            seg_g = d["gyro"][start:start + win]
            X.append(window_features(seg_a, seg_g))
            y.append(d["label"])
            groups.append("%s#%d" % (os.path.basename(d["path"]), start // block))
    return np.asarray(X), np.asarray(y), np.asarray(groups)


# ------------------------------- 训练与评估 ------------------------------- #
def _jobs_for(n_windows):
    """小数据集用单进程（Windows 上多进程启动开销比训练本身还大）。"""
    return 1 if n_windows < 5000 else -1


def train_model(dataset_dir=None, out_dir=None, n_estimators=15, max_depth=8,
                min_samples_leaf=10, win_sec=1.0, hop_sec=0.25, log=print):
    """完整训练流程（供命令行和上位机界面共用）。

    返回 dict：accuracy / report / importances / model_path 等。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, accuracy_score

    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    win = max(10, int(round(win_sec * FS)))
    hop = max(1, int(round(hop_sec * FS)))

    data = load_dataset(dataset_dir)
    if not data:
        log("数据集为空，请先采集数据（上位机 → 数据采集（打标签））")
        return None

    X, y, groups = build_windows(data, win=win, hop=hop)
    log("文件 %d 个，窗口 %d 个，特征 %d 维，各类窗口数 %s"
        % (len(data), X.shape[0], X.shape[1], dict(Counter(y.tolist()))))
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape[1], len(FEATURE_NAMES))
    n_groups = len(set(groups.tolist()))
    log("交叉验证分组数（每组 = 一个 10 秒数据块）：%d" % n_groups)

    # 特征标准化参数（作为模型的一部分导出，C 端用同样公式先标准化再进树）
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    Xs = (X - feat_mean) / feat_std

    rf = RandomForestClassifier(n_estimators=int(n_estimators),
                                max_depth=int(max_depth),
                                min_samples_leaf=int(min_samples_leaf),
                                class_weight="balanced", random_state=42,
                                n_jobs=_jobs_for(X.shape[0]))
    cv = GroupKFold(n_splits=min(5, max(2, n_groups)))
    log("开始分组交叉验证……")
    y_pred = cross_val_predict(rf, Xs, y, cv=cv, groups=groups,
                               n_jobs=_jobs_for(X.shape[0]))
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, labels=[0, 1, 2, 3],
                                   target_names=CLASS_ZH, digits=3,
                                   zero_division=0, output_dict=True)
    log("交叉验证总体准确率：%.1f%%" % (acc * 100))
    for zh in CLASS_ZH:
        r = report[zh]
        log("   %-4s 精确率 %.1f%%  召回率 %.1f%%  F1 %.3f"
            % (zh, r["precision"] * 100, r["recall"] * 100, r["f1-score"]))
    log("用全部数据训练最终模型……")
    rf.fit(Xs, y)
    n_leaves = sum(int(est.tree_.n_leaves) for est in rf.estimators_)
    log("随机森林：%d 棵树，最大深度 %d，叶子总数 %d"
        % (len(rf.estimators_), max_depth, n_leaves))

    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])[:15]
    log("最重要的 15 个特征：")
    for k, v in imp:
        log("   %-12s %.4f" % (k, v))

    model_path = os.path.join(out_dir, "model.h")

    export_c_model(rf, feat_mean, feat_std, out_dir)
    log("已生成：%s" % model_path)
    return {
        "accuracy": float(acc),
        "report": report,
        "importances": [[k, float(v)] for k, v in imp],
        "model_path": model_path,
        "n_windows": int(X.shape[0]),
        "n_groups": int(n_groups),
        "n_leaves": int(n_leaves),
        "n_trees": int(len(rf.estimators_)),
        "window_sec": float(win / FS),
        "hop_sec": float(hop / FS),
    }


def train(dataset_dir=None, out_dir=None):
    """命令行入口：用默认参数训练。"""
    res = train_model(dataset_dir=dataset_dir, out_dir=out_dir)
    if res:
        print("\n训练完成，准确率 %.1f%%" % (res["accuracy"] * 100))
    return res


def export_c_model(rf, mean, std, out_dir=None):
    """导出纯 C 数组：标准化参数 + 每棵树的节点（叶子存 4 类概率百分比）。"""
    out_dir = out_dir or OUT_DIR
    L = []
    L.append("/* 自动生成：MPU6050 四类动作识别（静止/走路/跑步/跌倒）随机森林模型")
    L.append(" * 特征数 %d，树 %d 棵；输入请先用 FEAT_MEAN/FEAT_STD 标准化 */"
             % (len(mean), len(rf.estimators_)))
    L.append("#ifndef MPU_MODEL_H")
    L.append("#define MPU_MODEL_H")
    L.append("#define MODEL_N_FEATURES %d" % len(mean))
    L.append("#define MODEL_N_CLASSES  %d" % len(CLASS_ZH))
    L.append("#define MODEL_N_TREES    %d" % len(rf.estimators_))
    L.append("")
    L.append("static const float FEAT_MEAN[MODEL_N_FEATURES] = {")
    for i in range(0, len(mean), 6):
        L.append("  " + ", ".join("%.6ff" % v for v in mean[i:i + 6]) + ",")
    L.append("};")
    L.append("static const float FEAT_STD[MODEL_N_FEATURES] = {")
    for i in range(0, len(std), 6):
        L.append("  " + ", ".join("%.6ff" % v for v in std[i:i + 6]) + ",")
    L.append("};")
    L.append("")
    L.append("typedef struct {")
    L.append("  short feature;        /* -1 表示叶子节点 */")
    L.append("  float threshold;")
    L.append("  short left, right;    /* 子节点下标 */")
    L.append("  unsigned char proba[MODEL_N_CLASSES];   /* 叶子：各类概率 0~100 */")
    L.append("} rf_node_t;")
    L.append("")
    for ti, est in enumerate(rf.estimators_):
        tree = est.tree_
        L.append("static const rf_node_t RF_TREE_%d[%d] = {" % (ti, tree.node_count))
        for node in range(tree.node_count):
            if tree.children_left[node] == -1:
                cnt = tree.value[node][0]
                tot = cnt.sum() if cnt.sum() > 0 else 1.0
                p = cnt / tot * 100.0
                L.append("  {-1, 0.0f, 0, 0, {%s}},"
                         % ", ".join("%d" % round(v) for v in p))
            else:
                L.append("  {%d, %.6ff, %d, %d, {0,0,0,0}},"
                         % (int(tree.feature[node]), tree.threshold[node],
                            int(tree.children_left[node]), int(tree.children_right[node])))
        L.append("};")
        L.append("")
    L.append("static const rf_node_t *const RF_TREES[MODEL_N_TREES] = {")
    L.append("  " + ", ".join("RF_TREE_%d" % i for i in range(len(rf.estimators_))))
    L.append("};")
    L.append("")
    L.append("#endif /* MPU_MODEL_H */")
    with open(os.path.join(out_dir, "model.h"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(L) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "qc"
    if mode in ("qc", "all"):
        qc()
    if mode in ("train", "all"):
        train()
