# -*- coding: utf-8 -*-
"""随机森林训练窗口，复用 模型训练\训练随机森林.py。"""
from __future__ import annotations

import importlib
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
DATASET_DIR = os.path.join(APP_DIR, "数据集")
TRAIN_DIR = os.path.join(PROJECT_DIR, "模型训练")

sys.path.insert(0, TRAIN_DIR)
trainlib = importlib.import_module("训练随机森林")


class TrainWorker(QtCore.QThread):
    """后台训练线程，避免界面卡死。"""

    log_line = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            res = trainlib.train_model(log=self.log_line.emit, **self.params)
            if res is None:
                self.failed.emit("训练失败：数据集为空")
            else:
                self.finished_ok.emit(res)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            self.failed.emit("%s\n%s" % (exc, traceback.format_exc()))


class TrainDialog(QtWidgets.QDialog):
    """训练参数、日志和分类指标。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("随机森林模型训练")
        self.resize(940, 700)
        self.worker = None
        self.result = None
        self._build_ui()
        self.refresh_dataset_info()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        box1 = QtWidgets.QGroupBox("1. 数据集")
        g1 = QtWidgets.QGridLayout(box1)
        g1.addWidget(QtWidgets.QLabel("目录"), 0, 0)
        self.dir_edit = QtWidgets.QLineEdit(DATASET_DIR)
        g1.addWidget(self.dir_edit, 0, 1)
        for text, slot in (("选择…", self.pick_dir), ("打开", self.open_dir)):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            g1.addWidget(btn, 0, g1.columnCount())
        self.info_label = QtWidgets.QLabel("--")
        self.info_label.setStyleSheet("color:#9e9e9e;")
        g1.addWidget(self.info_label, 1, 0, 1, 6)
        root.addWidget(box1)

        box2 = QtWidgets.QGroupBox("2. 训练参数")
        g2 = QtWidgets.QGridLayout(box2)
        self.tree_box = self._spin(g2, 0, 0, "树的数量", 5, 200, 15)
        self.depth_box = self._spin(g2, 0, 2, "最大深度", 2, 20, 8)
        self.leaf_box = self._spin(g2, 0, 4, "叶子最小样本", 2, 100, 10)
        self.win_box = self._spin_double(g2, 1, 0, "窗口长度(秒)", 0.5, 3.0, 1.0, 0.5)
        self.hop_box = self._spin_double(g2, 1, 2, "步长(秒)", 0.05, 1.0, 0.25, 0.05)
        tip = QtWidgets.QLabel(
            "提示：窗口 1.0 秒 / 步长 0.25 秒 与固件一致；如果改窗口长度，"
            "必须同步修改 reco.h 里的 RECO_WIN_SAMPLES（当前 100）并重新编译固件。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#b58900;")
        g2.addWidget(tip, 2, 0, 1, 6)
        root.addWidget(box2)

        row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("▶  开始训练")
        self.train_btn.setMinimumHeight(34)
        self.train_btn.clicked.connect(self.start_train)
        row.addWidget(self.train_btn)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)
        root.addLayout(row)

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, 1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family:Consolas,'Microsoft YaHei'; font-size:12px;")
        body.addWidget(self.log_view, 3)

        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, 2)
        self.acc_label = QtWidgets.QLabel("准确率：--")
        f = QtGui.QFont("Microsoft YaHei", 13)
        f.setBold(True)
        self.acc_label.setFont(f)
        self.acc_label.setStyleSheet("color:#4caf50;")
        right.addWidget(self.acc_label)
        self.table = QtWidgets.QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(["精确率", "召回率", "F1", "窗口数"])
        self.table.setVerticalHeaderLabels(trainlib.CLASS_ZH)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        right.addWidget(self.table)

    @staticmethod
    def _spin(layout, row, col, text, lo, hi, val):
        layout.addWidget(QtWidgets.QLabel(text), row, col)
        sp = QtWidgets.QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        layout.addWidget(sp, row, col + 1)
        return sp

    @staticmethod
    def _spin_double(layout, row, col, text, lo, hi, val, step):
        layout.addWidget(QtWidgets.QLabel(text), row, col)
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(2)
        sp.setValue(val)
        layout.addWidget(sp, row, col + 1)
        return sp

    def pick_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dir_edit.text())
        if path:
            self.dir_edit.setText(path)
            self.refresh_dataset_info()

    def open_dir(self):
        path = self.dir_edit.text()
        if os.path.isdir(path):
            os.startfile(path)
        else:
            QtWidgets.QMessageBox.warning(self, "提示", "目录不存在：%s" % path)

    def refresh_dataset_info(self):
        folder = self.dir_edit.text()
        files = []
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        cnt = {}
        for f in files:
            key = f.split("_")[0]
            cnt[key] = cnt.get(key, 0) + 1
        detail = "，".join("%s %d" % (k, v) for k, v in sorted(cnt.items())) or "（空）"
        self.info_label.setText("文件 %d 个：%s" % (len(files), detail))

    # ----------------------------- 训练 ----------------------------- #
    def start_train(self):
        if self.worker is not None:
            return
        self.refresh_dataset_info()
        params = dict(
            dataset_dir=self.dir_edit.text(),
            out_dir=TRAIN_DIR,
            n_estimators=self.tree_box.value(),
            max_depth=self.depth_box.value(),
            min_samples_leaf=self.leaf_box.value(),
            win_sec=self.win_box.value(),
            hop_sec=self.hop_box.value(),
        )
        self.log_view.clear()
        self.table.clearContents()
        self.acc_label.setText("训练中…")
        self.train_btn.setEnabled(False)
        self.worker = TrainWorker(params)
        self.worker.log_line.connect(self.append_log)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def append_log(self, text):
        self.log_view.appendPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def on_failed(self, msg):
        self.append_log("训练失败：" + msg)
        self.acc_label.setText("训练失败")
        self.train_btn.setEnabled(True)
        self.worker = None

    def on_finished(self, res):
        self.result = res
        self.acc_label.setText("交叉验证准确率：%.1f%%" % (res["accuracy"] * 100))
        rep = res["report"]
        for i, zh in enumerate(trainlib.CLASS_ZH):
            r = rep[zh]
            values = [r["precision"], r["recall"], r["f1-score"], r["support"]]
            for j, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(("%.3f" % v) if j < 3 else ("%d" % v))
                self.table.setItem(i, j, item)

        target = os.path.join(PROJECT_DIR, "model.h")
        try:
            shutil.copyfile(res["model_path"], target)
            self.append_log("")
            self.append_log("已更新 %s —— Keil 重新编译（F7）即可生效" % target)
        except Exception as exc:                      # noqa: BLE001
            self.append_log("复制 model.h 失败：%s" % exc)
        self.train_btn.setEnabled(True)
        self.worker = None

    def closeEvent(self, event):
        if self.worker is not None:
            QtWidgets.QMessageBox.information(self, "提示", "训练还在进行，等它跑完再关闭。")
            event.ignore()
            return
        event.accept()
