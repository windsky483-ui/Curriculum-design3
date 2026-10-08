# -*- coding: utf-8 -*-
"""MPU6050 串口采集、实时波形显示、CSV 记录和带标签数据采集。"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:                       # pragma: no cover
    print("缺少 pyserial：请执行  python -m pip install pyserial")
    raise

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


ACC_LSB_PER_G = 8192.0        # ±4g 量程（±2g=16384 / ±4g=8192 / ±8g=4096 / ±16g=2048）
GYRO_LSB_PER_DPS = 65.5       # ±500dps 量程
TEMP_LSB_PER_C = 340.0
TEMP_OFFSET_C = 36.53

CH_COLORS = {
    "ax": "#ff5252", "ay": "#4caf50", "az": "#42a5f5",
    "gx": "#ffb74d", "gy": "#ba68c8", "gz": "#26c6da",
}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "数据记录")      # CSV / PNG 默认保存位置
DATASET_DIR = os.path.join(APP_DIR, "数据集")     # 带标签的训练数据集

LABELS = [
    (0, "still", "静止"),
    (1, "walk", "走路"),
    (2, "run", "跑步"),
    (3, "fall", "跌倒"),
]


def parse_line(line):
    """解析固件输出的 CSV 数据行。"""
    if not line:
        return None
    parts = line.strip().split(",")
    if len(parts) != 9:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def to_physical(sample):
    """原始值 → 物理量：(ax,ay,az)[g], (gx,gy,gz)[°/s], temp[℃]"""
    seq, t_ms, ax, ay, az, temp, gx, gy, gz = sample
    acc = (ax / ACC_LSB_PER_G, ay / ACC_LSB_PER_G, az / ACC_LSB_PER_G)
    gyro = (gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS)
    t_c = temp / TEMP_LSB_PER_C + TEMP_OFFSET_C
    return acc, gyro, t_c


COMPACT_QSS = """
QPushButton { padding: 3px 8px; font-size: 12px; min-height: 20px; }
QComboBox, QSpinBox { padding: 1px 4px; font-size: 12px; min-height: 18px; }
QLabel { font-size: 12px; }
QGroupBox { font-size: 12px; }
"""


class DataSource(QtCore.QThread):
    """串口读取线程（也支持模拟数据）。"""

    sample_ready = QtCore.pyqtSignal(object)
    result_ready = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, mode="serial", port=None, baud=115200, parent=None):
        super().__init__(parent)
        self.mode = mode                          # 'serial' 或 'mock'
        self.port = port
        self.baud = baud
        self._running = True
        self.ser = None

        self.bytes_read = 0                       # 统计用
        self.lines_total = 0
        self.lines_bad = 0

    def stop(self):
        self._running = False

    def run(self):
        if self.mode == "mock":
            self._run_mock()
        else:
            self._run_serial()

    # ---------------------------- 真实串口 ---------------------------- #
    def _run_serial(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.2,
                                dsrdtr=False, rtscts=False)
        except Exception as exc:                  # 端口被占用 / 不存在
            self.failed.emit("打开串口失败：%s" % exc)
            return

        self.ser = ser
        self.reset_board()          # 关键：放开 DTR/RTS 并复位一次，让 MCU 正常运行
        self.status_changed.emit("已连接 %s @ %d" % (self.port, self.baud))
        buf = b""
        last_report = time.time()
        try:
            while self._running:
                chunk = ser.read(4096)
                if chunk:
                    self.bytes_read += len(chunk)
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        text = raw.decode("ascii", "ignore").strip()
                        if text.startswith("R,"):          # 板子的识别结果行
                            parts = text.split(",")
                            if len(parts) >= 4:
                                try:
                                    self.result_ready.emit((int(parts[1]), int(parts[2]),
                                                            int(parts[3]),
                                                            parts[4] if len(parts) > 4 else ""))
                                except ValueError:
                                    pass
                            continue
                        self.lines_total += 1
                        sample = parse_line(text)
                        if sample is None:
                            self.lines_bad += 1
                        else:
                            self.sample_ready.emit((sample, time.time()))

                now = time.time()
                if now - last_report >= 1.0:      # 每秒刷新一次状态栏
                    last_report = now
                    self.status_changed.emit(
                        "%s 已接收 %.1f KB，无效行 %d"
                        % (self.port, self.bytes_read / 1024.0, self.lines_bad)
                    )
        except Exception as exc:
            self.failed.emit("串口读取异常：%s" % exc)
        finally:
            try:
                ser.close()
            except Exception:
                pass
            self.ser = None
            self.status_changed.emit("串口已关闭")

    def reset_board(self):
        """给板子一个复位脉冲（解决"点连接后 LED 不闪/没有数据"的问题）。

        板上 CH340 的 RTS# 接 BOOT0、DTR# 接 RESET（一键下载电路）：
          * 打开串口时 pyserial 默认会拉低这两个信号，等于把 MCU 按在复位/下载模式；
          * 这里先把 RTS 放开（BOOT0=0，从 Flash 启动），再用 DTR 发一个复位脉冲。
        """
        if self.ser is None:
            return False
        try:
            self.ser.rts = False          # BOOT0 = 0
            self.ser.dtr = True           # 拉复位
            time.sleep(0.08)
            self.ser.dtr = False          # 放开 → 从 Flash 正常启动
            time.sleep(0.25)
            return True
        except Exception:
            return False

    # ---------------------------- 模拟数据 ---------------------------- #
    def _run_mock(self):
        """生成类似真实姿态变化的假数据，方便先调试界面。"""
        self.status_changed.emit("模拟数据模式运行中（未使用串口）")
        t0 = time.time()
        i = 0
        while self._running:
            t = i / 100.0
            ax = 0.25 * math.sin(2 * math.pi * 0.4 * t) + random.gauss(0, 0.01)
            ay = 0.35 * math.sin(2 * math.pi * 0.23 * t + 1.0) + random.gauss(0, 0.01)
            az = 1.0 + 0.05 * math.sin(2 * math.pi * 0.11 * t)
            gx = 25.0 * math.sin(2 * math.pi * 0.8 * t) + random.gauss(0, 0.6)
            gy = 18.0 * math.sin(2 * math.pi * 0.5 * t + 0.7)
            gz = 8.0 * math.sin(2 * math.pi * 1.3 * t)
            temp = 30.0 + 0.8 * math.sin(2 * math.pi * 0.02 * t)

            raw = (
                i, int(t * 1000),
                int(ax * ACC_LSB_PER_G), int(ay * ACC_LSB_PER_G), int(az * ACC_LSB_PER_G),
                int((temp - TEMP_OFFSET_C) * TEMP_LSB_PER_C),
                int(gx * GYRO_LSB_PER_DPS), int(gy * GYRO_LSB_PER_DPS),
                int(gz * GYRO_LSB_PER_DPS),
            )
            self.lines_total += 1
            self.sample_ready.emit((raw, time.time()))
            i += 1
            delay = t0 + i / 100.0 - time.time()   # 模拟 100Hz 采样
            if delay > 0:
                time.sleep(delay)


# ================================= 主窗口 ================================= #
class MainWindow(QtWidgets.QMainWindow):
    """实时波形 + 采集控制界面。"""

    MAX_POINTS = 2500        # 环形缓冲长度（25 秒 @100Hz，够 20 秒窗口用）
    UI_INTERVAL_MS = 20      # 波形刷新周期（20ms = 50FPS，可在界面上调整）

    def __init__(self, start_mock=False):
        super().__init__()
        self.setWindowTitle("MPU6050 六轴数据采集上位机  ——  普中-天马 STM32F407")
        self.resize(1280, 800)
        self.setMinimumSize(980, 600)
        self.setStyleSheet(COMPACT_QSS)

        # ---------------- 数据缓冲 ---------------- #
        self.sources = {}
        for key in ("ax", "ay", "az", "gx", "gy", "gz"):
            self.sources[key] = deque(maxlen=self.MAX_POINTS)
        self.time_axis = deque(maxlen=self.MAX_POINTS)

        self.worker = None
        self.acquiring = False
        self.t_start = None
        self.last_seq = None
        self.lost_frames = 0
        self.sample_count = 0
        self.rate_times = deque(maxlen=200)
        self.refresh_times = deque(maxlen=40)  # 用于显示实际刷新率

        # ---------------- 记录相关 ---------------- #
        self.recording = False
        self.rec_file = None
        self.rec_writer = None
        self.rec_buffer = []
        self.rec_rows = 0
        self.rec_label_id = -1                 # 当前录制文件的标签
        self.rec_label_name = "未标注"
        self.rec_start_time = None
        self.countdown_left = 0                # 倒计时（秒）
        self.pending_label = None              # 倒计时结束后要录的标签

        self._build_ui()

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setTimerType(QtCore.Qt.PreciseTimer)   # 高精度，避免刷新节奏被拖慢
        self.ui_timer.timeout.connect(self._on_refresh)
        self.ui_timer.start(self.UI_INTERVAL_MS)
        self._apply_y_ranges()

        if start_mock:
            self.mock_check.setChecked(True)
            self._toggle_connection()

    # ============================== 界面 ============================== #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addLayout(self._build_toolbar())

        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, 1)
        body.addWidget(self._build_plots(), 3)
        body.addWidget(self._build_panel(), 0)

        self.status = self.statusBar()
        self.status.showMessage("就绪：选择串口后点「连接」")

    def _build_toolbar(self):
        bar = QtWidgets.QHBoxLayout()

        bar.addWidget(QtWidgets.QLabel("串口："))
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setMinimumWidth(170)
        bar.addWidget(self.port_box)

        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        bar.addWidget(self.refresh_btn)

        bar.addWidget(QtWidgets.QLabel("波特率："))
        self.baud_box = QtWidgets.QComboBox()
        self.baud_box.addItems(["115200", "230400", "460800", "57600", "9600"])
        bar.addWidget(self.baud_box)

        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_connection)
        bar.addWidget(self.connect_btn)

        self.mock_check = QtWidgets.QCheckBox("模拟数据（无硬件演示）")
        bar.addWidget(self.mock_check)

        bar.addStretch(1)
        return bar

    def _build_plots(self):
        pg.setConfigOptions(antialias=True, background="#101418", foreground="#d0d0d0")
        self.plot_widget = pg.GraphicsLayoutWidget()

        self.plot_acc = self.plot_widget.addPlot(row=0, col=0, title="加速度 (g)")
        self.plot_gyro = self.plot_widget.addPlot(row=1, col=0, title="角速度 (°/s)")
        self.plot_gyro.setXLink(self.plot_acc)

        for p in (self.plot_acc, self.plot_gyro):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("bottom", "时间", units="s")
            p.addLegend(offset=(10, 10))
            view = p.getViewBox()
            view.setAutoVisible(y=True)          # Y 轴按可见数据自动缩放
            view.enableAutoRange(axis="y")

        self.curves = {
            "ax": self.plot_acc.plot(pen=pg.mkPen(CH_COLORS["ax"], width=2), name="AX"),
            "ay": self.plot_acc.plot(pen=pg.mkPen(CH_COLORS["ay"], width=2), name="AY"),
            "az": self.plot_acc.plot(pen=pg.mkPen(CH_COLORS["az"], width=2), name="AZ"),
            "gx": self.plot_gyro.plot(pen=pg.mkPen(CH_COLORS["gx"], width=2), name="GX"),
            "gy": self.plot_gyro.plot(pen=pg.mkPen(CH_COLORS["gy"], width=2), name="GY"),
            "gz": self.plot_gyro.plot(pen=pg.mkPen(CH_COLORS["gz"], width=2), name="GZ"),
        }
        self.plot_acc.setXRange(-5, 0)
        return self.plot_widget

    def _build_panel(self):
        """右侧控制面板：紧凑排版 + 可滚动（窗口再小也不会有点不到的按钮）。"""
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        # ---- 运行状态（实时数值面板已按需求去掉，直接看波形） ---- #
        self.stats_label = QtWidgets.QLabel("采样率：-- Hz    波形刷新：-- FPS\n样本：0    丢帧：0    无效行：0")
        self.stats_label.setStyleSheet("color:#9e9e9e;")
        lay.addWidget(self.stats_label)

        # ---- 板子识别结果（由下位机的 R 行回传） ---- #
        self.result_label = QtWidgets.QLabel("识别：--")
        f = QtGui.QFont("Microsoft YaHei", 11)
        f.setBold(True)
        self.result_label.setFont(f)
        self.result_label.setStyleSheet("color:#9e9e9e; padding:2px 0;")
        lay.addWidget(self.result_label)

        # ---- 显示设置 ---- #
        lay.addWidget(self._hline("显示设置"))
        grid2 = QtWidgets.QGridLayout()
        grid2.addWidget(QtWidgets.QLabel("显示窗口"), 0, 0)
        self.win_box = QtWidgets.QComboBox()
        self.win_box.addItems(["5 秒", "2 秒", "10 秒", "20 秒"])
        grid2.addWidget(self.win_box, 0, 1)

        grid2.addWidget(QtWidgets.QLabel("加速度 Y 轴"), 1, 0)
        self.y_acc_box = QtWidgets.QComboBox()
        self.y_acc_box.addItems(["自动", "±0.2 g", "±0.5 g", "±1 g", "±2 g"])
        grid2.addWidget(self.y_acc_box, 1, 1)

        grid2.addWidget(QtWidgets.QLabel("角速度 Y 轴"), 2, 0)
        self.y_gyro_box = QtWidgets.QComboBox()
        self.y_gyro_box.addItems(["自动", "±5 °/s", "±20 °/s", "±100 °/s", "±500 °/s"])
        grid2.addWidget(self.y_gyro_box, 2, 1)

        grid2.addWidget(QtWidgets.QLabel("波形刷新率"), 3, 0)
        self.fps_box = QtWidgets.QComboBox()
        self.fps_box.addItems(["50 FPS（默认）", "60 FPS", "30 FPS", "20 FPS"])
        grid2.addWidget(self.fps_box, 3, 1)
        # 只在用户改变选项时调整 Y 轴（不要在每帧刷新时重复设置，否则会拖慢刷新）
        self.y_acc_box.currentTextChanged.connect(lambda _t: self._apply_y_ranges())
        self.y_gyro_box.currentTextChanged.connect(lambda _t: self._apply_y_ranges())
        self.fps_box.currentTextChanged.connect(self._apply_fps)
        lay.addLayout(grid2)

        # ---- 采集控制（两列排布，省空间） ---- #
        lay.addWidget(self._hline("采集控制"))
        self.start_btn = QtWidgets.QPushButton("▶  开始采集")
        self.start_btn.clicked.connect(self.start_acquisition)
        self.pause_btn = QtWidgets.QPushButton("⏸  暂停显示")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.clear_btn = QtWidgets.QPushButton("🗑  清除波形")
        self.clear_btn.clicked.connect(self.clear_data)
        self.reset_btn = QtWidgets.QPushButton("🔄  复位板子")
        self.reset_btn.clicked.connect(self.reset_board)

        grid3 = QtWidgets.QGridLayout()
        grid3.setSpacing(4)
        grid3.addWidget(self.start_btn, 0, 0)
        grid3.addWidget(self.pause_btn, 0, 1)
        grid3.addWidget(self.clear_btn, 1, 0)
        grid3.addWidget(self.reset_btn, 1, 1)
        lay.addLayout(grid3)

        # ---- 带标签的数据采集（课设数据集） ---- #
        lay.addWidget(self._hline("数据采集（打标签：键盘 1/2/3/4）"))

        lab_grid = QtWidgets.QGridLayout()
        lab_grid.setSpacing(4)
        self.label_btns = {}
        for i, (lid, en, zh) in enumerate(LABELS):
            btn = QtWidgets.QPushButton("%d  %s" % (i + 1, zh))
            btn.clicked.connect(
                lambda _checked=False, item=(lid, en, zh): self.start_label_record(*item))
            lab_grid.addWidget(btn, i // 2, i % 2)
            self.label_btns[lid] = btn
        lay.addLayout(lab_grid)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("倒计时"))
        self.count_box = QtWidgets.QSpinBox()
        self.count_box.setRange(0, 10)
        self.count_box.setValue(3)
        self.count_box.setSuffix(" 秒")
        row.addWidget(self.count_box)
        stop_lab_btn = QtWidgets.QPushButton("⏹ 停止")
        stop_lab_btn.clicked.connect(self.stop_label_record)
        row.addWidget(stop_lab_btn)
        lay.addLayout(row)

        self.label_status = QtWidgets.QLabel("未录制")
        self.label_status.setStyleSheet("color:#ffd54f;")
        self.label_status.setWordWrap(True)
        lay.addWidget(self.label_status)

        self.open_ds_btn = QtWidgets.QPushButton("📂  打开数据集文件夹")
        self.open_ds_btn.clicked.connect(self.open_dataset_dir)
        lay.addWidget(self.open_ds_btn)

        # ---- 模型训练（随机森林） ---- #
        lay.addWidget(self._hline("模型训练（随机森林）"))
        self.train_dialog_btn = QtWidgets.QPushButton("🧠  打开训练窗口")
        self.train_dialog_btn.setToolTip("调参数训练随机森林并导出 model.h")
        self.train_dialog_btn.clicked.connect(self.open_train_window)
        lay.addWidget(self.train_dialog_btn)

        # ---- 普通记录（不带标签） ---- #
        lay.addWidget(self._hline("数据记录（不标注）"))
        self.rec_btn = QtWidgets.QPushButton("⏺  开始记录 CSV")
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self._toggle_record)
        self.open_dir_btn = QtWidgets.QPushButton("📂  打开数据文件夹")
        self.open_dir_btn.clicked.connect(self.open_data_dir)
        grid4 = QtWidgets.QGridLayout()
        grid4.setSpacing(4)
        grid4.addWidget(self.rec_btn, 0, 0, 1, 2)
        grid4.addWidget(self.open_dir_btn, 1, 0)
        lay.addLayout(grid4)

        self.rec_label = QtWidgets.QLabel("未记录")
        self.rec_label.setStyleSheet("color:#9e9e9e;")
        self.rec_label.setWordWrap(True)
        lay.addWidget(self.rec_label)
        lay.addStretch(1)

        # 放进滚动区：窗口拉小时也能滑动看到所有按钮
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(290)
        return scroll

    @staticmethod
    def _hline(text):
        lab = QtWidgets.QLabel("<b>%s</b>" % text)
        lab.setStyleSheet("color:#cfcfcf; padding-top:4px;")
        return lab

    # ============================ 串口操作 ============================ #
    def refresh_ports(self):
        current = self.port_box.currentText()
        ports = []
        for p in list_ports.comports():
            desc = (p.description or "").strip()
            ports.append(("%s  %s" % (p.device, desc)).strip())
        self.port_box.clear()
        if not ports:
            ports = ["(未发现串口)"]
        self.port_box.addItems(ports)
        for i, item in enumerate(ports):
            if current and item.startswith(current.split(" ")[0]):
                self.port_box.setCurrentIndex(i)
                break
        self.status.showMessage("发现 %d 个串口（CH340 一般显示为 USB-SERIAL CH340）"
                                % len(list_ports.comports()))

    def _selected_port(self):
        text = self.port_box.currentText().strip()
        if not text or text.startswith("("):
            return None
        return text.split(" ")[0]

    def _toggle_connection(self):
        if self.worker is not None:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        mock = self.mock_check.isChecked()
        port = None if mock else self._selected_port()
        if not mock and not port:
            QtWidgets.QMessageBox.warning(
                self, "提示", "没有选择串口。\n请插好板子后点「刷新」，或勾选「模拟数据」。")
            self.connect_btn.setChecked(False)
            return

        baud = int(self.baud_box.currentText())
        self.worker = DataSource("mock" if mock else "serial", port, baud)
        self.worker.sample_ready.connect(self._on_sample)
        self.worker.result_ready.connect(self._on_result)
        self.worker.status_changed.connect(self.status.showMessage)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self.connect_btn.setText("断开")
        self.connect_btn.setChecked(True)
        self.last_seq = None
        self.lost_frames = 0
        self.start_acquisition()

    def _disconnect(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None
        self.acquiring = False
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏸  暂停显示")
        self.connect_btn.setText("连接")
        self.connect_btn.setChecked(False)
        if self.recording:
            self._toggle_record()

    def _on_failed(self, msg):
        QtWidgets.QMessageBox.critical(self, "串口错误", msg)
        self._disconnect()

    # ============================ 采集控制 ============================ #
    def start_acquisition(self):
        """开始（或继续）采集：时间轴接着已有数据往后走。"""
        self.acquiring = True
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏸  暂停显示")
        self.t_start = time.time() - (self.time_axis[-1] if self.time_axis else 0.0)
        self.status.showMessage("采集中……")

    def _toggle_pause(self):
        paused = self.pause_btn.isChecked()
        self.pause_btn.setText("▶  继续显示" if paused else "⏸  暂停显示")
        self.acquiring = not paused
        self.status.showMessage("已暂停显示（串口仍在接收，记录不受影响）"
                                if paused else "采集中……")

    def clear_data(self):
        for buf in self.sources.values():
            buf.clear()
        self.time_axis.clear()
        self.rate_times.clear()
        self.sample_count = 0
        self.lost_frames = 0
        self.t_start = time.time()
        self.status.showMessage("波形已清除")

    def reset_board(self):
        """手动给 STM32 一个复位脉冲（板上 CH340 的 DTR 接了 RESET）。"""
        if self.worker is not None and self.worker.mode == "serial":
            if self.worker.reset_board():
                self.status.showMessage("已给板子发送复位脉冲，MCU 重新开始采集")
            else:
                self.status.showMessage("复位失败（串口可能已断开）")
        else:
            self.status.showMessage("只有连接真实串口时才能复位板子")

    # ============================ 数据回调 ============================ #
    def _on_sample(self, payload):
        """在 GUI 线程里接收一帧数据。"""
        sample, host_time = payload
        seq = sample[0]

        if self.last_seq is not None:
            gap = seq - self.last_seq - 1
            if 0 < gap < 1000:
                self.lost_frames += gap
        self.last_seq = seq

        acc, gyro, t_c = to_physical(sample)
        if self.t_start is None:
            self.t_start = host_time
        t_rel = host_time - self.t_start

        if self.acquiring:
            self.time_axis.append(t_rel)
            self.sources["ax"].append(acc[0])
            self.sources["ay"].append(acc[1])
            self.sources["az"].append(acc[2])
            self.sources["gx"].append(gyro[0])
            self.sources["gy"].append(gyro[1])
            self.sources["gz"].append(gyro[2])
            self.sample_count += 1
            self.rate_times.append(host_time)

        if self.recording:
            self.rec_buffer.append((
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                seq, sample[1],
                acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2], t_c,
                self.rec_label_id, self.rec_label_name,
            ))

    def _on_result(self, payload):
        """板子回传的识别结果：R,时间,类别号,置信度%,名称"""
        _t_ms, cls, conf, name = payload
        zh = {"still": "静止", "walk": "走路", "run": "跑步", "fall": "跌倒"}.get(name, name)
        color = "#ff5252" if cls == 3 else ("#4caf50" if cls == 0 else "#42a5f5")
        self.result_label.setText("识别：%s   %d%%" % (zh or cls, conf))
        self.result_label.setStyleSheet("color:%s; padding:2px 0;" % color)
        self.last_result = (cls, conf)

    def _on_refresh(self):
        """定时刷新波形与统计（默认 40FPS）。"""
        self._flush_record()

        # 1) 更新波形
        if self.time_axis:
            x = np.fromiter(self.time_axis, dtype=float, count=len(self.time_axis))
            for key, curve in self.curves.items():
                y = np.fromiter(self.sources[key], dtype=float,
                                count=len(self.sources[key]))
                curve.setData(x, y)

            span = self._window_seconds()
            t_now = x[-1]
            for p in (self.plot_acc, self.plot_gyro):
                p.setXRange(t_now - span, t_now, padding=0)

        # 3) 统计信息
        if self.recording and self.rec_start_time is not None:
            self.label_status.setText("● 正在录制「%s」：%.1f 秒 / %d 帧"
                                      % (self.rec_label_name,
                                         time.time() - self.rec_start_time,
                                         self.rec_rows + len(self.rec_buffer)))
        self.refresh_times.append(time.time())
        fps = 0.0
        if len(self.refresh_times) >= 2:
            dur = self.refresh_times[-1] - self.refresh_times[0]
            if dur > 0:
                fps = (len(self.refresh_times) - 1) / dur
        rate = 0.0
        if len(self.rate_times) >= 2:
            dur = self.rate_times[-1] - self.rate_times[0]
            if dur > 0:
                rate = (len(self.rate_times) - 1) / dur
        bad = self.worker.lines_bad if self.worker else 0
        self.stats_label.setText(
            "采样率：%.1f Hz    波形刷新：%.0f FPS\n样本：%d    丢帧：%d    无效行：%d"
            % (rate, fps, self.sample_count, self.lost_frames, bad))

    def _window_seconds(self):
        return float(self.win_box.currentText().split(" ")[0])

    def _apply_y_ranges(self):
        """按当前下拉框设置应用 Y 轴量程（只在设置变化时调用）。"""
        self._apply_y_range(self.plot_acc, self.y_acc_box.currentText())
        self._apply_y_range(self.plot_gyro, self.y_gyro_box.currentText())

    def _apply_fps(self, text):
        """切换波形刷新率（机器慢的时候可以调低）。"""
        try:
            fps = int("".join(c for c in text if c.isdigit()))
        except ValueError:
            return
        if fps > 0:
            self.ui_timer.setInterval(max(10, int(1000 / fps)))
            self.status.showMessage("波形刷新率已设为 %d FPS" % fps)

    @staticmethod
    def _apply_y_range(plot, text):
        """text 形如 '自动' 或 '±0.5 g' / '±20 °/s'。"""
        view = plot.getViewBox()
        if text.startswith("自动"):
            view.setAutoVisible(y=True)
            view.enableAutoRange(axis="y")
            return
        nums = [c for c in text if c.isdigit() or c == "."]
        try:
            half = float("".join(nums))
        except ValueError:
            return
        view.setAutoVisible(y=False)
        view.disableAutoRange(axis="y")
        view.setYRange(-half, half, padding=0)

    # ============================ 记录 / 导出 ============================ #
    def _start_record(self, label_id=-1, label_name="未标注",
                      folder=None, file_tag=None, note=""):
        """打开一个新的 CSV 文件开始记录（带标签/不带标签共用）。"""
        if self.recording:
            return False
        folder = folder or DATA_DIR
        os.makedirs(folder, exist_ok=True)
        tag = file_tag or "MPU6050"
        name = "%s_%s.csv" % (tag, datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = os.path.join(folder, name)
        try:
            self.rec_file = open(path, "w", newline="", encoding="utf-8-sig")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "无法创建文件", str(exc))
            return False
        self.rec_writer = csv.writer(self.rec_file)
        self.rec_writer.writerow(["# MPU6050 六轴数据记录（普中-天马 F407 板载 MPU6050，腰侧佩戴）"])
        self.rec_writer.writerow(["# 加速度 g（±4g，8192 LSB/g）；角速度 °/s（±500dps，65.5 LSB/dps）；温度 ℃"])
        self.rec_writer.writerow(["# 动作标签：%s（label_id=%d）%s" % (label_name, label_id, note)])
        self.rec_writer.writerow(["pc_time", "seq", "t_ms",
                                  "ax_g", "ay_g", "az_g",
                                  "gx_dps", "gy_dps", "gz_dps", "temp_c",
                                  "label_id", "label_name"])
        self.recording = True
        self.rec_rows = 0
        self.rec_label_id = label_id
        self.rec_label_name = label_name
        self.rec_start_time = time.time()
        self.rec_path = path
        return True

    def _stop_record(self):
        """停止记录，返回 (行数, 时长秒)。"""
        self.recording = False
        self._flush_record()
        if self.rec_file:
            self.rec_file.close()
        self.rec_file = None
        self.rec_writer = None
        secs = (time.time() - self.rec_start_time) if self.rec_start_time else 0.0
        self.rec_start_time = None
        return self.rec_rows, secs

    # ---------------- 带标签的数据采集（课设数据集） ---------------- #
    def start_label_record(self, label_id, name_en, name_zh):
        """点按钮或按键盘 1~4：倒计时结束后开始录制该类动作。"""
        if self.worker is None:
            self.status.showMessage("请先点「连接」（或勾选「模拟数据」）再开始录制")
            return
        if self.recording:
            self._stop_record()
        self.pending_label = (label_id, name_en, name_zh)
        seconds = int(self.count_box.value())
        if seconds <= 0:
            self._begin_label_record()
        else:
            self.countdown_left = seconds
            self._tick_countdown()

    def _tick_countdown(self):
        if self.pending_label is None:
            return
        zh = self.pending_label[2]
        if self.countdown_left > 0:
            self.label_status.setText("准备录制「%s」：%d…" % (zh, self.countdown_left))
            self.status.showMessage("准备录制「%s」，%d 秒后开始" % (zh, self.countdown_left))
            self.countdown_left -= 1
            QtCore.QTimer.singleShot(1000, self._tick_countdown)
        else:
            self._begin_label_record()

    def _begin_label_record(self):
        if self.pending_label is None:
            return
        label_id, name_en, name_zh = self.pending_label
        self.pending_label = None
        if self._start_record(label_id, name_zh, folder=DATASET_DIR, file_tag=name_en):
            self.label_status.setText("● 正在录制「%s」" % name_zh)
            self.status.showMessage("正在录制「%s」→ %s" % (name_zh, self.rec_path))
        else:
            self.label_status.setText("录制启动失败")

    def stop_label_record(self):
        if self.pending_label is not None and not self.recording:
            self.pending_label = None
            self.label_status.setText("已取消倒计时")
            return
        if not self.recording:
            return
        rows, secs = self._stop_record()
        self.label_status.setText("✔ 已保存「%s」：%.1f 秒 / %d 帧\n%s"
                                  % (self.rec_label_name, secs, rows,
                                     os.path.basename(self.rec_path)))
        self.status.showMessage("已保存 %s" % self.rec_path)

    def open_dataset_dir(self):
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.startfile(DATASET_DIR)

    def open_train_window(self):
        """打开随机森林训练窗口（训练逻辑复用 模型训练\\训练随机森林.py）。"""
        try:
            import 训练窗口                       # 同目录下的模块
        except Exception as exc:                  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "无法打开训练窗口",
                "加载训练模块失败：%s\n（需要 numpy / scikit-learn / matplotlib）" % exc)
            return
        if getattr(self, "_train_dialog", None) is None:
            self._train_dialog = 训练窗口.TrainDialog(self)
        self._train_dialog.show()
        self._train_dialog.raise_()
        self._train_dialog.activateWindow()

    def _toggle_record(self):
        """普通记录按钮（不标注）。"""
        if not self.recording:
            if self._start_record():
                self.rec_btn.setText("⏹  停止记录 CSV")
                self.rec_label.setText("记录中：%s" % os.path.basename(self.rec_path))
                self.status.showMessage("正在记录到 %s" % self.rec_path)
            else:
                self.rec_btn.setChecked(False)
        else:
            rows, _secs = self._stop_record()
            self.rec_btn.setText("⏺  开始记录 CSV")
            self.rec_btn.setChecked(False)
            self.rec_label.setText("已保存 %d 行到「数据记录」文件夹" % rows)
            self.status.showMessage("记录已停止并保存")

    def keyPressEvent(self, event):
        """键盘 1/2/3/4 快速打标签，Esc 停止。"""
        key = event.key()
        if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_4:
            idx = key - QtCore.Qt.Key_1
            if idx < len(LABELS):
                label_id, name_en, name_zh = LABELS[idx]
                self.start_label_record(label_id, name_en, name_zh)
                return
        if key == QtCore.Qt.Key_Escape:
            self.stop_label_record()
            return
        super().keyPressEvent(event)

    def _flush_record(self):
        if self.rec_writer and self.rec_buffer:
            self.rec_writer.writerows(self.rec_buffer)
            self.rec_rows += len(self.rec_buffer)
            self.rec_buffer = []
            self.rec_file.flush()

    def open_data_dir(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.startfile(DATA_DIR)                       # Windows 专用

    def closeEvent(self, event):
        if self.recording:
            rows, secs = self._stop_record()
            self.status.showMessage("关闭前已保存 %d 帧（%.1f 秒）" % (rows, secs))
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft YaHei", 10))
    win = MainWindow()
    win.refresh_ports()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
