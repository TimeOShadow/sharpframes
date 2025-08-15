import os, sys, threading, queue, traceback
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QRadioButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QTextEdit
)
from PySide6.QtCore import QTimer

# 复用现有逻辑
sys.path.append(r"D:\sharpframes")
import sharpframes_cli as core

class Worker(threading.Thread):
    def __init__(self, params, log_cb, done_cb):
        super().__init__(daemon=True)
        self.params = params
        self.log = log_cb
        self.done = done_cb

    def run(self):
        try:
            p = self.params
            self.log("Pass1: scoring...")
            scores, n_total = core.pass_one_score(p["input"], every=max(1, p["sample_every"]))
            if not scores:
                self.log("No frames scored. Check input.")
                self.done()
                return
            vals = [s.score for s in scores]
            if p["thr_mode"] == "auto":
                thr = core.robust_auto_threshold(vals, k=p["k"])
            else:
                thr = float(p["thr_val"])
            candidates = sorted([s.index for s in scores if s.score >= thr])
            selected = core.select_by_ratio(candidates, keep_ratio=p["keep_ratio"], min_interval=p["min_interval"])
            self.log(f"Total={n_total}, Scored={len(scores)}, Candidates={len(candidates)}, Selected={len(selected)}")
            if not selected:
                self.log("Nothing selected. Lower threshold or increase ratio.")
                self.done()
                return
            self.log("Pass2: exporting DNG...")
            core.pass_two_export(p["input"], selected, p["output"], save_format="dng")
            self.log(f"Done. Output: {p['output']}")
        except Exception as e:
            self.log("ERROR:\n" + traceback.format_exc())
        finally:
            self.done()

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sharpframes GUI (Qt, DNG)")
        self.log_q = queue.Queue()
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drain_logs)
        self.timer.start(80)

    def build_ui(self):
        # 输入视频
        self.in_edit = QLineEdit()
        in_btn = QPushButton("浏览")
        in_btn.clicked.connect(self.pick_input)

        # 输出目录
        self.out_edit = QLineEdit()
        out_btn = QPushButton("选择")
        out_btn.clicked.connect(self.pick_output)

        # 阈值
        thr_group = QGroupBox("清晰度阈值")
        self.auto_radio = QRadioButton("自动阈值")
        self.auto_radio.setChecked(True)
        self.k_spin = QDoubleSpinBox(); self.k_spin.setDecimals(2); self.k_spin.setRange(-10.0, 10.0); self.k_spin.setValue(2.0)
        self.fix_radio = QRadioButton("固定阈值")
        self.thr_val = QDoubleSpinBox(); self.thr_val.setDecimals(0); self.thr_val.setRange(0, 1e9); self.thr_val.setValue(0.0)

        thr_l1 = QHBoxLayout(); thr_l1.addWidget(self.auto_radio); thr_l1.addWidget(QLabel("k=")); thr_l1.addWidget(self.k_spin); thr_l1.addStretch(1)
        thr_l2 = QHBoxLayout(); thr_l2.addWidget(self.fix_radio); thr_l2.addWidget(QLabel("阈值=")); thr_l2.addWidget(self.thr_val); thr_l2.addStretch(1)
        thr_v = QVBoxLayout(); thr_v.addLayout(thr_l1); thr_v.addLayout(thr_l2); thr_group.setLayout(thr_v)

        # 密度/步长
        self.ratio = QDoubleSpinBox(); self.ratio.setDecimals(2); self.ratio.setRange(0.0, 1.0); self.ratio.setValue(0.2)
        self.minint = QSpinBox(); self.minint.setRange(1, 100000); self.minint.setValue(2)
        self.sample = QSpinBox(); self.sample.setRange(1, 100000); self.sample.setValue(1)

        dens = QGroupBox("密度/步长")
        dl = QHBoxLayout()
        dl.addWidget(QLabel("保留比例(0~1)")); dl.addWidget(self.ratio)
        dl.addWidget(QLabel("最小间隔")); dl.addWidget(self.minint)
        dl.addWidget(QLabel("评分步长")); dl.addWidget(self.sample)
        dl.addStretch(1)
        dens.setLayout(dl)

        # 日志与按钮
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.go_btn = QPushButton("开始")
        self.go_btn.clicked.connect(self.start)

        # 顶层布局
        l = QVBoxLayout()
        row1 = QHBoxLayout(); row1.addWidget(QLabel("输入视频")); row1.addWidget(self.in_edit); row1.addWidget(in_btn)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("输出目录")); row2.addWidget(self.out_edit); row2.addWidget(out_btn)
        l.addLayout(row1); l.addLayout(row2); l.addWidget(thr_group); l.addWidget(dens); l.addWidget(self.log_view); l.addWidget(self.go_btn)
        self.setLayout(l)

    def pick_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm *.MP4 *.MOV)")
        if path: self.in_edit.setText(path)

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if path: self.out_edit.setText(path)

    def log(self, msg):
        self.log_q.put(str(msg))

    def drain_logs(self):
        while True:
            try:
                msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            self.log_view.append(msg)

    def start(self):
        inp = self.in_edit.text().strip()
        outp = self.out_edit.text().strip()
        if not inp or not os.path.isfile(inp):
            self.log("请输入有效的输入视频路径")
            return
        if not outp:
            self.log("请选择输出目录")
            return
        params = {
            "input": inp,
            "output": outp,
            "thr_mode": "auto" if self.auto_radio.isChecked() else "fix",
            "k": float(self.k_spin.value()),
            "thr_val": float(self.thr_val.value()),
            "keep_ratio": float(self.ratio.value()),
            "min_interval": int(self.minint.value()),
            "sample_every": int(self.sample.value()),
        }
        self.go_btn.setEnabled(False)
        def done(): self.go_btn.setEnabled(True)
        Worker(params, self.log, done).start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main()
    w.resize(820, 560)
    w.show()
    sys.exit(app.exec())