import os
import time
import threading
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt, Slot, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches

from .constants import *
from .model_utils import load_or_train_model, build_small_ann
from .ga import GeneticOptimizer
from .plotting import draw_electrodes, draw_threshold_marker, draw_threshold_history

class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(dict)
    plot_update = Signal(object)

class ElectrodeOptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NNGA-Optimizer:Sciatic Nerve Stimulation Electrode Geometry Optimizer")
        self.setMinimumSize(1000, 650)

        self.signals = WorkerSignals()
        self.signals.log.connect(self.append_log)
        self.signals.progress.connect(self.update_progress_label)
        self.signals.plot_update.connect(self.update_plot_partial)
        self.signals.finished.connect(self.optimization_finished)

        self.model = None
        self.scaler = None

        self._create_ui()
        self._connect_ui()

        try:
            if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_SCALER_PATH):
                self.model, self.scaler = load_or_train_model(model_path=DEFAULT_MODEL_PATH,
                                                              scaler_path=DEFAULT_SCALER_PATH,
                                                              csv_path=None, signals=self.signals)
                self.append_log("Model and scaler loaded on startup.")
            else:
                self.append_log("No pretrained model found. Use File -> Load model or provide CSV to train.")
        except Exception as e:
            self.append_log(f"Auto-load failed: {e}")


    def _create_ui(self):
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        controls = QGroupBox("Controls")
        c_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Area (µm^2):"))
        self.area_input = QDoubleSpinBox(); self.area_input.setRange(100.0, 10000.0); self.area_input.setDecimals(3); self.area_input.setSingleStep(1.0); self.area_input.setValue(10000.0)
        row.addWidget(self.area_input)

        row.addWidget(QLabel("Pitch (µm):"))
        self.pitch_input = QDoubleSpinBox(); self.pitch_input.setRange(50.0, 1000.0); self.pitch_input.setDecimals(3); self.pitch_input.setSingleStep(1.0); self.pitch_input.setValue(1000.0)
        row.addWidget(self.pitch_input)
        c_layout.addLayout(row)

        self.btn_load_model = QPushButton("Load Model & Scaler")
        self.btn_load_csv = QPushButton("Load CSV (trainable)")
        self.btn_predict = QPushButton("Predict Threshold")
        self.btn_optimize = QPushButton("Run Optimizer (NNGA)"); self.btn_optimize.setStyleSheet("font-weight:bold;")
        self.btn_save_results = QPushButton("Save Results CSV"); self.btn_save_results.setEnabled(False)
        self.btn_export_png = QPushButton("Export Electrode PNG"); self.btn_export_png.setEnabled(False)

        btn_row = QHBoxLayout(); btn_row.addWidget(self.btn_load_model); btn_row.addWidget(self.btn_load_csv)
        c_layout.addLayout(btn_row)
        c_layout.addWidget(self.btn_predict); c_layout.addWidget(self.btn_optimize); c_layout.addWidget(self.btn_save_results); c_layout.addWidget(self.btn_export_png)

        ga_group = QGroupBox("Optimizer Parameters")
        ga_layout = QHBoxLayout()
        self.spin_pop = QSpinBox(); self.spin_pop.setRange(10,200); self.spin_pop.setValue(POP_SIZE)
        self.spin_gen = QSpinBox(); self.spin_gen.setRange(5,200); self.spin_gen.setValue(N_GENERATIONS)
        self.dspin_mut = QDoubleSpinBox(); self.dspin_mut.setRange(0.0,1.0); self.dspin_mut.setDecimals(2); self.dspin_mut.setValue(MUTATION_RATE)
        ga_layout.addWidget(QLabel("Pop:")); ga_layout.addWidget(self.spin_pop); ga_layout.addWidget(QLabel("Gens:")); ga_layout.addWidget(self.spin_gen); ga_layout.addWidget(QLabel("Mut:")); ga_layout.addWidget(self.dspin_mut)
        ga_group.setLayout(ga_layout); c_layout.addWidget(ga_group)

        self.progress_label = QLabel("Progress: idle")
        self.log_label = QLabel(""); self.log_label.setWordWrap(True); self.log_label.setAlignment(Qt.AlignTop); self.log_label.setMinimumHeight(150); self.log_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        c_layout.addWidget(self.progress_label); c_layout.addWidget(self.log_label)

        controls.setLayout(c_layout); left.addWidget(controls)

        right = QVBoxLayout()
        self.fig = Figure(figsize=(6,5)); self.canvas = FigureCanvas(self.fig); right.addWidget(self.canvas)
        self.ax_elec = self.fig.add_subplot(121); self.ax_plot = self.fig.add_subplot(122); self.fig.tight_layout()

        layout.addLayout(left, 1); layout.addLayout(right, 2)

    def _connect_ui(self):
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_load_csv.clicked.connect(self.on_load_csv)
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_optimize.clicked.connect(self.on_optimize)
        self.btn_save_results.clicked.connect(self.on_save_results)
        self.btn_export_png.clicked.connect(self.on_export_png)

    @Slot()
    def on_load_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Keras model (h5)", "", "H5 Files (*.h5)")
        if not model_file:
            return
        scaler_file, _ = QFileDialog.getOpenFileName(self, "Select scaler (joblib)", "", "Joblib Files (*.joblib *.pkl)")
        if not scaler_file:
            return
        try:
            self.model, self.scaler = load_or_train_model(model_path=model_file, scaler_path=scaler_file, csv_path=None, signals=self.signals)
            self.append_log(f"Loaded model: {model_file} and scaler: {scaler_file}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load model/scaler: {e}")

    @Slot()
    def on_load_csv(self):
        csv_file, _ = QFileDialog.getOpenFileName(self, "Select training CSV", "", "CSV Files (*.csv)")
        if not csv_file:
            return
        reply = QMessageBox.question(self, "Train model?", "Train a small model on the chosen CSV now? This will run in the app (may take minutes).", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
        t = threading.Thread(target=self._train_from_csv_thread, args=(csv_file,))
        t.start()

    def _train_from_csv_thread(self, csv_file):
        try:
            self.signals.log.emit("Starting training thread...")
            model, scaler = load_or_train_model(model_path=DEFAULT_MODEL_PATH, scaler_path=DEFAULT_SCALER_PATH, csv_path=csv_file, epochs=60, verbose=0, signals=self.signals)
            self.model, self.scaler = model, scaler
            self.signals.log.emit("Training finished and model saved.")
        except Exception as e:
            self.signals.log.emit(f"Training failed: {e}")

    @Slot()
    def on_predict(self):
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "No model", "Load or train a model first.")
            return
        area = float(self.area_input.value()); pitch = float(self.pitch_input.value())
        inp = np.array([[FIXED_DISPLACEMENT, area, pitch]], dtype=float)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            inp_scaled = self.scaler.transform(inp)
        pred = float(self.model.predict(inp_scaled, verbose=0)[0][0])
        self.append_log(f"Prediction -> Area={area:.3f}µm, Pitch={pitch:.3f}µm : Threshold = {pred:.6f} mA")
        draw_electrodes(self.ax_elec, self.canvas, area, pitch, optimized=None)
        draw_threshold_marker(self.ax_plot, self.canvas, area, pitch, pred, label="Input")

    @Slot()
    def on_optimize(self):
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "No model", "Load or train a model first.")
            return
        pop, gens, mut = int(self.spin_pop.value()), int(self.spin_gen.value()), float(self.dspin_mut.value())
        init_area, init_pitch = float(self.area_input.value()), float(self.pitch_input.value())
        init = np.array([init_area, init_pitch], dtype=float)
        self._set_ui_enabled(False)
        self.append_log(f"Starting Parameters: pop={pop}, gens={gens}, mut={mut:.3f}")
        t = threading.Thread(target=self._run_ga_thread, args=(init, pop, gens, mut))
        t.start()

    def _run_ga_thread(self, init, pop, gens, mut):
        try:
            ga = GeneticOptimizer(self.model, self.scaler, fixed_displacement=FIXED_DISPLACEMENT, pop_size=pop, n_generations=gens, mutation_rate=mut, tournament_size=TOURNAMENT_SIZE, bounds=BOUNDS, signals=self.signals, rng=None)
            ga.run(init_individual=init)
        except Exception as e:
            self.signals.log.emit(f"error: {e}")
            self._set_ui_enabled(True)

    @Slot(dict)
    def optimization_finished(self, result):
        best = result["best_individual"]; best_pred = result["best_pred"]; best_hist, mean_hist = result["best_history"], result["mean_history"]
        self.append_log(f"Optimization result -> Area={best[0]:.3f} µm, Pitch={best[1]:.3f} µm, Threshold={best_pred:.6f} mA")
        draw_electrodes(self.ax_elec, self.canvas, self.area_input.value(), self.pitch_input.value(), optimized=best)
        draw_threshold_history(self.ax_plot, self.canvas, best_hist, mean_hist, model=self.model, scaler=self.scaler, init=(self.area_input.value(), self.pitch_input.value(), None))
        self._set_ui_enabled(True); self.btn_save_results.setEnabled(True); self.btn_export_png.setEnabled(True)
        self._last_result = { "initial_area": float(self.area_input.value()), "initial_pitch": float(self.pitch_input.value()), "opt_area": float(best[0]), "opt_pitch": float(best[1]), "opt_threshold": float(best_pred), "history_best": best_hist, "history_mean": mean_hist }

    def update_plot_partial(self, data):
        best = data.get("best_history", []); mean = data.get("mean_history", []); gen = data.get("generation", 0)
        ax = self.ax_plot; ax.clear()
        ax.plot(range(1, len(best)+1), best, label='Best', marker='o')
        ax.plot(range(1, len(mean)+1), mean, label='Mean', marker='x')
        ax.set_title(f"NNGA Optimizer Progress (gen {gen})"); ax.set_xlabel("Generation"); ax.set_ylabel("Threshold (mA)"); ax.grid(True); ax.legend(); self.canvas.draw_idle()

    def append_log(self, text):
        old = self.log_label.text(); new = old + ("\n" if old else "") + f"[{time.strftime('%H:%M:%S')}] {text}"
        if len(new) > 4000: new = new[-4000:]; self.log_label.setText(new)

    def update_progress_label(self, percent):
        self.progress_label.setText(f"Progress: {percent}%")

    def _set_ui_enabled(self, enabled):
        self.btn_load_model.setEnabled(enabled); self.btn_load_csv.setEnabled(enabled); self.btn_predict.setEnabled(enabled); self.btn_optimize.setEnabled(enabled); self.spin_pop.setEnabled(enabled); self.spin_gen.setEnabled(enabled); self.dspin_mut.setEnabled(enabled)

    def on_save_results(self):
        if not hasattr(self, "_last_result"): QMessageBox.information(self, "No results", "No results to save yet."); return
        fname, _ = QFileDialog.getSaveFileName(self, "Save results CSV", "optimization_results.csv", "CSV Files (*.csv)"); 
        if not fname: return
        df = pd.DataFrame([self._last_result]); df.to_csv(fname, index=False); QMessageBox.information(self, "Saved", f"Results saved to {fname}")

    def on_export_png(self):
        if not hasattr(self, "_last_result"): QMessageBox.information(self, "No results", "Run optimization first."); return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Electrode Geometry PNG", "electrodes.png", "PNG Files (*.png)")
        if not fname: return
    
        fig_tmp = Figure(figsize=(6,5)); ax_tmp = fig_tmp.add_subplot(111)
        best = self._last_result
        draw_electrodes(self.ax_elec, self.canvas, best['initial_area'], best['initial_pitch'], optimized=[best['opt_area'], best['opt_pitch']])
        for patch in list(self.ax_elec.patches):
            ax_tmp.add_patch(patches.Rectangle((patch.get_x(), patch.get_y()), patch.get_width(), patch.get_height(),
                                              fill=False, edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth(), linestyle=patch.get_linestyle(), label=patch.get_label()))
        ax_tmp.set_xlim(self.ax_elec.get_xlim()); ax_tmp.set_ylim(self.ax_elec.get_ylim()); ax_tmp.set_aspect('equal', 'box')
        ax_tmp.set_title(self.ax_elec.get_title()); ax_tmp.set_xlabel(self.ax_elec.get_xlabel()); ax_tmp.set_ylabel(self.ax_elec.get_ylabel())
        fig_tmp.tight_layout(); fig_tmp.savefig(fname, dpi=300)
        QMessageBox.information(self, "Saved", f"Electrode geometry saved as PNG: {fname}")

