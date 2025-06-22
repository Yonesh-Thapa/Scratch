from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class LearningCurvePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.figure, self.axs = plt.subplots(2, 2, figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        self.setLayout(layout)
        self.metrics = {}
        self.last_symbol = None
        self.load_metrics()
        self.plot_curves()

    def load_metrics(self):
        # Load metrics for all available symbols (from test_closed_loop.py output)
        metrics_dir = os.path.join(os.path.dirname(__file__), '../tests')
        self.metrics = {}
        for fname in os.listdir(metrics_dir):
            if fname.startswith('metrics_') and fname.endswith('.npz'):
                symbol = fname[len('metrics_'):-4]
                try:
                    data = np.load(os.path.join(metrics_dir, fname))
                    self.metrics[symbol] = {
                        'vision_acc': data['vision_acc'],
                        'hand_sim': data['hand_sim'],
                        'ear_acc': data['ear_acc'],
                        'mouth_sim': data['mouth_sim']
                    }
                except Exception:
                    continue
        if self.metrics:
            self.last_symbol = sorted(self.metrics.keys())[0]

    def plot_curves(self):
        self.load_metrics()
        self.figure.clf()
        axs = self.figure.subplots(2, 2)
        if not self.metrics:
            for ax in axs.flat:
                ax.set_title('No metrics found')
                ax.axis('off')
            self.canvas.draw()
            return
        # Plot for the first symbol (or last viewed)
        symbol = self.last_symbol
        data = self.metrics[symbol]
        axs[0,0].plot(data['vision_acc'], label='Vision Acc')
        axs[0,0].set_title('Vision Accuracy')
        axs[0,1].plot(data['hand_sim'], label='Hand Sim')
        axs[0,1].set_title('Hand Similarity')
        axs[1,0].plot(data['ear_acc'], label='Ear Acc')
        axs[1,0].set_title('Ear Accuracy')
        axs[1,1].plot(data['mouth_sim'], label='Mouth Sim')
        axs[1,1].set_title('Mouth Similarity')
        for ax in axs.flat:
            ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def get_stats(self):
        # Show stats for the last symbol
        if not self.metrics or not self.last_symbol:
            return ["No stats available."]
        data = self.metrics[self.last_symbol]
        stats = [
            f"Symbol: {self.last_symbol}",
            f"Vision Acc: {np.mean(data['vision_acc']):.2f}",
            f"Hand Sim: {np.mean(data['hand_sim']):.4f}",
            f"Ear Acc: {np.mean(data['ear_acc']):.2f}",
            f"Mouth Sim: {np.mean(data['mouth_sim']):.4f}"
        ]
        return stats
