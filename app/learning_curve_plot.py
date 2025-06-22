from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class LearningCurvePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        # Create four separate figures/canvases for each module
        self.figures = []
        self.canvases = []
        self.labels = ['Vision Accuracy', 'Hand Similarity', 'Ear Accuracy', 'Mouth Similarity']
        for i in range(4):
            fig, ax = plt.subplots(figsize=(6, 3))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.figures.append((fig, ax))
            self.canvases.append(canvas)
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        self.setLayout(layout)
        self.metrics = {}
        self.last_symbol = None
        self.load_metrics()
        self.plot_curves()
        # Add timer for real-time updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.plot_curves)
        self.timer.start(1000)  # update every 1000 ms (1 second)

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
        metric_keys = ['vision_acc', 'hand_sim', 'ear_acc', 'mouth_sim']
        for i, (fig, ax) in enumerate(self.figures):
            ax.clear()
            if not self.metrics:
                ax.set_title('No metrics found')
                ax.axis('off')
                self.canvases[i].draw()
                continue
            for symbol, data in self.metrics.items():
                ax.plot(data[metric_keys[i]], label=f'{symbol}')
            ax.set_title(self.labels[i])
            ax.legend(title='Symbol', fontsize='small')
            fig.tight_layout()
            self.canvases[i].draw()

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
