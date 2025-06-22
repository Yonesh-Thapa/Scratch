import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtCore import QTimer
from .learning_curve_plot import LearningCurvePlot

class ModuleManager:
    def __init__(self):
        self.processes = {}

    def start_module(self, name, cmd, output_callback=None):
        if name not in self.processes or self.processes[name] is None:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.processes[name] = proc
            if output_callback:
                from threading import Thread
                def stream_output():
                    for line in proc.stdout:
                        output_callback(name, line.rstrip())
                Thread(target=stream_output, daemon=True).start()
        else:
            print(f"{name} already running.")

    def stop_module(self, name):
        if name in self.processes and self.processes[name] is not None:
            self.processes[name].terminate()
            self.processes[name] = None

    def stop_all(self):
        for name in list(self.processes.keys()):
            self.stop_module(name)

    def is_running(self, name):
        return self.processes.get(name) is not None and self.processes[name].poll() is None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI System Launcher")
        self.manager = ModuleManager()
        layout = QVBoxLayout()

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.curve_plot = LearningCurvePlot(self)
        layout.addWidget(self.curve_plot)
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        self.buttons = {}
        self.test_module_name = "Test"
        modules = {
            "Vision": ["python", "-m", "vision.vision_module", "--rule", "delta", "--lr", "0.01"],
            "Hand": ["python", "hand/hand_module.py", "--lr", "0.01"],
            "Ear": ["python", "ear/ear_module.py", "--lr", "0.01"],
            "Mouth": ["python", "mouth/mouth_module.py", "--lr", "0.01"],
            "Test": ["python", "tests/test_closed_loop.py"]
        }
        for name, cmd in modules.items():
            btn = QPushButton(f"Start {name}")
            btn.clicked.connect(lambda checked, n=name, c=cmd: self.toggle_module(n, c))
            layout.addWidget(btn)
            self.buttons[name] = btn

        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visuals)
        self.timer.start(2000)  # update every 2 seconds

    def toggle_module(self, name, cmd):
        if self.manager.is_running(name):
            self.manager.stop_module(name)
            self.log.append(f"Stopped {name}")
            print(f"Stopped {name}")
            self.buttons[name].setText(f"Start {name}")
            if name == self.test_module_name:
                self.manager.stop_all()
                self.log.append("Stopped all modules because Test was stopped.")
                print("Stopped all modules because Test was stopped.")
                for n in self.buttons:
                    self.buttons[n].setText(f"Start {n}")
        else:
            self.manager.start_module(name, cmd, self.append_log)
            self.log.append(f"Started {name}")
            print(f"Started {name}")
            self.buttons[name].setText(f"Stop {name}")

    def append_log(self, name, line):
        self.log.append(f"[{name}] {line}")
        print(f"[{name}] {line}")  # Print to terminal for real-time feedback

    def update_visuals(self):
        self.curve_plot.plot_curves()
        stats = self.curve_plot.get_stats()
        self.stats_label.setText("\n".join(stats))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())