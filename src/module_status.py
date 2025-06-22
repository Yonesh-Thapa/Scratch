# Module Status Checker for RL Log Visualizer
# This code adds a section to the RL log visualizer to check if core modules are running (by attempting to connect to their ports)
import socket

MODULES = [
    {"name": "Vision", "host": "127.0.0.1", "port": 5001},
    {"name": "Hand", "host": "127.0.0.1", "port": 5002},
    {"name": "Ear", "host": "127.0.0.1", "port": 5003},
    {"name": "Mouth", "host": "127.0.0.1", "port": 5004},
]

def check_module_status():
    status = {}
    for mod in MODULES:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect((mod["host"], mod["port"]))
            status[mod["name"]] = "Online"
        except Exception:
            status[mod["name"]] = "Offline"
        finally:
            s.close()
    return status

def print_module_status():
    status = check_module_status()
    print("\n=== Module Status ===")
    for name, stat in status.items():
        print(f"{name}: {stat}")
    print("====================\n")

# To use in rl_log_visualizer.py, import and call print_module_status() at the start of main
