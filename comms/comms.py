"""
Communication Layer: Provides message passing (IPC, sockets, queues) for all modules.
Optimized for energy efficiency and minimal data retention.
"""
import socket
import pickle
import time

class CommsClient:
    def __init__(self, host, port, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send(self, msg):
        data = pickle.dumps(msg)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            s.sendall(data)
            response = b''
            while True:
                try:
                    packet = s.recv(4096)
                    if not packet:
                        break
                    response += packet
                except socket.timeout:
                    break
            if response:
                return pickle.loads(response)
            return None

# Example usage:
# client = CommsClient('127.0.0.1', 5001)
# resp = client.send({'cmd': 'recognize', 'data': ...})

# This module can be imported by any process to communicate with other modules.
# No global state, no central controller, only direct message passing.
