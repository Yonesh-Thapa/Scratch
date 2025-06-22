"""
Communication Layer: Provides message passing (IPC, sockets, queues) for all modules.
Optimized for energy efficiency and minimal data retention.
"""
import socket
import pickle
import struct
import time

class CommsClient:
    def __init__(self, host, port, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send(self, msg):
        data = pickle.dumps(msg)
        msg_len = struct.pack('>I', len(data))  # 4-byte big-endian
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            s.sendall(msg_len + data)
            # --- Receive length-prefixed reply ---
            header = self._recvall(s, 4)
            if not header:
                return None
            resp_len = struct.unpack('>I', header)[0]
            resp_data = self._recvall(s, resp_len)
            if resp_data:
                return pickle.loads(resp_data)
            return None

    def _recvall(self, sock, n):
        """Helper to receive n bytes or return None if EOF."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
