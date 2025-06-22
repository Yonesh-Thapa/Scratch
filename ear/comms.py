import asyncio
import json
import uuid
import time
import logging

class AsyncModuleServer:
    def __init__(self, host, port, module_name, handler_registry=None, logger=None):
        self.host = host
        self.port = port
        self.module_name = module_name
        self.handler_registry = handler_registry or {}
        self.logger = logger or logging.getLogger(module_name)
        self.server = None

    async def handle_client(self, reader, writer):
        try:
            data = await reader.read(4)
            if not data or len(data) < 4:
                writer.close()
                await writer.wait_closed()
                return
            msg_len = int.from_bytes(data, 'big')
            msg_data = await reader.read(msg_len)
            msg = json.loads(msg_data.decode('utf-8'))
            self.logger.info(f"[AsyncServer] Received: {msg}")
            cmd = msg.get('cmd')
            handler = self.handler_registry.get(cmd)
            if handler:
                resp = await handler(msg)
            else:
                resp = {'error': f'Unknown command: {cmd}'}
            resp_bytes = json.dumps(resp).encode('utf-8')
            writer.write(len(resp_bytes).to_bytes(4, 'big') + resp_bytes)
            await writer.drain()
        except Exception as e:
            self.logger.error(f"[AsyncServer] Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.logger.info(f"[AsyncServer] Listening on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

class AsyncModuleClient:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("AsyncModuleClient")

    async def send(self, host, port, msg, timeout=5):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            msg_bytes = json.dumps(msg).encode('utf-8')
            writer.write(len(msg_bytes).to_bytes(4, 'big') + msg_bytes)
            await writer.drain()
            data = await asyncio.wait_for(reader.read(4), timeout=timeout)
            if not data or len(data) < 4:
                writer.close()
                await writer.wait_closed()
                return None
            msg_len = int.from_bytes(data, 'big')
            resp_data = await asyncio.wait_for(reader.read(msg_len), timeout=timeout)
            resp = json.loads(resp_data.decode('utf-8'))
            writer.close()
            await writer.wait_closed()
            self.logger.info(f"[AsyncClient] Sent to {host}:{port}, got: {resp}")
            return resp
        except Exception as e:
            self.logger.error(f"[AsyncClient] Error: {e}")
            return {'error': str(e)}

def make_message(src, dst, cmd, payload=None):
    return {
        'msg_id': str(uuid.uuid4()),
        'timestamp': time.time(),
        'src': src,
        'dst': dst,
        'cmd': cmd,
        'payload': payload or {},
    }
