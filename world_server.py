"""Local observer UI for Firstfolk. The UI is never an agent observation."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from living_world import World


class Simulation:
    def __init__(self, path, seed):
        self.path = Path(path)
        self.world = World.load(path) if self.path.exists() else World(seed=seed)
        self.lock = threading.RLock()
        self.running = False
        self.speed = 8
        self.error = None
        self.stop = threading.Event()

    def worker(self):
        while not self.stop.wait(.1):
            with self.lock:
                if self.running:
                    try:
                        self.world.step(self.speed)
                        if not self.world.living():
                            self.running = False
                        if self.world.data['tick'] % 40 < self.speed:
                            self.world.save(self.path)
                    except Exception as error:
                        self.error = str(error)
                        self.running = False

    def snapshot(self):
        return {**self.world.snapshot(), 'running': self.running,
                'speed': self.speed, 'error': self.error}


def handler(sim):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value, kind='application/json'):
            payload = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self.reply(200, Path(__file__).with_name('world.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/world':
                with sim.lock:
                    self.reply(200, sim.snapshot())
            elif path == '/api/export':
                with sim.lock:
                    sim.world.save(sim.path)
                    self.reply(200, sim.path.read_bytes())
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            # Local observer mutations require an explicit JSON request header,
            # and reject cross-origin browser requests. No remote binding.
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.reply(415, {'error': 'application/json required'})
                return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                self.reply(403, {'error': 'Cross-origin mutation rejected'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 4096:
                    raise ValueError('Request too large')
                body = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')
                with sim.lock:
                    path = urlparse(self.path).path
                    if path == '/api/control':
                        if set(body)-{'running', 'speed'}:
                            raise ValueError('Unknown control')
                        if 'running' in body and type(body['running']) is not bool:
                            raise ValueError('running must be boolean')
                        if 'speed' in body and (type(body['speed']) is not int or not 1 <= body['speed'] <= 40):
                            raise ValueError('speed must be integer 1..40')
                        sim.running = body.get('running', sim.running)
                        sim.speed = body.get('speed', sim.speed)
                    elif path == '/api/step':
                        if set(body)-{'steps'}:
                            raise ValueError('Unknown step field')
                        sim.world.step(body.get('steps', 1))
                    elif path == '/api/save':
                        if body:
                            raise ValueError('save accepts no arguments')
                    else:
                        self.reply(404, {'error': 'Not found'})
                        return
                    sim.world.save(sim.path)
                    self.reply(200, sim.snapshot())
            except (ValueError, TypeError) as error:
                self.reply(400, {'error': str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save', default='data/firstfolk/world-v7.json')
    args = parser.parse_args()
    sim = Simulation(args.save, args.seed)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(sim))
    worker = threading.Thread(target=sim.worker, daemon=True)
    worker.start()
    print(f'Firstfolk observer: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop.set()
        worker.join(5)
        with sim.lock:
            sim.world.save(sim.path)
        server.server_close()


if __name__ == '__main__':
    main()
