"""Local JSON HTTP API. No third-party packages required."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from service import Service
from cognition import think, reason, perceive


def handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.dispatch('GET')

        def do_POST(self):
            self.dispatch('POST')

        def dispatch(self, method):
            try:
                url = urlparse(self.path)
                parts = url.path.strip('/').split('/')
                body = {}
                if method == 'POST':
                    size = int(self.headers.get('Content-Length', '0'))
                    max_size=1_500_000 if len(parts)==3 and parts[0]=='agents' and parts[2] in ('perceive','cognition','profile') else 65536
                    if not 0 <= size <= max_size:
                        raise ValueError('Request body exceeds route limit')
                    body = json.loads(self.rfile.read(size) or b'{}')
                    if not isinstance(body, dict):
                        raise ValueError('JSON body must be an object')
                if method == 'GET' and parts == ['health']:
                    result = {'status': 'ok', 'engine': 'adaptive-cognitive-v6'}
                elif method=='GET' and parts==['brain','map']:
                    result=service.brain_map()
                elif method=='GET' and parts==['inventory']:
                    result=service.inventory()
                elif method=='GET' and parts==['research','sources']:
                    result=service.sources()
                elif parts == ['agents'] and method == 'GET':
                    result = service.list()
                elif parts == ['agents'] and method == 'POST':
                    result = service.create(**body)
                elif parts == ['simulation', 'tick'] and method == 'POST':
                    result = service.tick(**body)
                elif parts == ['teach'] and method == 'POST':
                    result = service.teach(**body)
                elif parts==['messages'] and method=='POST':
                    result=service.message(**body)
                elif parts==['teach-skill'] and method=='POST':
                    result=service.teach_skill(**body)
                elif len(parts) >= 2 and parts[0] == 'agents':
                    agent_id = parts[1]
                    if len(parts) == 2 and method == 'GET':
                        result = service.get(agent_id)
                    elif len(parts) == 3 and parts[2] == 'memories' and method == 'GET':
                        query = parse_qs(url.query).get('q', ['experience'])[0]
                        result = service.memories(agent_id, query)
                    elif len(parts) == 3 and parts[2] == 'experiences' and method == 'POST':
                        result = service.experience(agent_id, **body)
                    elif len(parts) == 3 and parts[2] == 'relationships' and method == 'GET':
                        result = service.relationships(agent_id)
                    elif len(parts) == 3 and parts[2] == 'events' and method == 'GET':
                        result = service.events(agent_id)
                    elif len(parts) == 3 and parts[2] == 'think' and method == 'POST':
                        if set(body)!={'task'}:
                            raise ValueError('think accepts only task; verified tool results are supplied internally')
                        result = think(service, agent_id, **body)
                    elif len(parts)==3 and parts[2]=='reason' and method=='POST':
                        result=reason(service,agent_id,**body)
                    elif len(parts)==3 and parts[2]=='perceive' and method=='POST':
                        result=perceive(service,agent_id,**body)
                    elif len(parts)==3 and parts[2]=='brain' and method=='GET':
                        result=service.brain_map(agent_id)
                    elif len(parts)==3 and parts[2]=='inventory' and method=='GET':
                        result=service.inventory(agent_id)
                    elif len(parts)==3 and parts[2]=='observe' and method=='POST':
                        result=service.observe(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='goals' and method=='POST':
                        result=service.goal(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='consolidate' and method=='POST':
                        result=service.consolidate(agent_id)
                    elif len(parts)==3 and parts[2]=='lesions' and method=='POST':
                        result=service.lesion(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='solve' and method=='POST':
                        result=service.solve(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='train' and method=='POST':
                        result=service.train(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='profile' and method=='POST':
                        result=service.profile(agent_id,**body)
                    elif len(parts)==3 and parts[2]=='cognition' and method=='POST':
                        result=service.cognition_operation(agent_id,**body)
                    else:
                        raise LookupError('Route not found')
                else:
                    raise LookupError('Route not found')
                self.respond(200, result)
            except LookupError as exc:
                self.respond(404, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.log_error('Internal request failure')
                self.respond(500, {'error': 'Internal server error'})

        def respond(self, status, result):
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--db', default=str(Path(__file__).parent / 'data' / 'agents.sqlite3'))
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Service(args.db)))
    print(f'Agent backend: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
