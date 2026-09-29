"""Local, model-free UI for the existing Stage3 human audit CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from targeted.human_audit import HUMAN_FIELDS, load_plans, read_human_audit


class ConflictError(ValueError):
    pass


class AuditStore:
    def __init__(self, plans_path, csv_path):
        self.plans_path = Path(plans_path).resolve()
        self.csv_path = Path(csv_path).resolve()
        self.plans = load_plans(self.plans_path)
        self.ids = [row['target']['sample_id'] for row in self.plans]
        self.lock = threading.Lock()
        self.images = {}
        for index, row in enumerate(self.plans):
            directory = (self.plans_path.parent / row['audit_directory']).resolve()
            if not directory.is_relative_to(self.plans_path.parent):
                raise ValueError('Invalid audit directory')
            for kind, filename in (('source', 'source.png'), ('crop', 'context_crop.png')):
                path = directory / filename
                if not path.is_file():
                    raise ValueError(f'Missing audit image: {path}')
                self.images[index, kind] = path
        self.snapshot()  # Require the existing CSV; never fabricate labels.

    def _read(self):
        before = self.csv_path.read_bytes()
        labels = read_human_audit(self.csv_path, set(self.ids))
        if set(labels) != set(self.ids):
            raise ValueError('CSV sample IDs must exactly match the selected audit plans')
        if self.csv_path.read_bytes() != before:
            raise ConflictError('CSV changed while reading; reload and retry')
        return before, labels

    def snapshot(self):
        with self.lock:
            data, labels = self._read()
            return dict(revision=hashlib.sha256(data).hexdigest(), csv_path=str(self.csv_path),
                        samples=[dict(sample_id=sample_id, label=labels[sample_id]) for sample_id in self.ids])

    def save(self, record, revision):
        if not isinstance(record, dict) or set(record) != set(HUMAN_FIELDS):
            raise ValueError('Expected exactly the six human audit fields')
        if any(not isinstance(value, str) for value in record.values()):
            raise ValueError('All labels must be strings')
        if record['sample_id'] not in self.ids:
            raise ValueError('Unknown sample_id')
        with self.lock:
            data, labels = self._read()
            if revision != hashlib.sha256(data).hexdigest():
                raise ConflictError('CSV 已被其他窗口或编辑器修改。请先重新载入，避免覆盖新标签。')
            labels[record['sample_id']] = record
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                                 dir=self.csv_path.parent, prefix='.human-audit-', delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=HUMAN_FIELDS)
                    writer.writeheader()
                    writer.writerows(labels.values())
                    stream.flush()
                    os.fsync(stream.fileno())
                read_human_audit(temporary, set(self.ids))  # Same enums and cross-field validation as evaluation.
                if self.csv_path.read_bytes() != data:
                    raise ConflictError('CSV changed during save; reload and retry')
                # Immutable, content-addressed backup before every actual replacement.
                backup_dir = self.csv_path.parent / 'human_audit_backups'
                backup_dir.mkdir(exist_ok=True)
                backup = backup_dir / f'{self.csv_path.stem}.{hashlib.sha256(data).hexdigest()}.csv'
                if not backup.exists():
                    with backup.open('xb') as stream:
                        stream.write(data)
                os.chmod(temporary, self.csv_path.stat().st_mode & 0o777)
                os.replace(temporary, self.csv_path)
                return dict(revision=hashlib.sha256(self.csv_path.read_bytes()).hexdigest())
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def make_server(store, port):
    token = secrets.token_urlsafe(32)
    html = (ROOT / 'targeted/annotation_ui.html').read_text(encoding='utf-8').replace('__SESSION_TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, content, mime='application/json; charset=utf-8'):
            data = content if isinstance(content, bytes) else json.dumps(content, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def trusted_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.trusted_host():
                return self.send(403, dict(error='Local host only'))
            path = urlsplit(self.path).path
            try:
                if path == '/':
                    return self.send(200, html.encode(), 'text/html; charset=utf-8')
                if path == '/api/state':
                    return self.send(200, store.snapshot())
                if path.startswith('/image/'):
                    _, _, number, kind = path.split('/')
                    image = store.images[int(number), kind]
                    return self.send(200, image.read_bytes(), 'image/png')
                self.send(404, dict(error='Not found'))
            except (ValueError, KeyError, OSError) as exc:
                self.send(400, dict(error=str(exc)))

        def do_POST(self):
            if not self.trusted_host() or self.headers.get('X-Audit-Token') != token:
                return self.send(403, dict(error='Invalid local session'))
            if self.path != '/api/save':
                return self.send(404, dict(error='Not found'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or set(data) != {'record', 'revision'}:
                    raise ValueError('Invalid save request')
                self.send(200, store.save(data['record'], data['revision']))
            except ConflictError as exc:
                self.send(409, dict(error=str(exc)))
            except (ValueError, KeyError, OSError) as exc:
                self.send(400, dict(error=str(exc)))

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, default=ROOT / 'outputs/stage3_targeted/planner_v2/targeted_edit_plans.jsonl')
    parser.add_argument('--csv', type=Path, default=ROOT / 'outputs/stage3_targeted/planner_human_audit.csv')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the browser automatically')
    args = parser.parse_args(argv)
    store = AuditStore(args.plans, args.csv)
    server = make_server(store, args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'人工标注：{url}\nCSV：{store.csv_path}\nCtrl+C 停止。只有点击保存才写入标签。', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
