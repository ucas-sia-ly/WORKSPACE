"""Annotation persistence/security checks on temporary files, never real human labels."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('annotate_targeted',ROOT/'scripts/annotate_targeted.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.csv_path=self.root/'human.csv'
        rows=[]
        for sid in ('a','b'):
            folder=self.root/sid;folder.mkdir()
            for name in ('source.png','context_crop.png'):(folder/name).write_bytes(b'image fixture')
            rows.append(dict(schema_version=2,contract='TargetedEditPlan',target=dict(sample_id=sid),
                             audit_directory=sid,status='rejected',decision=dict(editable=False)))
        self.plans=self.root/'plans.jsonl';self.plans.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write_csv([dict(sample_id='a'),dict(sample_id='b',comment='previous human note')])
        self.store=module.AuditStore(self.plans,self.csv_path)

    def write_csv(self,rows):
        with self.csv_path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=module.HUMAN_FIELDS)
            writer.writeheader();writer.writerows(rows)

    def record(self,**changes):
        row=dict(sample_id='a',human_region_type='road',human_support_surface='paved_ground',
                 human_editable='true',human_family='traffic_cones',comment='manual, 中文\nsecond line')
        row.update(changes);return row

    def test_load_is_readonly_and_does_not_expose_predictions(self):
        before=self.csv_path.read_bytes();state=self.store.snapshot()
        self.assertEqual(self.csv_path.read_bytes(),before)
        self.assertEqual(set(state['samples'][0]),{'sample_id','label'})
        self.assertEqual(state['samples'][1]['label']['comment'],'previous human note')

    def test_save_keeps_other_rows_and_backs_up_exact_old_bytes(self):
        before=self.csv_path.read_bytes();state=self.store.snapshot()
        result=self.store.save(self.record(),state['revision'])
        current=self.store.snapshot()
        self.assertEqual(current['revision'],result['revision'])
        self.assertEqual(current['samples'][0]['label'],self.record())
        self.assertEqual(current['samples'][1]['label']['comment'],'previous human note')
        backups=list((self.root/'human_audit_backups').glob('*.csv'))
        self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),before)

    def test_external_modification_conflict_preserves_new_file(self):
        revision=self.store.snapshot()['revision']
        self.write_csv([dict(sample_id='a',comment='external edit'),dict(sample_id='b')])
        before=self.csv_path.read_bytes()
        with self.assertRaises(module.ConflictError):self.store.save(self.record(),revision)
        self.assertEqual(self.csv_path.read_bytes(),before)

    def test_invalid_labels_rejected_without_touching_csv(self):
        before=self.csv_path.read_bytes();revision=self.store.snapshot()['revision']
        for record in (self.record(human_family='none'),self.record(human_region_type='roof'),self.record(sample_id='other'),dict(sample_id='a')):
            with self.assertRaises(ValueError):self.store.save(record,revision)
            self.assertEqual(self.csv_path.read_bytes(),before)
        self.assertFalse(list(self.root.glob('.human-audit-*')))

    def test_partial_label_allowed(self):
        self.store.save(self.record(human_editable='',human_family=''),self.store.snapshot()['revision'])
        self.assertEqual(self.store.snapshot()['samples'][0]['label']['human_editable'],'')

    def test_http_image_whitelist_and_save_session_token(self):
        server=module.make_server(self.store,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join()))
        base=f'http://127.0.0.1:{server.server_port}'
        with urlopen(base) as response:html=response.read().decode()
        self.assertIn('Stage3',html)
        token=html.split("const TOKEN='")[1].split("'")[0]
        with urlopen(base+'/api/state') as response:state=json.load(response)
        with urlopen(base+'/image/0/crop') as response:self.assertEqual(response.read(),b'image fixture')
        request=Request(base+'/api/save',data=json.dumps(dict(record=self.record(),revision=state['revision'])).encode(),headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as caught:urlopen(request)
        self.assertEqual(caught.exception.code,403)
        request.add_header('X-Audit-Token',token)
        with urlopen(request) as response:self.assertEqual(response.status,200)
        with self.assertRaises(HTTPError):urlopen(base+'/image/0/arbitrary-file')
        with self.assertRaises(HTTPError) as caught:urlopen(Request(base+'/api/state',headers={'Host':'untrusted.example'}))
        self.assertEqual(caught.exception.code,403)


if __name__=='__main__':unittest.main()
