"""Read-only full-cohort existing-rule filter. Durable rows; no generation API."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,signal,sqlite3,time
from pathlib import Path
from experiments.qwen_curriculum import common,campaign
STOP=False

def dumps(v):return json.dumps(v,ensure_ascii=False,allow_nan=False)
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(dumps(value));tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def connect(root):
 c=sqlite3.connect(Path(root)/'audit.sqlite',timeout=60);c.row_factory=sqlite3.Row
 c.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-32768;
 CREATE TABLE IF NOT EXISTS pairs (ordinal INTEGER PRIMARY KEY,sample_id TEXT UNIQUE NOT NULL,city TEXT NOT NULL,place_id INTEGER NOT NULL,payload TEXT NOT NULL,filter_status TEXT,quality TEXT,review_status TEXT,review TEXT,repair_status TEXT,repair TEXT,UNIQUE(city,place_id));
 CREATE INDEX IF NOT EXISTS filter_idx ON pairs(filter_status,ordinal);
 CREATE INDEX IF NOT EXISTS review_idx ON pairs(filter_status,review_status,ordinal);
 CREATE INDEX IF NOT EXISTS repair_idx ON pairs(repair_status,ordinal);''');return c

def prepare(root,manifest):
 from experiments.qwen_curriculum import balanced_night
 root=Path(root);manifest=Path(manifest).resolve();root.mkdir(parents=True,exist_ok=True)
 quality=balanced_night.runtime().quality_definition(clip_device='cpu')
 expected=json.loads((manifest.parent/'complete.json').read_text());digest=sha(manifest)
 if digest!=expected['manifest_sha256']:raise ValueError('Canonical manifest changed')
 cfg=campaign.frozen(root/'filter_protocol.json',dict(version=1,manifest=str(manifest),manifest_sha256=digest,expected=expected['count'],quality=quality,
  module_sha256=sha(__file__),seed=42,threads=4,clip_device='cpu',matcher_device='cuda',generation_enabled=False,
  rule_policy='Unmodified current project CandidateVerifier; existing thresholds, no semantic recalibration'))
 c=connect(root)
 if c.execute('select count(*) from pairs').fetchone()[0]!=cfg['expected']:
  with manifest.open() as f,c:
   for i,line in enumerate(f):
    r=json.loads(line);old=c.execute('select payload from pairs where ordinal=?',(i,)).fetchone()
    if old and json.loads(old[0])!=r:raise ValueError('Queue differs from canonical row')
    if not old:c.execute('insert into pairs(ordinal,sample_id,city,place_id,payload) values(?,?,?,?,?)',(i,r['sample_id'],r['city'],int(r['place_id']),dumps(r)))
  if i+1!=cfg['expected']:raise ValueError('Canonical count mismatch')
 if c.execute('select count(*) from pairs').fetchone()[0]!=cfg['expected']:raise ValueError('Queue count mismatch')
 c.close();return cfg

def status(root,phase=None):
 root=Path(root);c=connect(root);cfg=campaign.config_file(root/'filter_protocol.json')
 def counts(column):return {r[0] or 'pending':r[1] for r in c.execute(f'select {column},count(*) from pairs group by {column}')}
 value=dict(expected=cfg['expected'],filter=counts('filter_status'),review=counts('review_status'),repair=counts('repair_status'),generation_enabled=False,
  updated_unix=time.time(),phase=phase or 'status',paused=(root/'pause_requested').exists())
 c.close();atomic(root/'progress.json',value)
 (root/'report.md').write_text('# 全量夜景规则筛选、逐对视觉审核和Prompt修订\n\n当前为持续更新的运行报告，未完成项不代表通过。未调用生图，不修改原图库。\n\n```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```\n\n第一轮使用当前项目规则；仅通过项进入第二轮。第二轮是本地Qwen逐对视觉审核，不冒称人工逐一检查6万张。疑难、失败、缺证据单列。仅明确问题生成Prompt补丁或同地点换源图计划，不生成新图片。\n')
 return value

def check_resources():
 mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if ':' in x}
 if mem['MemAvailable']<1024*1024:raise RuntimeError('Host available RAM below emergency 1GiB; safely stop, resume later')

def run(root,limit=0):
 global STOP
 os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('ADAPTVPR_CLIP_LOCAL_FILES_ONLY','1')
 import torch,numpy as np,cv2
 from PIL import Image
 from experiments.qwen_curriculum import balanced_night
 root=Path(root);cfg=campaign.config_file(root/'filter_protocol.json')
 if sha(__file__)!=cfg['module_sha256'] or sha(cfg['manifest'])!=cfg['manifest_sha256']:raise ValueError('Frozen filter inputs/code changed')
 if cfg['quality']!=balanced_night.runtime().quality_definition(clip_device='cpu'):raise ValueError('Existing project rule changed')
 torch.set_num_threads(4);torch.set_num_interop_threads(2)
 c=connect(root);done=0;start=time.monotonic();last=0
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
 with common_lock(root/'filter.lock'):
  verifier=None
  for row in c.execute('select ordinal,payload from pairs where filter_status is null order by ordinal'):
   if STOP or (root/'pause_requested').exists():break
   if limit and done>=limit:break
   check_resources();r=json.loads(row['payload'])
   try:
    for key in ['source','output']:
     if sha(r[key+'_path'])!=r[key+'_sha256']:raise ValueError(key+' image hash mismatch')
    with Image.open(r['source_path']) as im:a=im.convert('RGB')
    with Image.open(r['output_path']) as im:b=im.convert('RGB')
   except (OSError,ValueError) as e:
    q=dict(error=str(e),error_type='integrity_error');state='error'
   else:
    if verifier is None:verifier=balanced_night.runtime().CandidateVerifier(device='cuda',clip_device='cpu')
    seed=int(hashlib.sha256(r['sample_id'].encode()).hexdigest()[:8],16)%2147483647
    torch.manual_seed(seed);np.random.seed(seed);cv2.setRNGSeed(seed)
    with torch.inference_mode():q=verifier.evaluate(a,b,'night')
    state='pass' if q['passed'] else 'reject'
   c.execute('update pairs set filter_status=?,quality=? where ordinal=?',(state,dumps(q),row['ordinal']));c.commit();done+=1
   if time.monotonic()-last>=20 or done==1:
    v=status(root,'filter');atomic(root/'filter_runtime.json',dict(session_completed=done,session_seconds=time.monotonic()-start,pairs_per_second=done/(time.monotonic()-start),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3));print(dumps(v),flush=True);last=time.monotonic()
  v=status(root,'filter');c.close();print(dumps(v),flush=True)
 return not v['filter'].get('pending',0)

@contextlib.contextmanager
def common_lock(path):
 import fcntl
 with open(path,'a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def export(root):
 root=Path(root);c=connect(root)
 for name,column,value in [('filter_pass.jsonl','filter_status','pass'),('filter_reject.jsonl','filter_status','reject'),('filter_errors.jsonl','filter_status','error')]:
  tmp=root/(name+'.tmp')
  with tmp.open('w') as f:
   for r in c.execute('select sample_id,payload,quality from pairs where '+column+'=? order by ordinal',(value,)):
    f.write(dumps(dict(sample_id=r['sample_id'],source_record=json.loads(r['payload']),quality=json.loads(r['quality']),eligible_for_training=False))+'\n')
  tmp.replace(root/name)
 c.close();return status(root)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','filter','status','export']);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,default=common.WORKSPACE_ROOT/'outputs/night_dataset_v1/manifest.jsonl');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 if a.phase=='prepare':print(dumps(prepare(a.root,a.manifest)))
 elif a.phase=='filter':run(a.root,a.limit)
 elif a.phase=='export':export(a.root)
 else:print(dumps(status(a.root)))
if __name__=='__main__':main()
