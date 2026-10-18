"""Retry only the failed complete public MGB validation file, no selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time,urllib.request
ROOT=Path(__file__).resolve().parents[3]
plan=json.loads((ROOT/'research/english_v2_data/audit/maga_acquisition_plan.json').read_text())
spec=next(r for r in plan['files'] if r['native_path']=='val/MGB_val.jsonl')
path=ROOT/spec['local_path'];part=path.with_suffix('.jsonl.retry.part')
assert not path.exists() and not part.exists()
start=time.monotonic();h=hashlib.sha256();count=0
req=urllib.request.Request(spec['url']+'?download=true',headers={'User-Agent':'english-v2-public-data-audit/1.0'})
with urllib.request.urlopen(req,timeout=120) as response,part.open('xb') as f:
    assert response.status==200
    etag=response.headers.get('etag')
    while chunk:=response.read(1024*1024):f.write(chunk);h.update(chunk);count+=len(chunk)
assert count==spec['bytes'] and h.hexdigest()==spec['sha256']
os.replace(part,path)
record=dict(spec,complete=True,observed_bytes=count,observed_sha256=h.hexdigest(),network_seconds=time.monotonic()-start,completed_at_utc=datetime.now(timezone.utc).isoformat(),retry_reason='Original pinned URL transport raised URLError; retry pinned URL with download=true, same official LFS SHA enforced',etag=etag)
path.with_suffix('.jsonl.retry.receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'file':spec['native_path'],'complete':True,'bytes':count,'seconds':record['network_seconds']}))
