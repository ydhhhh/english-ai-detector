"""Acquire whole public pinned Beemo parquet; never preview text."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time,urllib.request
ROOT=Path(__file__).resolve().parents[3]
focus=ROOT/'research/english_v2_data/modern_candidates/focus_recommendation.json'
info=json.loads(focus.read_text())['Beemo'];spec=info['files'][0]
assert info['dataset_revision']=='9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162'
path=ROOT/'research/english_v2_data/raw/beemo/train-00000-of-00001.parquet'
part=path.with_suffix('.parquet.part');assert not path.exists() and not part.exists()
start=time.monotonic();h=hashlib.sha256();count=0
req=urllib.request.Request(spec['download_url'],headers={'User-Agent':'english-v2-public-data-audit/1.0'})
with urllib.request.urlopen(req,timeout=120) as r,part.open('xb') as out:
    assert r.status==200
    etag=r.headers.get('etag')
    while chunk:=r.read(1024*1024):out.write(chunk);h.update(chunk);count+=len(chunk)
assert count==spec['size_bytes'] and h.hexdigest()==spec['lfs_sha256']
os.replace(part,path)
receipt={'kind':'complete_official_Beemo_acquisition','completed_at_utc':datetime.now(timezone.utc).isoformat(),'revision':info['dataset_revision'],'url':spec['download_url'],'bytes':count,'sha256':h.hexdigest(),'network_seconds':time.monotonic()-start,'etag':etag,'local_path':str(path.relative_to(ROOT)),'complete':True,'previewed_text':False,'roles_assigned':False,'detector_loaded':False}
(ROOT/'research/english_v2_data/audit/beemo_acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
