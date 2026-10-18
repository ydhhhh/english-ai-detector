"""Cache official historical S2ORC licensing and immediate mirror metadata."""
from pathlib import Path
from datetime import datetime,timezone
import urllib.request,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit/rights_evidence'
receipts=[]
def get(name,url):
    req=urllib.request.Request(url,headers={'User-Agent':'english-v2-public-data-audit/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:body=r.read();status=r.status
    p=OUT/name;p.write_bytes(body);receipts.append({'path':str(p.relative_to(ROOT)),'url':url,'status':status,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()});return body
history=json.loads(get('s2orc_pre2023_README_commit.json','https://api.github.com/repos/allenai/s2orc/commits?path=README.md&until=2022-12-31T23:59:59Z&per_page=1'))
rev=history[0]['sha']
get('s2orc_pre2023_README.md',f'https://raw.githubusercontent.com/allenai/s2orc/{rev}/README.md')
# This is metadata/card only; no upstream corpus payload downloaded.
api=json.loads(get('embedding_training_data_api.json','https://huggingface.co/api/datasets/sentence-transformers/embedding-training-data?blobs=true'))
get('embedding_training_data_README.md',f'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/{api["sha"]}/README.md')
(OUT/'s2orc_history_receipts.json').write_text(json.dumps({'historical_official_revision':rev,'embedding_mirror_revision':api['sha'],'receipts':receipts},indent=2)+'\n')
print(json.dumps({'historical_official_revision':rev,'embedding_mirror_revision':api['sha'],'receipts':[{'path':r['path'],'bytes':r['bytes']} for r in receipts]}))
