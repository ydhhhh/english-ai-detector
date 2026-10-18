"""Cache small official licensing/version statements, never corpus samples."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/english_v2_data/audit/rights_evidence';OUT.mkdir(exist_ok=True)
receipts=[]
def get(name,url):
    request=urllib.request.Request(url,headers={'User-Agent':'english-v2-public-data-audit/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        content=response.read();status=response.status
    (OUT/name).write_bytes(content)
    receipts.append({'path':str((OUT/name).relative_to(ROOT)),'url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'http_status':status,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    return content
for repo,filename,key in [('google-research-datasets/natural-questions','nq_open/README.md','nq'),('allenai/s2orc','README.md','s2orc')]:
    history=json.loads(get(key+'_README_latest_commit.json',f'https://api.github.com/repos/{repo}/commits?path={filename}&per_page=1'))
    revision=history[0]['sha']
    get(key+'_README_pinned.md',f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}')
get('wikimedia_Terms_of_Use.html','https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use')
(OUT/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps([{'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256']} for r in receipts]))
