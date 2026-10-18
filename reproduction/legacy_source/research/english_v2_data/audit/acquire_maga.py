"""Acquire pinned complete official MAGA files, hash before any parsing.

No authorization credentials, dataset preview, model or train/dev split.
"""
import argparse,concurrent.futures,hashlib,json,os,shutil,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
AUDIT=ROOT/'research/english_v2_data/audit'
RAW=ROOT/'research/english_v2_data/raw/maga'
REV='34c51fc6e71561bf1203945d90d253269f9f2ab8'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    api_path=ROOT/'research/english_v2_data/modern_candidates/maga_hf_api.json'
    api=json.loads(api_path.read_text())
    assert api['sha']==REV and not api.get('gated') and not api.get('private')
    names=['train/MGB_train.jsonl','val/MGB_val.jsonl','train/MAGA_train.jsonl','val/MAGA_val.jsonl']
    lookup={r['rfilename']:r for r in api['siblings']}
    specs=[]
    for name in names:
        r=lookup[name]
        assert r['size']==r['lfs']['size'] and len(r['lfs']['sha256'])==64
        specs.append({'native_path':name,'revision':REV,'url':f'https://huggingface.co/datasets/anyangsong/MAGA/resolve/{REV}/{name}','bytes':r['size'],'sha256':r['lfs']['sha256'],'local_path':str((RAW/name).relative_to(ROOT))})
    RAW.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(RAW).free>sum(r['bytes'] for r in specs)+1024**3
    plan={'kind':'fixed_complete_file_acquisition_no_role_assignment','created_at_utc':datetime.now(timezone.utc).isoformat(),'official_repo':'https://huggingface.co/datasets/anyangsong/MAGA','revision':REV,'metadata_receipt_sha256':sha(api_path),'files':specs,'total_bytes':sum(r['bytes'] for r in specs),'parallel_downloads':2,'selection':'All four complete MGB and MAGA English train/validation files, no row-body selection. Official split is provenance only, not project role.','quarantine':'Raw is evidence only, not a train release. Entire wikiHow parent/variant closure is forbidden for v2; all other source rights/quality/old-used barriers still require audit. No redistribution implied.'}
    planpath=AUDIT/'maga_acquisition_plan.json'
    if planpath.exists():
        old=json.loads(planpath.read_text());assert old['files']==specs
    else: planpath.write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic()
    def one(spec):
        path=ROOT/spec['local_path'];path.parent.mkdir(parents=True,exist_ok=True)
        record=dict(spec,started_at_utc=datetime.now(timezone.utc).isoformat())
        if path.exists():
            assert path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256']
            return dict(record,completed_from='already_complete_verified',network_seconds=0.0)
        part=path.with_suffix(path.suffix+'.part')
        if part.exists():raise RuntimeError('Unexpected partial; preserve and inspect before retry')
        t=time.monotonic();h=hashlib.sha256();size=0
        req=urllib.request.Request(spec['url'],headers={'User-Agent':'english-v2-public-data-audit/1.0'})
        with urllib.request.urlopen(req,timeout=120) as response,part.open('xb') as output:
            record['http_status']=response.status
            record['x_repo_commit']=response.headers.get('x-repo-commit')
            record['etag']=response.headers.get('ETag')
            assert response.status==200
            while chunk:=response.read(1024*1024):
                output.write(chunk);h.update(chunk);size+=len(chunk)
        record.update(network_seconds=time.monotonic()-t,observed_bytes=size,observed_sha256=h.hexdigest())
        assert size==spec['bytes'] and h.hexdigest()==spec['sha256'], 'Incomplete or changed payload is never admitted'
        os.replace(part,path)
        record.update(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat())
        (path.with_suffix(path.suffix+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'file':spec['native_path'],'complete':True,'bytes':size,'seconds':round(record['network_seconds'],2)}),flush=True)
        return record
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(one,r):r for r in specs}
        for task in concurrent.futures.as_completed(pending):
            spec=pending[task]
            try:results.append(task.result())
            except Exception as e:
                results.append(dict(spec,complete=False,error_type=type(e).__name__,error=str(e)))
                print(json.dumps({'file':spec['native_path'],'complete':False,'error':type(e).__name__}),flush=True)
    result={'kind':'complete_file_acquisition_receipts','created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(planpath),'wall_seconds':time.monotonic()-started,'files':results,'all_complete':all(r.get('complete') or r.get('completed_from')=='already_complete_verified' for r in results),'models_loaded':False,'dataset_text_previewed':False,'roles_constructed':False}
    (AUDIT/'maga_acquisition_receipts.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_complete']:raise SystemExit(1)
if __name__=='__main__':main()
