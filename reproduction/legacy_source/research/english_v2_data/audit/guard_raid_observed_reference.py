"""Transparent first100native-row superset for an earlier automatic landing preview.

No manual preview, no detector; persist only reference hashes/lineage/near edges.
The complete earlier base signature/near evidence stays unchanged.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,sqlite3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit';BASE=ROOT/'research/english_v2_data/raw/raid_observed_reference_v1';CACHE=ROOT/'.cache/english_v2_lexical_v1'
def sh(x):return hashlib.sha256(x.encode()).hexdigest()
def fs(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    helper=module('lexical_signature',OUT/'lexical_signature.py');worker=module('base_near',OUT/'build_full_lexical_near.py');worker.worker_init()
    path=BASE/'rows_000000_100.json';receipt=json.loads((BASE/'receipt.json').read_text());assert fs(path)==receipt['sha256'] and receipt['revision']=='865cac74188466cb0c3b7574a10204007b57a459'
    raw=json.loads(path.read_text());assert len(raw['rows'])==100
    db=sqlite3.connect('file:'+str(CACHE/'signatures.sqlite3')+'?mode=ro',uri=True)
    count=db.execute('SELECT count(*) FROM signatures').fetchone()[0];allnorm={h:u for u,h in db.execute('SELECT uid,norm_sha FROM signatures')}
    refs=[];new=[];refshingles={};keyowners=defaultdict(set);indexed={};need=defaultdict(list)
    for item in raw['rows']:
        assert not item.get('truncated_cells');r=item['row'];text=r['generation'];normal=helper.normalized(text);nh=sh(normal);sig,n=helper.signature(text);uid=allnorm.get(nh)
        if uid is None:
            uid=count+len(new);allnorm[nh]=uid;new.append({'uid':uid,'normalized_sha256':nh,'raw_sha256':sh(text),'signature':sig.tobytes(),'shingle_count':n})
        refshingles[nh]=helper.shingle_set(text);keys=[sh('raw:'+sh(text)),sh('norm:'+nh)]
        tokens=helper.TOKEN.findall(normal)
        if len(tokens)>=32:keys.append(sh('prefix32:'+sh(' '.join(tokens[:32]))))
        for k in ['id','source_id','adv_source_id']:
            if isinstance(r.get(k),str) and r[k].strip() and r[k].casefold() not in {'none','null','nan'}:keys.append(sh('RAID_identifier:'+r[k]))
        if isinstance(r.get('prompt'),str) and r['prompt'].strip():keys.append(sh('prompt:'+sh(helper.normalized(r['prompt']))))
        refs.append({'native_row_index':item['row_idx'],'id':r['id'],'source_id':r['source_id'],'adv_source_id':r.get('adv_source_id'),'domain':r['domain'],'model':r['model'],'attack':r['attack'],'normalized_sha256':nh,'raw_sha256':sh(text),'norm_uid':uid,'observable_key_hashes':sorted(set(keys))})
        if n>=20:
            indexed[nh]=(uid,sig,n)
            for band in range(8):need[(band,sig[band*4:(band+1)*4].tobytes())].append(nh)
    base_bucket_members=defaultdict(list)
    for uid,nh,blob,n in db.execute('SELECT uid,norm_sha,signature,shingle_count FROM signatures WHERE shingle_count>=20'):
        sig=np.frombuffer(blob,dtype=np.uint32)
        for band in range(8):
            key=(band,sig[band*4:(band+1)*4].tobytes())
            if key in need:base_bucket_members[key].append((uid,nh,n))
    pairs={};conservative=[]
    for key,refhashes in need.items():
        members={nh:(u,n) for u,nh,n in base_bucket_members[key]}
        for nh in refhashes:members[nh]=(indexed[nh][0],indexed[nh][2])
        if len(members)>=100:
            anchor=min(members)
            for nh in sorted(members):
                if nh!=anchor:conservative.append({'left_normalized_sha256':anchor,'right_normalized_sha256':nh,'kind':'conservative_oversized_bucket','jaccard':None,'band':key[0]})
        else:
            for nh in refhashes:
                for other,(uid,n) in members.items():
                    if other==nh:continue
                    a,b=sorted([nh,other]);pairs[(a,b)]=(uid,n,nh,other)
    edges=[]
    for (a,b),(uid,n,reference,other) in pairs.items():
        aa=refshingles[reference]
        if min(len(aa),n)/max(len(aa),n)<.85:continue
        bb=refshingles[other] if other in refshingles else worker.shingles(uid)
        inter=len(aa&bb);j=inter/max(1,len(aa)+len(bb)-inter)
        if j>=.85:edges.append({'left_normalized_sha256':a,'right_normalized_sha256':b,'kind':'verified_jaccard_ge_0.85','jaccard':j,'band':None})
    edges+=conservative
    schema=pq.read_schema(OUT/'full_lexical_signature_index.parquet');p1=OUT/'raid_observed_reference_new_signatures.parquet';pq.write_table(pa.Table.from_pylist(new,schema=schema),p1,compression='zstd')
    p2=OUT/'raid_observed_reference_metadata.parquet';pq.write_table(pa.Table.from_pylist(refs),p2,compression='zstd')
    es=pa.schema([('left_normalized_sha256',pa.string()),('right_normalized_sha256',pa.string()),('kind',pa.string()),('jaccard',pa.float64()),('band',pa.int16())]);p3=OUT/'raid_observed_reference_near_edges.parquet';pq.write_table(pa.Table.from_pylist(edges,schema=es),p3,compression='zstd')
    summary={'kind':'transparent_observed_reference_superset_increment','created_at_utc':datetime.now(timezone.utc).isoformat(),'reason':'Earlier HF landing automatically rendered abstracts data preview without retained exact visible IDs. Root approved conservative pinned native train first100 rows as a transparent superset, not an assertion all100were actually read.','original_base_signature_and_near_files_unchanged':True,'first_native_rows':100,'native_parent_components':len({r['source_id'] for r in refs}),'new_normalized_identities':len(new),'domains':sorted({r['domain'] for r in refs}),'verified_edges':sum(r['kind']=='verified_jaccard_ge_0.85' for r in edges),'conservative_edges':len(conservative),'no_body_preview':True,'no_models_or_scores':True,'fresh_roles_prohibited_whole_native_exact_near_component':True,'train_or_observed_diagnostic_only_subject_to_other_barriers':True,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':fs(p)} for p in [path,BASE/'receipt.json',CACHE/'complete.json',OUT/'full_lexical_near_completion.json',Path(__file__)]],'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':fs(p),'rows':pq.ParquetFile(p).metadata.num_rows} for p in [p1,p2,p3]]}
    (OUT/'raid_observed_reference_guard.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['first_native_rows','native_parent_components','new_normalized_identities','domains','verified_edges','conservative_edges']}))
if __name__=='__main__':main()
