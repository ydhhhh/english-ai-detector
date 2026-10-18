"""Full old/new lexical LSH graph, persisted hash-only evidence. CPU, no roles/models."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from functools import lru_cache
import argparse,hashlib,importlib.util,json,sqlite3,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit';CACHE=ROOT/'.cache/english_v2_lexical_v1';W=None

def fs(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ro(p):return sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
def worker_init():
    global W
    spec=importlib.util.spec_from_file_location('lexical_signature',OUT/'lexical_signature.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    W=(ro(CACHE/'signatures.sqlite3'),ro(CACHE/'new_texts.sqlite3'),ro(ROOT/'research/english_v1_data/augmentation_v2/texts.sqlite3'),m)
@lru_cache(maxsize=2000)
def shingles(uid):
    db,texts,old,helper=W
    kind,key,raw=db.execute('SELECT ref_kind,ref_key,raw_sha FROM signatures WHERE uid=?',(int(uid),)).fetchone()
    if kind=='old_sql':text=old.execute('SELECT text FROM texts WHERE i=?',(int(key),)).fetchone()[0]
    else:
        assert kind=='cache' and key==raw;text=texts.execute('SELECT text FROM texts WHERE raw_sha=?',(key,)).fetchone()[0]
    assert hashlib.sha256(text.encode()).hexdigest()==raw
    return helper.shingle_set(text)
def verify(pairs):
    accepted=[]
    for key in pairs:
        a,b=int(key>>np.uint64(32)),int(key&np.uint64(0xffffffff));aa,bb=shingles(a),shingles(b);inter=len(aa&bb);j=inter/max(1,len(aa)+len(bb)-inter)
        if j>=.85:accepted.append((a,b,j))
    return accepted

def bucket_pairs(uids):
    # Unique normalized bodies only. Alias documents are exact-unioned by the
    # metadata graph; normal buckets are exhaustive, oversized buckets whole-star.
    uids=np.sort(uids.astype(np.uint64))
    if len(uids)>=100:return (uids[0]<<np.uint64(32))|uids[1:],True
    aa,bb=np.triu_indices(len(uids),1)
    return (uids[aa]<<np.uint64(32))|uids[bb],False

def synthetic_check():
    keys,big=bucket_pairs(np.arange(101,dtype=np.uint64));assert big and len(keys)==100
    p=list(range(101))
    def root(i):
        while p[i]!=i:i=p[i]
        return i
    for key in keys:
        a,b=int(key>>np.uint64(32)),int(key&np.uint64(0xffffffff));p[root(b)]=root(a)
    assert len({root(i) for i in range(101)})==1
    small,big=bucket_pairs(np.array([7,2,5],dtype=np.uint64));assert not big and set(small.tolist())=={(2<<32)|5,(2<<32)|7,(5<<32)|7}
    assert bucket_pairs(np.array([2],dtype=np.uint64))[0].size==0
    return {'passed':True,'checks':3,'oversized_existing_members_all_star_connected':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();assert 1<=args.workers<=8
    started=time.monotonic();checks=synthetic_check();completion=json.loads((CACHE/'complete.json').read_text());assert completion['cache_complete']
    for item in completion['outputs']:assert fs(ROOT/item['path'])==item['sha256']
    resultpath=OUT/'full_lexical_near_completion.json';assert not resultpath.exists(),'Reuse complete graph; do not silently replace'
    db=ro(CACHE/'signatures.sqlite3');rows=list(db.execute('SELECT uid,norm_sha,raw_sha,signature,shingle_count FROM signatures ORDER BY uid'));db.close()
    n=len(rows);assert [r[0] for r in rows]==list(range(n));assert n<2**32
    sigs=np.frombuffer(b''.join(r[3] for r in rows),dtype=np.uint32).reshape(n,32);counts=np.array([r[4] for r in rows],dtype=np.int32);eligible=np.flatnonzero(counts>=20).astype(np.uint64)
    sigpath=OUT/'full_lexical_signature_index.parquet';pq.write_table(pa.table({'uid':np.arange(n,dtype=np.uint32),'normalized_sha256':[r[1] for r in rows],'raw_sha256':[r[2] for r in rows],'signature':[r[3] for r in rows],'shingle_count':counts}),sigpath,compression='zstd')
    del rows
    exactparts=[];conservative=[];members=[];statistics=[]
    for band in range(8):
        values=np.ascontiguousarray(sigs[eligible.astype(np.int64),band*4:(band+1)*4]).view('V16').reshape(-1);order=np.argsort(values,kind='stable');ordered=values[order]
        boundaries=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(order)];buckets=Counter();pair_n=0;large_n=0
        for start,stop in zip(boundaries[:-1],boundaries[1:]):
            size=int(stop-start)
            if size<2:continue
            ids=eligible[order[start:stop]];keys,big=bucket_pairs(ids);buckets[size]+=1
            if big:
                large_n+=1;bucketkey=bytes(ordered[start]).hex();anchor=int(ids.min())
                conservative.extend((anchor,int(uid),None,'conservative_oversized_bucket',band,bucketkey) for uid in ids if int(uid)!=anchor)
                members.extend((band,bucketkey,size,int(uid)) for uid in ids)
            else:exactparts.append(keys);pair_n+=len(keys)
        statistics.append({'band':band,'collision_buckets':sum(buckets.values()),'bucket_size_histogram':dict(buckets),'exact_candidate_occurrences':pair_n,'oversized_buckets':large_n})
        print(json.dumps({'phase':'LSH_band','band':band,'candidate_occurrences':pair_n,'large_buckets':large_n,'elapsed':time.monotonic()-started}),flush=True)
    pairs=np.unique(np.concatenate(exactparts)) if exactparts else np.array([],dtype=np.uint64);del exactparts
    left=(pairs>>np.uint64(32)).astype(np.int64);right=(pairs&np.uint64(0xffffffff)).astype(np.int64)
    possible=(np.minimum(counts[left],counts[right])/np.maximum(counts[left],counts[right]))>=.85
    retrieved=len(pairs);pruned=int((~possible).sum());pairs=pairs[possible];del left,right,possible
    pairpath=CACHE/'exact_candidate_pairs.npy';np.save(pairpath,pairs)
    print(json.dumps({'phase':'verify','unique_normal_bucket_pairs':retrieved,'impossible_count_ratio_pruned':pruned,'exact_pairs_to_verify':len(pairs),'elapsed':time.monotonic()-started}),flush=True)
    schema=pa.schema([('left_uid',pa.uint32()),('right_uid',pa.uint32()),('jaccard',pa.float64()),('kind',pa.string()),('band',pa.int16()),('bucket_key',pa.string())]);edgepath=OUT/'full_lexical_near_edges.parquet';writer=pq.ParquetWriter(edgepath,schema,compression='zstd')
    def write(rr):
        if rr:writer.write_table(pa.Table.from_pylist([dict(zip(schema.names,r)) for r in rr],schema=schema))
    for i in range(0,len(conservative),10000):write(conservative[i:i+10000])
    accepted_n=0;verified_n=0;pending={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        for i in range(0,len(pairs),1000):
            chunk=pairs[i:i+1000];pending[pool.submit(verify,chunk)]=len(chunk)
            if len(pending)>=args.workers*3:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for job in done:
                    rr=job.result();verified_n+=pending.pop(job);accepted_n+=len(rr);write([(a,b,j,'verified_jaccard_ge_0.85',None,None) for a,b,j in rr])
            if i%100000==0:print(json.dumps({'phase':'exact_progress','submitted':i+len(chunk),'verified':verified_n,'accepted':accepted_n,'elapsed':time.monotonic()-started}),flush=True)
        for job,size in pending.items():
            rr=job.result();verified_n+=size;accepted_n+=len(rr);write([(a,b,j,'verified_jaccard_ge_0.85',None,None) for a,b,j in rr])
    writer.close();assert verified_n==len(pairs)
    mpath=OUT/'full_lexical_oversized_bucket_members.parquet';mschema=pa.schema([('band',pa.int16()),('bucket_key',pa.string()),('bucket_size',pa.int32()),('uid',pa.uint32())]);pq.write_table(pa.Table.from_pylist([dict(zip(mschema.names,r)) for r in members],schema=mschema),mpath,compression='zstd')
    result={'kind':'complete_old_new_lexical_near_graph_NOT_ROLE_FREEZE','created_at_utc':datetime.now(timezone.utc).isoformat(),'complete':True,'synthetic_checks':checks,'unique_normalized_bodies':n,'signature_eligible_ge20shingles':len(eligible),'normal_bucket_unique_retrieved_pairs':retrieved,'pairs_pruned_by_impossible_shingle_count_ratio':pruned,'pairs_exactly_verified':verified_n,'verified_edges':accepted_n,'conservative_edges_including_crossband_duplicates':len(conservative),'oversized_bucket_members_including_crossband_duplicates':len(members),'bands':statistics,'elapsed_seconds':time.monotonic()-started,'workers':args.workers,'body_previewed':False,'models_or_scores':False,'roles_allocated':False,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':fs(p)} for p in [CACHE/'complete.json',OUT/'lexical_signature.py',Path(__file__)]],'outputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':fs(p)} for p in [sigpath,edgepath,mpath,pairpath]],'limitations':['Approximate LSH candidate recall, first3000 lexical tokens, not complete semantic or hidden-parent isolation.','Signature indexed once per identical normalized body; all alias documents remain native/exact graph nodes. Small buckets exhaustively pair, count-ratio is an exact mathematical impossibility prune.','Static oversized>=100 distinct-normalized-body buckets star-connect ALL existing members to the minimumuid. This is conservative v2 graph behavior, not the original incremental v1 firstanchor-only rule. Similarity is null, never falsely reported as verified.','All saved old component edges/keys must also be preserved by the joint graph consumer; these edges do not independently encode native parent/title relations.']}
    resultpath.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['normal_bucket_unique_retrieved_pairs','pairs_exactly_verified','verified_edges','conservative_edges_including_crossband_duplicates','elapsed_seconds']}))
if __name__=='__main__':main()
