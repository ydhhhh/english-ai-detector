"""One persistent lexical-signature pass over all old/new observed bodies.

CPU only; no detector imports, predictions, roles or printed text. The cache is
an internal data-engineering artifact, not redistributed text or a dataset view.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,ast,hashlib,importlib.util,json,re,sqlite3,time,unicodedata
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];AUDIT=ROOT/'research/english_v2_data/audit';CACHE=ROOT/'.cache/english_v2_lexical_v1'
HELPER=AUDIT/'lexical_signature.py'
WORK_HELPER=None

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def fsha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_helper():
    spec=importlib.util.spec_from_file_location('lexical_signature',HELPER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def worker_init():
    global WORK_HELPER
    WORK_HELPER=load_helper()
def batch_signatures(batch):
    result=[]
    for uid,text in batch:
        sig,n=WORK_HELPER.signature(text,seed=20261017)
        assert sig.shape==(32,) and str(sig.dtype)=='uint32'
        result.append((uid,sig.tobytes(),int(n)))
    return result

def doc_sources(references):
    graphpath=ROOT/'data/english_calibration_v2/graph_index.parquet'
    graph=pq.read_table(graphpath,columns=['id','raw_sha256','normalized_sha256','origin']).to_pylist()
    dbpath=ROOT/'research/english_v1_data/augmentation_v2/texts.sqlite3'
    refs=[graphpath,dbpath]
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    assert db.execute('SELECT count(*) FROM texts').fetchone()[0]==626732
    for i,text in db.execute('SELECT i,text FROM texts ORDER BY i'):
        expected=graph[i]
        yield 'old_base_sqlite',str(i),text,expected['raw_sha256'],expected['normalized_sha256'],'old_sql',str(i)
    db.close()
    by_id={r['id']:(r['raw_sha256'],r['normalized_sha256']) for r in graph if r['origin'].startswith('RAID')}
    oldraid=ROOT/'research/raid_external_v1/candidates.jsonl';refs.append(oldraid)
    seen=set()
    with oldraid.open() as stream:
        for line in stream:
            r=json.loads(line);identifier=r['id'];assert identifier not in seen;seen.add(identifier)
            raw,norm=by_id[identifier]
            yield 'old_RAID_bridge',identifier,r['text'],raw,norm,'cache',None
    assert seen==set(by_id), 'Every old RAID bridge body must be available'
    for split in ['train','validation','calibration']:
        path=ROOT/f'data/english_v1/pilot/{split}.parquet';refs.append(path)
        for r in pq.read_table(path,columns=['id','text']).to_pylist():
            yield 'old_pilot_'+split,r['id'],r['text'],None,None,'cache',None
    planpath=AUDIT/'maga_acquisition_plan.json';plan=json.loads(planpath.read_text());refs.append(planpath)
    expected_maga={}
    meta=ROOT/'research/english_v2_data/audit/maga_metadata_index.parquet';refs.append(meta)
    for r in pq.read_table(meta,columns=['file','line_number','raw_sha256','normalized_sha256']).to_pylist():expected_maga[(r['file'],r['line_number'])]=(r['raw_sha256'],r['normalized_sha256'])
    for spec in plan['files']:
        path=ROOT/spec['local_path'];refs.append(path)
        assert path.stat().st_size==spec['bytes'] and fsha(path)==spec['sha256']
        with path.open() as stream:
            for i,line in enumerate(stream,1):
                r=json.loads(line);raw,norm=expected_maga[(spec['native_path'],i)]
                yield 'new_MAGA',spec['native_path']+':'+str(i),r['text'],raw,norm,'cache',None
    beemo=ROOT/'research/english_v2_data/raw/beemo/train-00000-of-00001.parquet';refs.append(beemo)
    assert fsha(beemo)=='5df88f7941e873db3c6e2be4e4085db50fa824f7fc626685f0e317e935a188db'
    for r in pq.read_table(beemo).to_pylist():
        for field in ('human_output','model_output','human_edits'):
            text=r.get(field)
            if isinstance(text,str) and text.strip():yield 'new_Beemo',str(r['id'])+':'+field,text,None,None,'cache',None
        for field in ('gpt-4o_edits','llama-3.1-70b_edits'):
            v=r[field]
            if isinstance(v,str):
                try:v=json.loads(v)
                except json.JSONDecodeError:v=ast.literal_eval(v)
            for part in v:
                for prompt,text in part.items():
                    if isinstance(text,str) and text.strip():yield 'new_Beemo',str(r['id'])+':'+field+':'+prompt,text,None,None,'cache',None
    completion=ROOT/'research/english_v2_data/raw/raid_paired_range_v1/COMPLETED.json';refs.append(completion)
    done=json.loads(completion.read_text());assert done['all_8_files_complete'] is True
    seen=set()
    for file in done['files']:
        path=ROOT/file['payload']['path'];refs.append(path);assert fsha(path)==file['payload']['sha256']
        if pq.ParquetFile(path).metadata.num_rows==0:continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000,columns=['id','generation','raw_sha256','normalized_sha256']):
            for r in batch.to_pylist():
                assert r['id'] not in seen;seen.add(r['id'])
                yield 'new_RAID',r['id'],r['generation'],r['raw_sha256'],r['normalized_sha256'],'cache',None
    assert len(seen)==128196
    # Earlier manual-review bodies are observed references, not new training.
    path=ROOT/'research/data_audit_20261017/community_evidence/manual_sample_review.json';refs.append(path)
    for r in json.loads(path.read_text())['samples']:
        yield 'prior_observed_manual36',str(r['sample_index']),r['text'],r['text_sha256'],None,'cache',None
    base=ROOT/'research/english_v2_data/modern_candidates/maga_upstream'
    for name in ['sentence-transformers__natural-questions.README.md','sentence-transformers__s2orc.README.md']:
        path=base/name;refs.append(path)
        for i,block in enumerate(re.findall(r'```(?:python)?\s*(.*?)```',path.read_text(),re.S)):
            try:value=ast.literal_eval(block.strip())
            except (ValueError,SyntaxError):continue
            if isinstance(value,dict):
                for field,text in value.items():
                    if isinstance(text,str) and text.strip():yield 'prior_documentation_example',name+':'+str(i)+':'+field,text,None,None,'cache',None
    for path in refs:references[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':fsha(path)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    assert 1<=args.workers<=8
    if not HELPER.is_file():raise SystemExit('Signature helper not yet available')
    helper=load_helper();helper.synthetic_selfcheck()
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'complete.json').exists():raise SystemExit('Cache already complete: reuse rather than recalculate')
    db=sqlite3.connect(CACHE/'signatures.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.executescript('CREATE TABLE IF NOT EXISTS signatures(uid INTEGER PRIMARY KEY,norm_sha TEXT UNIQUE NOT NULL,raw_sha TEXT NOT NULL,ref_kind TEXT NOT NULL,ref_key TEXT,signature BLOB,shingle_count INTEGER);CREATE TABLE IF NOT EXISTS raw_refs(raw_sha TEXT PRIMARY KEY,norm_sha TEXT NOT NULL,ref_kind TEXT NOT NULL,ref_key TEXT);')
    texts=sqlite3.connect(CACHE/'new_texts.sqlite3');texts.execute('PRAGMA journal_mode=WAL');texts.execute('PRAGMA synchronous=NORMAL');texts.execute('CREATE TABLE IF NOT EXISTS texts(raw_sha TEXT PRIMARY KEY,text TEXT NOT NULL)')
    # Refuse a partial-cache provenance change; source metadata stay immutable.
    bound_paths=[ROOT/'data/english_calibration_v2/graph_index.parquet',ROOT/'research/english_v1_data/augmentation_v2/texts.sqlite3',ROOT/'research/raid_external_v1/candidates.jsonl',ROOT/'research/english_v2_data/raw/beemo/train-00000-of-00001.parquet',ROOT/'research/data_audit_20261017/community_evidence/manual_sample_review.json',*[ROOT/f'data/english_v1/pilot/{split}.parquet' for split in ['train','validation','calibration']],*[ROOT/'research/english_v2_data/modern_candidates/maga_upstream'/name for name in ['sentence-transformers__natural-questions.README.md','sentence-transformers__s2orc.README.md']]]
    header={'cache_schema_version':2,'bound_sources':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':fsha(p)} for p in bound_paths},'algorithm':'legacy normalized ASCII lexical word5gram, first3000tokens,CRC32,32affine minhash, seed20261017; cache by normalized text identity','helper_sha256':fsha(HELPER),'legacy_algorithm_sha256':fsha(ROOT/'experiments/english_v1/data_grouping.py'),'MAGA_metadata_sha256':fsha(AUDIT/'maga_metadata_index.parquet'),'RAID_completion_sha256':fsha(ROOT/'research/english_v2_data/raw/raid_paired_range_v1/COMPLETED.json')}
    hp=CACHE/'header.json'
    if hp.exists():assert json.loads(hp.read_text())==header,'Cache provenance/algorithm changed'
    else:hp.write_text(json.dumps(header,indent=2)+'\n')
    existing={norm:uid for uid,norm in db.execute('SELECT uid,norm_sha FROM signatures')};pending_norms=set();raw_existing={r[0] for r in db.execute('SELECT raw_sha FROM raw_refs')};next_uid=max(existing.values(),default=-1)+1
    references={};counts=Counter();started=time.monotonic();completed=0;batch=[];pending=set()
    def collect(task):
        nonlocal completed
        for uid,sig,n in task.result():db.execute('UPDATE signatures SET signature=?,shingle_count=? WHERE uid=?',(sig,n,uid));completed+=1
    def submit(pool):
        nonlocal batch
        if batch:pending.add(pool.submit(batch_signatures,batch));batch=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        for kind,key,text,expected_raw,expected_norm,refkind,refkey in doc_sources(references):
            assert isinstance(text,str)
            raw=sha(text);norm=normalized(text);normsha=sha(norm)
            assert expected_raw is None or raw==expected_raw, 'Source raw hash mismatch: '+kind+':'+key
            assert expected_norm is None or normsha==expected_norm, 'Source normalized hash mismatch: '+kind+':'+key
            counts[kind]+=1
            # Rehydrate cache bodies even if a previous two-database commit was
            # interrupted after the signature registry committed. Extra bodies
            # whose first reference is old_sql are harmless internal aliases.
            if refkind!='old_sql':texts.execute('INSERT OR IGNORE INTO texts VALUES (?,?)',(raw,text));refkey=raw
            if raw not in raw_existing:
                db.execute('INSERT INTO raw_refs VALUES (?,?,?,?)',(raw,normsha,refkind,refkey));raw_existing.add(raw)
            if normsha not in existing:
                uid=next_uid;next_uid+=1;existing[normsha]=uid
                # Normalize once in the streaming registry; the legacy helper
                # re-normalizes idempotently without changing its signature.
                if refkind!='old_sql':refkey=raw
                db.execute('INSERT INTO signatures(uid,norm_sha,raw_sha,ref_kind,ref_key) VALUES(?,?,?,?,?)',(uid,normsha,raw,refkind,refkey));pending_norms.add(normsha);batch.append((uid,norm))
            elif normsha not in pending_norms:
                uid=existing[normsha]
                if db.execute('SELECT signature IS NULL FROM signatures WHERE uid=?',(uid,)).fetchone()[0]:pending_norms.add(normsha);batch.append((uid,norm))
            if len(batch)>=128:submit(pool)
            if len(pending)>=args.workers*3:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for task in done:collect(task)
            if sum(counts.values())%25000==0:
                texts.commit();db.commit()
                progress={'at_utc':datetime.now(timezone.utc).isoformat(),'documents_streamed':sum(counts.values()),'unique_normalized_signatures':len(existing),'new_signatures_completed':completed,'elapsed_seconds':time.monotonic()-started,'by_source':dict(counts)}
                with (CACHE/'progress.jsonl').open('a') as out:out.write(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
        submit(pool)
        for task in pending:collect(task)
    texts.commit();db.commit()
    assert db.execute('SELECT count(*) FROM signatures WHERE signature IS NULL').fetchone()[0]==0
    body_keys={r[0] for r in texts.execute('SELECT raw_sha FROM texts')}
    for query in ['SELECT raw_sha,ref_key FROM raw_refs WHERE ref_kind="cache"','SELECT raw_sha,ref_key FROM signatures WHERE ref_kind="cache"']:
        assert all(raw==key and key in body_keys for raw,key in db.execute(query)), 'A cached body reference is missing or inconsistent'
    # Every saved old bridge must have a signature identity (including aliases).
    old_norms=set(pq.read_table(ROOT/'data/english_calibration_v2/graph_index.parquet',columns=['normalized_sha256'])['normalized_sha256'].to_pylist())
    assert old_norms<=existing.keys()
    total=db.execute('SELECT count(*) FROM signatures').fetchone()[0];eligible=db.execute('SELECT count(*) FROM signatures WHERE shingle_count>=20').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');texts.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();texts.close()
    result={'kind':'complete_reusable_lexical_signature_cache','completed_at_utc':datetime.now(timezone.utc).isoformat(),'cache_complete':True,'signature_count':total,'signature_indexed_count_ge20shingles':eligible,'raw_text_identities':len(raw_existing),'all_old_graph_normalized_identities_covered':len(old_norms),'document_source_counts':dict(counts),'elapsed_seconds':time.monotonic()-started,'workers':args.workers,'no_detector_or_scores':True,'no_roles_assigned':True,'no_text_previewed':True,'normalized_aliases_compute_signature_once':True,'header':header,'sources':references,'outputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':fsha(p)} for p in [CACHE/'signatures.sqlite3',CACHE/'new_texts.sqlite3']]}
    (CACHE/'complete.json').write_text(json.dumps(result,indent=2)+'\n');(AUDIT/'lexical_cache_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['signature_count','signature_indexed_count_ge20shingles','raw_text_identities','document_source_counts','elapsed_seconds']}))
if __name__=='__main__':main()
