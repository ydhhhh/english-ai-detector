"""Hash/length/lineage/English-diagnostic RAID index; no model or role allocation."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,re,time,unicodedata,zipfile
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit';BASE=ROOT/'research/english_v2_data/raw/raid_paired_range_v1'
LEX=re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def fs(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    started=time.monotonic();completion=BASE/'COMPLETED.json';done=json.loads(completion.read_text());assert done['all_8_files_complete'] is True
    quality=module('maga_quality',OUT/'index_maga_metadata.py')
    lg=module('scope_language',ROOT/'experiments/english_v1/language_guard.py')
    from langid.langid import LanguageIdentifier,model
    guard=lg.LanguageGuard(identifier=LanguageIdentifier.from_modelstring(model,norm_probs=True));scope_cache={}
    titlepath=ROOT/'research/english_v2_data/raw/raid_paired_range_v1_titles/parent_titles.parquet'
    titles={r['source_id']:r for r in pq.read_table(titlepath).to_pylist()};assert len(titles)==3561
    bbcpath=ROOT/'research/raid_external_v1/raw/bbc-fulltext.zip';bbc={};bbcprefix={}
    with zipfile.ZipFile(bbcpath) as z:
        for entry in z.infolist():
            if not entry.filename.endswith('.txt') or entry.file_size>1024*1024:continue
            t=z.read(entry).decode('utf-8',errors='replace')
            for value in [t,'\n'.join(t.splitlines()[1:])]:
                value=norm(value);bbc[sha(value)]=entry.filename
                if len(value)>=160:bbcprefix[sha(value[:160])]=entry.filename
    paths=[];parents={};records=[]
    for item in done['files']:
        p=ROOT/item['payload']['path'];assert fs(p)==item['payload']['sha256'];paths.append({'path':str(p.relative_to(ROOT)),'sha256':fs(p)})
        if pq.ParquetFile(p).metadata.num_rows==0:continue
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1000):
            for r in batch.to_pylist():
                text=r.pop('generation');normalized=norm(text);assert sha(text)==r['raw_sha256'] and sha(normalized)==r['normalized_sha256']
                y=0 if r['model']=='human' and r['attack']=='none' else 1 if r['attack']=='paraphrase' else None
                assert y==r['y_ai'] and y is not None
                sid=r['source_id'];title=titles[sid];assert title['domain']==r['domain']
                tokens=LEX.findall(normalized);words=r['word_like_tokens'];reason=quality.simple_quality(text,words)
                if r['normalized_sha256'] not in scope_cache:scope_cache[r['normalized_sha256']]=guard.check(text)
                scope=scope_cache[r['normalized_sha256']]
                human_rewrite=y==1 and r['model']=='human'
                record={**r,'collector':'RAID','source':'RAID/'+r['domain'],'parent':'RAID:'+sid,'characters':len(text),'whitespace_words':len(text.split()),'wordlike_tokens':words,'prefix32_sha256':sha(' '.join(tokens[:32])) if len(tokens)>=32 else None,'title_sha256':sha(title['canonical_title']),'canonical_title':title['canonical_title'],'publisher_English_evidence':'RAID publisher English benchmark, books=CMU Wikipedia plot summaries/news=BBC English collection','publisher_English':True,'language_supported':bool(scope['supported']),'language_identified':scope.get('identified'),'language_probability':scope.get('probability'),'ascii_letter_ratio':scope.get('ascii_letter_ratio'),'language_evidence':lg.METHOD,'provisional_quality_reason':reason,'structural_length_core':reason is None,'runtime_supported_structural_core':reason is None and scope['supported'],'view_human_origin_rewrite':y==0 or human_rewrite,'view_AI_origin_rewrite':y==0 or y==1 and not human_rewrite,'mask_family':False,'mask_model':False,'nominal_original_model':r['model'],'rewrite_kind':'DIPPER_paraphrase' if y==1 else 'unmodified_human_control'}
                if y==0:
                    method='normalized_full_text_or_body_exact' if r['normalized_sha256'] in bbc else 'normalized_first160_character_excerpt_match' if sha(normalized[:160]) in bbcprefix else None
                    parents[sid]={'domain':r['domain'],'human_id':r['id'],'BBC_match_method':method if r['domain']=='news' else None,'BBC_file':bbc.get(r['normalized_sha256']) or bbcprefix.get(sha(normalized[:160])) if r['domain']=='news' else None}
                records.append(record)
        print(json.dumps({'complete_payload':str(p.relative_to(ROOT)),'rows_indexed':len(records)}),flush=True)
    assert len(records)==128196 and len(parents)==3561
    for r in records:
        p=parents[r['source_id']]
        r['source_rights_allowed']=r['domain']=='books' or p['BBC_match_method']=='normalized_full_text_or_body_exact'
        r['source_rights_status']='CMU_CC_BY_SA_3_0_US_source_mapping_not_item_authentication' if r['domain']=='books' else 'BBC_noncommercial_research_original_exact_match' if r['source_rights_allowed'] else 'BBC_upstream_item_match_unverified_quarantine'
        r['BBC_match_method']=p['BBC_match_method']
    def counts(rr):
        h={r['source_id'] for r in rr if r['y_ai']==0};ha={r['source_id'] for r in rr if r['y_ai']==1 and r['model']=='human'};aa={r['source_id'] for r in rr if r['y_ai']==1 and r['model']!='human'}
        return {'rows':len(rr),'human_rows':sum(r['y_ai']==0 for r in rr),'AI_rows':sum(r['y_ai']==1 for r in rr),'human_parents':len(h),'AI_parents':len(ha|aa),'human_origin_AI_parents':len(ha),'AI_origin_AI_parents':len(aa),'paired_parents':len(h&(ha|aa)),'three_way_paired_parents':len(h&ha&aa)}
    output=OUT/'raid_metadata_index.parquet';pq.write_table(pa.Table.from_pylist(records),output,compression='zstd')
    summary={'kind':'RAID_complete_acquired_subset_capacity_NOT_ROLE_FREEZE','created_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'roles_allocated':False,'body_previewed':False,'model_loaded_or_predictions':False,'language_guard_is_diagnostic_not_fresh_view_eligibility':True,'all':counts(records),'rights_structural_core_before_old_near_barrier':counts([r for r in records if r['source_rights_allowed'] and r['structural_length_core']]),'rights_runtime_supported_structural_core_diagnostic':counts([r for r in records if r['source_rights_allowed'] and r['runtime_supported_structural_core']]),'BBC_human_match_methods':dict(Counter(p['BBC_match_method'] for p in parents.values() if p['domain']=='news')),'by_domain':[{'domain':d,'all':counts([r for r in records if r['domain']==d]),'rights_structural_core':counts([r for r in records if r['domain']==d and r['source_rights_allowed'] and r['structural_length_core']]),'quality_reasons':dict(Counter(r['provisional_quality_reason'] for r in records if r['domain']==d)),'language_supported_diagnostic':dict(Counter(str(r['language_supported']) for r in records if r['domain']==d))} for d in ['books','news']],'metadata_index':{'path':str(output.relative_to(ROOT)),'sha256':fs(output),'bytes':output.stat().st_size},'inputs':paths+[{'path':str(p.relative_to(ROOT)),'sha256':fs(p)} for p in [completion,titlepath,bbcpath,OUT/'index_maga_metadata.py',ROOT/'experiments/english_v1/language_guard.py']],'limitations':['Not the full RAID dataset: fixed converted partial range subset and receipts, no complete original11.8GB SHA recomputed.','Every paraphrase is AI-positive even when original model=human. Original human and35transforms are one source parent.','DIPPER is last rewriter; original model field is not a unique final model attribution. All fine supervision masks false.','Publisher English provenance does not guarantee runtime language support. Unsupported structural-core rows remain intended fresh view candidates and count as abstentions under deployment.','Quality heuristics are prospective capacity predicates, not human verification. Whole-old/new/native/lexical-near barrier still required.','Books permission is publisher CMU/Wikipedia source mapping, not per-summary origin authentication; no raw redistribution asserted. News requires exact source-body/original match to official BBC archive in this index.']}
    (OUT/'raid_actual_capacity.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['all','rights_structural_core_before_old_near_barrier','rights_runtime_supported_structural_core_diagnostic','BBC_human_match_methods','elapsed_seconds']},indent=2))
if __name__=='__main__':main()
