"""Create hash/lineage-only capacity index from complete pinned MAGA files.

No detector, split allocation, scores or text preview. Original text stays in
quarantined complete acquisition files, never in this metadata index/report.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,importlib.util,json,re,unicodedata,time
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/english_v2_data/audit'
WORD=re.compile(r'\b\w+\b',re.UNICODE)
LEXICAL=re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
THINK=re.compile(r'<\s*/?\s*(?:think|analysis|reasoning)\b|(?:^|\n)\s*(?:analysis|chain.of.thought)\s*:',re.I)
TRUNC=re.compile(r'\[(?:truncated|cut[ -]off)\]|<truncated>|\btruncated (?:for|to|due to)\b',re.I)

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha_file(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def identity(value):
    if value is None:return None
    value=str(value).strip()
    return None if value.casefold() in {'','none','null','nan','na','n/a','<na>'} else value

def simple_quality(text,words):
    if words<128 or not 30<=len(text.split())<=3000:return 'outside_provisional_core_length'
    if THINK.search(text):return 'visible_think_or_analysis'
    end=text.rstrip().rstrip('\"\'”’)]}').rstrip()
    if TRUNC.search(text) or end.endswith(('...','…')) or not end.endswith(('.','!','?')) or text.count('```')%2:return 'suspected_visible_truncation'
    return None

def main():
    started=time.monotonic()
    from langid.langid import LanguageIdentifier,model as LANGID_MODEL
    guardpath=ROOT/'experiments/english_v1/language_guard.py'
    spec=importlib.util.spec_from_file_location('capacity_language_guard',guardpath);guardmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(guardmodule)
    guard=guardmodule.LanguageGuard(identifier=LanguageIdentifier.from_modelstring(LANGID_MODEL,norm_probs=True))
    language_cache={}
    focus=json.loads((ROOT/'research/english_v2_data/modern_candidates/focus_recommendation.json').read_text())
    catalog=focus['MAGA']['modern_generators_reported_by_author'];documented_generators={x for x in catalog if x!='open_models'}|set(catalog['open_models'])
    planpath=OUT/'maga_acquisition_plan.json'
    plan=json.loads(planpath.read_text());assert plan['revision']=='34c51fc6e71561bf1203945d90d253269f9f2ab8'
    inputs=[]
    # Never admit a partial, even for metadata capacity. All four fixed inputs
    # must match official LFS size and SHA before the first row is parsed.
    for spec in plan['files']:
        path=ROOT/spec['local_path']
        assert path.is_file() and path.stat().st_size==spec['bytes']
        digest=sha_file(path);assert digest==spec['sha256']
        inputs.append({'path':spec['local_path'],'native_path':spec['native_path'],'bytes':spec['bytes'],'sha256':digest,'revision':spec['revision'],'url':spec['url']})
    rows=[];field_shapes=Counter();missing=Counter();human_by_parent=defaultdict(set);all_by_parent=defaultdict(list)
    model_domain=Counter();seen_id={};conflicts=[]
    for item in inputs:
        config='MGB' if '/MGB_' in item['native_path'] else 'MAGA'
        original_split='train' if item['native_path'].startswith('train/') else 'validation'
        with (ROOT/item['path']).open('r',encoding='utf-8') as f:
            for line_number,line in enumerate(f,1):
                value=json.loads(line);field_shapes[tuple(sorted(value))]+=1
                text=value.get('text');assert isinstance(text,str), 'Non-string text must not silently pass'
                native_id=identity(value.get('id'));parent=identity(value.get('human_source_id'));prompt=identity(value.get('prompt_id'))
                model=identity(value.get('model'));domain=identity(value.get('domain'))
                for name,v in [('id',native_id),('human_source_id',parent),('model',model),('domain',domain)]:
                    if v is None:missing[name]+=1
                canonical_domain=re.sub(r'[\s_-]+','',domain.casefold()) if domain else None
                y=0 if model and model.casefold()=='human' else 1 if model in documented_generators else None
                normalized=norm(text);tokens=LEXICAL.findall(normalized)
                raw_sha=sha_bytes(text.encode());norm_sha=sha_bytes(normalized.encode());words=len(WORD.findall(text))
                if norm_sha not in language_cache and text.strip():language_cache[norm_sha]=guard.check(text)
                scope=language_cache.get(norm_sha,{})
                title=value.get('title');system=value.get('system_prompt');user=value.get('user_prompt')
                record={'file':item['native_path'],'line_number':line_number,'config':config,'original_split':original_split,'id':native_id,'human_source_id':parent,'prompt_id':prompt,'model':model,'domain':domain,'canonical_domain':canonical_domain,'y_ai':y,'raw_sha256':raw_sha,'normalized_sha256':norm_sha,'prefix32_sha256':sha_bytes(' '.join(tokens[:32]).encode()) if len(tokens)>=32 else None,'title_sha256':sha_bytes(norm(title).encode()) if isinstance(title,str) and title.strip() else None,'system_prompt_sha256':sha_bytes(norm(system).encode()) if isinstance(system,str) and system.strip() else None,'full_prompt_sha256':sha_bytes(norm((system if isinstance(system,str) else '')+'\n'+user).encode()) if isinstance(user,str) and user.strip() else None,'user_prompt_sha256':sha_bytes(norm(user).encode()) if isinstance(user,str) and user.strip() else None,'characters':len(text),'whitespace_words':len(text.split()),'wordlike_tokens':words,'provisional_quality_reason':simple_quality(text,words),'wikiHow_parent_forbidden':canonical_domain=='wikihow','label_evidence':'publisher_model_field','source_revision':plan['revision'],'language_evidence':guardmodule.METHOD,'language_supported':scope.get('supported',False),'language_identified':scope.get('identified'),'language_probability':scope.get('probability'),'ascii_letter_ratio':scope.get('ascii_letter_ratio'),'unexpected_generator_quarantined':y is None,'temperature':str(value['temperature']) if value.get('temperature') is not None else None,'top_p':str(value['top_p']) if value.get('top_p') is not None else None,'top_k':str(value['top_k']) if value.get('top_k') is not None else None,'repetition_penalty':str(value['repetition_penalty']) if value.get('repetition_penalty') is not None else None}
                if native_id:
                    signature=(raw_sha,parent,model,domain)
                    if native_id in seen_id and seen_id[native_id]!=signature:conflicts.append(native_id)
                    seen_id[native_id]=signature
                if y==0 and parent:human_by_parent[parent].add(norm_sha)
                if parent:all_by_parent[parent].append(record)
                rows.append(record);model_domain[(config,domain,model)]+=1
        print(json.dumps({'file_complete':item['native_path'],'metadata_rows':sum(r['file']==item['native_path'] for r in rows)}),flush=True)
    # Parent closure excludes even records whose domain spelling differs.
    forbidden={r['human_source_id'] for r in rows if r['wikiHow_parent_forbidden'] and r['human_source_id']}
    for r in rows:r['wikiHow_parent_forbidden']=r['human_source_id'] in forbidden or r['wikiHow_parent_forbidden']
    def summary(subset):
        humans={r['human_source_id'] for r in subset if r['y_ai']==0 and r['human_source_id']}
        ai={r['human_source_id'] for r in subset if r['y_ai']==1 and r['human_source_id']}
        return {'rows_including_cross_config_repeats':len(subset),'human_rows':sum(r['y_ai']==0 for r in subset),'AI_rows':sum(r['y_ai']==1 for r in subset),'unique_ids':len({r['id'] for r in subset if r['id']}),'unique_raw_texts':len({r['raw_sha256'] for r in subset}),'human_parents':len(humans),'AI_parents':len(ai),'paired_parents':len(humans&ai),'distinct_human_texts':len({r['normalized_sha256'] for r in subset if r['y_ai']==0})}
    rights_path=OUT/'source_rights_register.json'
    rights=json.loads(rights_path.read_text());domain_status={re.sub(r'[\s_-]+','',r['domain'].casefold()):r for r in rights['domains']}
    for r in rows:
        entry=domain_status.get(r['canonical_domain'],{})
        r['source_rights_status']=entry.get('status','unrecognized_domain_rights_quarantine')
        r['allow_noncommercial_research']=entry.get('allow_noncommercial_research') is True
    allowed=[r for r in rows if not r['wikiHow_parent_forbidden']]
    core=[r for r in allowed if r['provisional_quality_reason'] is None and r['language_supported'] and r['y_ai'] is not None]
    licensed_core=[r for r in core if r['allow_noncommercial_research']]
    licensed_core_pairs={r['human_source_id'] for r in licensed_core if r['y_ai']==0}&{r['human_source_id'] for r in licensed_core if r['y_ai']==1}
    paired_all={r['human_source_id'] for r in allowed if r['y_ai']==0}&{r['human_source_id'] for r in allowed if r['y_ai']==1}
    paired_core={r['human_source_id'] for r in core if r['y_ai']==0}&{r['human_source_id'] for r in core if r['y_ai']==1}
    by_domain=[]
    for d in sorted({r['domain'] for r in rows}):
        a=[r for r in allowed if r['domain']==d];c=[r for r in core if r['domain']==d]
        by_domain.append({'domain':d,'all':summary(a),'provisional_core':summary(c),'source_rights_status':domain_status.get(re.sub(r'[\s_-]+','',d.casefold()),{}).get('status'),'allow_noncommercial_research':domain_status.get(re.sub(r'[\s_-]+','',d.casefold()),{}).get('allow_noncommercial_research') is True,'quality_reasons':dict(Counter(r['provisional_quality_reason'] for r in a))})
    table=pa.Table.from_pylist(rows)
    index=OUT/'maga_metadata_index.parquet';pq.write_table(table,index,compression='zstd')
    meta={'kind':'MAGA_full_payload_metadata_capacity_not_v2_freeze','created_at_utc':datetime.now(timezone.utc).isoformat(),'revision':plan['revision'],'elapsed_seconds':time.monotonic()-started,'language_unique_texts_checked':len(language_cache),'language_guard_sha256':sha_file(guardpath),'unexpected_model_counts':dict(Counter(r['model'] for r in rows if r['unexpected_generator_quarantined'])),'all_four_complete_LFS_verified_before_parsing':True,'raw_body_previewed':False,'detector_loaded_or_scores_read':False,'roles_constructed':False,'all_records':summary(rows),'excluding_entire_wikiHow_parent_closure':summary(allowed),'provisional_core_excluding_wikiHow':summary(core),'core_paired_parent_capacity_before_licence_language_oldbarrier_near_checks':len(paired_core),'rights_confirmed_core_before_oldbarrier_near':summary(licensed_core),'rights_confirmed_core_paired_parents_before_oldbarrier_near':len(licensed_core_pairs),'source_rights_register_sha256':sha_file(rights_path),'unfiltered_paired_parent_capacity_before_licence_language_oldbarrier_near_checks':len(paired_all),'by_domain':by_domain,'by_config_domain_model':[dict(config=k[0],domain=k[1],model=k[2],rows=v) for k,v in sorted(model_domain.items(),key=lambda x:str(x[0]))],'missing_required_fields':dict(missing),'conflicting_native_ids':len(set(conflicts)),'human_parent_multiple_distinct_texts':sum(len(v)>1 for v in human_by_parent.values()),'observed_field_schemas':[{'fields':list(k),'rows':v} for k,v in field_shapes.items()],'metadata_index':{'path':str(index.relative_to(ROOT)),'bytes':index.stat().st_size,'sha256':sha_file(index),'columns':table.schema.names},'inputs':inputs,'limitations':['Provisional core uses fixed legacy128wordlike/30..3000whitespace/visible-think/truncation checks only for capacity disclosure, not a frozen v2 rule.','The old-used observable graph barrier and lexical-near replay have not happened yet; counts are upperbounds. Language uses finite-sampling existing LanguageGuard rather than human verification.','Source licenses are separate layered rights and not implied by the MAGA MIT declaration. WikiHow closure is excluded; other unconfirmed domains require quarantine.','Human UUIDs are stable inside this release but original native identifiers were discarded by the author sampler, so cross-source overlaps require text/prefix/title/near checks.','Model fields are publisher nominal generation claims, not authentication of weights/execution. MGB is baseline generation without alignment; MAGA is fused alignment configuration and not equivalent to sentence paraphrase.','Across-config human repetitions do not create new independent controls. Each complete parent and all configs/variants must stay in one project role.']}
    (OUT/'maga_actual_capacity.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all':meta['all_records'],'no_wikihow':meta['excluding_entire_wikiHow_parent_closure'],'provisional_core':meta['provisional_core_excluding_wikiHow'],'core_paired_parents':len(paired_core),'rights_confirmed_core':summary(licensed_core),'rights_confirmed_core_paired_parents':len(licensed_core_pairs),'domains':by_domain,'conflicting_native_ids':len(set(conflicts))},indent=2))
if __name__=='__main__':main()
