"""Content-derived Beemo capacity metadata; no text preview or detector use."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,re,unicodedata,time
import pyarrow as pa
import pyarrow.parquet as pq
from langid.langid import LanguageIdentifier,model as LANGID_MODEL
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/english_v2_data/audit'
WORD=re.compile(r'\b\w+\b');TOKEN=re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
THINK=re.compile(r'<\s*/?\s*(?:think|analysis|reasoning)\b|(?:^|\n)\s*(?:analysis|chain.of.thought)\s*:',re.I)
TRUNCATED=re.compile(r'\[(?:truncated|cut[ -]off)\]|<truncated>|\btruncated (?:for|to|due to)\b',re.I)

def sh(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def qualify(text,words,minimum=128):
    if words<minimum or not 30<=len(text.split())<=3000:return 'length'
    if THINK.search(text):return 'visible_think_or_analysis'
    ending=text.rstrip().rstrip('\"\'”’)]}').rstrip()
    if TRUNCATED.search(text) or ending.endswith(('...','…')) or not ending.endswith(('.','!','?')) or text.count('```')%2:return 'suspected_visible_truncation'
    return None

class UF:
    def __init__(self,n):self.p=list(range(n));self.size=[1]*n
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,x,y):
        x,y=self.find(x),self.find(y)
        if x==y:return
        if self.size[x]<self.size[y]:x,y=y,x
        self.p[y]=x;self.size[x]+=self.size[y]

def main():
    started=time.monotonic()
    path=ROOT/'research/english_v2_data/raw/beemo/train-00000-of-00001.parquet'
    assert path.stat().st_size==8345056 and file_sha(path)=='5df88f7941e873db3c6e2be4e4085db50fa824f7fc626685f0e317e935a188db'
    eventpath=ROOT/'research/english_v2_data/modern_candidates/beemo/observed_parent_role_constraints.json'
    events=json.loads(eventpath.read_text());observed={r['prompt_id'] for r in events['observed']};assert len(observed)==9
    guardpath=ROOT/'experiments/english_v1/language_guard.py'
    spec=importlib.util.spec_from_file_location('capacity_language_guard',guardpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    guard=module.LanguageGuard(identifier=LanguageIdentifier.from_modelstring(LANGID_MODEL,norm_probs=True))
    native=pq.read_table(path).to_pylist();assert len(native)==2187
    nativeIDs={r['id'] for r in native};assert len(nativeIDs)==2187
    parents={r['prompt_id'] for r in native};assert len(parents)==2187 and observed<=parents
    rows=[];parse_errors=Counter();schemas=Counter();language_cache={}
    owners={};uf=UF(len(native));indices={r['prompt_id']:i for i,r in enumerate(native)}
    keys_per_parent=defaultdict(set)
    def add(parent,source_id,category,base_model,kind,text,editor=None,edit_prompt=None):
        if not isinstance(text,str) or not text.strip():parse_errors[kind+':empty_or_nonstring']+=1;return
        norm=normalized(text);tokens=TOKEN.findall(norm)
        raw=sh(text);normsha=sh(norm);wordlike=len(WORD.findall(text));quality=qualify(text,wordlike)
        # Actual lightweight language/length gate, no detector/model scores.
        if normsha not in language_cache:language_cache[normsha]=guard.check(text)
        scope=language_cache[normsha]
        keyset={'raw:'+raw,'norm:'+normsha}
        prefix=sh(' '.join(tokens[:32])) if len(tokens)>=32 else None
        if prefix:keyset.add('prefix32:'+prefix)
        keys_per_parent[parent].update(keyset)
        row={'id':f'Beemo:{source_id}:{kind}:{edit_prompt or "original"}','native_id':source_id,'prompt_id':parent,'category':category,'original_generator':base_model if kind!='human_output' else None,'last_editor':editor,'edit_prompt':edit_prompt,'variant':kind,'y_ai':0 if kind=='human_output' else None if kind=='human_edits' else 1,'raw_sha256':raw,'normalized_sha256':normsha,'prefix32_sha256':prefix,'characters':len(text),'whitespace_words':len(text.split()),'wordlike_tokens':wordlike,'quality_reason':quality,'nonlength_quality_reason':qualify(text,wordlike,minimum=0),'language_supported':scope['supported'],'language_identified':scope['identified'],'language_probability':scope['probability'],'ascii_letter_ratio':scope['ascii_letter_ratio'],'provisional_core':quality is None and scope['supported'],'observed_parent':parent in observed,'source_revision':'9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162','fine_source_mask':False,'license_scope':'NoRobots_CC-BY-NC-4.0' if kind=='human_output' else 'publisher_generated_editor_underlying_model_terms' if kind!='human_edits' else 'mixed_provenance_MIT_subject_to_base_model_terms'}
        rows.append(row)
    for r in native:
        schemas[tuple(sorted(r))]+=1;parent=r['prompt_id'];i=indices[parent]
        prompt=r.get('prompt')
        if isinstance(prompt,str) and prompt.strip():keys_per_parent[parent].update({'prompt:'+sh(normalized(prompt)),'question:'+sh(normalized(prompt))})
        for k in ('human_output','model_output','human_edits'):
            add(parent,r['id'],r['category'],r['model'],k,r.get(k),'human_expert' if k=='human_edits' else None)
        for k,editor in [('gpt-4o_edits','GPT-4o'),('llama-3.1-70b_edits','Llama-3.1-70B-Instruct')]:
            value=r.get(k)
            if isinstance(value,str):
                try:value=json.loads(value)
                except (json.JSONDecodeError,ValueError):
                    try:value=ast.literal_eval(value)
                    except (ValueError,SyntaxError):parse_errors[k+':unparseable_literal']+=1;continue
            if isinstance(value,dict):value=[value]
            if not isinstance(value,list):parse_errors[k+':not_list']+=1;continue
            seen=set()
            for part in value:
                if not isinstance(part,dict):parse_errors[k+':not_mapping']+=1;continue
                for p,text in part.items():
                    if p not in {'P1','P2','P3'} or p in seen:parse_errors[k+':unexpected_or_duplicate_prompt']+=1;continue
                    seen.add(p);add(parent,r['id'],r['category'],r['model'],k,text,editor,p)
            if seen!={'P1','P2','P3'}:parse_errors[k+':incomplete_prompt_set']+=1
        if (i+1)%500==0:print(json.dumps({'native_rows_processed':i+1,'text_variants_indexed':len(rows)}),flush=True)
    for p,keys in keys_per_parent.items():
        i=indices[p]
        for key in keys:
            if key in owners:uf.union(i,owners[key])
            else:owners[key]=i
    members=defaultdict(list)
    for p,i in indices.items():members[uf.find(i)].append(p)
    component={p:sh('Beemo:component:'+min(members[uf.find(i)])) for p,i in indices.items()}
    observed_components={component[p] for p in observed}
    for r in rows:
        r['component_id']=component[r['prompt_id']]
        r['component_has_observed_parent']=r['component_id'] in observed_components
    core=[r for r in rows if r['provisional_core'] and not r['component_has_observed_parent']]
    h={r['component_id'] for r in core if r['variant']=='human_output'}
    edited={r['component_id'] for r in core if r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'}}
    direct={r['component_id'] for r in core if r['variant']=='model_output'}
    def summary(subset):
        return {'rows':len(subset),'components':len({r['component_id'] for r in subset}),'native_prompt_groups':len({r['prompt_id'] for r in subset}),'unique_normalized_texts':len({r['normalized_sha256'] for r in subset})}
    categories=[]
    for c in sorted({r['category'] for r in rows}):
        rr=[r for r in core if r['category']==c]
        hh={r['component_id'] for r in rr if r['variant']=='human_output'}
        aa={r['component_id'] for r in rr if r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'}}
        categories.append({'category':c,'core_human':summary([r for r in rr if r['variant']=='human_output']),'core_edited_AI':summary([r for r in rr if r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'}]),'paired_human_edited_components':len(hh&aa)})
    threshold_capacity=[]
    for minimum in (32,64,80,100,128):
        scoped=[r for r in rows if r['wordlike_tokens']>=minimum and r['nonlength_quality_reason'] is None and r['language_supported'] and not r['component_has_observed_parent']]
        humans={r['component_id'] for r in scoped if r['variant']=='human_output'}
        edited_parents={r['component_id'] for r in scoped if r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'}}
        threshold_capacity.append({'minimum_wordlike_tokens':minimum,'human_components':len(humans),'edited_AI_rows':sum(r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'} for r in scoped),'edited_AI_components':len(edited_parents),'human_edited_paired_components':len(humans&edited_parents),'by_editor_prompt':[{'editor':editor,'edit_prompt':prompt,'rows':len(rr:=[r for r in scoped if r['last_editor']==editor and r['edit_prompt']==prompt]),'paired_components':len(humans&{r['component_id'] for r in rr})} for editor in ['GPT-4o','Llama-3.1-70B-Instruct'] for prompt in ['P1','P2','P3']]})
    length_quantiles=[]
    import numpy as np
    for variant in sorted({r['variant'] for r in rows}):
        rr=[r for r in rows if r['variant']==variant]
        length_quantiles.append({'variant':variant,'wordlike_quantiles_0_25_50_75_95_100':np.quantile([r['wordlike_tokens'] for r in rr],[0,.25,.5,.75,.95,1]).tolist(),'characters_max':max(r['characters'] for r in rr),'nonlength_quality_reasons':dict(Counter(str(r['nonlength_quality_reason']) for r in rr))})
    index=OUT/'beemo_variant_metadata.parquet';table=pa.Table.from_pylist(rows);pq.write_table(table,index,compression='zstd')
    parent_index=OUT/'beemo_parent_keys.parquet';pq.write_table(pa.Table.from_pylist([{'prompt_id':p,'component_id':component[p],'keys_sha256':[sh(k) for k in sorted(keys)],'observed_parent':p in observed,'component_has_observed_parent':component[p] in observed_components} for p,keys in sorted(keys_per_parent.items())]),parent_index,compression='zstd')
    info={'kind':'Beemo_actual_core_metadata_capacity_not_v2_freeze','created_at_utc':datetime.now(timezone.utc).isoformat(),'complete_file':{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':file_sha(path),'revision':'9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162'},'no_text_preview':True,'no_detector_or_predictions':True,'no_new_roles_assigned':True,'native_rows':len(native),'native_prompt_groups':len(parents),'within_source_observable_components':len(members),'text_variants':len(rows),'parse_errors':dict(parse_errors),'prospective_scope_capacity_curves_not_a_rule_change':threshold_capacity,'variant_length_distribution':length_quantiles,'schemas':[{'fields':list(k),'rows':v} for k,v in schemas.items()],'observed_native_prompt_groups':len(observed),'observed_connected_components':len(observed_components),'observed_connected_native_prompt_groups':sum(r['component_has_observed_parent'] for r in pq.read_table(parent_index).to_pylist()),'core_rule':'>=128wordlike;30..3000whitespace;05visible_think/truncation heuristics; existing shared deployment LanguageGuard finite sampling .8/en and ASCIIletters>=.9','language_guard_sha256':file_sha(guardpath),'language_unique_texts_checked':len(language_cache),'core_unobserved_human':summary([r for r in core if r['variant']=='human_output']),'core_unobserved_direct_AI':summary([r for r in core if r['variant']=='model_output']),'core_unobserved_edited_AI':summary([r for r in core if r['variant'] in {'gpt-4o_edits','llama-3.1-70b_edits'}]),'paired_human_edited_components_before_oldbarrier_near':len(h&edited),'paired_human_direct_components_before_oldbarrier_near':len(h&direct),'per_variant':[{'variant':v,'all':summary([r for r in rows if r['variant']==v]),'core_unobserved':summary([r for r in core if r['variant']==v]),'quality_reasons':dict(Counter(str(r['quality_reason']) for r in rows if r['variant']==v))} for v in sorted({r['variant'] for r in rows})],'by_category':categories,'metadata_outputs':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha(p),'bytes':p.stat().st_size} for p in [index,parent_index]],'observed_role_constraints':{'path':str(eventpath.relative_to(ROOT)),'sha256':file_sha(eventpath)},'elapsed_seconds':time.monotonic()-started,'limitations':['Old-used graph observable and lexical near barriers remain to run jointly with MAGA; these are capacity upperbounds, not final eligibility.','Within-source components union original prompt, raw/normalized output hashes and first32 lexical prefixes across ALL variants, including rejected/short/expert-edited bridge texts.','Human expert edits remain mixed provenance diagnostics, never automatic human negatives.','Each output has original generator and last editor separately; AI-edited text is AI positive without inventing unique fine-model identity.','NoRobots human/prompt NC license is allowed for this noncommercial project, but is not MIT and raw GitHub redistribution is not granted by the source code MIT license.','Repeated edited variants and repeated controls do not create independent human samples; counts disclose connected components.']}
    (OUT/'beemo_actual_capacity.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:info[k] for k in ('native_rows','within_source_observable_components','text_variants','parse_errors','observed_connected_native_prompt_groups','core_unobserved_human','core_unobserved_edited_AI','paired_human_edited_components_before_oldbarrier_near','by_category','prospective_scope_capacity_curves_not_a_rule_change','variant_length_distribution','elapsed_seconds')},indent=2))
if __name__=='__main__':main()
