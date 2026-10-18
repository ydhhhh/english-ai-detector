"""All preserved old/new native/observable/lexical components, without allocating roles.

Only metadata projections are exported; programmatic old-text reads below derive
consistent wordlike/structural predicates. No detector, scores or body preview.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,re,sqlite3,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit';CACHE=ROOT/'.cache/english_v2_lexical_v1'

def sh(x):return hashlib.sha256(x.encode()).hexdigest()
def fs(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity(x):
    if x is None:return None
    x=str(x).strip();return None if x.casefold() in {'','none','null','nan','na','n/a','<na>'} else x
class UF:
    def __init__(self,n):self.p=list(range(n));self.z=[1]*n
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def union(self,i,j):
        a,b=self.find(i),self.find(j)
        if a==b:return
        if self.z[a]<self.z[b]:a,b=b,a
        self.p[b]=a;self.z[a]+=self.z[b]

def main():
    started=time.monotonic();inputs=[]
    def read(name,columns=None):
        p=ROOT/name;cols=columns or pq.read_schema(p).names
        assert not {'text','generation','prompt','title','native_title','human_output','model_output','human_edits','gpt-4o_edits','llama-3.1-70b_edits'}&set(cols)
        inputs.append({'path':name,'sha256':fs(p),'columns':cols});return pq.read_table(p,columns=cols).to_pylist()
    oldroles=read('research/english_v2_data/audit/existing_component_roles.parquet');oldpos={r['component_id']:i for i,r in enumerate(oldroles)}
    maga=read('research/english_v2_data/audit/maga_metadata_index.parquet');beemo=read('research/english_v2_data/audit/beemo_variant_metadata.parquet');beemokeys=read('research/english_v2_data/audit/beemo_parent_keys.parquet');raid=read('research/english_v2_data/audit/raid_metadata_index.parquet')
    titles=read('research/english_v2_data/audit/article_title_bridge_metadata.parquet')
    parentkeys=defaultdict(set);parentflags=defaultdict(set);rows=[]
    def contentkeys(parent,r):
        if r.get('normalized_sha256')==sh(''):return
        for namespace,column in [('raw','raw_sha256'),('norm','normalized_sha256'),('prefix32','prefix32_sha256'),('prompt','full_prompt_sha256'),('prompt','user_prompt_sha256')]:
            if r.get(column):parentkeys[parent].add(sh(namespace+':'+r[column]))
    for r in maga:
        parent='MAGA:'+str(r['human_source_id'] or 'missing:'+r['file']+':'+str(r['line_number']));contentkeys(parent,r)
        if r.get('id'):parentkeys[parent].add(sh('MAGA_variant_ID:'+r['id']))
        if r.get('prompt_id'):parentkeys[parent].add(sh('MAGA_prompt_ID:'+r['prompt_id']))
        if r.get('title_sha256'):
            parentkeys[parent].add(sh('MAGA_source_title:'+r['canonical_domain']+':'+r['title_sha256']))
            if r['canonical_domain']=='naturalquestions':parentkeys[parent].add(sh('question:'+r['title_sha256']))
        if r['wikiHow_parent_forbidden']:parentflags[parent].add('wikiHow_forbidden')
        if not r['human_source_id']:parentflags[parent].add('missing_parent_ID')
        if r['canonical_domain']=='naturalquestions':parentflags[parent].add('draft_NQ_source_holdout')
        if r['model']=='Gemini-2.0-flash':parentflags[parent].add('draft_Gemini_nominal_holdout')
        rows.append({'row_record_id':'MAGA:'+r['file']+':'+str(r['line_number']),'file':r['file'],'line_number':r['line_number'],'parent':parent,'collector':'MAGA','source':'MAGA/'+r['domain'],'domain':r['domain'],'config':r['config'],'variant':'human' if r['y_ai']==0 else 'direct' if r['config']=='MGB' else 'fused_alignment','id':'MAGA:'+str(r['id']),'raw_id':r['id'],'y_ai':r['y_ai'],'normalized_sha256':r['normalized_sha256'],'raw_sha256':r['raw_sha256'],'wordlike_tokens':r['wordlike_tokens'],'structural_core':r['provisional_quality_reason'] is None and r['y_ai'] is not None,'structural_quality_reason':r['provisional_quality_reason'],'runtime_language_supported_diagnostic':r['language_supported'],'publisher_English':True,'source_rights_allowed':r['allow_noncommercial_research'],'nominal_generator':r['model'],'last_editor':None,'canonical_domain':r['canonical_domain'],'title_sha256':r.get('title_sha256'),'view_human_origin_rewrite':False,'view_AI_origin_rewrite':False})
    for r in beemokeys:
        parent='Beemo:'+r['prompt_id'];parentkeys[parent].update(r['keys_sha256'])
        if r['component_has_observed_parent']:parentflags[parent].add('Beemo_documentation_or_viewer_observed')
    for r in beemo:
        rows.append({'row_record_id':r['id'],'parent':'Beemo:'+r['prompt_id'],'collector':'Beemo','source':'Beemo/'+r['category'],'domain':r['category'],'config':'native','variant':r['variant'],'id':r['id'],'raw_id':str(r['native_id']),'y_ai':r['y_ai'],'normalized_sha256':r['normalized_sha256'],'raw_sha256':r['raw_sha256'],'wordlike_tokens':r['wordlike_tokens'],'structural_core':r['provisional_core'] and r['y_ai'] is not None,'structural_quality_reason':None if r['provisional_core'] else 'short_or_structural_Beemo_prior_gate','runtime_language_supported_diagnostic':r.get('language_supported',False),'publisher_English':True,'source_rights_allowed':True,'nominal_generator':r['original_generator'],'last_editor':r['last_editor'],'canonical_domain':r['category'].casefold(),'title_sha256':None,'view_human_origin_rewrite':False,'view_AI_origin_rewrite':False})
    for r in raid:
        parent=r['parent'];contentkeys(parent,r)
        rows.append({'row_record_id':'RAID:'+r['id'],'parent':parent,'collector':'RAID','source':r['source'],'domain':r['domain'],'config':'raid_train_converted_partial','variant':'human' if r['y_ai']==0 else 'human_origin_DIPPER' if r['model']=='human' else 'AI_origin_DIPPER','id':'RAID:'+r['id'],'raw_id':r['id'],'y_ai':r['y_ai'],'normalized_sha256':r['normalized_sha256'],'raw_sha256':r['raw_sha256'],'wordlike_tokens':r['wordlike_tokens'],'structural_core':r['structural_length_core'],'structural_quality_reason':r['provisional_quality_reason'],'runtime_language_supported_diagnostic':r['language_supported'],'publisher_English':r['publisher_English'],'source_rights_allowed':r['source_rights_allowed'],'nominal_generator':r['model'],'last_editor':r['last_editor'],'canonical_domain':r['domain'],'title_sha256':r['title_sha256'],'view_human_origin_rewrite':r['view_human_origin_rewrite'],'view_AI_origin_rewrite':r['view_AI_origin_rewrite']})
    completion=ROOT/'research/english_v2_data/raw/raid_paired_range_v1/COMPLETED.json';done=json.loads(completion.read_text());assert done['all_8_files_complete'];inputs.append({'path':str(completion.relative_to(ROOT)),'sha256':fs(completion)})
    screened_n=0;screened_ids=set()
    for item in done['files']:
        name=str(Path(item['payload']['path']).parent/'screened_lineage_metadata.parquet')
        for r in read(name):
            sid=identity(r.get('source_id'));assert sid is not None;parent='RAID:'+sid
            for key in ['id','source_id','adv_source_id']:
                v=identity(r.get(key))
                if v:parentkeys[parent].add(sh('RAID_identifier:'+v))
            screened_n+=1;screened_ids.add(r['id'])
    for r in titles:
        parentkeys[r['parent']].update(r['observable_key_hashes'])
        if r['ambiguous_title_quarantine']:parentflags[r['parent']].add('ambiguous_Wikipedia_article_title_quarantine')
    observed_references=read('research/english_v2_data/audit/raid_observed_reference_metadata.parquet')
    for r in observed_references:parentkeys['RAID:'+r['source_id']].update(r['observable_key_hashes'])
    parents=sorted(parentkeys);parentpos={p:len(oldroles)+i for i,p in enumerate(parents)}
    signatures=read('research/english_v2_data/audit/full_lexical_signature_index.parquet',['uid','normalized_sha256','raw_sha256','shingle_count']);signatures+=read('research/english_v2_data/audit/raid_observed_reference_new_signatures.parquet',['uid','normalized_sha256','raw_sha256','shingle_count']);normpos={r['normalized_sha256']:r['uid'] for r in signatures};sigoffset=len(oldroles)+len(parents);n=sigoffset+len(signatures);uf=UF(n)
    assert [r['uid'] for r in signatures]==list(range(len(signatures)))
    owners={}
    for parent in parents:
        i=parentpos[parent]
        for key in sorted(parentkeys[parent]):
            if key in owners:uf.union(i,owners[key])
            else:owners[key]=i
    for r in rows:uf.union(parentpos[r['parent']],sigoffset+normpos[r['normalized_sha256']])
    for r in observed_references:uf.union(parentpos['RAID:'+r['source_id']],sigoffset+normpos[r['normalized_sha256']])
    graphpath=ROOT/'data/english_calibration_v2/graph_index.parquet';inputs.append({'path':str(graphpath.relative_to(ROOT)),'sha256':fs(graphpath),'columns':['id','component_id','normalized_sha256','observable_key_hashes']});old_nodes=0;oldnew_keyhits=0;old_id_norm={}
    for batch in pq.ParquetFile(graphpath).iter_batches(batch_size=8192,columns=['id','component_id','normalized_sha256','observable_key_hashes']):
        for r in batch.to_pylist():
            old_nodes+=1;old_id_norm[r['id']]=r['normalized_sha256'];i=oldpos[r['component_id']];uf.union(i,sigoffset+normpos[r['normalized_sha256']])
            for key in r['observable_key_hashes']:
                if key in owners:uf.union(i,owners[key]);oldnew_keyhits+=1
    observed={parentpos['RAID:'+r['source_id']] for r in observed_references}|{sigoffset+normpos[r['normalized_sha256']] for r in observed_references}
    for filename in ['upstream_documentation_example_access.json','prior_manual_review_access_constraints.json']:
        p=OUT/filename;access=json.loads(p.read_text());inputs.append({'path':str(p.relative_to(ROOT)),'sha256':fs(p)})
        constraints=access.get('observed_content_hash_constraints',access.get('constraints',[]))
        for c in constraints:
            if c.get('normalized_sha256') in normpos:observed.add(sigoffset+normpos[c['normalized_sha256']])
            for key in c.get('key_sha256',[]):
                if key in owners:observed.add(owners[key])
            card=c.get('source_card','');field=c.get('field');domains={'naturalquestions'} if 'natural-questions' in card else {'s2orc'} if 's2orc' in card else {'wikipedia'}
            title=c.get('title_sha256') or (c.get('normalized_sha256') if field in {'query','question','title'} else None)
            if title:observed.update(parentpos[r['parent']] for r in rows if r['collector']=='MAGA' and r['canonical_domain'] in domains and r['title_sha256']==title)
    google_path=OUT/'upstream_documentation_example_root_google.json';google=json.loads(google_path.read_text());inputs.append({'path':str(google_path.relative_to(ROOT)),'sha256':fs(google_path)})
    observed.update(parentpos[r['parent']] for r in rows if r['collector']=='MAGA' and r['canonical_domain'] in google['canonical_domains'] and r['title_sha256']==google['title_sha256'])
    seedflags=defaultdict(set)
    for r in oldroles:
        i=oldpos[r['component_id']]
        if r['forbid_fresh']:seedflags[i].add('old_used_or_reserved_forbid_fresh')
        if r['touches_nontraining_assigned'] or r['reserved05']:seedflags[i].add('old_nontrain_or_reserved_forbid_new_training')
        if r['touches_actual_fit']:seedflags[i].add('old_actual_fit_kin')
        if r['forbidden_wikihow']:seedflags[i].add('wikiHow_forbidden')
        if r['observed_manual_review']:seedflags[i].add('prior_manual_review_observed')
    for parent in parents:seedflags[parentpos[parent]].update(parentflags[parent])
    for i in observed:seedflags[i].add('upstream_documentation_example_observed')
    def label(i):
        if i<len(oldroles):return 'old:'+oldroles[i]['component_id']
        if i<sigoffset:return parents[i-len(oldroles)]
        return 'norm:'+signatures[i-sigoffset]['normalized_sha256']
    def component_state():
        flags=defaultdict(set);anchors={};roots=[]
        for i in range(n):
            root=uf.find(i);roots.append(root);s=label(i)
            if root not in anchors or s<anchors[root]:anchors[root]=s
        for i,ff in seedflags.items():flags[uf.find(i)].update(ff)
        ids={root:sh('v2_joint_component:'+anchor) for root,anchor in anchors.items()}
        return roots,flags,ids
    preroot,preflags,preids=component_state();pre_component=[preids[i] for i in preroot];pre_nodeflags=[sorted(preflags[i]) for i in preroot]
    nearproof=OUT/'full_lexical_near_completion.json';near=json.loads(nearproof.read_text());assert near['complete'] is True
    cache=json.loads((CACHE/'complete.json').read_text());assert fs(OUT/'lexical_signature.py')==cache['header']['helper_sha256'];assert fs(ROOT/'experiments/english_v1/data_grouping.py')==cache['header']['legacy_algorithm_sha256']
    for o in near['outputs']:assert fs(ROOT/o['path'])==o['sha256']
    inputs.append({'path':str(nearproof.relative_to(ROOT)),'sha256':fs(nearproof)})
    direct_edges=Counter();annotated=[]
    all_near_edges=read('research/english_v2_data/audit/full_lexical_near_edges.parquet')
    for edge in read('research/english_v2_data/audit/raid_observed_reference_near_edges.parquet'):
        all_near_edges.append({'left_uid':normpos[edge['left_normalized_sha256']],'right_uid':normpos[edge['right_normalized_sha256']],'jaccard':edge['jaccard'],'kind':edge['kind'],'band':edge['band'],'bucket_key':None})
    for r in all_near_edges:
        a,b=sigoffset+r['left_uid'],sigoffset+r['right_uid'];fa,fb=set(pre_nodeflags[a]),set(pre_nodeflags[b]);different=pre_component[a]!=pre_component[b]
        crossfitheld=('old_actual_fit_kin' in fa and 'old_nontrain_or_reserved_forbid_new_training' in fb) or ('old_actual_fit_kin' in fb and 'old_nontrain_or_reserved_forbid_new_training' in fa)
        direct_edges[(r['kind'],'different_pre_components')]+=int(different);direct_edges[(r['kind'],'old_actualfit_to_nontrain_or_reserved_across_pre_components')]+=int(crossfitheld and different)
        annotated.append({**r,'left_pre_component':pre_component[a],'right_pre_component':pre_component[b],'left_pre_barrier_flags':pre_nodeflags[a],'right_pre_barrier_flags':pre_nodeflags[b],'different_pre_components':different,'cross_old_actualfit_nontrain_or_reserved':crossfitheld});uf.union(a,b)
    roots,flags,component_ids=component_state();del preroot,preflags
    norm_labels=defaultdict(set)
    for r in rows:
        if r['y_ai'] is not None:norm_labels[r['normalized_sha256']].add(r['y_ai'])
    for filename in ['data/english_v1/full_index.parquet','data/english_augmentation_v2/full_index.parquet','data/english_external_v1/full_index.parquet','data/raid_external_v1/full_index.parquet','data/raid_external_v2/full_index.parquet']:
        for r in read(filename,['normalized_sha256','y_ai']):
            if r['y_ai'] in [0,1]:norm_labels[r['normalized_sha256']].add(r['y_ai'])
    for split in ['train','validation','calibration']:
        for r in read(f'data/english_v1/pilot/{split}.parquet',['id','y_ai']):norm_labels[old_id_norm[r['id']]].add(r['y_ai'])
    conflicts={h for h,yy in norm_labels.items() if len(yy)>1}
    forbidden_capacity={'wikiHow_forbidden','missing_parent_ID','ambiguous_Wikipedia_article_title_quarantine'}
    fresh_barrier={'old_used_or_reserved_forbid_fresh','Beemo_documentation_or_viewer_observed','upstream_documentation_example_observed','prior_manual_review_observed'}
    for r in rows:
        i=parentpos[r['parent']];f=flags[roots[i]];r['pre_near_component_id']=pre_component[i];r['component_id']=component_ids[roots[i]];r['barrier_flags']=sorted(f);r['norm_uid']=normpos[r['normalized_sha256']];r['binary_conflicting_normalized_text']=r['normalized_sha256'] in conflicts
        r['capacity_qualified']=r['structural_core'] and r['publisher_English'] and r['source_rights_allowed'] and not r['binary_conflicting_normalized_text'] and not forbidden_capacity&f
        r['candidate_fresh']=r['capacity_qualified'] and not fresh_barrier&f
        r['candidate_train']=r['capacity_qualified'] and 'old_nontrain_or_reserved_forbid_new_training' not in f
        r['candidate_train_under_draft_holdouts']=r['candidate_train'] and not {'draft_NQ_source_holdout','draft_Gemini_nominal_holdout'}&f
    # Existing MAGE/community candidates retain prior-admission limits, with
    # consistent wordlike+structure derived from their raw cache references.
    candidates=read('research/english_v2_data/audit/existing_graph_candidate_index.parquet');refdb=sqlite3.connect('file:'+str(CACHE/'signatures.sqlite3')+'?mode=ro',uri=True);bodydb=sqlite3.connect('file:'+str(CACHE/'new_texts.sqlite3')+'?mode=ro',uri=True);olddb=sqlite3.connect('file:'+str(ROOT/'research/english_v1_data/augmentation_v2/texts.sqlite3')+'?mode=ro',uri=True)
    spec=importlib.util.spec_from_file_location('structural_quality',OUT/'index_maga_metadata.py');quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
    for r in candidates:
        i=oldpos[r['component_id']];r['old_component_id']=r['component_id'];r['pre_near_component_id']=pre_component[i];r['component_id']=component_ids[roots[i]];f=flags[roots[i]];r['barrier_flags']=sorted(f);r['norm_uid']=normpos[r['normalized_sha256']]
        reason='not_previously_admitted';words=None
        if r['capacity_gate']=='prior_admitted':
            kind,key=refdb.execute('SELECT ref_kind,ref_key FROM raw_refs WHERE raw_sha=?',(r['raw_sha256'],)).fetchone()
            text=olddb.execute('SELECT text FROM texts WHERE i=?',(int(key),)).fetchone()[0] if kind=='old_sql' else bodydb.execute('SELECT text FROM texts WHERE raw_sha=?',(key,)).fetchone()[0]
            assert sh(text)==r['raw_sha256'];words=len(quality.WORD.findall(text));reason=quality.simple_quality(text,words)
        r['wordlike_tokens']=words;r['structural_quality_reason']=reason;r['binary_conflicting_normalized_text']=r['normalized_sha256'] in conflicts;r['capacity_qualified']=reason is None and r['y_ai'] in [0,1] and not r['binary_conflicting_normalized_text'] and not forbidden_capacity&f;r['candidate_fresh']=r['capacity_qualified'] and not fresh_barrier&f
    refdb.close();bodydb.close();olddb.close()
    def summary(rr):
        h={r['component_id'] for r in rr if r['y_ai']==0};a={r['component_id'] for r in rr if r['y_ai']==1};hr={r['normalized_sha256'] for r in rr if r['y_ai']==0};ar={r['normalized_sha256'] for r in rr if r['y_ai']==1}
        return {'variant_rows':len(rr),'human_rows':sum(r['y_ai']==0 for r in rr),'AI_rows':sum(r['y_ai']==1 for r in rr),'distinct_human_texts':len(hr),'distinct_AI_texts':len(ar),'human_components':len(h),'AI_components':len(a),'paired_components':len(h&a),'components':len(h|a)}
    def three_way(rr):
        h={r['component_id'] for r in rr if r['y_ai']==0};ha={r['component_id'] for r in rr if r['variant']=='human_origin_DIPPER'};aa={r['component_id'] for r in rr if r['variant']=='AI_origin_DIPPER'}
        hp={r['parent'] for r in rr if r['y_ai']==0};hap={r['parent'] for r in rr if r['variant']=='human_origin_DIPPER'};aap={r['parent'] for r in rr if r['variant']=='AI_origin_DIPPER'}
        return {'three_way_components':len(h&ha&aa),'three_way_native_parents':len(hp&hap&aap)}
    outputfiles=[]
    def save(name,rr):
        p=OUT/name;pq.write_table(pa.Table.from_pylist(rr),p,compression='zstd');outputfiles.append({'path':str(p.relative_to(ROOT)),'rows':len(rr),'sha256':fs(p),'bytes':p.stat().st_size})
    save('joint_binary_conflict_hashes.parquet',[{'normalized_sha256':h,'labels':sorted(norm_labels[h])} for h in sorted(conflicts)]);save('joint_new_candidate_index.parquet',rows);save('joint_existing_candidate_index.parquet',candidates);save('joint_near_annotated_edges.parquet',annotated)
    nodenames=[]
    for i in range(n):
        nodenames.append({'node_index':i,'kind':'old_component' if i<len(oldroles) else 'native_parent' if i<sigoffset else 'normalized_body','identity':oldroles[i]['component_id'] if i<len(oldroles) else parents[i-len(oldroles)] if i<sigoffset else signatures[i-sigoffset]['normalized_sha256'],'norm_uid':i-sigoffset if i>=sigoffset else None,'pre_near_component_id':pre_component[i],'component_id':component_ids[roots[i]],'seed_barrier_flags':sorted(seedflags[i]),'pre_near_barrier_flags':pre_nodeflags[i],'barrier_flags':sorted(flags[roots[i]])})
    save('joint_component_node_index.parquet',nodenames)
    parent_evidence=[{'parent':p,'node_index':parentpos[p],'observable_key_hashes':sorted(parentkeys[p]),'seed_barrier_flags':sorted(parentflags[p]),'pre_near_component_id':pre_component[parentpos[p]],'component_id':component_ids[roots[parentpos[p]]]} for p in parents];save('joint_native_parent_keys.parquet',parent_evidence)
    bysource=[]
    for collector,domain in sorted({(r['collector'],r['domain']) for r in rows}):
        rr=[r for r in rows if r['collector']==collector and r['domain']==domain];fresh=[r for r in rr if r['candidate_fresh']]
        bysource.append({'collector':collector,'domain':domain,'qualified_before_final_barrier':summary([r for r in rr if r['capacity_qualified']]),'fresh_after_full_observable_near_graph':summary(fresh),'fresh_runtime_supported_diagnostic':summary([r for r in fresh if r['runtime_language_supported_diagnostic']]),'train_under_draft_source_generator_holdouts':summary([r for r in rr if r['candidate_train_under_draft_holdouts']]),'fresh_rewrite_three_way':three_way(fresh) if collector=='RAID' else None})
    result={'kind':'full_joint_observable_lexical_capacity_NOT_ROLE_FREEZE','created_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'models_or_scores':False,'body_previewed':False,'roles_allocated':False,'language_supported_is_not_fresh_eligibility_filter':True,'old_graph_bridge_rows':old_nodes,'old_component_nodes':len(oldroles),'new_native_parent_nodes':len(parents),'normalized_body_nodes':len(signatures),'joint_graph_nodes':n,'joint_components':len(component_ids),'old_new_observable_key_matches':oldnew_keyhits,'all_screened_RAID_lineage_rows':screened_n,'distinct_screened_RAID_ids':len(screened_ids),'new_binary_conflict_normalized_hashes':len(conflicts),'newsource_fresh':summary([r for r in rows if r['candidate_fresh']]),'old_unused_fresh':summary([r for r in candidates if r['candidate_fresh']]),'RAID_fresh_three_way':three_way([r for r in rows if r['collector']=='RAID' and r['candidate_fresh']]),'by_collector_domain':bysource,'MAGA_fresh_Gemini_parent_rows':summary([r for r in rows if r['collector']=='MAGA' and r['candidate_fresh'] and 'draft_Gemini_nominal_holdout' in r['barrier_flags']]),'near_direct_edge_annotations':[{'kind':k[0],'statistic':k[1],'count':v} for k,v in sorted(direct_edges.items())],'old_cross_training_nontraining_components_before_near':sum('old_actual_fit_kin' in ff and 'old_nontrain_or_reserved_forbid_new_training' in ff for ff in {pre_component[i]:set(pre_nodeflags[i]) for i in range(len(oldroles))}.values()),'old_cross_training_nontraining_components_after_near':sum('old_actual_fit_kin' in flags[root] and 'old_nontrain_or_reserved_forbid_new_training' in flags[root] for root in {roots[i] for i in range(len(oldroles))}),'outputs':outputfiles,'inputs':inputs,'limitations':['No role assignment, post-quota train capacity or score evidence. Counts remain tied to source-rights and structural predicates, and a frozen allocation protocol is still required.','All old raw bridges, every new native parent/config/observed variant, title aliases and all known excluded references remain graph nodes regardless eligibility.','LSH is approximate first3000token lexical candidate retrieval, not proof of hidden original prompt, semantic or pretrained-model independence.','Fresh source English provenance is independent of runtime language_supported. Unsupported English rows remain in future true-class denominators as abstentions; calibration fitting uses registered supported-core only.','Prior-admitted unused old rows are rechecked for consistent wordlike/structural criteria; other old admission/rights exclusions are not silently relaxed.','All v1 old actual-fit/nontrain links found by new bridges/near are prospective v2 barriers and disclosed errata; frozen v1 files and historical results are not rewritten.','Canonical CMU/Wikipedia-title bridges are conservative source-level clues, not per-article authentication; ambiguous/generic titles quarantined.']}
    (OUT/'joint_full_near_capacity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['old_graph_bridge_rows','joint_graph_nodes','joint_components','newsource_fresh','old_unused_fresh','RAID_fresh_three_way','by_collector_domain','near_direct_edge_annotations','elapsed_seconds']},indent=2))
if __name__=='__main__':main()
