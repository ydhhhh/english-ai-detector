"""Metadata-only provisional v2 capacity; no text column/model imports.

The existing graph is an observable barrier upper-bound, not a new split.
All policy decisions await a new frozen v2 protocol.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json
from datetime import datetime, timezone
import pyarrow as pa
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/english_v2_data/audit'
READS = []

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(rel, columns):
    assert 'text' not in columns and 'generation' not in columns and 'prompt' not in columns
    path=ROOT/rel
    table=pq.read_table(path,columns=columns)
    READS.append({'path':rel,'columns':columns,'rows':table.num_rows,'bytes':path.stat().st_size,'sha256':sha(path)})
    return table.to_pylist()

def count(rows):
    return {'rows':len(rows),'human':sum(r['y_ai']==0 for r in rows),'AI':sum(r['y_ai']==1 for r in rows),'components':len({r['component_id'] for r in rows})}

def breakdown(rows, keys):
    result=[]
    for k, values in sorted(group(rows, keys).items(),key=lambda item:str(item[0])):
        result.append({**dict(zip(keys,k)),**count(values)})
    return result

def group(rows, keys):
    output=defaultdict(list)
    for r in rows:
        output[tuple(r.get(k) for k in keys)].append(r)
    return output

def main():
    graph=read('data/english_calibration_v2/graph_index.parquet',['id','origin','component_id','reserved_role','raw_sha256','observable_key_hashes'])
    comp_by_id=defaultdict(set)
    for r in graph: comp_by_id[r['id']].add(r['component_id'])
    assert all(len(v)==1 for v in comp_by_id.values())
    comp_by_id={k:next(iter(v)) for k,v in comp_by_id.items()}
    primary=read('data/english_v1/full_index.parquet',['id','y_ai','source','domain','family','model','canonical_model_id','mask_family','mask_model','source_revision','original_split','eligible','rejection_reason','language','language_probability','words','characters','output_terms_review','admission','split','raw_sha256','normalized_sha256'])
    aug=read('data/english_augmentation_v2/full_index.parquet',['id','y_ai','source','domain','family','model','canonical_model_id','mask_family','mask_model','source_revision','original_split','eligible','admission_eligible','rejection_reason','language','language_probability','whitespace_words','split','raw_sha256','normalized_sha256'])
    pilot=read('data/english_v1/pilot/train.parquet',['id'])
    pilotval=read('data/english_v1/pilot/validation.parquet',['id'])
    pilotcal=read('data/english_v1/pilot/calibration.parquet',['id'])
    mixed=read('data/english_augmentation_v2/train_mixed100k_with_official.parquet',['id','y_ai'])
    adapted=read('data/english_adaptation_v3/training_index.parquet',['id','y_ai','adaptation_cohort'])
    primary_fit={r['id'] for r in primary if r['split']=='train'}
    mixed_fit={r['id'] for r in mixed}
    adapted_fit={r['id'] for r in adapted}
    pilot_fit={r['id'] for r in pilot}
    actual_fit=primary_fit|mixed_fit|adapted_fit|pilot_fit
    seeds={}
    seeds['main_v1_assigned_all_roles']={r['id'] for r in primary if r['split']!='excluded_or_duplicate_or_subsampled'}
    seeds['actual_neural_fit_and_stat_vocab_superset']=actual_fit
    seeds['pilot_all_assigned']={r['id'] for r in pilot+pilotval+pilotcal}
    seeds['HC3_allowed_development']={r['id'] for r in read('data/english_external_v1/dev_backup_allowed.parquet',['id'])}
    seeds['HC3_allowed_final']={r['id'] for r in read('data/english_external_v1/final_sealed_allowed.parquet',['id'])}
    seeds['RAID_v1_selected']={r['id'] for r in read('data/raid_external_v1/final_sealed_allowed.parquet',['id'])}
    seeds['RAID_v2_selected']={r['id'] for r in read('data/raid_external_v2/final_sealed_allowed.parquet',['id'])}
    extra_old_finals=['data/english_v1/challenge_paraphrase.parquet','data/english_v1/challenge_unseen_gpt4.parquet']
    for path in extra_old_finals:
        seeds['old_observed_final_'+Path(path).stem]={r['id'] for r in read(path,['id'])}
    reserved_components={r['component_id'] for r in graph if r['reserved_role']!='unassigned'}
    allseeds=set().union(*seeds.values())
    missing=allseeds-comp_by_id.keys()
    assert not missing, sorted(missing)[:10]
    manual_path=OUT/'prior_manual_review_access_constraints.json'
    manual=json.loads(manual_path.read_text());manual_hashes={r['raw_sha256'] for r in manual['constraints']};manual_keys={key for r in manual['constraints'] for key in r['key_sha256']}
    manual_components={r['component_id'] for r in graph if r['raw_sha256'] in manual_hashes or any(key in manual_keys for key in r['observable_key_hashes'])}
    protected_components={comp_by_id[i] for i in allseeds}|reserved_components|manual_components
    # Role summaries are audit metadata, not a new dataset assignment.
    nontrain_ids={r['id'] for r in primary if r['split'] not in {'train','excluded_or_duplicate_or_subsampled'} and r['id'] not in actual_fit}
    for name,ids in seeds.items():
        if name.startswith(('HC3_','RAID_','old_observed_final_')):nontrain_ids.update(ids)
    nontrain_ids.update(r['id'] for r in pilotval+pilotcal)
    fit_components={comp_by_id[i] for i in actual_fit}
    nontrain_components={comp_by_id[i] for i in nontrain_ids}
    wikihow_components={comp_by_id[r['id']] for r in primary if str(r.get('domain','')).casefold().replace('_','')=='wikihow'}
    all_components={r['component_id'] for r in graph}
    component_roles=[]
    for component in sorted(all_components):
        component_roles.append({'component_id':component,'touches_actual_fit':component in fit_components,'touches_nontraining_assigned':component in nontrain_components,'reserved05':component in reserved_components,'observed_manual_review':component in manual_components,'forbidden_wikihow':component in wikihow_components,'forbid_fresh':component in protected_components or component in wikihow_components,'may_replay_train_subject_to_source_rights':component in fit_components and component not in nontrain_components and component not in reserved_components and component not in wikihow_components})
    roles_path=OUT/'existing_component_roles.parquet'
    pq.write_table(pa.Table.from_pylist(component_roles),roles_path,compression='zstd')
    seed_components={name:{comp_by_id[i] for i in ids} for name,ids in seeds.items()}
    candidates=[]
    all_candidates=[]
    for origin,rows in [('primary',primary),('augmentation',aug)]:
        for r in rows:
            r=dict(r,origin=origin,component_id=comp_by_id[r['id']])
            if r['component_id'] in protected_components: continue
            if origin=='primary':
                if r['split']!='excluded_or_duplicate_or_subsampled': raise AssertionError('Assigned primary escaped seed')
                r['length_evidence_words']=r.pop('words')
                r['capacity_gate']='prior_admitted' if r['eligible'] and r['language']=='en' else 'not_prior_admitted'
            else:
                r['length_evidence_words']=r.pop('whitespace_words')
                r['capacity_gate']='prior_admitted' if r['admission_eligible'] and r['language']=='en' else 'new_role_language_quality_review_required'
            r['actually_fit_id']=r['id'] in actual_fit
            assert not r['actually_fit_id']
            all_candidates.append(r)
            if r['capacity_gate']=='prior_admitted' and r['length_evidence_words']>=128:
                candidates.append(r)
    # Bind every real training family using metadata, without reading any metric.
    producer_proofs=[]
    for path in sorted((ROOT/'experiments/english_v1/runs').glob('**/detector.json')):
        metadata=json.loads(path.read_text())
        config=metadata.get('training_config',{})
        train=config.get('train')
        if not train or train.startswith('experiments/english_v1/runs/integration_cpu/'):
            continue  # Synthetic engineering fixtures, no real corpus input.
        assert train in {'data/english_v1/train.parquet','data/english_augmentation_v2/train_mixed100k_with_official.parquet','data/english_adaptation_v3/train_replay50k_wp.parquet'}
        assert config.get('max_train') in (None,60000)
        producer_proofs.append({'metadata':str(path.relative_to(ROOT)),'metadata_sha256':sha(path),'training_config_only':{'train':train,'max_train':config.get('max_train'),'seed':config.get('seed')},'input_sha256':sha(ROOT/train),'coverage_argument':'Entire train input included in actual-fit ID superset, so deterministic max_train subset cannot add new IDs.'})
    statistical_proofs=[]
    for path in sorted((ROOT/'results/english_v1/statistics').glob('*/metadata.json')):
        metadata=json.loads(path.read_text())
        fit=metadata['fit']
        source_inputs={}
        for role,item in metadata.get('inputs',{}).items():
            clean_path=item.get('path','')
            if str(ROOT)+'/' in clean_path: clean_path=clean_path[len(str(ROOT))+1:]
            source_inputs[role]={'path':clean_path,'sha256':item.get('sha256'),'purpose':item.get('purpose')}
        statistical_proofs.append({'metadata':str(path.relative_to(ROOT)),'metadata_sha256':sha(path),'inputs':source_inputs,'train_input_rows':fit['train_input_rows'],'feature_fit_rows':fit['feature_fit_rows'],'official_head_added_rows':fit.get('official_nominal_refinement',{}).get('official_training_rows_used_in_selected_head',0),'coverage_argument':'Statistical features and heads fit a subset of input train; entire primary train, pilot train and official_train are already in the neural/pilot ID superset.'})
    additional_old_final_coverage=[]
    for path in extra_old_finals:
        ids={r['id'] for r in read(path,['id'])}
        assert ids<=allseeds
        additional_old_final_coverage.append({'path':path,'IDs':len(ids),'all_explicitly_protected_including_later_views':True,'ids_outside_original_primary_assigned_role':len(ids-seeds['main_v1_assigned_all_roles'])})
    # One ID may exist in graph aliases, but the two parent corpora have distinct IDs.
    assert len({r['id'] for r in candidates})==len(candidates)
    by_comp=group(candidates,['component_id'])
    paired=[rows for rows in by_comp.values() if {r['y_ai'] for r in rows}=={0,1}]
    conditional=[r for r in all_candidates if r['capacity_gate']!='prior_admitted' and r['length_evidence_words']>=128]
    columns=['id','origin','component_id','y_ai','source','domain','family','model','canonical_model_id','mask_family','mask_model','source_revision','original_split','language','language_probability','length_evidence_words','capacity_gate','raw_sha256','normalized_sha256','output_terms_review','admission','split','rejection_reason']
    schema=pa.schema([(c,pa.int64() if c in {'y_ai','length_evidence_words'} else pa.float64() if c=='language_probability' else pa.bool_() if c in {'mask_family','mask_model','output_terms_review'} else pa.string()) for c in columns])
    # Do not publish text or designate a new split. Audited candidate index only.
    path=OUT/'existing_graph_candidate_index.parquet'
    pq.write_table(pa.Table.from_pylist([{c:r.get(c) for c in columns} for r in all_candidates],schema=schema),path,compression='zstd')
    payload={
      'stage':'metadata_capacity_audit_not_dataset_freeze','created_at_utc':datetime.now(timezone.utc).isoformat(),'read_text':False,'loaded_model':False,'created_train_dev_cal_final_split':False,
      'policy':'All actual-fit and assigned/observed roles are protected for fresh roles. Actual training kin may only be considered train replay under a new protocol. All original raw rows remain graph bridges; unassigned raw existence alone is not a protection seed.',
      'actual_fit_ID_union':{'primary_neural_v1_all_train':len(primary_fit),'Mini2_all_mixed100k':len(mixed_fit),'Mini3_replay50k_plus_WP6104':len(adapted_fit),'pilot_statistical_train':len(pilot_fit),'unique_union':len(actual_fit),'new_Mini3_IDs_outside_primary_and_mixed':len(adapted_fit-(primary_fit|mixed_fit)),'Modern60k_is_subset_of_Mini2_full100k':True,'statistical_binary_family_model_and_vocabulary_fit_are_subset_of_primary_train_plus_official2159_already_in_Mini2':True},
      'graph':{'nodes':len(graph),'components':len({r['component_id'] for r in graph}),'protection_seed_ID_union':len(allseeds),'reserved_05_components':len(reserved_components),'prior_manual_review_observed_components':len(manual_components),'prior_manual_review_constraints_sha256':sha(manual_path),'protected_components_union':len(protected_components),'per_seed_role':{k:{'ids':len(v),'components':len(seed_components[k])} for k,v in seeds.items()}},
      'prior_admitted_English_length_at_least128_candidate_upperbound':{**count(candidates),'by_origin':breakdown(candidates,['origin']),'by_source_domain':breakdown(candidates,['source','domain']),'by_family':breakdown(candidates,['family']),'by_fine_label_mask':breakdown(candidates,['mask_model']),'mixed_H_AI_components':len(paired),'rows_in_mixed_H_AI_components':sum(map(len,paired)),'independent_human_components':len({r['component_id'] for r in candidates if r['y_ai']==0}),'independent_AI_components':len({r['component_id'] for r in candidates if r['y_ai']==1})},
      'unprotected_but_not_previously_admitted_length_at_least128_requires_new_review':{**count(conditional),'by_source_domain':breakdown(conditional,['source','domain']),'by_rejection_reason':breakdown(conditional,['rejection_reason'])},
      'actual_fit_provenance_proofs':{'neural_training_metadata':producer_proofs,'statistical_fit_metadata':statistical_proofs,'additional_final_coverage':additional_old_final_coverage,'statistical_fit_implementation_sha256':sha(ROOT/'experiments/english_v1/statistical.py'),'neural_read_records_implementation_sha256':sha(ROOT/'experiments/english_v1/neural.py')},
      'component_role_index':{'path':str(roles_path.relative_to(ROOT)),'rows':len(component_roles),'sha256':sha(roles_path),'counts':{key:sum(r[key] for r in component_roles) for key in component_roles[0] if key!='component_id'},'not_a_new_role_assignment':True},
      'candidate_index':{'path':str(path.relative_to(ROOT)),'rows':len(all_candidates),'sha256':sha(path),'no_text_columns':True},
      'limitations':['Counts are provisional upperbounds under the saved observable graph, not data-use authorization, semantic quality certification or a frozen assignment.','The corrected05 graph replays cross-protected lexical near edges for WP-related candidates only. Unused non-WP candidates need a new full cross-protected near-duplicate pass before fresh roles; inherited augmentation groups alone cannot recover all old-to-new near-only mappings.','128 threshold here uses previously saved words/whitespace_words, not a newly recomputed wordlike count. Final new protocol must use a consistent tokenizer and quality/language checks.','Old restrictive whole-domain flags are not silently relaxed: old not-admitted rows appear only as conditional review capacity. Existing Yelp final documents and all connected components remain protected.','Stored source/model names are publisher assertions, not execution/weight authentication. Modern nominal catalog scores are not independent model truth.','MAGE released CSV lacks full original prompt/parent mapping. Observable grouping cannot establish complete source-document/prompt independence. A parent-linked external source is preferred for new blind evaluation.', 'NC use is permitted for this noncommercial research. Unknown rights are not authorized; raw redistribution and training/output licensing remain separate.'],
      'input_metadata_projections':READS,
    }
    (OUT/'existing_capacity.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'actual_fit_union':len(actual_fit),'protected_components':len(protected_components),'candidate_upperbound':count(candidates),'conditional_review':count(conditional),'by_origin':payload['prior_admitted_English_length_at_least128_candidate_upperbound']['by_origin'],'by_source_domain':payload['prior_admitted_English_length_at_least128_candidate_upperbound']['by_source_domain']},indent=2))
if __name__=='__main__':main()
