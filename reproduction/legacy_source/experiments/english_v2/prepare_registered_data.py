"""Single frozen-protocol whole-component data allocation; never load a detector.

--describe-contract writes only the prospective algorithm contract. main() may
allocate only after root freezes this exact code/contract and metadata graph QA.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,math,re,sqlite3
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];AUDIT=ROOT/'research/english_v2_data/audit';SEED=20261018
CONTRACT={
 'algorithm':'english_v2_whole_component_fixed_hash_v1','seed':SEED,
 'salt_template':'english-v2-allocation-20261018:{role}:{stratum}:{identity}',
 'priority':['final_source_domain','final_modern_Gemini_minimum','final_modern_remaining','final_paraphrase','final_main','cal_source_domain','cal_modern','cal_paraphrase','cal_general','cal_historical_WP','dev_source_domain','dev_modern','dev_paraphrase','dev_general','dev_legacy_main','dev_legacy_creative','new_train_pairs','new_train_unused_MAGE','old_train_WP_replay','old_train_other_replay'],
 'fresh_core':'publisher-English supported source rights + >=128wordlike/30..3000whitespace + fixed visible-think/truncation predicate; runtime language gate DIAGNOSTIC only',
 'whole_component':'All native/config/transform/excluded siblings keep the selected component reservation; no role overlap. One human and chosen positive per component for pairs; rewrite views share same human and select one positive per origin.',
 'holdouts':'All NQ or Gemini-nominal MAGA components final_source_domain/final_modern only or unselected protected; never train/dev/cal/final_main/final_paraphrase. WikiHow and ambiguous title parents excluded from new raw/replay. Actual oldfit/nontrain/observed/reserved graph flags retained.',
 'pair_strata':'Within requested domain, round-robin available nominal original-generator strata; fixed-hash component/native-parent order; balance MGB/MAGA configuration by lower selected count with fixed-hash ties; no score-based choice.',
 'unpaired_strata':'Per-class source/domain round-robin fixed hash; unused MAGE new-training uses matched per-domain class quotas from largest-remainder min(Hcapacity,AIcapacity), then same class/source/domain round-robin. Components at most one row per class in a role.',
 'Gemini_minimum':'Exactly300 first, Wikipedia/S2ORC quotas largest-remainder proportional to eligible Gemini component capacities and capped by final domain quotas700/300; never fallback if unavailable.',
 'RAID_domain_quotas':'Each paraphrase role equal books/news parents: final300+300,cal200+200,dev150+150; each parent same cleanH+one human-originDIPPER+one AI-originDIPPER.',
 'new_training':'Up to10000pairs perclass: remaining usable matched parent pairs in fixed cell order MAGA/Wikipedia,MAGA/S2ORC,RAID/books,RAID/news, recomputing capacity after each preceding whole-component reservation and taking min(remaining10000,cellcapacity), first (oneAI origin half by parent hash, deterministic fallback if unavailable); then unused MAGE matched-domain unpaired rows. Require>=6000new/class. Fill eachclass30000 using actual oldfit IDs only. No replacement.',
 'WP_replay':'All eligible actual-oldfit WP human up to1061 and sameN eligible WP AI; one row/class/component, balanced. Remaining replay source/domain round-robin. No invented WP humans.',
 'legacy_reuse':'Legacy main/WP dev and historicalWP CAL read only registered original input IDs; global old roles must be exactly development or calibration respectively, excluding actualfit, any oldfinal and multi-role ancestor components.',
 'files':{'train_registered':60000,'validation_general':2000,'dev_source_domain':800,'dev_modern':1600,'dev_paraphrase':900,'dev_legacy_main':2000,'dev_legacy_creative':2000,'calibration':8000,'final_main':3000,'final_source_domain':800,'final_modern':2000,'final_paraphrase':1800},
 'calibration_history':'8000rows=3800H/4200AI: fresh general2000+S2ORC800+Wikipedia2000+RAID1200+oldWP CAL2000. Original WP4000 unchanged and allunselected CAL siblings protected. Cal fitting later supported-core only, coverage exclusions disclosed.',
 'fresh_final':'7600rows=3500H/4100AI; modern700Wiki+300S2ORC paired controls,>=300nominalGemini. Finalparaphrase600H+600human-origin+600AI-origin, sharedH views not independent.',
 'no_scores_or_models':'Allocation and capacity cannot read model outputs or change after scores. Inability to meet exact quotas fails before publishing; no automatic relaxation.'}

def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def fs(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def key(role,stratum,identity):return sha(f'english-v2-allocation-{SEED}:{role}:{stratum}:{identity}')
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def apportion(target,counts,role):
 if target<0 or target>sum(counts.values()):raise ValueError(f'Insufficient capacity for {role}: {counts} target={target}')
 if not target:return {k:0 for k in counts}
 total=sum(counts.values());q={k:target*v//total for k,v in counts.items()}
 order=sorted(counts,key=lambda k:(-(target*counts[k]%total),key(role,'apportion',str(k))))
 for k in order[:target-sum(q.values())]:q[k]+=1
 assert sum(q.values())==target and all(q[k]<=counts[k] for k in q)
 return q

def selfcheck():
 assert apportion(7,{'a':3,'b':6,'c':1},'fixture')=={'a':2,'b':4,'c':1}
 assert sum(apportion(0,{'a':0},'fixture').values())==0
 try:apportion(2,{'a':1},'fixture')
 except ValueError:pass
 else:raise AssertionError('must fail capacity')
 assert key('r','s','a')==key('r','s','a') and key('r','s','a')!=key('r2','s','a')
 def row(parent,y,variant='direct',config='MGB',flags=()):
  return {'collector':'MAGA','domain':'Wikipedia','candidate_fresh':True,'candidate_train_under_draft_holdouts':not bool(flags),'component_id':parent,'parent':parent,'nominal_generator':'g','y_ai':y,'row_record_id':parent+':'+str(y)+':'+config+':'+variant,'variant':variant,'config':config,'barrier_flags':list(flags)}
 fixture=[]
 for parent in ['p1','p2']:
  fixture.extend([row(parent,0,'human'),row(parent,1,config='MGB'),row(parent,1,config='MAGA')])
 alloc=Allocator(fixture);assert alloc.pairs('dev_modern',2,'MAGA','Wikipedia',gemini=False)==2
 assert len(alloc.selected['dev_modern'])==4 and Counter(r['config'] for r in alloc.selected['dev_modern'] if r['y_ai']==1)=={'MGB':1,'MAGA':1}
 assert len({r['component_id'] for r in alloc.selected['dev_modern']})==2
 try:alloc.pairs('cal_modern',1,'MAGA','Wikipedia',gemini=False)
 except ValueError:pass
 else:raise AssertionError('whole component reservation must block another role')
 held=[row('held',0,'human',flags=['draft_Gemini_nominal_holdout']),row('held',1,flags=['draft_Gemini_nominal_holdout'])]
 try:Allocator(held).pairs('dev_modern',1,'MAGA','Wikipedia',gemini=False)
 except ValueError:pass
 else:raise AssertionError('nonGemini variant of held component must not enter dev')
 tri=[row('tri',0,'human'),row('tri',1,'human_origin_DIPPER'),row('tri',1,'AI_origin_DIPPER')]
 for r in tri:r['collector']='RAID';r['domain']='books'
 tri_alloc=Allocator(tri);tri_alloc.pairs('final_paraphrase',1,'RAID','books',rewrite=True)
 assert len(tri_alloc.selected['final_paraphrase'])==3 and Counter(r['y_ai'] for r in tri_alloc.selected['final_paraphrase'])=={0:1,1:2}
 assert Allocator(fixture).pair_capacity('MAGA','Wikipedia',training=True)==2
 return {'passed':True,'checks':11,'real_data_read':False,'whole_component_and_holdout_fixtures':True}

class Allocator:
 def __init__(self,rows):self.rows=rows;self.reserved={};self.selected=defaultdict(list);self.selected_ids=set();self.used_class_components=set();self.configuration_counts=defaultdict(Counter);self.positive_generator_counts=defaultdict(Counter)
 def allowed(self,comp,role):return comp not in self.reserved or self.reserved[comp]==role
 def add(self,r,role):
  comp=r['component_id'];assert self.allowed(comp,role);assert r['row_record_id'] not in self.selected_ids
  assert (role,comp,r['y_ai']) not in self.used_class_components or role.endswith('paraphrase') and r['y_ai']==1
  self.reserved[comp]=role;self.selected_ids.add(r['row_record_id']);self.used_class_components.add((role,comp,r['y_ai']));self.selected[role].append(r)
 def pairs(self,role,n,collector,domain,*,gemini=None,rewrite=False,training=False):
  rr=[r for r in self.rows if r['collector']==collector and r['domain']==domain and r['candidate_fresh'] and self.allowed(r['component_id'],role) and (not training or r['candidate_train_under_draft_holdouts']) and (role in {'final_source_domain','final_modern'} or not {'draft_NQ_source_holdout','draft_Gemini_nominal_holdout'}&set(r['barrier_flags']))]
  byparent=defaultdict(list)
  for r in rr:byparent[r['parent']].append(r)
  options=[]
  for parent,values in byparent.items():
   hh=[r for r in values if r['y_ai']==0];aa=[r for r in values if r['y_ai']==1]
   if gemini is not None:aa=[r for r in aa if (r['nominal_generator']=='Gemini-2.0-flash')==gemini]
   if not hh or not aa:continue
   human=min(hh,key=lambda r:key(role,'human',r['row_record_id']))
   if (role,human['component_id'],0) in self.used_class_components:continue
   if rewrite:
    originH=[r for r in aa if r['variant']=='human_origin_DIPPER'];originA=[r for r in aa if r['variant']=='AI_origin_DIPPER']
    if not originH or not originA:continue
   model=min(aa,key=lambda r:key(role,'parent-generator',r['nominal_generator']))['nominal_generator']
   options.append((human['component_id'],parent,model,human,aa))
  strata=defaultdict(list)
  for option in options:strata[option[2]].append(option)
  for label in strata:strata[label].sort(key=lambda x:key(role,'component-parent',x[0]+':'+x[1]))
  order=sorted(strata,key=lambda label:key(role,'generator-stratum',label));positions=Counter();chosen=[];seen=set()
  while len(chosen)<n:
   changed=False
   for label in order:
    values=strata[label]
    while positions[label]<len(values):
     option=values[positions[label]];positions[label]+=1;comp=option[0]
     if comp in seen or not self.allowed(comp,role) or (role,comp,0) in self.used_class_components:continue
     chosen.append(option);seen.add(comp);changed=True;break
    if len(chosen)==n:break
   if not changed:break
  if len(chosen)!=n:raise ValueError(f'Insufficient paired capacity {role}/{collector}/{domain}/{gemini}: {len(chosen)}<{n}')
  for comp,parent,label,human,aa in chosen:
   self.add(human,role)
   if rewrite:
    for variant in ['human_origin_DIPPER','AI_origin_DIPPER']:
     options=[r for r in aa if r['variant']==variant];gc=self.positive_generator_counts[role];positive=min(options,key=lambda r:(gc[r['nominal_generator']],key(role,variant,r['row_record_id'])));gc[positive['nominal_generator']]+=1;self.add(positive,role)
   else:
    if training and collector=='RAID':
     variant='human_origin_DIPPER' if int(key(role,'rewrite-origin',parent),16)%2==0 else 'AI_origin_DIPPER';options=[r for r in aa if r['variant']==variant] or aa
    else:options=aa
    cfg=self.configuration_counts[role];gc=self.positive_generator_counts[role];positive=min(options,key=lambda r:(cfg[r['config']],gc[r['nominal_generator']],key(role,'AI-variant',r['row_record_id'])));cfg[positive['config']]+=1;gc[positive['nominal_generator']]+=1;self.add(positive,role)
  return len(chosen)
 def pair_capacity(self,collector,domain,*,gemini=None,training=False):
  byparent=defaultdict(set);component={}
  for r in self.rows:
   if r['collector']!=collector or r['domain']!=domain or not r['candidate_fresh'] or r['component_id'] in self.reserved or training and not r['candidate_train_under_draft_holdouts']:continue
   if r['y_ai']==1 and gemini is not None and (r['nominal_generator']=='Gemini-2.0-flash')!=gemini:continue
   byparent[r['parent']].add(r['y_ai']);component[r['parent']]=r['component_id']
  return len({component[p] for p,v in byparent.items() if v=={0,1}})
 def unpaired(self,role,pool,targets,domain=None):
  for y,target in targets.items():
   rr=[r for r in pool if r['y_ai']==y and (domain is None or r['domain']==domain) and self.allowed(r['component_id'],role) and (role,r['component_id'],y) not in self.used_class_components]
   strata=defaultdict(list)
   for r in rr:strata[(r['source'],r['domain'])].append(r)
   for k in strata:strata[k].sort(key=lambda r:key(role,f'class{y}',r['row_record_id']))
   labels=sorted(strata,key=lambda k:key(role,'source-domain',str(k)));positions=Counter();n=0
   while n<target:
    changed=False
    for label in labels:
     values=strata[label]
     while positions[label]<len(values):
      r=values[positions[label]];positions[label]+=1
      if not self.allowed(r['component_id'],role) or (role,r['component_id'],y) in self.used_class_components:continue
      self.add(r,role);n+=1;changed=True;break
     if n==target:break
    if not changed:break
   if n!=target:raise ValueError(f'Insufficient unpaired capacity {role}/class{y}: {n}<{target}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--describe-contract');p.add_argument('--protocol');p.add_argument('--expected-protocol-sha');p.add_argument('--graph-qa');p.add_argument('--output-root',default='data/english_v2');args=p.parse_args()
 if args.describe_contract:dump(Path(args.describe_contract),CONTRACT);print(json.dumps(selfcheck()));return
 if not args.protocol or not args.expected_protocol_sha or not args.graph_qa:p.error('Frozen protocol exact SHA and graphQA required')
 protocol_path=(ROOT/args.protocol).resolve();protocol_bytes=protocol_path.read_bytes();assert hashlib.sha256(protocol_bytes).hexdigest()==args.expected_protocol_sha;protocol=json.loads(protocol_bytes)
 assert protocol['status']=='FROZEN_BEFORE_V2_TRAINING_AND_SCORES' and protocol['data_allocation_contract']==CONTRACT
 assert protocol['role_builder_sha256']==fs(Path(__file__))
 graphpath=AUDIT/'joint_full_near_capacity.json';graph=json.loads(graphpath.read_text());graphsha=fs(graphpath);assert protocol['joint_graph_manifest_sha256']==graphsha
 qa_path=(ROOT/args.graph_qa).resolve();qa=json.loads(qa_path.read_text());assert qa['passed'] is True and qa['graph_manifest_sha256']==graphsha
 out=ROOT/args.output_root;assert not out.exists(),'Never overwrite data release or a partial role allocation';out.mkdir(parents=True)
 inputs={str(p.relative_to(ROOT)):{'sha256':fs(p),'bytes':p.stat().st_size} for p in [protocol_path,graphpath,qa_path,Path(__file__)]}
 def read(name,columns=None):
  path=ROOT/name;cols=columns or pq.read_schema(path).names;assert not {'text','generation','prompt'}&set(cols);inputs[name]={'sha256':fs(path),'bytes':path.stat().st_size,'columns':cols};return pq.read_table(path,columns=cols).to_pylist()
 conflicts={r['normalized_sha256'] for r in read('research/english_v2_data/audit/joint_binary_conflict_hashes.parquet')};rows=read('research/english_v2_data/audit/joint_new_candidate_index.parquet');oldunused=read('research/english_v2_data/audit/joint_existing_candidate_index.parquet');nodes=read('research/english_v2_data/audit/joint_component_node_index.parquet',['kind','identity','component_id','barrier_flags']);oldmap={r['identity']:r for r in nodes if r['kind']=='old_component'}
 for item in graph['outputs']:assert inputs.get(item['path'],{}).get('sha256',fs(ROOT/item['path']))==item['sha256']
 assert fs(AUDIT/'old_component_role_memberships.parquet')=='4b29c09062fed8d7efb243f9ac83ad16b6b237c9e6e37eed02d6b77a906fa65d';ledger=read('research/english_v2_data/audit/old_component_role_memberships.parquet');global_roles=defaultdict(set)
 for r in ledger:global_roles[oldmap[r['component_id']]['component_id']].update(r['roles'])
 graphold=read('data/english_calibration_v2/graph_index.parquet',['id','component_id','raw_sha256','normalized_sha256']);idgraph={r['id']:r for r in graphold}
 cache=ROOT/'.cache/english_v2_lexical_v1';db=sqlite3.connect('file:'+str(cache/'signatures.sqlite3')+'?mode=ro',uri=True);newdb=sqlite3.connect('file:'+str(cache/'new_texts.sqlite3')+'?mode=ro',uri=True);olddb=sqlite3.connect('file:'+str(ROOT/'research/english_v1_data/augmentation_v2/texts.sqlite3')+'?mode=ro',uri=True)
 def body(raw):
  kind,k=db.execute('SELECT ref_kind,ref_key FROM raw_refs WHERE raw_sha=?',(raw,)).fetchone();text=olddb.execute('SELECT text FROM texts WHERE i=?',(int(k),)).fetchone()[0] if kind=='old_sql' else newdb.execute('SELECT text FROM texts WHERE raw_sha=?',(k,)).fetchone()[0];assert sha(text)==raw;return text
 spec=importlib.util.spec_from_file_location('structural',AUDIT/'index_maga_metadata.py');quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
 oldmeta={}
 for name in ['data/english_v1/full_index.parquet','data/english_augmentation_v2/full_index.parquet','data/english_v1/pilot/train.parquet','data/english_v1/pilot/validation.parquet','data/english_v1/pilot/calibration.parquet']:
  cols=[c for c in ['id','y_ai','source','domain','family','model','canonical_model_id','mask_family','mask_model','out_of_model_catalog','source_revision','language'] if c in pq.read_schema(ROOT/name).names]
  for r in read(name,cols):oldmeta.setdefault(r['id'],r)
 def oldrecord(identifier,variant):
  m=oldmeta[identifier];g=idgraph[identifier];node=oldmap[g['component_id']];text=body(g['raw_sha256']);words=len(quality.WORD.findall(text));reason=quality.simple_quality(text,words);f=set(node['barrier_flags'])
  return {**m,'row_record_id':identifier,'parent':'old:'+g['component_id'],'collector':'MAGE' if str(m.get('source','')).casefold().startswith('mage') else 'NTL_or_community','variant':variant,'config':'legacy','raw_sha256':g['raw_sha256'],'normalized_sha256':g['normalized_sha256'],'component_id':node['component_id'],'barrier_flags':sorted(f),'wordlike_tokens':words,'structural_quality_reason':reason,'structural_core':reason is None and g['normalized_sha256'] not in conflicts,'candidate_fresh':False,'runtime_language_supported_diagnostic':None,'source_rights_allowed':True,'publisher_English':True,'nominal_generator':m.get('model'),'last_editor':None,'canonical_domain':str(m.get('domain','')).casefold(),'view_human_origin_rewrite':False,'view_AI_origin_rewrite':False}
 A=Allocator(rows)
 # All selected siblings are reserved in this shared global map; the separate
 # NQ/Gemini holdout flags bar nonfinal use even when the component is unselected.
 A.pairs('final_source_domain',400,'MAGA','Natural Questions')
 gemini={d:A.pair_capacity('MAGA',d,gemini=True) for d in ['Wikipedia','S2ORC']};gq=apportion(300,gemini,'final_modern_Gemini_minimum');assert gq['Wikipedia']<=700 and gq['S2ORC']<=300
 for d,target in [('Wikipedia',700),('S2ORC',300)]:
  A.pairs('final_modern',gq[d],'MAGA',d,gemini=True);A.pairs('final_modern',target-gq[d],'MAGA',d,gemini=False)
 for d in ['books','news']:A.pairs('final_paraphrase',300,'RAID',d,rewrite=True)
 oldpool=[]
 for r in oldunused:
  if r['candidate_fresh']:
   oldpool.append({**r,'row_record_id':r['id'],'parent':'old:'+r['old_component_id'],'collector':'MAGE' if str(r.get('source','')).casefold().startswith('mage') else 'NTL_or_community','variant':'unused_legacy','config':'legacy','runtime_language_supported_diagnostic':None,'publisher_English':True,'source_rights_allowed':True,'nominal_generator':r.get('model'),'last_editor':None,'canonical_domain':str(r.get('domain','')).casefold(),'view_human_origin_rewrite':False,'view_AI_origin_rewrite':False})
 # General fresh cohorts are old-but-unfitted MAGE only, not AI-only community.
 magepool=[r for r in oldpool if r['collector']=='MAGE' and not {'draft_NQ_source_holdout','draft_Gemini_nominal_holdout'}&set(r['barrier_flags'])]
 A.unpaired('final_main',magepool,{0:1500,1:1500})
 for role,pairs in [('cal_source_domain',400),('dev_source_domain',400)]:
  if role.startswith('dev'):continue
  A.pairs(role,pairs,'MAGA','S2ORC',gemini=False)
 A.pairs('cal_modern',1000,'MAGA','Wikipedia',gemini=False)
 for d in ['books','news']:A.pairs('cal_paraphrase',200,'RAID',d,rewrite=True)
 A.unpaired('cal_general',magepool,{0:1000,1:1000})
 def inherited(name,desired_role,variant):
  ids=[r['id'] for r in read(name,['id'])];pool=[]
  for identifier in ids:
   r=oldrecord(identifier,variant)
   if r['structural_core'] and global_roles[r['component_id']]=={desired_role} and not {'wikiHow_forbidden','draft_NQ_source_holdout','draft_Gemini_nominal_holdout'}&set(r['barrier_flags']):pool.append(r)
  return pool
 wpcal=inherited('data/english_calibration_v2/wp_calibration.parquet','calibration','historical_WP_CAL_replay');A.unpaired('cal_historical_WP',wpcal,{0:1000,1:1000})
 A.pairs('dev_source_domain',400,'MAGA','S2ORC',gemini=False);A.pairs('dev_modern',800,'MAGA','Wikipedia',gemini=False)
 for d in ['books','news']:A.pairs('dev_paraphrase',150,'RAID',d,rewrite=True)
 A.unpaired('dev_general',magepool,{0:1000,1:1000})
 mainlegacy=inherited('data/english_v1/validation.parquet','development','observed_legacy_main_dev');wplegacy=inherited('data/english_v1/dev_challenge_source_domain.parquet','development','observed_legacy_creative_dev')
 A.unpaired('dev_legacy_main',mainlegacy,{0:1000,1:1000});A.unpaired('dev_legacy_creative',wplegacy,{0:1000,1:1000})
 # New source paired training precedes old-unused rows. Native source caps are
 # enforced by one row/class/component; no 35variant RAID count inflation.
 for collector,d in [('MAGA','Wikipedia'),('MAGA','S2ORC'),('RAID','books'),('RAID','news')]:
  target=min(10000-len(A.selected['train_registered'])//2,A.pair_capacity(collector,d,training=True))
  A.pairs('train_registered',target,collector,d,gemini=False if collector=='MAGA' else None,training=True)
 new_pair_n=len(A.selected['train_registered'])//2
 remaining=10000-new_pair_n
 counts={}
 for domain in {r['domain'] for r in magepool}:
  counts[domain]=min(len({r['component_id'] for r in magepool if r['domain']==domain and r['y_ai']==y and A.allowed(r['component_id'],'train_registered') and ('train_registered',r['component_id'],y) not in A.used_class_components}) for y in [0,1])
 take=min(remaining,sum(counts.values()));quotas=apportion(take,counts,'new_train_unused_MAGE')
 for domain,target in sorted(quotas.items()):A.unpaired('train_registered',magepool,{0:target,1:target},domain=domain)
 new_counts=Counter(r['y_ai'] for r in A.selected['train_registered']);assert new_counts[0]==new_counts[1] and 6000<=new_counts[0]<=10000, f'new capacity feasibility failed {new_counts}'
 for r in A.selected['train_registered']:r['new_vs_replay']='new'
 actualfit=set()
 for path in ['data/english_v1/train.parquet','data/english_augmentation_v2/train_mixed100k_with_official.parquet','data/english_adaptation_v3/train_replay50k_wp.parquet','data/english_v1/pilot/train.parquet']:actualfit.update(r['id'] for r in read(path,['id']))
 assert len(actualfit)==174019
 replay=[]
 for identifier in sorted(actualfit):
  r=oldrecord(identifier,'legacy_replay');f=set(r['barrier_flags'])
  if r['structural_core'] and global_roles[r['component_id']]=={'train_fit'} and not {'wikiHow_forbidden','ambiguous_Wikipedia_article_title_quarantine','old_nontrain_or_reserved_forbid_new_training','draft_NQ_source_holdout','draft_Gemini_nominal_holdout'}&f and not r['component_id'] in A.reserved:replay.append(r)
 wppool=[r for r in replay if str(r['domain']).casefold()=='wp'];wpn=min(1061,*[len({r['component_id'] for r in wppool if r['y_ai']==y}) for y in [0,1]])
 A.unpaired('train_registered',wppool,{0:wpn,1:wpn})
 for y in [0,1]:A.unpaired('train_registered',[r for r in replay if str(r['domain']).casefold()!='wp'],{y:30000-sum(r['y_ai']==y for r in A.selected['train_registered'])})
 for r in A.selected['train_registered']:r.setdefault('new_vs_replay','replay')
 assert Counter(r['y_ai'] for r in A.selected['train_registered'])=={0:30000,1:30000}
 outputs={};records_by_role={}
 def materialize(role,selected,split):
  result=[]
  for r in selected:
   text=body(r['raw_sha256']);assert len(quality.WORD.findall(text))==r['wordlike_tokens']
   result.append({'id':r['id'],'text':text,'eligible':True,'y_ai':r['y_ai'],'family':r.get('family'),'model':r.get('model') if r['collector'] not in ['MAGA','RAID','Beemo'] else None,'mask_family':False if r['collector'] in ['MAGA','RAID','Beemo'] else bool(r.get('mask_family',False)),'mask_model':False if r['collector'] in ['MAGA','RAID','Beemo'] else bool(r.get('mask_model',False)),'out_of_model_catalog':r.get('out_of_model_catalog'),'canonical_model_id':r.get('canonical_model_id'),'source':r['source'],'domain':r['domain'],'collector':r['collector'],'group_id':r['component_id'],'native_parent_id':r['parent'],'split':split,'dataset_role':role,'language':'en','language_evidence':'publisher-English provenance plus frozen structural eligibility; runtime support recorded separately' if r['collector'] in ['MAGA','RAID'] else 'previous frozen English-admission provenance','runtime_language_supported_diagnostic':r['runtime_language_supported_diagnostic'],'wordlike_tokens':r['wordlike_tokens'],'raw_sha256':r['raw_sha256'],'normalized_sha256':r['normalized_sha256'],'variant':r['variant'],'config':r['config'],'new_vs_replay':r.get('new_vs_replay','historical_calibration_replay' if role=='cal_historical_WP' or r['variant']=='historical_WP_CAL_replay' else 'new' if role not in ['dev_legacy_main','dev_legacy_creative'] else 'observed_development_replay'),'rewrite_kind':'DIPPER_paraphrase' if r['collector']=='RAID' and r['y_ai']==1 else r['variant'],'original_generator':r['nominal_generator'],'last_editor':r['last_editor'],'view_human_origin_rewrite':r['view_human_origin_rewrite'],'view_AI_origin_rewrite':r['view_AI_origin_rewrite'],'source_revision':r.get('source_revision') or ('34c51fc6e71561bf1203945d90d253269f9f2ab8' if r['collector']=='MAGA' else '865cac74188466cb0c3b7574a10204007b57a459' if r['collector']=='RAID' else None)})
  # Stable final row order, independent of dictionary insertion and batch order.
  result.sort(key=lambda r:key(role,'output-order',r['id']));assert len({r['id'] for r in result})==len(result)
  path=out/(role+'.parquet');pq.write_table(pa.Table.from_pylist(result),path,compression='zstd');outputs[role]={'path':str(path.relative_to(ROOT)),'role':split,'rows':len(result),'groups':len({r['group_id'] for r in result}),'class_counts':dict(Counter(str(r['y_ai']) for r in result)),'core_counts':{'n':len(result),'human_n':sum(r['y_ai']==0 for r in result),'AI_n':sum(r['y_ai']==1 for r in result),'groups':len({r['group_id'] for r in result})},'runtime_unsupported_diagnostic_rows':sum(r['runtime_language_supported_diagnostic'] is False for r in result),'runtime_support_unknown_diagnostic_rows':sum(r['runtime_language_supported_diagnostic'] is None for r in result),'sha256':fs(path),'bytes':path.stat().st_size};records_by_role[role]=result
 for role,selected in A.selected.items():materialize('validation_general' if role=='dev_general' else role,selected,'train' if role=='train_registered' else 'validation' if role=='dev_general' else 'dev_challenge' if role.startswith('dev_') else 'calibration' if role.startswith('cal_') else 'final_sealed')
 calroles=['cal_general','cal_source_domain','cal_modern','cal_paraphrase','cal_historical_WP'];materialize('calibration',[r for role in calroles for r in A.selected[role]],'calibration')
 # Reuse materialized records to form trainer stress view, no new observations.
 stress=[r for role in ['dev_source_domain','dev_modern','dev_paraphrase'] for r in records_by_role[role]];path=out/'dev_challenge_stress.parquet';pq.write_table(pa.Table.from_pylist(stress),path,compression='zstd');outputs['dev_challenge_stress']={'path':str(path.relative_to(ROOT)),'role':'dev_challenge','rows':len(stress),'sha256':fs(path),'bytes':path.stat().st_size,'alias_of_roles':['dev_source_domain','dev_modern','dev_paraphrase']}
 for role,target in CONTRACT['files'].items():assert outputs[role]['rows']==target,(role,outputs[role]['rows'],target)
 # Metadata-only selection and all-component reservations for independent QA.
 selection=[]
 for role,selected in A.selected.items():
  for r in selected:selection.append({k:r.get(k) for k in ['row_record_id','id','parent','component_id','raw_sha256','normalized_sha256','y_ai','source','domain','collector','variant','config','wordlike_tokens','runtime_language_supported_diagnostic','new_vs_replay','nominal_generator']}|{'reserved_role':role})
 indexpath=out/'selection_index.parquet';pq.write_table(pa.Table.from_pylist(selection),indexpath,compression='zstd')
 reservepath=out/'component_reservations.parquet';pq.write_table(pa.Table.from_pylist([{'component_id':c,'reserved_role':role} for c,role in sorted(A.reserved.items())]),reservepath,compression='zstd')
 manifest={'status':'FROZEN_DATA_BEFORE_V2_MODEL_PREDICTIONS','frozen':True,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'path':str(protocol_path.relative_to(ROOT)),'sha256':args.expected_protocol_sha},'role_builder_sha256':fs(Path(__file__)),'graph_manifest_sha256':graphsha,'graph_metadata_QA':{'path':str(qa_path.relative_to(ROOT)),'sha256':fs(qa_path)},'allocation_contract':CONTRACT,'new_training_each_class':new_counts[0],'old_replay_each_class':30000-new_counts[0],'balanced_WP_replay_each_class':wpn,'final_modern_Gemini_domain_quotas':gq,'actual_fit_ID_superset':174019,'output_files':outputs,'training_inputs_sha256':{'train_registered':outputs['train_registered']['sha256']},'selection_index':{'path':str(indexpath.relative_to(ROOT)),'sha256':fs(indexpath)},'component_reservations':{'path':str(reservepath.relative_to(ROOT)),'sha256':fs(reservepath)},'input_files':inputs,'no_detector_or_model_scores_read':True,'no_body_previewed':True,'final_or_calibration_predictions_opened':False,'limits':['Whole observed lineage and approximate lexical grouping, not complete hidden/semantic/pretraining independence.','Allv1tenfinal remain observed diagnostics; nooldfinal kin allowed new training/dev/cal/freshfinal. Legacy dev/CAL exact-role ancestry excludes actualfit and multi-role.','Human controls in the two rewrite views are shared, not independent. Each required view retains full denominator including runtime abstentions.','Freshpublisher-English rows are not screened by the deployment language gate; calibration fitting later uses explicitly supported core.','Legacy runtime-language support diagnostics are unknown(null), not assumedtrue. Later export recomputes the frozen guard on actualtext; onlyfalse is countedunsupported, unknown separately.','OldWPcal replay is calibration history, not wholly fresh calibration; inheritedwarmstart traininghistory includes previouslytrainedWikiHow although this roundraw/replayexcludesit.','MAGA identities and alloriginal/editor model tags are publisher nominal claims; v2 binary supervision doesnotauthenticate versions.']}
 dump(out/'manifest.json',manifest);print(json.dumps({'manifest':str((out/'manifest.json').relative_to(ROOT)),'sha256':fs(out/'manifest.json'),'new_each_class':new_counts[0],'WP_replay_each_class':wpn,'rows':{k:v['rows'] for k,v in outputs.items()}},indent=2))
if __name__=='__main__':main()
