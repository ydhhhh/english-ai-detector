"""Summarize cached official metadata; no corpus payloads."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'research/english_v2_data/audit/public_metadata'
OUT=D.parent

def load(n): return json.loads((D/n).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
ref=load('m4_latest_ref.json')['object']['sha']
tree=load('m4_pinned_tree.json')['tree']
domains=('arxiv','peerread','reddit','wikihow','wikipedia')
files=[{'path':r['path'],'bytes':r['size'],'git_blob_sha1':r['sha'],'public_url':f'https://raw.githubusercontent.com/mbzuai-nlp/M4/{ref}/{r["path"]}'} for r in tree if any(r['path'].startswith('data/'+domain+'_') for domain in domains) and r['path'].endswith('.jsonl')]
rec=[]
for v in range(1,4):
    x=load(f'multitude_v{v}.json')
    rec.append({'version':f'v{v}','record_id':x['id'],'official_record_url':f'https://zenodo.org/records/{x["id"]}','access_right':x['metadata']['access_right'],'exposed_files':len(x.get('files',[])),'anonymous_download_available':False,'metadata_json_sha256':sha(D/f'multitude_v{v}.json'),'metadata_license_field':x['metadata'].get('license'),'access_conditions':x['metadata'].get('access_conditions')})
p={
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'kind':'official_metadata_audit_no_corpus_acquisition','read_text':False,'M4':{
 'repository':'https://github.com/mbzuai-nlp/M4','current_official_default_head_matches_existing_pin':ref=='628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d','revision':ref,'English_files':files,'English_all_file_bytes':sum(x['bytes'] for x in files),'English_excluding_wikihow_bytes':sum(x['bytes'] for x in files if not x['path'].startswith('data/wikihow_')),
 'schema':['prompt','human_text','machine_text','model','source','source_ID'],'paired_human_parent_paper_upperbound':14344,'including_nonparallel_peerread_human_paper_upperbound':17798,'actual_GPT4_named_download_file_found':False,'blanket_LICENSE_file_found':any(x['path'].rsplit('/',1)[-1].lower() in {'license','license.md','license.txt','copying'} for x in tree),
 'decision':'Backup only, no v2 payload downloaded. Old generators, unverified overall data licence and overlapping v1 M4 WikiHow make this inferior to modern parent-linked candidates. Upstream rights need source-level verification; do not infer arXiv abstract copyright from metadata CC0.',
 'wikiHow_current_policy':'Exclude WikiHow as an initial v2 candidate. New MAGA official human-source README discloses domain removal/DMCA; previous v1 M4 WikiHow use is immutable history, not fresh authorization for v2 replay.'},
 'MULTITuDE':{'records':rec,'decision':'Not an immediate anonymous free training source: official records expose zero files with restricted access; no institutional access request submitted, no account/terms changes.'},
 'priority_change':'Modern MAGA and parent-linked rewrite sources are primary audit candidates. Do not spend more acquisition effort on M4/MULTITuDE absent a new need.',
 'evidence_files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.glob('*.json')) if not p.name.endswith('.receipt.json')]
}
(OUT/'backup_source_findings.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
print(json.dumps({'M4_English_bytes':p['M4']['English_all_file_bytes'],'M4_without_wikihow_bytes':p['M4']['English_excluding_wikihow_bytes'],'MULTITuDE_restricted_versions':len(rec)}))
