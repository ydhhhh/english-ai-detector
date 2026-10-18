"""Article-title-only bridge metadata; does not inspect or print document bodies."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,re,unicodedata
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research/english_v2_data/audit'
DISAMBIG=re.compile(r'\s*\((?:(?:18|19|20)\d{2}\s+)?(?:novel|book|novella|short story|play|poem|poetry collection|short story collection)\)\s*$')
GENERIC={'introduction','abstract','untitled','summary','review','conclusion','home','index','unknown','none','nan','n/a','null','it','the','a','an'}
def norm(x):return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
def sh(x):return hashlib.sha256(x.encode()).hexdigest()
def fs(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    entries={};inputs=[]
    def add(parent,collector,title):
        if not isinstance(title,str) or not title.strip():return
        title=norm(title);base=DISAMBIG.sub('',title).strip();entries[(parent,title)]={'parent':parent,'collector':collector,'canonical_title_sha256':sh(title),'base_title_sha256':sh(base),'canonical_title_characters':len(title),'base_title_characters':len(base),'base_title_wordlike':len(re.findall(r'\b\w+\b',base)),'generic_or_short':base in GENERIC or len(base)<4,'used_trailing_disambiguator_alias':base!=title}
    planpath=OUT/'maga_acquisition_plan.json';plan=json.loads(planpath.read_text());inputs.append(planpath)
    for spec in plan['files']:
        path=ROOT/spec['local_path'];assert fs(path)==spec['sha256'];inputs.append(path)
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                if re.sub(r'[\s_-]+','',r['domain'].casefold())=='wikipedia':add('MAGA:'+r['human_source_id'],'MAGA_Wikipedia',r.get('title'))
    old=ROOT/'research/raid_external_v1/candidates.jsonl';inputs.append(old)
    with old.open() as f:
        for line in f:
            r=json.loads(line)
            if r.get('domain')=='books':add('RAID:'+r['source_id'],'RAID_books',r.get('title'))
    new=ROOT/'research/english_v2_data/raw/raid_paired_range_v1_titles/parent_titles.parquet';inputs.append(new)
    for r in pq.read_table(new).to_pylist():
        if r['domain']=='books':add('RAID:'+r['source_id'],'RAID_books',r['native_title'])
    full=defaultdict(set);base=defaultdict(set)
    for r in entries.values():full[r['canonical_title_sha256']].add(r['parent']);base[r['base_title_sha256']].add(r['parent'])
    rows=list(entries.values())
    for r in rows:
        r['full_title_parent_bucket_size']=len(full[r['canonical_title_sha256']]);r['base_title_parent_bucket_size']=len(base[r['base_title_sha256']]);r['ambiguous_title_quarantine']=r['generic_or_short'] or r['full_title_parent_bucket_size']>20 or r['base_title_parent_bucket_size']>20
        r['observable_key_hashes']=[] if r['ambiguous_title_quarantine'] else [sh('Wikipedia_article_title:'+r['canonical_title_sha256']),sh('Wikipedia_article_title:'+r['base_title_sha256'])]
    output=OUT/'article_title_bridge_metadata.parquet';pq.write_table(pa.Table.from_pylist(rows),output,compression='zstd')
    crossfull=[h for h,ps in full.items() if any(p.startswith('MAGA:') for p in ps) and any(p.startswith('RAID:') for p in ps)]
    crossbase=[h for h,ps in base.items() if any(p.startswith('MAGA:') for p in ps) and any(p.startswith('RAID:') for p in ps)]
    summary={'kind':'prospective_specific_Wikipedia_article_title_bridge_rules_no_roles','created_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'native_parents':len({r['parent'] for r in rows}),'by_collector':dict(Counter(r['collector'] for r in rows)),'cross_MAGA_RAID_full_title_buckets':len(crossfull),'cross_MAGA_RAID_base_title_buckets':len(crossbase),'ambiguous_quarantine_parents':len({r['parent'] for r in rows if r['ambiguous_title_quarantine']}),'rules':{'normalization':'NFKC casefold whitespace collapse','shared_namespace':'Wikipedia_article_title:sha256(normalized title), only MAGA Wikipedia and RAID CMU/Wikipedia books','trailing_alias_regex':DISAMBIG.pattern,'ambiguity':'generic set or normalized base length<4characters or full/base title>20distinct native parent bucket: do not make a shared generic title edge; retain exact/native graph and quarantine that parent from new train/fresh capacity until resolved','generic_title_set':sorted(GENERIC),'NQ_questions':'never treated as Wikipedia article titles','news_titles':'never treated as Wikipedia article titles'},'body_previewed':False,'roles_allocated':False,'model_or_scores':False,'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':fs(p)} for p in inputs],'output':{'path':str(output.relative_to(ROOT)),'sha256':fs(output),'bytes':output.stat().st_size},'limits':['Book/article title alias is conservative observable lineage, not authentication of original Wikipedia page ID. Unavailable original IDs and other semantic overlaps remain unknown.','Ambiguity quarantine is prospective and score-independent; high-frequency/generic text is never used as an indiscriminate global merge.']}
    (OUT/'article_title_bridge_rules.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['rows','native_parents','cross_MAGA_RAID_full_title_buckets','cross_MAGA_RAID_base_title_buckets','ambiguous_quarantine_parents']}))
if __name__=='__main__':main()
