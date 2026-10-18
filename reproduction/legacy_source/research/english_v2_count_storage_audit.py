"""One counts/fingerprint-only audit; never loads probability array values."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.english_v1 import runtime  # noqa: F401
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    output=ROOT/'research/english_v2_count_storage_audit_v1.json'
    assert not output.exists()
    suites={'fresh_main':3000,'fresh_source_domain':800,'fresh_modern_with_human_controls':2000,'fresh_paraphrase_with_human_controls':1800}
    components=['minilm4_epoch1','minilm_v3_epoch2','modern3_epoch2','statistical_v2']
    records=[]
    for suite,n in suites.items():
        for component in components:
            p=ROOT/'results/english_v2/release_v1/raw_final'/suite/(component+'.npz')
            side=p.with_suffix('.provenance.json')
            before,side_sha=sha(p),sha(side)
            loaded=['lengths','scope_word_like_tokens','component_weights_sha256','component_metadata_sha256']
            with np.load(p,allow_pickle=False) as archive:
                values={key:archive[key] for key in loaded}
            lengths,scope=values['lengths'],values['scope_word_like_tokens']
            assert lengths.shape==scope.shape==(n,) and lengths.dtype.kind in'iu'and scope.dtype.kind in'iu'
            assert (lengths>=0).all()and(scope>=0).all()and np.array_equal(lengths,scope)
            identity={key:str(values[key].item())for key in ['component_weights_sha256','component_metadata_sha256']}
            receipt=json.loads(side.read_text())
            assert receipt['output_sha256']==before and receipt['weights_sha256']==identity['component_weights_sha256'] and receipt['component_metadata_sha256']==identity['component_metadata_sha256']
            assert sha(p)==before and sha(side)==side_sha
            records.append({'suite':suite,'component':component,'path':str(p.relative_to(ROOT)),'npz_sha256':before,
                'sidecar_path':str(side.relative_to(ROOT)),'sidecar_sha256':side_sha,'rows':n,
                'lengths_dtype':str(lengths.dtype),'scope_dtype':str(scope.dtype),'shape':[n],
                'nonnegative_integer_counts':True,'exact_equal_values':True,'different_values':int(np.count_nonzero(lengths!=scope)),
                'same_C_bytes':lengths.tobytes()==scope.tobytes(),
                'lengths_C_bytes_sha256':hashlib.sha256(lengths.tobytes()).hexdigest(),
                'scope_C_bytes_sha256':hashlib.sha256(scope.tobytes()).hexdigest(),**identity,
                'only_loaded_array_keys':loaded,'NPZ_and_sidecar_unchanged_before_after':True})
    result={'schema_version':1,'scope':'Frozen final raw count storage and fingerprint audit only; no metrics',
        'counts_only_no_probability_values_loaded':True,'no_corpus_or_model_loaded':True,'no_GPU_or_metrics_used':True,
        'all16_original_NPZ_and_sidecars_unchanged':True,'neural_same_dtype_and_bytes_files':12,
        'stat_int32_lengths_int64_scope_exact_value_equal_files':4,'raw_files':records,'audit_source_sha256':sha(__file__),
        'source_contract':{'stat_official_refine.py':'export_raw_predictions lengths dtype=int32',
            'neural.py':'export_predictions lengths dtype=int64','language_guard.py':'scope_arrays scope_word_like_tokens dtype=int64',
            'export_component.py':'requires np.array_equal(lengths,scope counts); representation identity not required'}}
    with output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'audit':str(output.relative_to(ROOT)),'sha256':sha(output),'files':len(records),'probability_values_loaded':False}))


if __name__=='__main__':main()
