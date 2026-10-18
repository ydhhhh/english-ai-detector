"""Synthetic validation-only compatibility checks; never opens actual data."""
from __future__ import annotations
import contextlib,copy,hashlib,io,json
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.english_v2 import count_storage_compat as compat
import numpy as np


def main():
    checks=[]
    def record(name,fn):fn();checks.append(name)
    def denied(name,fn):
        try:fn()
        except (ValueError,SystemExit):checks.append(name)
        else:raise AssertionError(name+' accepted')
    def fixture():
        return {'ids':np.asarray(['a','b','c','d']),'group_ids':np.asarray(['g1','g2','g3','g4']),
            'calibrated':np.asarray(False),'lengths':np.asarray([127,128,256,511],dtype=np.int32),
            'scope_word_like_tokens':np.asarray([127,128,256,511],dtype=np.int64),
            'component_weights_sha256':np.asarray(compat.STAT_SHA),'component_metadata_sha256':np.asarray(compat.STAT_SHA),
            'y_ai':np.asarray([1,0,1,0]),'p_ai':np.asarray([.9,.1,.8,.2]),'raw_binary':np.asarray([[0.,2.],[2.,0.],[0.,1.],[1.,0.]])}
    original=compat.bf.validate_raw_rows
    payload=fixture();before={k:(id(v),v.dtype.str,v.shape,v.tobytes())for k,v in payload.items()}
    denied('original_validator_exposes_storage_boundary',lambda:original(payload))
    validate=compat.storage_validator(original)
    def unchanged():
        validate(payload)
        assert validate.accepted_storage_exceptions==1
        assert {k:(id(v),v.dtype.str,v.shape,v.tobytes())for k,v in payload.items()}==before
    record('stat_same_values_accepted_all_original_arrays_unchanged',unchanged)
    def proxy_only():
        calls=[]
        def spy(p):
            calls.append(p)
            if p is payload:raise ValueError(compat.SCOPE_ERROR)
            assert p is not payload and p['scope_word_like_tokens']is payload['lengths']
            assert all(p[k]is payload[k]for k in payload if k!='scope_word_like_tokens')
            return original(p)
        compat.storage_validator(spy)(payload)
        assert len(calls)==2
    record('shallow_proxy_only_original_checks_reused',proxy_only)
    def neural():
        p=fixture();p['lengths']=p['lengths'].astype(np.int64);p['component_weights_sha256']=np.asarray('a'*64)
        v=compat.storage_validator(original);v(p);assert v.accepted_storage_exceptions==0
    record('original_neural_representation_unchanged',neural)
    mutations={
        'wrong_stat_weight':lambda p:p.update(component_weights_sha256=np.asarray('a'*64)),
        'wrong_stat_metadata':lambda p:p.update(component_metadata_sha256=np.asarray('a'*64)),
        'different_values':lambda p:p.update(scope_word_like_tokens=np.asarray([128,128,256,511],dtype=np.int64)),
        'negative_counts':lambda p:(p['lengths'].__setitem__(0,-1),p['scope_word_like_tokens'].__setitem__(0,-1)),
        'floating_lengths':lambda p:p.update(lengths=p['lengths'].astype(np.float64)),
        'unsigned_lengths':lambda p:p.update(lengths=p['lengths'].astype(np.uint32)),
        'wrong_scope_dtype':lambda p:p.update(scope_word_like_tokens=p['scope_word_like_tokens'].astype(np.float64)),
        'scope_shape':lambda p:p.update(scope_word_like_tokens=p['scope_word_like_tokens'][:3]),
        'empty_groups':lambda p:p.update(group_ids=np.asarray(['','g2','g3','g4'])),
        'calibrated_input':lambda p:p.update(calibrated=np.asarray(True)),
    }
    for name,mutate in mutations.items():
        p=fixture();mutate(p);denied(name,lambda p=p:compat.storage_validator(original)(p))
    def arithmetic():
        p=fixture();m32=(p['lengths']>=128);m64=(p['scope_word_like_tokens']>=128)
        assert np.array_equal(m32,m64)
        r32=compat.bf._binary_metrics(p['y_ai'],p['p_ai'],.5,m32.astype(float))
        r64=compat.bf._binary_metrics(p['y_ai'],p['p_ai'],.5,m64.astype(float))
        assert r32==r64
    record('synthetic_count_mask_and_metric_arithmetic_equivalent',arithmetic)
    denied('fit_operation_forbidden_before_any_artifact_read',lambda:compat.prepare('missing', '0'*64,'fit',[]))
    with tempfile.TemporaryDirectory()as tmp:
        p=Path(tmp)/'draft.json';p.write_text(json.dumps({'schema_version':1,'status':'DRAFT','validation_only':True,'probabilities_or_thresholds_changed':False}))
        denied('DRAFT_amendment_not_executable',lambda:compat.prepare(p,compat.sha(p),'evaluate',[]))
        denied('wrong_amendment_SHA',lambda:compat.prepare(p,'0'*64,'evaluate',[]))
    with contextlib.redirect_stderr(io.StringIO()):
        denied('development_purpose_forbidden',lambda:compat.arguments('evaluate',['--protocol','p','--protocol-sha256','s','--blend','b','--calibration','c','--output-dir','o','--purpose','development','--input','i','--input-sha256','s','--predictions','x=y']))
    result={'schema_version':1,'status':'PASS','synthetic_only':True,'checks_passed':len(checks),'checks':checks,
        'actual_NPZ_calibration_corpus_or_model_opened':False,'GPU_or_actual_metrics_used':False,
        'wrapper_sha256':compat.sha(compat.__file__),'checks_source_sha256':compat.sha(__file__)}
    out=ROOT/'research/english_v2_count_storage_checks_v1.json'
    with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'checks_passed':len(checks),'report_sha256':compat.sha(out),'synthetic_only':True}))


if __name__=='__main__':main()
