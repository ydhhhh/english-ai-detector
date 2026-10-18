"""Exercise a real frozen bundle using synthetic engineering inputs only.

These checks establish inference consistency and input handling, not detection
accuracy. No training, calibration fitting or final evaluation data are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np

from experiments.english_v1.inference import EnglishDetector


def synthetic_passage(paragraphs=24):
    sentences=(
        "The residents compared the old plans with their measurements before deciding where to place the new footbridge.",
        "A volunteer recorded the water level each morning, while another checked whether the path remained accessible after rain.",
        "Their notes describe several small disagreements about cost, timing, and how to preserve the trees beside the river.",
        "After the meeting, the group agreed to publish the measurements and invite further comments from the neighborhood.",
    )
    return "\n\n".join(f"Entry {i+1}. "+" ".join(sentences[i%4:]+sentences[:i%4]) for i in range(paragraphs))


def probability_vector(result):
    return np.asarray([result["probabilities"]["AI"],
        *[v["probability"] for v in sorted(result["family_probabilities"],key=lambda r:r["label"])],
        *[v["probability"] for v in sorted(result["model_probabilities"],key=lambda r:r["label"])]],dtype=float)


def validate_result(result):
    binary=result["probabilities"]
    assert abs(binary["human"]+binary["AI"]-1)<1e-12
    family={r["label"]:r["probability"] for r in result["family_probabilities"]}
    leaves=result["model_probabilities"]
    assert len(family)==7 and len(leaves)==25
    assert abs(sum(family.values())-1)<1e-10 and abs(sum(r["probability"] for r in leaves)-1)<1e-10
    for label,mass in family.items():
        assert abs(sum(r["probability"] for r in leaves if r["family"]==label)-mass)<1e-10
    vector=probability_vector(result)
    assert np.isfinite(vector).all() and ((vector>=0)&(vector<=1)).all()
    assert result["quality"]["encoder_coverage"]==1
    for head in (result["family_probabilities"],leaves):
        assert all(a["probability"]>=b["probability"] for a,b in zip(head,head[1:]))


def run(bundle,output,*,device="cpu",compare_device=None,tolerance=1e-4):
    started=time.monotonic()
    model=EnglishDetector.load(bundle,device=device,batch_size=4)
    long=synthetic_passage()
    texts=["A short English sentence.",long,"\u8fd9\u662f\u4e00\u6bb5\u4e2d\u6587\u81ea\u7136\u8bed\u8a00\u6587\u672c，\u7528\u4e8e\u68c0\u67e5\u4e0d\u652f\u6301\u8bed\u8a00\u65f6\u7684\u5904\u7406。"*80]
    grouped=model.predict_many(texts)
    alone=[model.predict(t) for t in texts]
    reverse=model.predict_many(list(reversed(texts)))[::-1]
    for result in grouped+alone+reverse:validate_result(result)
    differences=[float(np.max(np.abs(probability_vector(a)-probability_vector(b))))
        for pairs in ((grouped,alone),(grouped,reverse)) for a,b in zip(*pairs)]
    assert max(differences)<=tolerance
    assert grouped[0]["status"]=="inconclusive" and grouped[0]["accepted_family"] is None and grouped[0]["accepted_model"] is None
    assert grouped[2]["status"]=="inconclusive" and grouped[2]["quality"]["language"]=="unsupported"
    assert grouped[1]["quality"]["window_counts"]>1
    attribution=model.models[model.attribution_name]
    count=len(attribution.tokenizer(long,add_special_tokens=False,truncation=False,padding=False,verbose=False)["input_ids"])
    assert grouped[1]["quality"]["encoder_tokens"]==count
    # A changed tail must reach the encoder, even if final probabilities happen
    # to be nearly identical for this deliberately repetitive synthetic passage.
    changed=long+"\nThe final inspection discovered a cracked support beam and postponed construction until spring."
    raw=attribution.predict_raw_scores([long,changed],batch_size=4,device=device,precision="fp32")
    assert int(raw["encoder_tokens"][1])>int(raw["encoder_tokens"][0])
    assert np.max(np.abs(raw["binary"][1]-raw["binary"][0]))>0
    for invalid in ("", "   ", None):
        try:model.predict(invalid)
        except ValueError:pass
        else:raise AssertionError("invalid text must fail explicitly")
    try:model.predict_many("This is a string, not a sequence of passages.")
    except ValueError:pass
    else:raise AssertionError("bare string batch must fail")
    reloaded=EnglishDetector.load(bundle,device=device,batch_size=4)
    repeat=reloaded.predict(texts[0])
    reload_error=float(np.max(np.abs(probability_vector(repeat)-probability_vector(grouped[0]))))
    assert reload_error<=tolerance
    del reloaded
    compare=None
    if compare_device:
        other=EnglishDetector.load(bundle,device=compare_device,batch_size=4)
        other_results=other.predict_many(texts)
        compare=float(max(np.max(np.abs(probability_vector(a)-probability_vector(b))) for a,b in zip(grouped,other_results)))
        assert compare<=tolerance
        del other
    # Corruption is rejected before trying to load any model weights.
    with tempfile.TemporaryDirectory(prefix="english-bundle-integrity-") as folder:
        root=Path(folder)
        original=(Path(bundle)/"bundle.json").read_bytes()
        (root/"bundle.json").write_bytes(original+b" ")
        (root/"bundle.sha256").write_text(hashlib.sha256(original).hexdigest()+"  bundle.json\n")
        try:EnglishDetector.load(root)
        except ValueError:pass
        else:raise AssertionError("corrupted manifest must be rejected")
    value={"passed":True,"synthetic_engineering_inputs_only":True,"model_id":model.model_id,
        "device":device,"compare_device":compare_device,"probability_atol":tolerance,
        "batch_order_max_absolute_probability_difference":max(differences),
        "reload_max_absolute_probability_difference":reload_error,"cross_device_max_difference":compare,
        "long_input_encoder_tokens":count,"long_input_windows":grouped[1]["quality"]["window_counts"],
        "checks":["single/batch/order invariance","binary and hierarchy mass","complete long-input token coverage and tail reaches encoder",
            "short text abstention","unsupported-language abstention","invalid input rejection","reload consistency","manifest corruption rejection"],
        "scope":"Engineering consistency only; generated synthetic text is not evidence of detection accuracy.",
        "elapsed_seconds":time.monotonic()-started}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps(value,indent=2))
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cpu");parser.add_argument("--compare-device")
    args=parser.parse_args()
    run(args.bundle,args.output,device=args.device,compare_device=args.compare_device)
