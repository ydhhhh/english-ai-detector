"""Load a frozen, calibrated English detector bundle for local inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from experiments.english_v1.calibration import BinaryPlatt, HierarchicalCalibration
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES, VERSION_INDICES, NOMINAL_MODELS
from experiments.english_v1.select_binary import blend_margin
from experiments.english_v1.language_guard import LanguageGuard


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


class EnglishDetector:
    def __init__(self):
        self.models = {}
        self.language_guard = LanguageGuard()

    @classmethod
    def load(cls, bundle_dir, device="cpu", *, batch_size=16):
        import torch
        obj=cls(); obj.root=Path(bundle_dir).resolve()
        checksum_file=obj.root/"bundle.sha256"
        if not checksum_file.is_file() or checksum_file.read_text().split()[0]!=_sha(obj.root/"bundle.json"):
            raise ValueError("bundle manifest checksum is missing or mismatched")
        obj.metadata=json.loads((obj.root/"bundle.json").read_text())
        if obj.metadata.get("schema_version") != 1:
            raise ValueError("unsupported detector bundle schema")
        obj.model_id=obj.metadata["model_id"]
        obj.device=torch.device(device)
        if obj.device.type=="mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; use device='cpu' or run with GPU access")
        torch.set_num_threads(int(obj.metadata.get("cpu_threads",6)))
        obj.batch_size=batch_size
        if not isinstance(batch_size,int) or batch_size < 1: raise ValueError("positive integer batch_size required")
        for item in obj.metadata.get("integrity",[]):
            path=obj.root/item["path"]
            if not path.is_file() or _sha(path)!=item["sha256"]:
                raise ValueError(f"bundle integrity check failed: {item['path']}")
        obj.blend=json.loads((obj.root/obj.metadata["binary_blend"]).read_text())
        value=json.loads((obj.root/obj.metadata["calibration"]).read_text())
        if value["provenance"]["blend_sha256"]!=_sha(obj.root/obj.metadata["binary_blend"]):
            raise ValueError("bundle blend differs from frozen probability calibration")
        if value["provenance"].get("language_rule_sha256")!=_sha(Path(__file__).with_name("language_guard.py")):
            raise ValueError("runtime language-scope rule differs from frozen calibration")
        if value["provenance"].get("deployment_decision_protocol_sha256")!=_sha(obj.root/obj.metadata["decision_protocol"]):
            raise ValueError("bundle decision protocol differs from frozen calibration")
        if obj.metadata.get("validation_report"):
            binding=obj.metadata.get("validation_binding") or {}
            observed={"report_sha256":_sha(obj.root/obj.metadata["validation_report"]),
                "blend_sha256":_sha(obj.root/obj.metadata["binary_blend"]),
                "calibration_sha256":_sha(obj.root/obj.metadata["calibration"])}
            if any(binding.get(key)!=observed_value for key,observed_value in observed.items()):
                raise ValueError("bundle validation report is not bound to these frozen artifacts")
        obj.binary=BinaryPlatt(**value["binary"])
        if obj.binary.fit_split!="calibration" or not np.isfinite([obj.binary.slope,obj.binary.intercept]).all() or obj.binary.slope<0:
            raise ValueError("invalid or uncalibrated deployment artifact")
        obj.hierarchy=HierarchicalCalibration.from_dict(value["hierarchy"])
        if obj.hierarchy.family_classes!=FAMILIES or obj.hierarchy.leaf_classes!=MODEL_LABELS:
            raise ValueError("bundle attribution class order mismatch")
        obj.thresholds=value["thresholds"]
        obj.version_refinement=value.get("version_refinement")
        if obj.version_refinement:
            ra=obj.version_refinement["artifact"]
            if ra.get("family_labels")!=FAMILIES or ra.get("neural_leaf_labels")!=MODEL_LABELS or ra.get("leaf_families")!=MODEL_FAMILIES:
                raise ValueError("version refinement neural class order mismatch")
        for key in ("AI","human","family","model"):
            t=obj.thresholds.get(key)
            if not isinstance(t,dict) or t.get("fit_split")!="calibration" or t.get("comparison")!=">=" or not np.isfinite(float(t["threshold"])):
                raise ValueError(f"invalid calibration threshold: {key}")
        if any(float(obj.thresholds[key]["threshold"])<=.5 for key in ("AI","human")):
            raise ValueError("AI/human thresholds must require a strict probability majority")
        obj.attribution_name=obj.metadata["attribution_model"]
        needed={name for name,w in zip(obj.blend["names"],obj.blend["selected"]["weights"]) if w!=0}
        needed.add(obj.attribution_name)
        if obj.version_refinement:
            needed.add(obj.version_refinement["statistical_model_name"])
            for key,name in (("reference",obj.attribution_name),("statistical",obj.version_refinement["statistical_model_name"])):
                fingerprint=obj.version_refinement["artifact"]["fingerprints"][key]["component_weights_sha256"]
                if obj.metadata["models"][name]["weights_sha256"]!=fingerprint:
                    raise ValueError("version refinement component fingerprint mismatch")
        obj.source_assertions={"valid":False,"family_enabled":False,"model_enabled":False,
            "errors":["A verified source-assertion release gate was not packaged."]}
        if obj.metadata.get("source_release_gate") and obj.metadata.get("source_release_protocol"):
            from experiments.english_v1.source_release_gate import load_gate
            gate_path=obj.root/obj.metadata["source_release_gate"]
            policy_path=obj.root/obj.metadata["source_release_protocol"]
            expected={"blend_sha256":_sha(obj.root/obj.metadata["binary_blend"]),
                "calibration_sha256":_sha(obj.root/obj.metadata["calibration"]),
                "language_rule_sha256":value["provenance"]["language_rule_sha256"],
                "deployment_decision_protocol_sha256":value["provenance"]["deployment_decision_protocol_sha256"],
                "source_assertion_release_protocol_sha256":_sha(policy_path),"gate_sha256":_sha(gate_path),
                "components":{name:{"weights_sha256":value["provenance"]["component_weights_sha256"][name],
                    "metadata_sha256":value["provenance"]["component_metadata_sha256"][name]} for name in needed}}
            obj.source_assertions=load_gate(gate_path,expected=expected,protocol_path=policy_path)
        for name in needed:
            entry=obj.metadata["models"][name]
            if entry["type"]=="neural":
                from experiments.english_v1.neural import NeuralDetector
                model=NeuralDetector.load_local(obj.root/entry["path"]).to(obj.device).eval()
            elif entry["type"]=="statistical":
                from experiments.english_v1.statistical import EnglishStatisticalDetector
                model=EnglishStatisticalDetector.load(obj.root/entry["path"])
            else: raise ValueError("unsupported component model type")
            obj.models[name]=model
        if obj.metadata["models"][obj.attribution_name]["type"]!="neural":
            raise ValueError("hierarchical attribution requires a neural component")
        return obj

    def predict_many(self,texts):
        if isinstance(texts,str): raise ValueError("predict_many expects a sequence of texts; use predict for one string")
        values=list(texts)
        if not values or any(not isinstance(t,str) or not t.strip() for t in values):
            raise ValueError("provide one or more nonempty natural-language texts")
        started=time.monotonic(); raw={}
        for name,model in self.models.items():
            entry=self.metadata["models"][name]
            if entry["type"]=="neural":
                raw[name]=model.predict_raw_scores(values,batch_size=self.batch_size,device=self.device,precision="fp32")
            else:
                raw[name]=model.predict_raw_scores(values,batch_size=self.batch_size)
        # Zero-weight candidates need no runtime weights or model files.
        binaries={name:(raw[name]["binary"] if name in raw else np.zeros((len(values),2))) for name in self.blend["names"]}
        p_ai=self.binary.predict(blend_margin(self.blend,binaries))
        attribution=raw[self.attribution_name]
        versions={f:attribution["version"][:,VERSION_INDICES[f]] for f in FAMILIES}
        p=self.hierarchy.predict(attribution["family"],versions)
        if self.version_refinement:
            from experiments.english_v1 import version_refinement
            vr=self.version_refinement
            model=self.models[vr["statistical_model_name"]]
            if list(model.classes_["model"])!=vr["artifact"]["stat_model_labels"]:
                raise ValueError("version refinement statistical class order mismatch")
            p["leaf"]=version_refinement.apply(vr["artifact"],p["family"],p["leaf"],attribution["version"],raw[vr["statistical_model_name"]]["model"])
        results=[]
        for i,text in enumerate(values):
            scope=self.language_guard.check(text)
            words=scope["word_like_tokens"]
            english=scope["supported"]
            core=words>=128
            warnings=[]
            if not english: warnings.append("The model is validated for English; language identification is uncertain or indicates another language.")
            if not core: warnings.append("Fewer than 128 word-like tokens: probabilities are exploratory; no confident decision or source assertion is issued.")
            q=p["family"][i]; leaves=p["leaf"][i]
            fi=int(np.argmax(q)); mi=int(np.argmax(leaves))
            ai_cut=float(self.thresholds["AI"]["threshold"])
            human_artifact=self.thresholds.get("human")
            human_ok=bool(human_artifact and 1-p_ai[i]>=float(human_artifact["threshold"]))
            ai_ok=bool(p_ai[i]>=ai_cut)
            status="inconclusive"
            if core and english:
                if ai_ok and not human_ok: status="AI-generated"
                elif human_ok and not ai_ok: status="likely human"
            af=am=None
            if status=="AI-generated" and self.source_assertions["family_enabled"]:
                if FAMILIES[fi]!="Other" and q[fi]>=float(self.thresholds["family"]["threshold"]): af=FAMILIES[fi]
                if self.source_assertions["model_enabled"] and MODEL_LABELS[mi] in NOMINAL_MODELS and leaves[mi]>=float(self.thresholds["model"]["threshold"]) and MODEL_FAMILIES[mi]==af: am=MODEL_LABELS[mi]
            if not self.source_assertions["family_enabled"]:
                warnings.append("Source-family assertions are disabled by release quality review; the displayed source probabilities are exploratory rankings.")
            elif not self.source_assertions["model_enabled"]:
                warnings.append("Exact-model assertions are disabled by release quality review; version probabilities are exploratory rankings.")
            if status=="AI-generated":
                conclusion="Likely AI-generated under the fixed calibrated operating threshold."
                if am: conclusion+=f" The strongest supported nominal source is {am}."
                elif af: conclusion+=f" The strongest supported family is {af}; the exact version is uncertain."
                else: conclusion+=" The source family and model version are uncertain."
            elif status=="likely human": conclusion="Likely human-written under the fixed calibrated operating threshold."
            else: conclusion="The available evidence is insufficient for a confident decision."
            coverage=float(attribution["encoder_coverage"][i])
            if coverage < 1-1e-8: raise AssertionError("incomplete input coverage")
            if not np.isfinite(p_ai[i]) or not np.isfinite(leaves).all(): raise FloatingPointError("nonfinite prediction")
            results.append({"schema_version":1,"model_id":self.model_id,"status":status,
                "probabilities":{"human":float(1-p_ai[i]),"AI":float(p_ai[i])},
                "family_probabilities":[{"label":FAMILIES[j],"probability":float(q[j])} for j in np.argsort(-q,kind="stable")],
                "model_probabilities":[{"label":MODEL_LABELS[j],"family":MODEL_FAMILIES[j],"probability":float(leaves[j])} for j in np.argsort(-leaves,kind="stable")],
                "accepted_family":af,"accepted_model":am,"conclusion":conclusion,
                "source_assertions":{key:self.source_assertions[key] for key in ("valid","family_enabled","model_enabled")},
                "quality":{"word_like_tokens":words,"core_length_supported":core,
                    "encoder_tokens":int(attribution["encoder_tokens"][i]),"encoder_coverage":coverage,
                    "window_counts":int(attribution["window_counts"][i]),
                    "language":"en" if english else "unsupported","language_evidence":{k:v for k,v in scope.items() if k not in {"supported","word_like_tokens"}}},
                "warnings":warnings,"probability_semantics":{"binary":"Calibrated on the held-out calibration cohort, with its class balance and domains; not a proof of authorship or a fraction of AI-written words.",
                    "family":"Conditional on AI generation; the Other bucket is the observed training catalog complement.",
                    "model":"Conditional on AI generation; includes other_version buckets; nominal publisher labels, not authenticated model identities."},
                "elapsed_seconds":time.monotonic()-started})
        return results

    def predict(self,text):
        return self.predict_many([text])[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",required=True)
    p.add_argument("--text")
    p.add_argument("--file",type=Path)
    p.add_argument("--stdin",action="store_true",help="Read UTF-8 text from standard input; all tokens are windowed, with no model-side length cutoff")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if sum((a.text is not None,a.file is not None,a.stdin))!=1: p.error("provide exactly one of --text, --file or --stdin")
    text=sys.stdin.read() if a.stdin else a.text
    if a.file:
        from experiments.english_v1.documents import extract_text
        text=extract_text(a.file.read_bytes(),a.file.name)
    result=EnglishDetector.load(a.bundle,device=a.device).predict(text)
    encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if a.output: a.output.write_text(encoded,encoding="utf-8")
    else: print(encoded,end="")


if __name__=="__main__": main()
