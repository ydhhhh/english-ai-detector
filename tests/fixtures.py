"""Constructed deterministic components; never load a trained checkpoint."""
import numpy as np
from english_ai_detector import EnglishDetector
from experiments.english_v1.calibration import BinaryPlatt, HierarchicalCalibration
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, VERSION_INDICES

PASSAGE = ("Our neighborhood library opens early on Saturday. A volunteer records the returned books, "
           "checks the shelves, and places a handwritten note beside the reading room. "
           "Visitors can ask for a quiet seat or borrow a map of the nearby walking paths. ") * 5

class ConstructedRawComponent:
    def predict_raw_scores(self, texts, **kwargs):
        n = len(texts)
        margin = np.array([0.0 if t.startswith("Balanced fixture.") else 4.0 for t in texts])
        return {"binary": np.column_stack((np.zeros(n), margin)),
                "family": np.zeros((n, len(FAMILIES))), "version": np.zeros((n, len(MODEL_LABELS))),
                "encoder_coverage": np.ones(n), "encoder_tokens": np.full(n, 300), "window_counts": np.ones(n)}

def detector():
    obj = EnglishDetector()
    obj.models = {"fixture": ConstructedRawComponent()}
    obj.metadata = {"models": {"fixture": {"type": "neural"}}}
    obj.model_id = "synthetic-engineering-fixture"
    obj.device = "cpu"
    obj.batch_size = 4
    obj.attribution_name = "fixture"
    obj.version_refinement = None
    obj.blend = {"names": ["fixture"], "mean": [0.0], "scale": [1.0],
                 "selected": {"weights": [1.0], "intercept": 0.0}}
    obj.binary = BinaryPlatt(fit_split="calibration")
    obj.hierarchy = HierarchicalCalibration(FAMILIES, {f: [MODEL_LABELS[i] for i in VERSION_INDICES[f]] for f in FAMILIES})
    obj.thresholds = {key: {"threshold": value} for key, value in (("AI", .7), ("human", .9), ("family", 1.), ("model", 1.))}
    obj.source_assertions = {"valid": False, "family_enabled": False, "model_enabled": False}
    return obj
