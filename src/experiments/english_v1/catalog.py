"""One immutable family and leaf order shared by training and inference."""

FAMILIES = ["Claude", "GPT", "Gemini", "Qwen", "GLM", "Kimi", "Other"]
NOMINAL_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol",
    "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
MODEL_LABELS = [leaf for f in FAMILIES[:-1] for leaf in
                ([m for m in NOMINAL_MODELS if m.startswith(f.casefold())] + [f"{f}:other_version"])] + ["other_family"]
LEAF_CLASSES = MODEL_LABELS
MODEL_FAMILIES = [f for f in FAMILIES[:-1] for _ in
                  ([m for m in NOMINAL_MODELS if m.startswith(f.casefold())] + [f"{f}:other_version"])] + ["Other"]
VERSION_INDICES = {f: [i for i, ff in enumerate(MODEL_FAMILIES) if ff == f] for f in FAMILIES}
FAMILY_CANONICAL = {f.casefold(): f for f in FAMILIES}


def canonical_family(value):
    """Return a declared family only; unknown raw strings do not imply Other."""
    return FAMILY_CANONICAL.get(str(value or "").casefold(), str(value or ""))


def catalog_dict():
    return {"family_classes": list(FAMILIES), "nominal_model_classes": list(NOMINAL_MODELS),
            "leaf_classes": list(MODEL_LABELS), "leaf_families": list(MODEL_FAMILIES),
            "version_indices": {k: list(v) for k, v in VERSION_INDICES.items()}}
