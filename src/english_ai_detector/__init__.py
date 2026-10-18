"""Local English AI detection with frozen calibrated bundles.

Compatibility modules preserve frozen serialization and numeric behavior.
"""
from experiments.english_v1.inference import EnglishDetector
from english_ai_detector.documents import DocumentError, extract_document, extract_text

__version__ = "2.0.0"
__all__ = ["EnglishDetector", "DocumentError", "extract_document", "extract_text", "__version__"]
