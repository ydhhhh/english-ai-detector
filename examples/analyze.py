"""Run: python examples/analyze.py /path/to/trusted/bundle /path/to/passage.txt"""
import argparse
import json
from pathlib import Path
from english_ai_detector import EnglishDetector, extract_text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle", type=Path)
parser.add_argument("document", type=Path)
args = parser.parse_args()
model = EnglishDetector.load(args.bundle, device="cpu")
text = extract_text(args.document.read_bytes(), args.document.name)
print(json.dumps(model.predict(text), indent=2, ensure_ascii=False))
