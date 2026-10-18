"""Predict locally or serve the English browser interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def predict(argv=None):
    parser = argparse.ArgumentParser(prog="english-ai-detector predict", description="Detect English text using a trusted local frozen bundle.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--file", type=Path)
    inputs.add_argument("--stdin", action="store_true", help="Read UTF-8 text from standard input")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from english_ai_detector import EnglishDetector, extract_text
    try:
        text = sys.stdin.read() if args.stdin else args.text
        if args.file is not None:
            text = extract_text(args.file.read_bytes(), args.file.name)
        result = EnglishDetector.load(args.bundle, device=args.device, batch_size=args.batch_size).predict(text)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            args.output.write_text(encoded, encoding="utf-8")
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: english-ai-detector {predict,serve} [options]\n\n"
              "  predict   Detect text or a document using a local bundle\n"
              "  serve     Start the local English browser interface\n\n"
              "Use english-ai-detector COMMAND --help for command options.")
        return 0
    if args == ["--version"]:
        from english_ai_detector import __version__
        print(__version__)
        return 0
    if args[0] == "serve":
        from english_ai_detector.serve import main as serve
        serve(args[1:])
        return 0
    if args[0] == "predict":
        return predict(args[1:])
    print("Error: choose 'predict' or 'serve'; use --help for usage.", file=sys.stderr)
    return 2
