"""Download the approved public archive and print its verified local bundle path."""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from bundle_archive import extract_bundle_archive

REPOSITORY = "ydh0822/english-ai-detector"
FILENAME = "english_ai_detector_v2_bundle.zip"
ARCHIVE_SHA256 = "da04e4314bdc4571e1dfff8414d203865ac03ad6d14aeff2a182ac070b5be164"
MANIFEST_SHA256 = "f588dc69355cc96105ea627e0aa20bdcd9bed78709038884466abf008a4271cc"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Pin a reviewed full Model Hub commit for reproducible downloads")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/english_detector_v2"))
    parser.add_argument("--archive", type=Path, help="Verify/extract this already downloaded approved archive without a network call")
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        archive = hf_hub_download(repo_id=REPOSITORY, filename=FILENAME, revision=args.revision, token=False)
    bundle = extract_bundle_archive(archive, expected_archive_sha256=ARCHIVE_SHA256,
        expected_manifest_sha256=MANIFEST_SHA256, cache_directory=args.cache_dir)
    print(bundle.resolve())

if __name__ == "__main__":
    main()
