"""Download the approved public archive and print its verified local bundle path."""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from bundle_archive import extract_bundle_archive

REPOSITORY = "ydh0822/english-ai-detector"
FILENAME = "english_ai_detector_v2_bundle.zip"
ARCHIVE_SHA256 = "8d040a57f38739ec10c8eec9b5930d94a6bcea303232a7b80a53fbf3a47877c4"
MANIFEST_SHA256 = "e0ba21adeb0bdd39cff9c7375edfa0c651150bf7ae1584c61a859cd24fa11c65"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="d54db158b99d4b408f950a99a37664f0c5f3d81b", help="Immutable Model Hub commit (defaults to the verified v2 release)")
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
