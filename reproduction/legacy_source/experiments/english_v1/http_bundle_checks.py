"""Check the original localhost server with one real CPU detector bundle.

Only newly authored synthetic English is used. Reports contain counts, error
codes and consistency differences, never input text or complete predictions.
This verifies engineering behavior, not detection accuracy. Example:

python -m experiments.english_v1.http_bundle_checks --bundle BUNDLE --output QA.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote
from xml.sax.saxutils import escape
import zipfile

from experiments.english_v1.documents import extract_document
from experiments.english_v1.serve import DetectorServer, Limits

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
OLE = bytes.fromhex("d0cf11e0a1b11ae1")


class CheckFailure(AssertionError):
    """Messages are fixed check identifiers, never copied response/input text."""


def require(condition, identifier):
    if not condition:
        raise CheckFailure(identifier)


def synthetic_body():
    paragraphs = [
        "On Monday morning, Mara and her neighbors met beside the community garden to plan a new wooden sign. "
        "They measured the gate, counted the spare brushes, and compared several sketches before choosing a simple design. "
        "The group wanted visitors to understand the instructions without having to ask a volunteer for help.",
        "The first table entry listed blue paint, two clean jars, a ruler, and a roll of string. "
        "Mara checked each item against the notebook and moved the tools onto a dry shelf. "
        "Another neighbor repaired a loose hinge while the children collected fallen leaves from the narrow path.",
        "The second entry described the afternoon inspection. Everyone checked that the painted letters were easy to read "
        "and that the sign did not block the view of the street. They left enough space for a small notice about opening hours.",
        "Before leaving, the volunteers wrote the date in the notebook and put the remaining materials away. "
        "They agreed to inspect the gate again after the next heavy rain and to share their notes at the following meeting. "
        "The finished garden was quiet, tidy, and ready for visitors the next morning.",
    ]
    return paragraphs, "\n".join(paragraphs)


def docx_fixture(paragraphs):
    def p(text):
        return '<w:p><w:r><w:t xml:space="preserve">' + escape(text) + "</w:t></w:r></w:p>"

    body = p(paragraphs[0])
    body += "<w:tbl><w:tr><w:tc>" + p(paragraphs[1]) + "</w:tc><w:tc>" + p(paragraphs[2]) + "</w:tc></w:tr></w:tbl>"
    body += p(paragraphs[3])
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr("word/document.xml", xml)
    return stream.getvalue()


def legacy_doc_fixture(text):
    converter = Path("/usr/bin/textutil")
    if not converter.is_file():
        return None, {"status": "skipped", "reason": "Native textutil legacy-DOC export unavailable; no substitute parser installed"}
    with tempfile.TemporaryDirectory(prefix="http-bundle-doc-") as directory:
        source = Path(directory) / "synthetic.txt"
        target = Path(directory) / "synthetic.doc"
        source.write_text(text, encoding="utf-8")
        command = [str(converter), "-format", "txt", "-inputencoding", "UTF-8", "-convert", "doc", "-strip", "-output", str(target), str(source)]
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, {"status": "failed", "reason": "native_export_failed", "error_type": type(exc).__name__}
        if result.returncode != 0 or not target.is_file():
            return None, {"status": "failed", "reason": "native_export_failed", "returncode": result.returncode}
        payload = target.read_bytes()
    if not payload.startswith(OLE):
        return None, {"status": "failed", "reason": "native_export_was_not_genuine_OLE_DOC"}
    return payload, {"status": "available", "converter": "textutil", "OLE_signature_verified": True, "bytes": len(payload)}


def probability_map(result):
    require(isinstance(result, dict), "prediction_not_object")
    require(result.get("schema_version") == 1, "schema_version")
    binary = result["probabilities"]
    require(set(binary) == {"AI", "human"}, "binary_labels")
    family_rows = result["family_probabilities"]
    leaf_rows = result["model_probabilities"]
    families = {row["label"]: row["probability"] for row in family_rows}
    require(len(family_rows) == len(families) == 7, "family_cardinality")
    require(len(leaf_rows) == len({row["label"] for row in leaf_rows}) == 25, "leaf_cardinality")
    values = {"binary:" + key: float(value) for key, value in binary.items()}
    values.update({"family:" + key: float(value) for key, value in families.items()})
    values.update({"leaf:" + row["label"]: float(row["probability"]) for row in leaf_rows})
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()), "probability_range")
    require(abs(sum(binary.values()) - 1) <= 1e-8, "binary_mass")
    require(abs(sum(families.values()) - 1) <= 1e-8, "family_mass")
    require(abs(sum(row["probability"] for row in leaf_rows) - 1) <= 1e-8, "leaf_mass")
    for family, mass in families.items():
        require(abs(sum(row["probability"] for row in leaf_rows if row["family"] == family) - mass) <= 1e-8, "hierarchy_mass")
    for rows in (family_rows, leaf_rows):
        require(all(a["probability"] >= b["probability"] for a, b in zip(rows, rows[1:])), "ranking_order")
    require(result["quality"]["encoder_coverage"] >= 1 - 1e-8, "full_encoder_coverage")
    return values


def compare_prediction(expected, actual, tolerance):
    a, b = probability_map(expected), probability_map(actual)
    require(set(a) == set(b), "prediction_label_sets")
    difference = max(abs(a[key] - b[key]) for key in a)
    require(difference <= tolerance, "HTTP_vs_direct_probability_tolerance")
    for key in ("model_id", "status", "accepted_family", "accepted_model"):
        require(expected[key] == actual[key], "HTTP_vs_direct_decision")
    for key in ("word_like_tokens", "encoder_tokens", "window_counts", "language", "core_length_supported"):
        require(expected["quality"][key] == actual["quality"][key], "HTTP_vs_direct_input_quality")
    return difference


def request(port, route, *, method="GET", body=None, headers=None, timeout=120):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, route, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "response_size_bound")
        require(response.getheader("Cache-Control") == "no-store", "response_cache_control")
        require(response.getheader("Content-Type", "").startswith("application/json"), "response_JSON_media")
        return response.status, json.loads(data.decode("utf-8"))
    finally:
        connection.close()


def exercise(detector, *, tolerance=1e-5, timeout=120):
    """Use the actual server class and the same supplied detector object.

    This internal seam also permits a clearly labeled lightweight wiring smoke
    check; only run() marks a report as real-bundle verification.
    """
    cases = []
    limits = Limits(max_request_bytes=64 * 1024, max_characters=4096, requests_per_minute=100)
    paragraphs, text = synthetic_body()
    require(128 <= len(re.findall(r"\b\w+\b", text)), "fixture_core_length")
    require(len(text) < limits.max_characters, "fixture_under_character_limit")
    direct = detector.predict(text)
    probability_map(direct)
    body = json.dumps({"text": text}).encode("utf-8")
    server = DetectorServer(("127.0.0.1", 0), detector, limits)
    require(server.detector is detector, "server_same_detector_object")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.server_port
    differences = []

    def check(name, function):
        started = time.monotonic()
        try:
            detail = function() or {}
            cases.append({"name": name, "status": "passed", **detail, "elapsed_seconds": time.monotonic() - started})
        except Exception as exc:
            detail = {"reason": str(exc)} if isinstance(exc, CheckFailure) else {}
            cases.append({"name": name, "status": "failed", "error_type": type(exc).__name__, **detail, "elapsed_seconds": time.monotonic() - started})

    def health():
        code, value = request(port, "/health", timeout=timeout)
        require(code == 200 and value.get("ready") is True and value.get("language") == "en", "health_readiness")
        require(value["limits"] == {"max_request_bytes": limits.max_request_bytes, "max_characters": limits.max_characters, "requests_per_minute": 100, "concurrent_detections": 1}, "health_test_limits")
        require("doc_parser" in value and "text_storage" in value and "window_policy" in value, "health_contract")
        return {"HTTP_status": code}

    def detect():
        code, value = request(port, "/detect", method="POST", body=body, headers={"Content-Type": "application/json"}, timeout=timeout)
        require(code == 200, "detect_HTTP_status")
        difference = compare_prediction(direct, value, tolerance)
        differences.append(difference)
        return {"HTTP_status": code, "max_absolute_probability_difference": difference}

    def upload(payload, filename):
        parsed = extract_document(payload, filename, max_bytes=limits.max_request_bytes, max_characters=limits.max_characters)
        require(parsed == text, "synthetic_format_exact_body_roundtrip")
        # Since parsing preserves exactly this text, the cached direct result is
        # exactly detector.predict(parsed); no second detector is constructed.
        code, value = request(port, "/detect-file", method="POST", body=payload, headers={"Content-Type": "application/octet-stream", "X-Filename": quote(filename)}, timeout=timeout)
        require(code == 200, "upload_HTTP_status")
        difference = compare_prediction(direct, value, tolerance)
        differences.append(difference)
        return {"HTTP_status": code, "upload_bytes": len(payload), "parsed_characters": len(parsed), "exact_body_roundtrip": True, "max_absolute_probability_difference": difference}

    def reject_and_recover(route, invalid, media, expected, filename=None):
        headers = {"Content-Type": media}
        if filename is not None:
            headers["X-Filename"] = filename
        code, value = request(port, route, method="POST", body=invalid, headers=headers, timeout=timeout)
        require(code == expected, "error_HTTP_status")
        require(isinstance(value, dict) and isinstance(value.get("error"), str) and set(value) == {"error"}, "error_JSON_contract")
        health()
        recovered = detect()
        return {"HTTP_status": code, "recovery_health_HTTP_status": 200, "recovery_detection_HTTP_status": recovered["HTTP_status"], "recovery_max_absolute_probability_difference": recovered["max_absolute_probability_difference"]}

    legacy = {"status": "not_attempted"}
    try:
        check("GET_health", health)
        check("POST_detect_matches_direct", detect)
        for name, payload in (("TXT_UTF8", text.encode("utf-8")), ("TXT_UTF8_BOM", text.encode("utf-8-sig")), ("TXT_UTF16_BOM", text.encode("utf-16"))):
            check(name + "_upload", lambda payload=payload: upload(payload, "synthetic garden.txt"))
        check("DOCX_body_and_table_upload", lambda: upload(docx_fixture(paragraphs), "synthetic garden.docx"))
        payload, legacy = legacy_doc_fixture(text)
        if payload is not None:
            check("genuine_legacy_DOC_upload", lambda: upload(payload, "synthetic garden.doc"))
        else:
            cases.append({"name": "genuine_legacy_DOC_upload", **legacy})
        invalids = [
            ("empty_text", "/detect", b'{"text":"   "}', "application/json", 400, None),
            ("empty_upload", "/detect-file", b"", "application/octet-stream", 400, "empty.txt"),
            ("wrong_JSON_media_type", "/detect", body, "text/plain", 415, None),
            ("malformed_JSON", "/detect", b"{", "application/json", 400, None),
            ("text_character_limit", "/detect", json.dumps({"text": "a" * (limits.max_characters + 1)}).encode(), "application/json", 413, None),
            # Parsed-document character failures are DocumentError/400 in serve;
            # request Content-Length violations and JSON text limits are 413.
            ("file_character_limit", "/detect-file", b"a" * (limits.max_characters + 1), "application/octet-stream", 400, "oversize.txt"),
            ("request_byte_limit", "/detect", b"x" * (limits.max_request_bytes + 1), "application/json", 413, None),
        ]
        for name, route, invalid, media, expected, filename in invalids:
            check(name + "_and_recovery", lambda route=route, invalid=invalid, media=media, expected=expected, filename=filename: reject_and_recover(route, invalid, media, expected, filename))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    require(not thread.is_alive(), "HTTP_server_thread_shutdown")
    failed = sum(c["status"] == "failed" for c in cases)
    skipped = sum(c["status"] == "skipped" for c in cases)
    return {"status": "failed" if failed else "passed_with_skips" if skipped else "passed", "passed": failed == 0,
            "checks_passed": sum(c["status"] == "passed" for c in cases), "checks_failed": failed, "checks_skipped": skipped,
            "cases": cases, "legacy_doc": legacy, "legacy_DOC_upload_verified": any(c["name"] == "genuine_legacy_DOC_upload" and c["status"] == "passed" for c in cases),
            "test_limits": {"max_request_bytes": limits.max_request_bytes, "max_characters": limits.max_characters, "requests_per_minute": 100},
            "bound_to": "127.0.0.1", "random_ephemeral_port": True, "same_detector_object": True,
            "probability_atol": tolerance, "max_absolute_probability_difference": max(differences, default=None),
            "synthetic_characters": len(text), "synthetic_word_like_tokens": len(re.findall(r"\b\w+\b", text)), "server_closed": True}


def run(bundle, output, *, tolerance=1e-5, timeout=120, batch_size=4):
    started = time.monotonic()
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Synthetic engineering consistency only; no real/final text or accuracy conclusions", "input_or_prediction_text_stored": False, "device": "cpu", "real_bundle_loaded": False, "bundle": str(Path(bundle).resolve())}
    phase = "load_real_CPU_bundle"
    try:
        from experiments.english_v1.inference import EnglishDetector
        detector = EnglishDetector.load(bundle, device="cpu", batch_size=batch_size)
        report.update(real_bundle_loaded=True, model_id=detector.model_id)
        phase = "real_detector_HTTP_checks"
        report.update(exercise(detector, tolerance=tolerance, timeout=timeout))
    except Exception as exc:
        report.update(status="failed", passed=False, failed_phase=phase, error_type=type(exc).__name__)
        if isinstance(exc, CheckFailure):
            report["reason"] = str(exc)
    report["elapsed_seconds"] = time.monotonic() - started
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "passed", "real_bundle_loaded", "checks_passed", "checks_failed", "checks_skipped", "elapsed_seconds") if k in report}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or not 0 < args.tolerance <= 1e-3:
        parser.error("tolerance must be finite, positive and at most1e-3")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.batch_size < 1:
        parser.error("positive timeout and batch-size required")
    report = run(args.bundle, args.output, tolerance=args.tolerance, timeout=args.timeout, batch_size=args.batch_size)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
