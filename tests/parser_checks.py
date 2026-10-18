"""Self-contained 43-case synthetic document QA; no corpora or detector models.

Run: python tests/parser_checks.py --output report.json
Native legacy DOC cases explicitly skip when macOS textutil is unavailable.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import argparse
import struct
import tempfile
import subprocess
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

from experiments.english_v1 import documents

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SAMPLE = (
    "On Monday morning, Mara painted a small wooden sign for the community garden. "
    "She chose cobalt blue for the letters and left a narrow white border.\n\n"
    "The sign reads, “Please close the gate.” Its purpose is simple: keep the rabbits "
    "outside while allowing neighbors to share the path. A jar beside the shed holds "
    "clean brushes, string, and two spare labels.\n\n"
    "At sunset, Mara checked the hinges, wrote the date in her notebook, and carried "
    "the empty paint tin home."
)


def paragraph(text):
    return '<w:p><w:r><w:t xml:space="preserve">' + escape(text) + '</w:t></w:r></w:p>'


def package(xml, *, duplicate=False, many_entries=False):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
            archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
            archive.writestr("word/document.xml", xml)
            archive.writestr("word/_rels/document.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="header" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/><Relationship Id="footer" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')
            archive.writestr("word/header1.xml", f'<w:hdr xmlns:w="{W}">'+paragraph("HEADER OUTSIDE BODY")+'</w:hdr>')
            archive.writestr("word/footer1.xml", f'<w:ftr xmlns:w="{W}">'+paragraph("FOOTER OUTSIDE BODY")+'</w:ftr>')
            if duplicate:
                archive.writestr("word/document.xml", xml)
            if many_entries:
                for i in range(2049):
                    archive.writestr(f"empty/{i}.xml", b"")
    return stream.getvalue()



def _additional_checks():
    extract_document = documents.extract_document
    DocumentError = documents.DocumentError
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    cases = []

    def paragraph(value):
        return f"<w:p><w:r><w:t>{value}</w:t></w:r></w:p>"

    def package(body, encoding="utf-8", doctype="", namespace=ns):
        declaration = "utf-16" if encoding.startswith("utf-16") else encoding
        xml = f'<?xml version="1.0" encoding="{declaration}"?>{doctype}<w:document xmlns:w="{namespace}"><w:body>{body}</w:body></w:document>'
        payload = xml.encode(encoding)
        if encoding == "utf-16-le": payload = b"\xff\xfe" + payload
        if encoding == "utf-16-be": payload = b"\xfe\xff" + payload
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", payload)
        return stream.getvalue()

    def okay(name, payload, expected):
        assert extract_document(payload, "synthetic.docx") == expected, name
        cases.append({"name": name, "passed": True})

    body = '<w:p><w:r><w:t>Before box.</w:t></w:r><w:r><w:drawing><w:txbxContent>' + paragraph("Textbox sentence.") + '</w:txbxContent></w:drawing></w:r><w:r><w:t>After box.</w:t></w:r></w:p>' + paragraph("Final paragraph.")
    okay("nested textbox once", package(body), "Before box.\nTextbox sentence.\nAfter box.\nFinal paragraph.")
    table = paragraph("First paragraph.") + '<w:tbl><w:tr><w:tc>' + paragraph("Left cell.") + '</w:tc><w:tc>' + paragraph("Right cell.") + '</w:tc></w:tr></w:tbl>' + paragraph("Last paragraph.")
    okay("table paragraph order", package(table), "First paragraph.\nLeft cell.\nRight cell.\nLast paragraph.")
    okay("tabs CR breaks", package('<w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:cr/><w:t>C</w:t><w:br/><w:t>D</w:t></w:r></w:p>'), "A\tB\nC\nD")
    okay("delText/instrText omitted", package('<w:p><w:del><w:r><w:delText>Deleted.</w:delText></w:r></w:del><w:r><w:instrText>PAGE</w:instrText><w:t>Visible.</w:t></w:r></w:p>'), "Visible.")
    for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
        okay("normal XML " + encoding, package(paragraph("Ordinary English body."), encoding), "Ordinary English body.")
        try:
            extract_document(package('<w:p><w:r><w:t>&label;</w:t></w:r></w:p>', encoding, '<!DOCTYPE w:document [<!ENTITY label "Synthetic label.">]>'), "synthetic.docx")
        except DocumentError as exc:
            assert "custom XML entities" in str(exc)
            cases.append({"name": "entity rejection " + encoding, "passed": True})
        else: raise AssertionError("entity declaration accepted")
    for declaration in ("UTF-16-LE", "unknown-invalid-encoding"):
        xml = f'<?xml version="1.0" encoding="{declaration}"?><w:document xmlns:w="{ns}"><w:body>{paragraph("Harmless body.")}</w:body></w:document>'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", xml.encode("utf-16-le") if declaration == "UTF-16-LE" else xml.encode())
        try: extract_document(stream.getvalue(), "synthetic.docx")
        except DocumentError: cases.append({"name": "unsupported XML encoding " + declaration, "passed": True})
        else: raise AssertionError("unsupported XML encoding accepted")
    damaged = bytearray(package(paragraph("Synthetic damaged compression.")))
    name_size, extra_size = struct.unpack_from("<HH", damaged, 26)
    offset = 30 + name_size + extra_size; damaged[offset:offset+4] = b"\xff" * 4
    try: extract_document(bytes(damaged), "synthetic.docx")
    except DocumentError: cases.append({"name": "damaged deflate mapped to DocumentError", "passed": True})
    else: raise AssertionError("damaged compression accepted")
    okay("Strict OOXML body and table", package(table, namespace="http://purl.oclc.org/ooxml/wordprocessingml/main"), "First paragraph.\nLeft cell.\nRight cell.\nLast paragraph.")
    assert len(cases) == 14
    return cases

def _run(work):
    fixtures = work / "fixtures"
    fixtures.mkdir(exist_ok=True)
    cases = []

    def success(name, payload, filename, expected, **kwargs):
        start = time.perf_counter()
        try:
            actual = documents.extract_text(payload, filename, **kwargs)
            passed = actual == expected
            item = {"name": name, "passed": passed, "characters": len(actual),
                    "exact_body_equal": passed, "whitespace_normalized_equal": " ".join(actual.split()) == " ".join(expected.split())}
            if not passed:
                item.update(expected_body_sha256=hashlib.sha256(expected.encode()).hexdigest(), actual_body_sha256=hashlib.sha256(actual.encode()).hexdigest())
        except Exception as error:
            item = {"name": name, "passed": False, "error_type": type(error).__name__, "message": str(error)}
        item["elapsed_seconds"] = round(time.perf_counter() - start, 6)
        cases.append(item)

    def reject(name, payload, filename, fragment, **kwargs):
        start = time.perf_counter()
        try:
            actual = documents.extract_text(payload, filename, **kwargs)
            item = {"name": name, "passed": False, "unexpected_acceptance_characters": len(actual)}
        except documents.DocumentError as error:
            item = {"name": name, "passed": any(value in str(error) for value in (fragment if isinstance(fragment, tuple) else (fragment,))), "error_type": "DocumentError", "message": str(error)}
        except Exception as error:
            item = {"name": name, "passed": False, "error_type": type(error).__name__, "message": str(error)}
        item["elapsed_seconds"] = round(time.perf_counter() - start, 6)
        cases.append(item)

    encodings = {"utf8": SAMPLE.encode("utf-8"), "utf8_bom": b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"),
                 "utf16_le_bom": b"\xff\xfe" + SAMPLE.encode("utf-16-le"),
                 "utf16_be_bom": b"\xfe\xff" + SAMPLE.encode("utf-16-be")}
    for name, payload in encodings.items():
        path = fixtures / (name + ".txt"); path.write_bytes(payload)
        success(name, payload, path.name, SAMPLE)
    success("uppercase_txt_suffix", encodings["utf8"], "SAMPLE.TXT", SAMPLE)

    body_lines = ["Garden equipment record", "Before the table: labels & brushes <ready>.",
                  "Morning\tblue marker\nSecond line of the paragraph.", "Row one, left cell.",
                  "Row one, right cell.", "Row two, left cell, first paragraph.",
                  "Row two, left cell, second paragraph.", "Row two, right cell.",
                  "After the table: every item is accounted for."]
    body = paragraph(body_lines[0]) + paragraph(body_lines[1])
    body += '<w:p><w:r><w:t>Morning</w:t><w:tab/><w:t>blue marker</w:t><w:br/><w:t>Second line of the paragraph.</w:t></w:r></w:p>'
    body += '<w:tbl><w:tr><w:tc>' + paragraph(body_lines[3]) + '</w:tc><w:tc>' + paragraph(body_lines[4]) + '</w:tc></w:tr><w:tr><w:tc>'
    body += paragraph(body_lines[5]) + paragraph(body_lines[6]) + '</w:tc><w:tc>' + paragraph(body_lines[7]) + '</w:tc></w:tr></w:tbl>'
    body += paragraph(body_lines[8]) + '<w:sectPr><w:headerReference w:type="default" r:id="header"/><w:footerReference w:type="default" r:id="footer"/></w:sectPr>'
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>{body}</w:body></w:document>'
    docx = package(xml)
    docx_path = fixtures / "body_and_table.docx"; docx_path.write_bytes(docx)
    expected_docx = "\n".join(body_lines)
    success("docx_body_table_row_major_tabs_breaks", docx, docx_path.name, expected_docx)
    success("uppercase_docx_suffix", docx, "TABLE.DOCX", expected_docx)

    legacy = {"attempted": False, "converter": "/usr/bin/textutil", "additional_parser_installed": False}
    legacy_names = ["genuine_ole_legacy_doc_roundtrip", "doc_output_byte_limit", "doc_timeout_error_branch_simulated_clock"]
    def skip(name, reason):
        cases.append({"name": name, "passed": None, "status": "skipped", "reason": reason})
    output = fixtures / "genuine_legacy.doc"
    if Path("/usr/bin/textutil").is_file():
        legacy["attempted"] = True
        command = ["/usr/bin/textutil", "-format", "txt", "-inputencoding", "UTF-8", "-convert", "doc", "-strip", "-output", str(output), str(fixtures / "utf8.txt")]
        try:
            converted = subprocess.run(command, capture_output=True, text=True, timeout=30, shell=False)
            legacy["conversion_returncode"] = converted.returncode
            if converted.returncode != 0 or not output.is_file():
                raise RuntimeError("Native textutil DOC export failed")
            payload = output.read_bytes()
            legacy.update(bytes=len(payload), signature_hex=payload[:8].hex(),
                          ole_compound_signature=payload.startswith(bytes.fromhex("d0cf11e0a1b11ae1")),
                          different_from_source_bytes=payload != encodings["utf8"])
            if not legacy["ole_compound_signature"]:
                raise RuntimeError("Native export did not produce an OLE Compound Word binary")
            success(legacy_names[0], payload, output.name, SAMPLE)
            reject(legacy_names[1], payload, output.name, "exceeds the resource limit", max_uncompressed=32)
            with patch.object(documents.time, "monotonic", side_effect=[0,16]):
                reject(legacy_names[2], payload, output.name, "conversion timed out")
        except Exception as error:
            # Converter exists but an attempted export failed: report failure,
            # never turn it into a passing real-DOC check.
            legacy.update(error_type=type(error).__name__, error=str(error))
            cases.append({"name": legacy_names[0], "passed": False,
                          "reason": "Native export failed or was not genuine OLE", "error_type": type(error).__name__})
            for name in legacy_names[1:]: skip(name, "Blocked by unsuccessful genuine DOC export")
    else:
        legacy["availability"] = "Native legacy DOC export unavailable; no replacement parser installed"
        for name in legacy_names: skip(name, legacy["availability"])

    reject("empty_input", b"", "empty.txt", "is empty")
    reject("unsupported_suffix", encodings["utf8"], "sample.pdf", "Supported formats are TXT")
    reject("invalid_utf8", b"\xffbad", "bad.txt", "Unsupported TXT encoding")
    reject("utf16_without_bom", SAMPLE.encode("utf-16-le"), "no_bom.txt", "binary data")
    reject("utf8_binary_nul", b"garden\x00label", "nul.txt", "binary data")
    reject("input_byte_limit", encodings["utf8"], "sample.txt", "Upload exceeds", max_bytes=8)
    reject("extracted_character_limit", encodings["utf8"], "sample.txt", "Text exceeds", max_characters=20)
    reject("not_a_docx_zip", b"not a ZIP archive", "broken.docx", "Cannot read DOCX")
    reject("docx_uncompressed_limit", docx, "large.docx", "resource limit", max_uncompressed=64)
    reject("docx_entry_count_limit", package(xml, many_entries=True), "many.docx", "resource limit")
    reject("docx_duplicate_main_body", package(xml, duplicate=True), "duplicate.docx", "exactly one DOCX document body")
    reject("docx_xml_malformed", package('<w:document xmlns:w="'+W+'"><broken>'), "xml.docx", "XML is malformed")
    reject("docx_body_missing", package('<w:document xmlns:w="'+W+'"/>'), "body.docx", "no document body")
    reject("docx_empty_body_no_ocr", package('<w:document xmlns:w="'+W+'"><w:body><w:p/></w:body></w:document>'), "empty_body.docx", "No body text")
    reject("docx_custom_xml_entities", package('<!DOCTYPE root [<!ENTITY x "garden">]>'+xml), "entities.docx", "custom XML entities")
    encrypted = bytearray(docx); central = encrypted.find(b"PK\x01\x02")
    encrypted[central+8:central+10] = (int.from_bytes(encrypted[central+8:central+10],"little")|1).to_bytes(2,"little")
    reject("docx_declared_zip_encrypted_flag", bytes(encrypted), "encrypted.docx", "Encrypted DOCX")
    with patch.object(documents, "doc_converter", return_value=(None,None)):
        reject("doc_converter_unavailable", b"synthetic body", "sample.doc", "No DOC converter")
    with patch.object(documents, "doc_converter", return_value=("textutil", "/synthetic-converter")), patch.object(documents.subprocess, "Popen", side_effect=OSError("Synthetic launch failure")):
        reject("doc_converter_launch_failure", b"synthetic body", "sample.doc", "converter is unavailable")
    if documents.doc_converter()[1]:
        # Native converters may reject corruption before returning empty text.
        reject("corrupt_ole_doc", bytes.fromhex("d0cf11e0a1b11ae1")+b"\x00"*512, "corrupt.doc", ("No body text", "DOC conversion failed"))
    else:
        skip("corrupt_ole_doc", "No installed DOC converter; unavailable branch checked separately")

    assert len(cases) == 29
    cases.extend(_additional_checks())
    assert len(cases) == 43
    for case in cases:
        case.setdefault("status", "passed" if case["passed"] else "failed")
    counts = {status: sum(c["status"] == status for c in cases) for status in ("passed", "failed", "skipped")}
    hashes = [{"path": str(path.relative_to(work)), "bytes": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(fixtures.iterdir()) if path.is_file()]
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": counts["failed"] == 0, "status": "failed" if counts["failed"] else ("passed_with_skips" if counts["skipped"] else "passed"),
        "planned_cases": 43, "executed_cases": 43-counts["skipped"], "case_counts": counts, "all43_passed": counts["passed"] == 43,
        "constructed_only": True,
        "scope": "Only self-authored nonsensitive synthetic English fixtures; no real final text, corpus, detector or prediction used.",
        "platform": {"system": platform.system(), "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "extractor": "experiments.english_v1.documents.extract_text (alias of extract_document)",
        "extractor_sha256": hashlib.sha256(Path(documents.__file__).read_bytes()).hexdigest(),
        "legacy_doc": legacy, "cases": cases,
        "format_semantics": {"txt": "UTF8/UTF8BOM and BOM-marked UTF16LE/BE preserve internal original characters and paragraphs. Leading/trailing whitespace is stripped by the API.",
                             "docx": "Body, nested textbox and table-cell paragraph order follow logical document XML, row-major in this fixture; each text node is visited once, tabs and explicit linebreaks retained. Table grid formatting is flattened to text paragraphs. Header/footer excluded; logical order does not certify Word visual layout.",
                             "doc": "Real native OLE binary roundtrip required for success; conversion semantics depend on installed textutil. No extension-only file used for the success case."},
        "resource_limits": {"upload_bytes_default": 8*1024*1024, "extracted_characters_default": 200000,
                            "docx_declared_uncompressed_bytes_default": 32*1024*1024, "docx_zip_entries_maximum": 2048,
                            "doc_output_bytes_default": 32*1024*1024, "doc_conversion_deadline_seconds": 15,
                            "tests": "Small fixtures lower configured limits to exercise guards; timeout branch uses a simulated elapsed clock, not a deliberately hung converter."},
        "limits": ["Synthetic roundtrips do not certify every historical Word variant, corrupt/encrypted file or complex layout.",
                   "No OCR; fields/headers/footers/layout are not a full Word rendering. DOC parsing delegates to the installed converter's format recognition.",
                   "XML/ZIP checks and time/output bounds were checked as implemented; this is not a security audit of macOS textutil internals."],
        "model_modules_loaded": any(k == "torch" or k.startswith("transformers") or k == "experiments.english_v1.neural" for k in sys.modules),
        "fixtures": hashes,
    }
    return result


def run():
    """Create and remove only self-authored fixtures; return counts and hashes."""
    if not __debug__:
        raise RuntimeError("Document assertion checks require Python without -O/-OO")
    with tempfile.TemporaryDirectory(prefix="english-document-checks-") as directory:
        return _run(Path(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run()
    except Exception as error:
        result = {"passed": False, "status": "harness_failed", "planned_cases": 43,
                  "constructed_only": True, "error_type": type(error).__name__, "message": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "status", "planned_cases", "case_counts", "all43_passed") if k in result}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
