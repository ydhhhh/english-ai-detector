"""Bounded plain text extraction; no macros, embedded code or shell expansion."""
from __future__ import annotations

import io
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib


class DocumentError(ValueError):
    pass


def _txt(payload):
    try:
        if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
            return payload.decode("utf-16")
        result = payload.decode("utf-8-sig")
        if "\x00" in result:
            raise DocumentError("TXT contains binary data; save it as UTF-8 text.")
        return result
    except UnicodeDecodeError:
        raise DocumentError("Unsupported TXT encoding; use UTF-8 or BOM-marked UTF-16.") from None


def _docx(payload, max_uncompressed):
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 2048 or sum(x.file_size for x in entries) > max_uncompressed:
                raise DocumentError("DOCX exceeds the uncompressed resource limit.")
            if any(x.flag_bits & 1 for x in entries):
                raise DocumentError("Encrypted DOCX is unsupported; remove password protection first.")
            names = [x.filename for x in entries]
            if names.count("word/document.xml") != 1:
                raise DocumentError("The file must contain exactly one DOCX document body.")
            info = archive.getinfo("word/document.xml")
            if info.file_size > max_uncompressed:
                raise DocumentError("DOCX body exceeds the resource limit.")
            xml = archive.read(info)
    except (zipfile.BadZipFile, KeyError, RuntimeError, NotImplementedError, OSError, zlib.error):
        raise DocumentError("Cannot read DOCX; check that the file is complete and not encrypted.") from None
    # XML may use UTF-16/32; ASCII-only byte matching misses declarations with
    # intervening NUL bytes. Inspect an ASCII-compatible declaration view while
    # leaving the original encoding and text bytes untouched for the parser.
    declarations = xml.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise DocumentError("Documents with custom XML entities are unsupported.")
    try:
        root = ET.fromstring(xml)
    except (ET.ParseError, ValueError, LookupError):
        raise DocumentError("DOCX body XML is malformed or uses an unsupported encoding.") from None
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    strict_ns = "{http://purl.oclc.org/ooxml/wordprocessingml/main}"
    if root.tag == strict_ns + "document":
        ns = strict_ns
    body = root.find(ns + "body")
    if body is None:
        raise DocumentError("DOCX has no document body.")
    # A textbox may contain w:p inside the outer paragraph's drawing. Walking
    # every paragraph and then all its descendants emits that inner text twice.
    # Visit each XML node once, keeping logical paragraph boundaries in order.
    parts = []
    stack = [(body, False)]
    while stack:
        node, leaving = stack.pop()
        if leaving:
            parts.append("\n")
            continue
        if node.tag == ns + "p":
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
            stack.append((node, True))
        elif node.tag == ns + "t":
            parts.append(node.text or "")
        elif node.tag in (ns + "br", ns + "cr"):
            parts.append("\n")
        elif node.tag == ns + "tab":
            parts.append("\t")
        stack.extend((child, False) for child in reversed(node))
    return "".join(parts)


def doc_converter():
    textutil = Path("/usr/bin/textutil")
    if textutil.exists():
        return "textutil", str(textutil)
    antiword = shutil.which("antiword")
    return ("antiword", antiword) if antiword else (None, None)


def _doc(payload, max_output):
    kind, executable = doc_converter()
    if not executable:
        raise DocumentError("No DOC converter is installed; save the file as DOCX or UTF-8 TXT.")
    with tempfile.TemporaryDirectory(prefix="english-detector-doc-") as directory:
        source = Path(directory) / "document.doc"
        source.write_bytes(payload)
        command = ([executable, "-convert", "txt", "-encoding", "UTF-8", "-stdout", str(source)] if kind == "textutil"
                   else [executable, "-m", "UTF-8.txt", str(source)])
        process = None
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
            chunks = []; count = 0; deadline = time.monotonic()+15
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, 15)
                    if not selector.select(min(remaining, .5)):
                        continue
                    chunk = os.read(process.stdout.fileno(), min(65536, max_output-count+1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > max_output:
                        raise DocumentError("DOC extraction exceeds the resource limit.")
                    chunks.append(chunk)
            if process.wait(timeout=max(.01, deadline-time.monotonic())):
                raise DocumentError("DOC conversion failed; remove password protection or save as DOCX.")
            return b"".join(chunks).decode("utf-8")
        except subprocess.TimeoutExpired:
            raise DocumentError("DOC conversion timed out; save as DOCX or TXT and retry.") from None
        except (OSError, UnicodeDecodeError):
            raise DocumentError("DOC converter is unavailable; save as DOCX or UTF-8 TXT.") from None
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdout:
                    process.stdout.close()


def extract_document(payload: bytes, filename: str, *, max_bytes=8*1024*1024,
                     max_characters=200_000, max_uncompressed=32*1024*1024):
    """Extract body only, preserving paragraph and table-cell paragraph order."""
    if not isinstance(payload, bytes) or not payload:
        raise DocumentError("The uploaded file is empty.")
    if len(payload) > max_bytes:
        raise DocumentError(f"Upload exceeds the {max_bytes//1024//1024} MiB limit.")
    suffix = Path(filename).suffix.casefold()
    if suffix == ".txt":
        text = _txt(payload)
    elif suffix == ".docx":
        text = _docx(payload, max_uncompressed)
    elif suffix == ".doc":
        text = _doc(payload, max_uncompressed)
    else:
        raise DocumentError("Supported formats are TXT, DOCX, and DOC when a system converter is available.")
    text = text.replace("\x00", "").strip()
    if not text:
        raise DocumentError("No body text was extracted; scanned images require OCR.")
    if len(text) > max_characters:
        raise DocumentError(f"Text exceeds the {max_characters:,}-character limit; no truncation or detection was performed.")
    return text


# Shared CLI name used by inference.py; this is exactly the same bounded parser.
extract_text = extract_document
