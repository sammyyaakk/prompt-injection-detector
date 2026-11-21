"""Decode-and-rescan preprocessing: a cheap patch for the base64 bypass.

Finds runs that look like base64 (24+ characters from the base64 alphabet),
decodes those that turn into mostly printable UTF-8 text, and appends the
decoded text to the document so the classifier sees the hidden instruction.
Text without such runs is returned unchanged.
"""
import base64
import binascii
import re

_B64_RUN = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def _try_decode(run):
    padded = run + "=" * (-len(run) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    if not text or printable / len(text) < 0.9 or not re.search(r"[A-Za-z]{3,}", text):
        return None
    return text


def decode_and_append(text):
    decoded = [d for d in (_try_decode(m.group()) for m in _B64_RUN.finditer(text)) if d]
    if not decoded:
        return text
    return text + "\n" + "\n".join(decoded)
