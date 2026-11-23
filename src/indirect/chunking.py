"""Split a document into overlapping character windows for chunk-level scoring.

Chunk-level scoring addresses the dilution effect found in the pilot: a short
injection inside a long benign document gets averaged away when the whole
document is scored at once. Scoring windows and taking the maximum lets the
most suspicious region decide, and that region doubles as the explanation.
"""
import re

WINDOW_CHARS = 400
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(<\[#])")


def _units(text):
    """Character spans of lines, with long lines further split into sentences."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        start, line_text = line.start(), line.group()
        if len(line_text) <= WINDOW_CHARS:
            spans.append((start, line.end()))
            continue
        cursor = 0
        for m in _SENTENCE_END.finditer(line_text):
            spans.append((start + cursor, start + m.start()))
            cursor = m.end()
        spans.append((start + cursor, line.end()))
    return [s for s in spans if text[s[0]:s[1]].strip()]


def windows(text, max_chars=WINDOW_CHARS):
    """Greedy windows of consecutive units up to `max_chars`, overlapping by one unit.

    Returns a list of (start, end) character offsets into `text`. A single unit
    longer than `max_chars` becomes its own window. Short texts yield one window.
    """
    units = _units(text)
    if not units:
        return [(0, len(text))]
    if units[-1][1] - units[0][0] <= max_chars:
        return [(units[0][0], units[-1][1])]
    out, i = [], 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1][1] - units[i][0] <= max_chars:
            j += 1
        out.append((units[i][0], units[j][1]))
        if j + 1 >= len(units):
            break
        i = j if j > i else j + 1  # overlap by one unit when the window holds 2+
    return out


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
