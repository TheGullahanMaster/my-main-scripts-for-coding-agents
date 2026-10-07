"""CSV Editor column kinds: the on-disk layout and the text/image encoders.

A CSV written by the GUI's CSV Editor is self-contained.  Besides its plain
numeric/category columns it holds reserved columns, all starting with
``#afpo:`` and always ignored by training:

  #afpo:src:<name>     the raw value of an encoded column, one per row: the
                       string of a text column, the file path of an image
  #afpo:schema:<spec>  one empty column whose *name* lists every editor
                       column and its kind (0 bool, 1 integer, 2 float,
                       3 category, 4 text encoded, 5 text char level,
                       6 image), so the kinds survive a save

and the encoded columns training reads:

  text (encoded)   <name>            one number per string (EasyNN-plus rule)
  text (chars)     <name>[1..L]      one character code per position, 0 = none
  image            <name>.PC/.EC/.BC three codes per image, or
                   <name>[r1c1].PC … three per grid cell

This file is also pasted verbatim into an exported best_model.py, so it must
stay self-contained: every top-level name starts with editor_/EDITOR_ and
Pillow is imported only when an image is actually encoded.
"""
import bisect
import math
import os
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd

EDITOR_PREFIX = "#afpo:"
EDITOR_SCHEMA_PREFIX = "#afpo:schema:"
EDITOR_SOURCE_PREFIX = "#afpo:src:"
EDITOR_SCHEMA_KEY = "__afpo_editor_schema__"
EDITOR_KINDS = ("bool", "int", "float", "category", "text", "chars", "image")
EDITOR_KIND_LABELS = ("Bool", "Integer", "Float", "Category", "Text (encoded)", "Text (char level)", "Image")
EDITOR_BOOL, EDITOR_INT, EDITOR_FLOAT, EDITOR_CATEGORY, EDITOR_TEXT, EDITOR_CHARS, EDITOR_IMAGE = range(7)
EDITOR_IMAGE_SIZE = 100       # every image is stretched to this square before encoding
EDITOR_MAX_GRID = 10
EDITOR_IMAGE_CODES = ("PC", "EC", "BC")
EDITOR_TEXT_GAP = 100         # two strings' codes are never this close...
EDITOR_TEXT_STEP = 101        # ...a clash moves the newer one up by this much


def editor_reserved(name):
    """True for a column the editor owns; training always ignores these."""
    return str(name).startswith(EDITOR_PREFIX)


def editor_source_column(name):
    return EDITOR_SOURCE_PREFIX + str(name)


# ───────────────────────── schema header ─────────────────────────

def editor_schema_header(columns):
    """The reserved column name recording [{name, kind, grid}] (cells stay empty)."""
    parts = []
    for column in columns:
        part = f"{quote(str(column['name']), safe='')}={int(column['kind'])}"
        if int(column["kind"]) == EDITOR_IMAGE:
            rows, cols = editor_grid(column.get("grid"))
            part += f"@{rows}x{cols}"
        parts.append(part)
    return EDITOR_SCHEMA_PREFIX + ";".join(parts)


def editor_parse_header(header):
    """[{name, kind, grid}] from a schema column name; None if it is not one."""
    header = str(header)
    if not header.startswith(EDITOR_SCHEMA_PREFIX):
        return None
    columns = []
    for part in header[len(EDITOR_SCHEMA_PREFIX):].split(";"):
        name, separator, spec = part.partition("=")
        if not separator:
            continue
        kind, _, grid = spec.partition("@")
        try:
            kind = int(kind)
        except ValueError:
            continue
        if not 0 <= kind < len(EDITOR_KINDS):
            continue
        column = {"name": unquote(name), "kind": kind}
        if kind == EDITOR_IMAGE:
            rows, _, cols = grid.partition("x")
            column["grid"] = list(editor_grid([rows or 1, cols or 1]))
        columns.append(column)
    return columns


def editor_grid(grid):
    """(rows, cols), each clamped to 1..EDITOR_MAX_GRID."""
    try:
        rows, cols = (int(float(value)) for value in (grid or (1, 1)))
    except (TypeError, ValueError):
        rows, cols = 1, 1
    return max(1, min(EDITOR_MAX_GRID, rows)), max(1, min(EDITOR_MAX_GRID, cols))


# ───────────────────────── text (encoded) ─────────────────────────
# EasyNN-plus: "the sum of the ASCII code multiplied by its position in the
# string. For example, the word 'Dog' is (68 * 3) + (111 * 2) + (103 * 1) =
# 529. ... If it is found to be within 100 of any other string then 101 is
# added to the numeric value and it is checked again."  So a code depends on
# the strings entered before it: stored codes are never recomputed.

def editor_text_base(text):
    text = str(text)
    return sum(ord(character) * (len(text) - index) for index, character in enumerate(text))


def editor_text_code(text, taken):
    """Code of a new string given the sorted list of codes already in use."""
    value = editor_text_base(text)
    while True:
        at = bisect.bisect_left(taken, value - EDITOR_TEXT_GAP)
        if at < len(taken) and taken[at] <= value + EDITOR_TEXT_GAP:
            value += EDITOR_TEXT_STEP
        else:
            return value


def editor_assign_codes(strings, codes):
    """Give every string not yet in ``codes`` a code, in the order met; returns ``codes``."""
    taken = sorted(codes.values())
    for text in strings:
        if text is None or text == "" or text in codes:
            continue
        codes[text] = editor_text_code(text, taken)
        bisect.insort(taken, codes[text])
    return codes


def editor_lookup_code(value, codes):
    """Code for one cell: a known string, a number (already a code), or a new string."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value)
    if text in codes:
        return float(codes[text])
    try:
        return float(text)
    except ValueError:
        pass
    if not text.strip():
        return float("nan")
    return float(editor_text_code(text, sorted(codes.values())))


def editor_decode_text(value, codes):
    """Nearest stored string, prefixed with 0-3 '~' by how far the value is from it.

    The distance is measured against the point where the answer would flip to
    the neighbouring string: no '~' within the first quarter of the way,
    then '~', '~~', and '~~~' for the last quarter."""
    if not codes or value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    ordered = sorted((code, text) for text, code in codes.items())
    numbers = [code for code, _ in ordered]
    at = bisect.bisect_left(numbers, value)
    choices = [index for index in (at - 1, at) if 0 <= index < len(ordered)]
    nearest = min(choices, key=lambda index: abs(numbers[index] - value))
    code, text = ordered[nearest]
    toward = nearest + (1 if value > code else -1)
    if 0 <= toward < len(ordered):
        half = abs(numbers[toward] - code) / 2
    else:
        # Beyond the outermost string there is nothing to flip to: reuse its
        # one neighbour's spacing, or the minimum spacing for a lone string.
        other = nearest - (1 if value > code else -1)
        half = abs(numbers[other] - code) / 2 if 0 <= other < len(ordered) else EDITOR_TEXT_STEP / 2
    share = abs(value - code) / half if half > 0 else 0.
    return "~" * (0 if share < .25 else 1 if share < .5 else 2 if share < .75 else 3) + text


# ───────────────────────── text (char level) ─────────────────────────

def editor_char_columns(name, length):
    return [f"{name}[{position}]" for position in range(1, int(length) + 1)]


def editor_char_codes(text, length):
    """Character codes (ASCII, else the Unicode code point), padded with 0 to ``length``."""
    text = "" if text is None else str(text)
    codes = [ord(character) for character in text[:int(length)]]
    return codes + [0] * (int(length) - len(codes))


def editor_decode_chars(values):
    """Text from predicted character codes: each rounded to the nearest code, 0 ends the string."""
    out = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            break
        code = int(round(float(value)))
        if code <= 0:
            break
        out.append(chr(min(code, 0x10FFFF)) if not 0xD800 <= code <= 0xDFFF else "?")
    return "".join(out)


# ───────────────────────── images ─────────────────────────
# EasyNN-plus names three codes but gives one sentence for each ("PC is
# derived by comparing a thousand pairs of pixels in the image. The positions
# of the top, bottom, left and right outer edges are used to produce EC. The
# position of the largest block of pixels with the same color is used to
# produce BC.") and no formula, so the definitions below are a reconstruction:
#
#   image  stretched to 100x100, grayscale 0..255
#   PC     how many of 1000 fixed pixel pairs have the first pixel brighter
#          (0..1000)
#   EC     bounding box of everything that differs from the background (the
#          commonest of 16 gray levels on the border), each side as 0..99:
#          top*1e6 + bottom*1e4 + left*100 + right; 0 for a blank image
#   BC     centre of the largest connected area of one gray level (of 16):
#          row*100 + column, each 0..99

def editor_image_columns(name, grid=None):
    rows, cols = editor_grid(grid)
    if rows == 1 and cols == 1:
        return [f"{name}.{code}" for code in EDITOR_IMAGE_CODES]
    return [f"{name}[r{r}c{c}].{code}" for r in range(1, rows + 1) for c in range(1, cols + 1) for code in EDITOR_IMAGE_CODES]


def editor_patch_spans(count, size=EDITOR_IMAGE_SIZE):
    """(start, stop) of ``count`` equal patches along one side; they overlap when the side does not divide evenly."""
    width = -(-size // count)
    if count == 1:
        return [(0, size)]
    return [(start, start + width) for start in (int(round(index * (size - width) / (count - 1))) for index in range(count))]


_EDITOR_PAIRS = []


def editor_pixel_pairs():
    """1000 fixed pixel pairs as fractions of the patch (own generator: identical on every machine)."""
    if not _EDITOR_PAIRS:
        state, values = 20261007, []
        for _ in range(4000):
            state = (state * 1103515245 + 12345) % 2147483648
            values.append(state / 2147483648)
        _EDITOR_PAIRS.append(np.asarray(values).reshape(1000, 4))
    return _EDITOR_PAIRS[0]


def editor_pair_code(patch):
    height, width = patch.shape
    pairs = editor_pixel_pairs()
    first = patch[(pairs[:, 0] * height).astype(int), (pairs[:, 1] * width).astype(int)]
    second = patch[(pairs[:, 2] * height).astype(int), (pairs[:, 3] * width).astype(int)]
    return float(np.count_nonzero(first > second))


def _editor_scaled(position, size):
    return int(round(position * 99 / (size - 1))) if size > 1 else 0


def editor_edge_code(patch):
    levels = patch // 16
    border = np.concatenate([levels[0], levels[-1], levels[:, 0], levels[:, -1]])
    content = levels != np.bincount(border, minlength=16).argmax()
    if not content.any():
        return 0.
    rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
    height, width = patch.shape
    top, bottom = _editor_scaled(rows[0], height), _editor_scaled(rows[-1], height)
    left, right = _editor_scaled(cols[0], width), _editor_scaled(cols[-1], width)
    return float(top * 1_000_000 + bottom * 10_000 + left * 100 + right)


def editor_block_code(patch):
    levels = (patch // 16).tolist()
    height, width = patch.shape
    seen = [[False] * width for _ in range(height)]
    best = (0, 0., 0.)
    for row in range(height):
        for col in range(width):
            if seen[row][col]:
                continue
            level, stack, count, row_sum, col_sum = levels[row][col], [(row, col)], 0, 0, 0
            seen[row][col] = True
            while stack:
                r, c = stack.pop()
                count += 1; row_sum += r; col_sum += c
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc] and levels[nr][nc] == level:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if count > best[0]:
                best = (count, row_sum / count, col_sum / count)
    return float(_editor_scaled(best[1], height) * 100 + _editor_scaled(best[2], width))


def editor_load_image(path):
    """The image as a 100x100 uint8 grayscale array."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image columns need the Pillow package (pip install pillow)") from error
    with Image.open(path) as image:
        image = image.convert("RGBA")
        # Transparent areas read as white instead of whatever colour hides under them.
        flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
        flat.alpha_composite(image)
        return np.asarray(flat.convert("L").resize((EDITOR_IMAGE_SIZE, EDITOR_IMAGE_SIZE), Image.BILINEAR), dtype=np.uint8)


_EDITOR_IMAGE_CACHE = {}


def editor_image_codes(path, grid=None):
    """[PC, EC, BC] per grid cell, row by row, in editor_image_columns order."""
    rows, cols = editor_grid(grid)
    path = os.path.abspath(os.path.expanduser(str(path)))
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size, rows, cols)
    hit = _EDITOR_IMAGE_CACHE.get(key)
    if hit is None:
        pixels = editor_load_image(path)
        hit = []
        for top, bottom in editor_patch_spans(rows):
            for left, right in editor_patch_spans(cols):
                patch = pixels[top:bottom, left:right]
                hit += [editor_pair_code(patch), editor_edge_code(patch), editor_block_code(patch)]
        if len(_EDITOR_IMAGE_CACHE) >= 4096:
            _EDITOR_IMAGE_CACHE.clear()
        _EDITOR_IMAGE_CACHE[key] = hit
    return list(hit)


def editor_resolve_image(path, base=None):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) or not base else os.path.join(base, path)


# ───────────────────────── schema of a loaded table ─────────────────────────

def editor_schema_from_frame(frame, base=None):
    """What prediction needs to rebuild the encoded columns of an editor CSV, or None.

    {"version": 1, "base": folder of the CSV (relative image paths),
     "columns": [{"name", "kind", "columns": [encoded column names],
                  "codes": {string: code} (text), "length": L (chars), "grid": [r, c] (image)}]}"""
    header = next((str(column) for column in frame.columns if str(column).startswith(EDITOR_SCHEMA_PREFIX)), None)
    parsed = editor_parse_header(header) if header else None
    if not parsed:
        return None
    present = [str(column) for column in frame.columns]
    columns = []
    for column in parsed:
        name, kind = column["name"], column["kind"]
        entry = {"name": name, "kind": kind}
        if kind == EDITOR_TEXT:
            entry["columns"] = [name] if name in present else []
            codes, source = {}, editor_source_column(name)
            if name in present and source in present:
                numbers = pd.to_numeric(frame[name], errors="coerce")
                for text, code in zip(frame[source], numbers):
                    if isinstance(text, str) and text != "" and math.isfinite(code):
                        codes.setdefault(text, int(code) if float(code).is_integer() else float(code))
            entry["codes"] = codes
        elif kind == EDITOR_CHARS:
            entry["columns"] = [item for item in present if item.startswith(name + "[") and item.endswith("]") and item[len(name) + 1:-1].isdigit()]
            entry["columns"].sort(key=lambda item: int(item[len(name) + 1:-1]))
            entry["length"] = len(entry["columns"])
        elif kind == EDITOR_IMAGE:
            entry["grid"] = list(editor_grid(column.get("grid")))
            entry["columns"] = [item for item in editor_image_columns(name, entry["grid"]) if item in present]
        else:
            entry["columns"] = [name] if name in present else []
        columns.append(entry)
    return {"version": 1, "base": None if base is None else str(base), "columns": columns}


def editor_input_only_columns(schema):
    """Encoded image columns: images are inputs only."""
    return {name for column in (schema or {}).get("columns", ()) if column["kind"] == EDITOR_IMAGE for name in column["columns"]}


def editor_prepare_frame(frame, schema):
    """Add the encoded columns a model reads when the frame holds the raw text or image paths instead.

    A text column may hold strings (looked up, or given a new code by the
    same rule), a char-level or image column may be given under its own name
    or its ``#afpo:src:`` name.  Encoded columns already present are kept."""
    if not schema:
        return frame
    frame = frame.copy()
    for column in schema.get("columns", ()):
        name, kind, encoded = column["name"], column["kind"], column.get("columns") or []
        source = name if name in frame.columns else editor_source_column(name) if editor_source_column(name) in frame.columns else None
        if kind == EDITOR_TEXT:
            if name in frame.columns:
                if not pd.api.types.is_numeric_dtype(frame[name]):
                    frame[name] = [editor_lookup_code(None if _editor_missing(value) else value, column.get("codes") or {}) for value in frame[name]]
            elif source is not None:
                frame[name] = [editor_lookup_code(None if _editor_missing(value) else value, column.get("codes") or {}) for value in frame[source]]
        elif kind == EDITOR_CHARS:
            if source is not None and not all(item in frame.columns for item in encoded):
                codes = [editor_char_codes(None if _editor_missing(value) else value, len(encoded)) for value in frame[source]]
                for position, item in enumerate(encoded):
                    frame[item] = [float(row[position]) for row in codes]
        elif kind == EDITOR_IMAGE:
            if source is not None and not all(item in frame.columns for item in encoded):
                names = editor_image_columns(name, column.get("grid"))
                rows = []
                for value in frame[source]:
                    if _editor_missing(value) or not str(value).strip():
                        rows.append([float("nan")] * len(names))
                    else:
                        rows.append(editor_image_codes(editor_resolve_image(value, schema.get("base")), column.get("grid")))
                for position, item in enumerate(names):
                    if item in encoded:
                        frame[item] = [row[position] for row in rows]
    return frame


def _editor_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def editor_decode_row(outputs, schema):
    """{"<name> (text)": decoded} for the text columns among predicted outputs ({name: number})."""
    decoded = {}
    for column in (schema or {}).get("columns", ()):
        name, kind = column["name"], column["kind"]
        if kind == EDITOR_TEXT and isinstance(outputs.get(name), (int, float)) and not isinstance(outputs.get(name), bool):
            decoded[f"{name} (text)"] = editor_decode_text(outputs[name], column.get("codes") or {})
        elif kind == EDITOR_CHARS:
            encoded = column.get("columns") or []
            if encoded and all(isinstance(outputs.get(item), (int, float)) for item in encoded):
                decoded[f"{name} (text)"] = editor_decode_chars([outputs[item] for item in encoded])
    return decoded


def editor_decode_frame(frame, schema):
    """A copy of a predictions frame with a '<name> (text)' column per predicted text column."""
    if not schema:
        return frame
    frame = frame.copy()
    for column in schema.get("columns", ()):
        name, kind = column["name"], column["kind"]
        if kind == EDITOR_TEXT and name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]):
            frame[f"{name} (text)"] = [editor_decode_text(value, column.get("codes") or {}) for value in frame[name]]
        elif kind == EDITOR_CHARS:
            encoded = column.get("columns") or []
            if encoded and all(item in frame.columns for item in encoded):
                frame[f"{name} (text)"] = [editor_decode_chars(row) for row in frame[encoded].to_numpy(float)]
    return frame
