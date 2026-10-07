"""The GUI's CSV Editor: one table held by the server, edited a cell at a time.

The browser only ever holds the rows it shows (a page at a time), so a phone
can scroll a large file and a reloaded tab finds the table as it was.  Cells
keep their raw value (a string for text, a file path for an image); the
encoded columns are produced when the table is saved (see editor_format).
"""
import csv
import hashlib
import io
import math
import os
import threading
from pathlib import Path

import pandas as pd

from afpo_lib import editor_format as fmt

PAGE_ROWS = 100
MAX_CELLS = 5_000_000          # above this a file is too large to hold as an editable table
IMAGE_FOLDER = "afpo_images"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
WIDE_WARNING = 100             # encoded columns per editor column worth a warning
# A category column offers its values as a menu in every cell.  Past this many
# (a column of names, ids, free text) the browser gets a count instead and
# shows a plain text box: 100 rows x 98,000 options froze the page.
CATEGORY_MENU_LIMIT = 200
TRUE_WORDS, FALSE_WORDS = ("1", "true", "t", "yes", "y", "on"), ("0", "false", "f", "no", "n", "off")


def round_half_away(value):
    return int(math.copysign(math.floor(abs(value) + .5), value))


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not a finite number")
    return number


def coerce(kind, value):
    """A cell value as its column kind stores it; None for an empty cell."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if kind == fmt.EDITOR_BOOL:
        if isinstance(value, str):
            word = value.strip().lower()
            if word in TRUE_WORDS: return 1
            if word in FALSE_WORDS: return 0
        return 1 if _number(value) != 0 else 0
    if kind == fmt.EDITOR_INT:
        return round_half_away(_number(value))
    if kind == fmt.EDITOR_FLOAT:
        return _number(value)
    return str(value)


def _plain(value):
    """A stored cell as CSV text."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    return str(value)


class CsvEditor:
    def __init__(self, home):
        self.home = Path(home)
        self.lock = threading.RLock()
        self._reset()

    def _reset(self):
        self.columns, self.rows, self.path, self.dirty = [], [], None, False

    # ───────────── requests ─────────────
    def handle(self, body):
        """One editor operation ({"op": ...}); answers with the table's state and any requested rows."""
        op = str(body.get("op") or "view")
        handler = getattr(self, "_op_" + op, None)
        if handler is None:
            raise ValueError(f"Unknown editor operation {op!r}")
        with self.lock:
            extra = handler(body) or {}
            out = self._meta()
            if "start" in body:
                start = max(0, min(int(body.get("start") or 0), len(self.rows)))
                count = max(1, min(int(body.get("count") or PAGE_ROWS), 1000))
                out.update(start=start, rows=[self._public_row(row) for row in self.rows[start:start + count]])
            out.update(extra)
            return out

    def _meta(self):
        warnings = []
        columns = []
        for column in self.columns:
            info = {"name": column["name"], "kind": column["kind"]}
            if column["kind"] == fmt.EDITOR_CATEGORY:
                if len(column["categories"]) <= CATEGORY_MENU_LIMIT:
                    info["categories"] = list(column["categories"])
                else:
                    info["many"] = len(column["categories"])
            if column["kind"] == fmt.EDITOR_IMAGE:
                info["grid"] = list(column["grid"])
                width = 3 * column["grid"][0] * column["grid"][1]
                info["encoded"] = width
                if width >= WIDE_WARNING:
                    warnings.append(f"{column['name']}: a {column['grid'][0]}x{column['grid'][1]} grid makes {width} input columns per image. That is a lot of data for the search.")
            columns.append(info)
        return {"columns": columns, "total": len(self.rows), "path": None if self.path is None else str(self.path), "dirty": self.dirty,
                "home": str(self.home), "kinds": list(fmt.EDITOR_KIND_LABELS), "max_grid": fmt.EDITOR_MAX_GRID, "warnings": warnings}

    def _public_row(self, row):
        return [None if value is None else {"path": value, "name": os.path.basename(value)} if column["kind"] == fmt.EDITOR_IMAGE else value
                for column, value in zip(self.columns, row)]

    def _column(self, index):
        index = int(index)
        if not 0 <= index < len(self.columns):
            raise ValueError("Unknown column")
        return index

    def _check_name(self, name, skip=None):
        name = str(name or "").strip()
        if not name:
            raise ValueError("A column needs a name")
        if fmt.editor_reserved(name):
            raise ValueError(f"Column names cannot start with {fmt.EDITOR_PREFIX!r}")
        if any(column["name"] == name for index, column in enumerate(self.columns) if index != skip):
            raise ValueError(f"There already is a column named {name!r}")
        return name

    @staticmethod
    def _new_column(name, kind, grid=None):
        kind = int(kind)
        if not 0 <= kind < len(fmt.EDITOR_KINDS):
            raise ValueError("Unknown column kind")
        return {"name": name, "kind": kind, "grid": list(fmt.editor_grid(grid)), "categories": [], "codes": {}}

    # ───────────── operations ─────────────
    def _op_view(self, body):
        return None

    def _op_new(self, body):
        self._reset()

    def _op_load(self, body):
        self.load(body["path"], body.get("delimiter") or ",")

    def _op_add_column(self, body):
        column = self._new_column(self._check_name(body.get("name")), body.get("kind", fmt.EDITOR_FLOAT), body.get("grid"))
        self.columns.append(column)
        for row in self.rows:
            row.append(None)
        self.dirty = True

    def _op_update_column(self, body):
        index = self._column(body["index"])
        column = self.columns[index]
        if "name" in body:
            column["name"] = self._check_name(body["name"], skip=index)
        if "grid" in body:
            column["grid"] = list(fmt.editor_grid(body["grid"]))
        dropped = 0
        if "kind" in body and int(body["kind"]) != column["kind"]:
            kind = self._new_column("x", body["kind"])["kind"]
            for row in self.rows:
                try:
                    row[index] = coerce(kind, row[index])
                except (TypeError, ValueError):
                    dropped += row[index] is not None
                    row[index] = None
            if kind == fmt.EDITOR_IMAGE and self.path is not None:
                # A column of file names in a loaded CSV: they are relative to that file.
                for row in self.rows:
                    if row[index] is not None:
                        row[index] = fmt.editor_resolve_image(row[index], str(self.path.parent))
            column["kind"] = kind
            # Codes belong to the strings as a text column saw them; a column that
            # becomes one starts its own table when it is next saved.
            column["codes"] = {}
            column["categories"] = sorted({row[index] for row in self.rows if row[index] is not None}) if kind == fmt.EDITOR_CATEGORY else []
        self.dirty = True
        return {"dropped": dropped}

    def _op_delete_column(self, body):
        index = self._column(body["index"])
        del self.columns[index]
        for row in self.rows:
            del row[index]
        if not self.columns:
            self.rows = []
        self.dirty = True

    def _op_move_column(self, body):
        index = self._column(body["index"])
        target = max(0, min(len(self.columns) - 1, index + int(body.get("delta") or 0)))
        if target != index:
            self.columns.insert(target, self.columns.pop(index))
            for row in self.rows:
                row.insert(target, row.pop(index))
            self.dirty = True

    def _op_add_category(self, body):
        column = self.columns[self._column(body["col"])]
        value = str(body.get("value") or "").strip()
        if column["kind"] != fmt.EDITOR_CATEGORY or not value:
            raise ValueError("A category needs a name")
        if value not in column["categories"]:
            column["categories"].append(value)

    def _op_set_cell(self, body):
        """Set one cell; a row index equal to the row count starts a new row."""
        col, row = self._column(body["col"]), int(body["row"])
        if not 0 <= row <= len(self.rows):
            raise ValueError("Unknown row")
        column = self.columns[col]
        try:
            value = coerce(column["kind"], body.get("value"))
        except (TypeError, ValueError):
            raise ValueError(f"{body.get('value')!r} is not a valid {fmt.EDITOR_KIND_LABELS[column['kind']]} value")
        if row == len(self.rows):
            self.rows.append([None] * len(self.columns))
        self.rows[row][col] = value
        if column["kind"] == fmt.EDITOR_CATEGORY and value is not None and value not in column["categories"]:
            column["categories"].append(value)
        self.dirty = True
        return {"changed": {"row": row, "values": self._public_row(self.rows[row])}}

    def _op_add_rows(self, body):
        if not self.columns:
            raise ValueError("Add a column first")
        count = max(1, min(int(body.get("count") or 1), 100_000))
        at = len(self.rows) if body.get("at") is None else max(0, min(int(body["at"]), len(self.rows)))
        self.rows[at:at] = [[None] * len(self.columns) for _ in range(count)]
        self.dirty = True

    def _op_delete_rows(self, body):
        doomed = {int(row) for row in body.get("rows") or []}
        if doomed:
            self.rows = [row for index, row in enumerate(self.rows) if index not in doomed]
            self.dirty = True

    def _op_save(self, body):
        return {"saved": str(self.save(body.get("name"), bool(body.get("overwrite"))))}

    # ───────────── images ─────────────
    def image_folder(self):
        folder = self.home / IMAGE_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_image(self, name, data, row=None, col=None):
        """Store an uploaded image under its content hash, encode it, and (optionally) put it in a cell."""
        name = os.path.basename(str(name or "image.png")).strip() or "image.png"
        stem, suffix = os.path.splitext(name)
        if suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{name} is not an image file ({', '.join(IMAGE_SUFFIXES)})")
        digest = hashlib.sha1(data).hexdigest()[:10]
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)[:60] or "image"
        # The same picture uploaded again (under any name) reuses the stored file.
        target = next(iter(sorted(self.image_folder().glob(f"*-{digest}{suffix.lower()}"))), None) or self.image_folder() / f"{safe}-{digest}{suffix.lower()}"
        if not target.exists():
            target.write_bytes(data)
        with self.lock:
            grid = None
            if col is not None and str(col) != "":
                column = self.columns[self._column(col)]
                if column["kind"] != fmt.EDITOR_IMAGE:
                    raise ValueError(f"{column['name']} is not an image column")
                grid = column["grid"]
            try:
                codes = fmt.editor_image_codes(target, grid)
            except Exception as error:
                target.unlink(missing_ok=True)
                if isinstance(error, RuntimeError):
                    raise
                raise ValueError(f"{name} could not be read as an image: {error}")
            out = {"image": {"path": str(target), "name": target.name, "codes": codes}}
            if grid is not None:
                out.update(self._op_set_cell({"row": row, "col": col, "value": str(target)}))
                out.update(self._meta())
            return out

    def thumbnail(self, path, size=96):
        """A small PNG of an image the editor stored or the table refers to."""
        target = Path(str(path)).expanduser().resolve()
        with self.lock:
            known = target.parent == self.image_folder().resolve() or any(
                row[index] and Path(row[index]).expanduser().resolve() == target
                for index, column in enumerate(self.columns) if column["kind"] == fmt.EDITOR_IMAGE for row in self.rows)
        if not known or not target.is_file():
            raise FileNotFoundError("unknown image")
        from PIL import Image
        with Image.open(target) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
        return buffer.getvalue()

    # ───────────── load ─────────────
    def load(self, path, delimiter=","):
        path = Path(str(path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, engine="python" if len(delimiter) > 1 else "c")
        if frame.shape[0] * max(1, frame.shape[1]) > MAX_CELLS:
            raise ValueError(f"{path.name} has {frame.shape[0]:,} rows x {frame.shape[1]} columns; the editor holds at most {MAX_CELLS:,} cells")
        names = [str(name) for name in frame.columns]
        header = next((name for name in names if name.startswith(fmt.EDITOR_SCHEMA_PREFIX)), None)
        schema = fmt.editor_parse_header(header) if header else []
        columns, data, claimed = [], [], set()
        for entry in schema or []:
            name, kind = entry["name"], entry["kind"]
            source = fmt.editor_source_column(name)
            column = self._new_column(name, kind, entry.get("grid"))
            if kind in (fmt.EDITOR_TEXT, fmt.EDITOR_LABEL, fmt.EDITOR_CHARS, fmt.EDITOR_IMAGE):
                encoded = ([name] if kind in fmt.EDITOR_STRING_CODES else
                           [item for item in names if item.startswith(name + "[") and item.endswith("]") and item[len(name) + 1:-1].isdigit()] if kind == fmt.EDITOR_CHARS
                           else fmt.editor_image_columns(name, column["grid"]))
                if source not in names:
                    continue            # the raw values are gone: its columns load as plain ones below
                claimed.update(encoded)
                values = [value if value != "" else None for value in frame[source]]
                if kind == fmt.EDITOR_TEXT and name in names:
                    for text, code in zip(values, frame[name]):
                        try:
                            number = float(code)
                        except ValueError:
                            continue
                        if text is not None and math.isfinite(number):
                            column["codes"].setdefault(text, int(number) if number.is_integer() else number)
                if kind == fmt.EDITOR_IMAGE:
                    values = [None if value is None else str(Path(fmt.editor_resolve_image(value, str(path.parent))).resolve()) for value in values]
            elif name in names:
                claimed.add(name)
                values = self._parse_values(kind, frame[name])
                if kind == fmt.EDITOR_CATEGORY:
                    column["categories"] = sorted({value for value in values if value is not None})
            else:
                continue
            columns.append(column); data.append(values)
        for name in names:
            if name in claimed or fmt.editor_reserved(name) or any(column["name"] == name for column in columns):
                continue
            kind = self._infer_kind(frame[name])
            column = self._new_column(name, kind)
            values = self._parse_values(kind, frame[name])
            if kind == fmt.EDITOR_CATEGORY:
                column["categories"] = sorted({value for value in values if value is not None})
            columns.append(column); data.append(values)
        self.columns, self.rows = columns, [list(row) for row in zip(*data)] if data else []
        self.path, self.dirty = path, False

    @staticmethod
    def _parse_values(kind, series):
        out = []
        for value in series:
            try:
                out.append(coerce(kind, value))
            except (TypeError, ValueError):
                out.append(None)
        return out

    @staticmethod
    def _infer_kind(series):
        numbers = []
        for value in series:
            if value.strip() == "":
                continue
            try:
                numbers.append(_number(value))
            except ValueError:
                return fmt.EDITOR_CATEGORY
        if not numbers or any(not number.is_integer() for number in numbers):
            return fmt.EDITOR_FLOAT
        return fmt.EDITOR_BOOL if set(numbers) <= {0., 1.} else fmt.EDITOR_INT

    # ───────────── save ─────────────
    def table(self, base=None):
        """(header, rows of CSV text) with the encoded and reserved columns in place."""
        header, blocks, sources = [], [], []
        for index, column in enumerate(self.columns):
            name, kind = column["name"], column["kind"]
            values = [row[index] for row in self.rows]
            if kind in fmt.EDITOR_STRING_CODES:
                if kind == fmt.EDITOR_TEXT:
                    fmt.editor_assign_codes(values, column["codes"])      # kept for good once given
                else:
                    column["codes"] = fmt.editor_label_codes(values)      # the sorted vocabulary, renumbered on every save
                header.append(name); blocks.append([["" if value is None else _plain(column["codes"][value])] for value in values])
                sources.append((name, [_plain(value) for value in values]))
            elif kind == fmt.EDITOR_CHARS:
                length = max([len(value) for value in values if value is not None] + [1])
                header += fmt.editor_char_columns(name, length)
                blocks.append([[""] * length if value is None else [str(code) for code in fmt.editor_char_codes(value, length)] for value in values])
                sources.append((name, [_plain(value) for value in values]))
            elif kind == fmt.EDITOR_IMAGE:
                names = fmt.editor_image_columns(name, column["grid"])
                header += names
                block, stored = [], []
                for value in values:
                    if value is None:
                        block.append([""] * len(names)); stored.append("")
                        continue
                    try:
                        block.append([_plain(code) for code in fmt.editor_image_codes(value, column["grid"])])
                    except OSError:
                        raise ValueError(f"{name}: the image file {value} is missing")
                    stored.append(self._relative(value, base))
                blocks.append(block); sources.append((name, stored))
            else:
                header.append(name); blocks.append([[_plain(value)] for value in values])
        if len(set(header)) != len(header):
            clash = next(item for item in header if header.count(item) > 1)
            raise ValueError(f"Two columns would both be saved as {clash!r}; rename one of them")
        rows = [[cell for block in blocks for cell in block[r]] for r in range(len(self.rows))]
        for name, values in sources:
            header.append(fmt.editor_source_column(name))
            for row, value in zip(rows, values):
                row.append(value)
        header.append(fmt.editor_schema_header(self.columns))
        for row in rows:
            row.append("")
        return header, rows

    @staticmethod
    def _relative(value, base):
        """An image path relative to the CSV's folder when it lies inside it (the pair stays movable)."""
        if base is None:
            return value
        try:
            return str(Path(value).resolve().relative_to(Path(base).resolve()))
        except ValueError:
            return value

    def save(self, name=None, overwrite=False):
        if not self.columns:
            raise ValueError("The table has no columns yet")
        name = str(name or "").strip()
        if name:
            target = Path(name).expanduser()
            if not target.is_absolute():
                target = self.home / target.name
            if target.suffix.lower() not in (".csv", ".tsv", ".txt", ".dat"):
                target = target.with_name(target.name + ".csv")
        elif self.path is not None:
            target = self.path
        else:
            raise ValueError("Give the file a name")
        target = target.resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists")
        if not target.parent.is_dir():
            raise FileNotFoundError(f"No such folder: {target.parent}")
        header, rows = self.table(target.parent)
        temporary = target.with_name(target.name + ".part")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        temporary.replace(target)
        self.path, self.dirty = target, False
        return target
