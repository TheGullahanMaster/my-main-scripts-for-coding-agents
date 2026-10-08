#!/usr/bin/env python3
"""Reader for .tvq files (Neural Planner Software: EasyNN-plus and JustNN).

A .tvq is an MFC CArchive stream (0xFFFF new-class tags, 0x8000|index known-class
tags, 0x7FFF "big" tags, WORD list counts, length-prefixed CStrings, little-endian
doubles).  There is no public spec; this was reverse-engineered from sample files
and cross-checked against the EasyNN-plus manual.  Layout, in file order:

    header (0xD8 bytes, kept raw; u32 at 0 is 0 for EasyNN-plus, 99000/100000 for JustNN)
    CGraphPoint*   learning-curve points            (index, 3 doubles)
    CInOut*        columns                          (name, role, mode, lowest, highest)
    CExample*      rows                             (name, type, raw values, scaled values)
    CWeight*       connections                      (from, to, weight, ., delta)
    CNode*         neurons                          (kind, index, layer, first/last-in, 7 doubles)
    CHashText*     text cell <-> numeric code       (column, code, text)
    CImage*        image cells                      (name, 3 doubles, path)
    CAssociation*  association results              (meaning unconfirmed)
    CExtra*        settings / history lines         (type, value, text)
    CNote*         free-text notes
    COutput*, CFreeze*, CClone*, CReport*           training bookkeeping

Column role: 0 input, 1 output, 2 exclude, 3 serial.  Column mode: 0 real, 1 integer,
2 bool, 3 text, 4 image.  Example type: 0 training, 1 validating, 2 querying,
3 exclude.  Node layer: 0 input, 1..3 hidden layer, 9 output.  A node's net input is
bias + sum(weight * source activation) and its activation is the logistic sigmoid;
the 7 node doubles are net input, activation, error, delta, bias, bias derivative,
delta bias.  JustNN-era files (header version 99000/100000) store weights at half
scale: the effective weight is twice the stored one.

Fields whose meaning is not confirmed keep neutral names.  Only the standard library
is needed, except network()/forward() which use numpy.
"""
import csv
import math
import os
import re
import struct
import sys

HEADER_SIZE = 0xD8
MISSING = 3.0000000054977558e+38     # stored in place of an empty cell

ROLE = {0: 'input', 1: 'output', 2: 'exclude', 3: 'serial'}
MODE = {0: 'real', 1: 'integer', 2: 'bool', 3: 'text', 4: 'image'}
EXAMPLE = {0: 'training', 1: 'validating', 2: 'querying', 3: 'exclude'}
JUSTNN_VERSIONS = (99000, 100000)


class Incompatible(ValueError):
    """The file is valid but cannot be used for the requested purpose; the message says why."""


class Reader:
    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos
        self.map = [None]       # MFC archive object/class map, index 0 = null
        self.classes = {}       # map index -> class name

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from('<H', self.d, self.p)[0]; self.p += 2; return v

    def u32(self):
        v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v

    def f64(self):
        v = struct.unpack_from('<d', self.d, self.p)[0]; self.p += 8; return v

    def raw(self, n):
        v = self.d[self.p:self.p + n]; self.p += n; return v

    def cstr(self):
        n = self.u8()
        if n == 0xFF:
            n = self.u16()
            if n == 0xFFFF:
                n = self.u32()
        b = self.raw(n)
        if b'\x00' in b:       # a wide (UTF-16) string; an odd length means one stray byte in front
            try:
                return (b[1:] if len(b) % 2 else b).decode('utf-16-le').rstrip('\x00')
            except UnicodeDecodeError:
                pass
        return b.decode('latin-1')

    def count(self):
        n = self.u16()
        if n == 0xFFFF:
            n = self.u32()
        return n

    def count_or0(self):
        """List count, or 0 when the file has already ended."""
        return self.count() if self.p + 2 <= len(self.d) else 0

    def doubles(self):
        return [self.f64() for _ in range(self.u16())]

    def obj_tag(self, expect=None):
        """Read an MFC object tag and register the object in the map.  Returns the class name."""
        t = self.u16()
        if t == 0xFFFF:
            self.u16()                                  # schema
            name = self.raw(self.u16()).decode('ascii')
            self.map.append(('class', name))
            self.classes[len(self.map) - 1] = name
        elif t & 0x8000:
            name = self.classes[t & 0x7FFF]
        elif t == 0x7FFF:           # MFC "big" tag: map index no longer fits in 15 bits
            dw = self.u32()
            if not dw & 0x80000000:
                raise ValueError('object back-reference at 0x%x not supported' % (self.p - 6))
            name = self.classes[dw & 0x7FFFFFFF]
        else:
            raise ValueError('object back-reference at 0x%x not supported' % (self.p - 2))
        if expect and name != expect:
            raise ValueError('expected %s, got %s at 0x%x' % (expect, name, self.p))
        self.map.append(('obj', name))
        return name


def _zero_lists(r, n, what):
    """Skip n u16 list counts that are expected to be 0 (empty lists)."""
    for _ in range(n):
        c = r.u16()
        if c:
            raise ValueError('unsupported non-empty %s list (count %d) at 0x%x' % (what, c, r.p - 2))


def read_tvq(path):
    """Parse a .tvq file into a dict of plain lists/dicts.  Raises ValueError/struct.error on
    a file that is not understood (an empty or truncated file included)."""
    with open(path, 'rb') as fh:
        data = fh.read()
    if len(data) <= HEADER_SIZE:
        raise ValueError('not a .tvq file (%d bytes)' % len(data))
    r = Reader(data, HEADER_SIZE)
    out = {'file': path, 'size': len(data), 'header': data[:HEADER_SIZE],
           'version': struct.unpack_from('<I', data, 0)[0]}

    out['graph'] = []           # learning curve: (index, 3 doubles)
    for _ in range(r.count()):
        r.obj_tag('CGraphPoint')
        out['graph'].append((r.u32(), r.f64(), r.f64(), r.f64()))

    out['columns'] = []
    for _ in range(r.count()):
        r.obj_tag('CInOut')
        out['columns'].append({'name': r.cstr(), 'role': r.u32(), 'mode': r.u32(),
                               'min': r.f64(), 'max': r.f64()})

    out['examples'] = []        # the first row is usually the "Query" row
    for _ in range(r.count()):
        r.obj_tag('CExample')
        out['examples'].append({'name': r.cstr(), 'type': r.u32(),
                                'raw': r.doubles(), 'norm': r.doubles()})

    out['weights'] = []
    for _ in range(r.count()):
        r.obj_tag('CWeight')
        out['weights'].append({'h': (r.u32(), r.u32()), 'src': r.u32(), 'dst': r.u32(),
                               'w': r.f64(), 'f1': r.f64(), 'delta': r.f64()})

    out['nodes'] = []
    for _ in range(r.count()):
        r.obj_tag('CNode')
        out['nodes'].append({'kind': r.u32(), 'index': r.u32(), 'layer': r.u32(),
                             'w_first': r.u32(), 'w_end': r.u32(),
                             'd': [r.f64() for _ in range(7)]})

    out['hashtext'] = []        # numeric code -> string for text columns
    for _ in range(r.count()):
        r.obj_tag('CHashText')
        out['hashtext'].append({'col': r.u32(), 'hash': r.f64(), 'text': r.cstr()})

    out['images'] = []
    for _ in range(r.count()):
        r.obj_tag('CImage')
        out['images'].append({'name': r.cstr(), 'a': r.u32(),
                              'd': [r.f64() for _ in range(3)], 'path': r.cstr()})

    out['associations'] = []    # meaning not confirmed
    for _ in range(r.count()):
        r.obj_tag('CAssociation')
        out['associations'].append({'col': r.u32(), 'b': r.u32(), 'v': r.f64(), 'z': r.f64()})
    _zero_lists(r, 1, 'unknown')

    out['extras'] = []          # settings / history lines: (type, value, text)
    for _ in range(r.count()):
        r.obj_tag('CExtra')
        out['extras'].append({'type': r.u32(), 'value': r.f64(), 'text': r.cstr()})
    if r.p + 6 <= len(r.d):
        _zero_lists(r, 2, 'unknown')
    out['notes'] = []
    for _ in range(r.count_or0()):
        r.obj_tag('CNote')
        out['notes'].append({'a': r.u32(), 'b': r.u32(), 'c': r.u32(), 'text': r.cstr()})

    out['outputs'] = []
    for _ in range(r.count_or0()):
        r.obj_tag('COutput')
        out['outputs'].append({'k': r.u32(), 'value': r.f64(), 'example': r.u32()})

    out['freeze'] = []
    for _ in range(r.count_or0()):
        r.obj_tag('CFreeze')
        out['freeze'].append({'k': r.u32(), 'm1': r.u32(), 'example': r.u32()})

    out['clones'] = []
    for _ in range(r.count_or0()):
        r.obj_tag('CClone')
        out['clones'].append([r.u32() for _ in range(6)])
    if r.p + 2 <= len(r.d):
        _zero_lists(r, 1, 'unknown')

    out['reports'] = []
    for _ in range(r.count_or0()):
        r.obj_tag('CReport')
        out['reports'].append((r.u32(), r.u32()))

    # A few autosave (.ed / .bu) files carry a longer report section than this reads; the
    # data and network above are unaffected.
    out['tail'] = data[r.p:]
    return out


# ───────────────────────────── dataset ─────────────────────────────
def weight_scale(t):
    """Effective weight = stored weight * this (JustNN-era files store half-scale weights)."""
    return 2.0 if t['version'] in JUSTNN_VERSIONS else 1.0


# Strings that pandas reads back as a missing value; a label spelled like one would vanish from the CSV.
NA_WORDS = frozenset(('', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                      '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'))


def safe_text(s):
    """The label as written to a CSV: spellings that readers treat as 'missing' get a trailing '_'."""
    return s + '_' if s in NA_WORDS else s


def _number_like(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def column_names(t):
    """Unique, non-empty column names (blank / repeated names get a suffix)."""
    names, seen = [], set()
    for i, c in enumerate(t['columns']):
        base = c['name'].strip() or 'col%d' % (i + 1)
        name, n = base, 1
        while name in seen:
            n += 1
            name = '%s_%d' % (base, n)
        seen.add(name)
        names.append(name)
    return names


def dataset(t, row_types=(0, 1), text='decoded', roles=(0, 1, 2, 3), label_column=True):
    """The data grid as {'columns': [...], 'rows': [[...]], 'notes': [...]}.

    row_types selects example types (default: training + validating; querying and
    excluded rows are left out).  Text cells are decoded through the hash table
    (text='decoded'), kept as EasyNN's numeric codes (text='codes'), or kept as codes with
    the strings returned separately in result['sources'][column name] (text='both').
    Empty cells are ''.  Image
    columns are dropped (their pictures are not stored in the file).  Each column entry
    has name, role, mode and index (its position in the file); a leading 'example'
    column holds row names when they are real labels rather than #0, #1, ..."""
    names = column_names(t)
    labels = {(h['col'], h['hash']): safe_text(h['text']) for h in t['hashtext'] if h['text']}
    renamed = {h['text'] for h in t['hashtext'] if h['text'] in NA_WORDS and h['text']}
    keep = [i for i, c in enumerate(t['columns']) if c['mode'] != 4 and c['role'] in roles]
    notes = []
    dropped = [names[i] for i, c in enumerate(t['columns']) if c['mode'] == 4]
    if dropped:
        notes.append('Image columns left out (the pictures are not stored in the file): ' + ', '.join(dropped))
    sel = [e for e in t['examples'] if e['type'] in row_types]
    label_col = label_column and any(not re.fullmatch(r'(#|T:|V:|Q:)?\d*', e['name'].strip()) for e in sel)
    # A text column is written as text only if reading the text back gives the same codes: every cell has a
    # label, labels and codes pair one-to-one, and no label looks like a number (a reader would take it for a code).
    # Otherwise the column keeps EasyNN's codes, which hashed-text columns accept as they are.
    as_codes = []
    if text == 'decoded':
        for i in keep:
            if t['columns'][i]['mode'] != 3:
                continue
            by_text, by_code, ok = {}, {}, True
            for e in sel:
                v = e['raw'][i]
                if v == MISSING:
                    continue
                s = labels.get((i, v))
                if s is None or _number_like(s) or by_text.setdefault(s, v) != v or by_code.setdefault(v, s) != s:
                    ok = False
                    break
            if not ok:
                as_codes.append(i)
    unresolved = 0
    rows = []
    sources = {names[i]: [] for i in keep if t['columns'][i]['mode'] == 3} if text == 'both' else {}
    for e in sel:
        cells = [e['name']] if label_col else []
        for i in keep:
            v = e['raw'][i]
            mode = t['columns'][i]['mode']
            if v == MISSING:
                cells.append('')
                if mode == 3 and text == 'both':
                    sources[names[i]].append('')
            elif mode == 3 and text != 'codes' and i not in as_codes:
                s = labels.get((i, v))
                if s is None:
                    unresolved += 1
                if text == 'both':
                    cells.append(v)
                    sources[names[i]].append('' if s is None else s)
                else:
                    cells.append('' if s is None else s)
            elif mode in (1, 2) and v == int(v):
                cells.append(int(v))
            else:
                cells.append(v)
        rows.append(cells)
    if unresolved:
        notes.append('%d text cells held a code with no matching text; %s' % (
            unresolved, 'the code was kept' if text == 'both' else 'they were left empty'))
    if as_codes:
        notes.append('text columns written as EasyNN codes because their labels are ambiguous or incomplete: '
                     + ', '.join(names[i] for i in as_codes))
    if renamed and text != 'codes':
        notes.append('labels that CSV readers treat as missing were renamed with a trailing _: ' + ', '.join(sorted(renamed)))
    cols = [{'name': names[i], 'role': t['columns'][i]['role'], 'mode': t['columns'][i]['mode'], 'index': i}
            for i in keep]
    if label_col:
        cols.insert(0, {'name': 'example', 'role': 2, 'mode': 3, 'index': -1})
    skipped = len(t['examples']) - len(sel)
    if skipped:
        notes.append('%d querying/excluded rows left out' % skipped)
    return {'columns': cols, 'rows': rows, 'notes': notes, 'sources': sources}


def codebook(t, col):
    """{text: code} of one text column (first code wins when a text appears twice)."""
    codes = {}
    for h in t['hashtext']:
        key = safe_text(h['text'])
        if h['col'] == col and h['text'] and key not in codes:
            codes[key] = int(h['hash']) if float(h['hash']).is_integer() else h['hash']
    return codes


def write_csv(ds, path):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow([c['name'] for c in ds['columns']])
        w.writerows(ds['rows'])
    return path


# ───────────────────────────── network ─────────────────────────────
def network(t):
    """The trained network as dense layers: {'sizes', 'weights': [(out, in) arrays], 'biases',
    'input_cols', 'output_cols', 'sparse', 'notes', 'scale'}.  Weights already include
    weight_scale(t).  Raises Incompatible unless it is a plain feed-forward net whose every
    connection goes from one layer to the next (missing connections become zero weights)."""
    import numpy as np
    nodes, weights, cols = t['nodes'], t['weights'], t['columns']
    if not nodes:
        raise Incompatible('the file has no trained network')
    for pos, n in enumerate(nodes):
        if n['index'] != pos:
            raise Incompatible('node numbering is not sequential')
        if (n['kind'], n['layer'] == 0, n['layer'] == 9) not in ((0, True, False), (1, False, False), (2, False, True)):
            raise Incompatible('unexpected node kind %d in layer %d' % (n['kind'], n['layer']))
    ids = sorted({n['layer'] for n in nodes})
    hidden_ids = [i for i in ids if i not in (0, 9)]
    if 0 not in ids or 9 not in ids or hidden_ids != list(range(1, len(hidden_ids) + 1)):
        raise Incompatible('layers are not input, consecutive hidden layers, output (found %s)' % ids)
    order = [0] + hidden_ids + [9]
    members = {lid: [i for i, n in enumerate(nodes) if n['layer'] == lid] for lid in order}
    where = {i: (lid, k) for lid in order for k, i in enumerate(members[lid])}
    scale = weight_scale(t)
    mats, biases, present, total = [], [], 0, 0
    for li in range(1, len(order)):
        lid, prev = order[li], order[li - 1]
        W = np.zeros((len(members[lid]), len(members[prev])))
        b = np.zeros(len(members[lid]))
        for k, ni in enumerate(members[lid]):
            n = nodes[ni]
            b[k] = n['d'][4]
            for wi in range(n['w_first'], n['w_end']):
                w = weights[wi]
                if w['dst'] != ni:
                    raise Incompatible('connection %d does not point at its node' % wi)
                slid, sk = where[w['src']]
                if slid != prev:
                    raise Incompatible('node %d takes input from layer %d, not the layer before it (skip connection)'
                                       % (ni, slid))
                W[k, sk] += w['w'] * scale
                present += 1
        total += W.size
        mats.append(W)
        biases.append(b)
    if not all(np.isfinite(m).all() for m in mats + biases):
        raise Incompatible('the network contains NaN / infinite values (its training diverged)')
    in_cols = [i for i, c in enumerate(cols) if c['role'] == 0 and c['mode'] != 4]
    out_cols = [i for i, c in enumerate(cols) if c['role'] == 1 and c['mode'] != 4]
    if len(in_cols) != len(members[0]) or len(out_cols) != len(members[9]):
        raise Incompatible('%d input / %d output nodes do not match %d input / %d output columns'
                           % (len(members[0]), len(members[9]), len(in_cols), len(out_cols)))
    notes = []
    if present < total:
        notes.append('%d of %d possible connections are present; the rest are zero weights' % (present, total))
    return {'sizes': [len(members[l]) for l in order], 'weights': mats, 'biases': biases,
            'input_cols': in_cols, 'output_cols': out_cols, 'sparse': present < total, 'notes': notes,
            'scale': scale}


def forward(net, x):
    """Run the logistic network on scaled inputs x (rows, inputs); returns (rows, outputs)."""
    import numpy as np
    a = np.asarray(x, dtype=float)
    for W, b in zip(net['weights'], net['biases']):
        a = 1.0 / (1.0 + np.exp(-np.clip(a @ W.T + b, -60.0, 60.0)))
    return a


def training_arrays(t, net, limit=2000):
    """(X, Y) scaled input / target arrays of the training rows with no empty cell."""
    import numpy as np
    ins, outs = net['input_cols'], net['output_cols']
    rows = [e for e in t['examples'] if e['type'] == 0 and all(e['raw'][i] != MISSING for i in ins + outs)]
    rows = rows[:limit]
    X = np.array([[e['norm'][i] for i in ins] for e in rows], dtype=float).reshape(len(rows), len(ins))
    Y = np.array([[e['norm'][i] for i in outs] for e in rows], dtype=float).reshape(len(rows), len(outs))
    return X, Y


def network_quality(t, net):
    """How well the stored network reproduces its own training targets: RMSE divided by the RMSE
    of always predicting the column means (well below 1 = the network learned something).
    None when there are too few rows."""
    X, Y = training_arrays(t, net)
    if len(X) < 3:
        return None
    base = math.sqrt(float(((Y - Y.mean(0)) ** 2).mean()))
    rmse = math.sqrt(float(((forward(net, X) - Y) ** 2).mean()))
    return {'rows': int(len(X)), 'rmse': rmse, 'baseline': base, 'ratio': (rmse / base) if base > 0 else None}


def network_report(t):
    """Compatibility summary used by the GUIs.  Never raises for an understood file."""
    rep = {'has_network': bool(t['nodes']), 'compatible': False, 'reasons': [], 'notes': []}
    if not t['nodes']:
        rep['reasons'].append('the file has no trained network')
        return rep
    try:
        net = network(t)
    except Incompatible as e:
        rep['reasons'].append(str(e))
        return rep
    for i in net['input_cols'] + net['output_cols']:
        c = t['columns'][i]
        if c['max'] <= c['min']:
            rep['reasons'].append("column '%s' has no range" % c['name'])
    rep['sizes'] = net['sizes']
    rep['notes'] = net['notes']
    if any(t['columns'][i]['mode'] == 3 for i in net['input_cols'] + net['output_cols']):
        rep['notes'].append('text columns reach the network as EasyNN hash codes; they import as hashed text')
    rep['weight_scale'] = net['scale']
    q = network_quality(t, net)
    rep['quality'] = q
    if q and q['ratio'] is not None and q['ratio'] >= 1.0:
        rep['notes'].append('the stored network does not predict its own training rows better than the column '
                            'means do; it may be untrained or half-trained')
    rep['compatible'] = not rep['reasons']
    return rep


# ───────────────────────────── command line ─────────────────────────────
def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    csv_dir = args.pop() if '--csv' in argv else None      # --csv <dir>: one CSV of the data per file
    for path in args:
        try:
            t = read_tvq(path)
        except Exception as e:
            print('%-45s FAIL %s' % (path[:45], e))
            continue
        line = '%-45s cols=%d rows=%d nodes=%d weights=%d tail=%s' % (
            path[:45], len(t['columns']), len(t['examples']), len(t['nodes']), len(t['weights']),
            t['tail'].hex()[:8] or '-')
        if t['nodes']:
            rep = network_report(t)
            line += ' net=%s' % ('ok' if rep['compatible'] else 'no: ' + '; '.join(rep['reasons'])[:60])
        print(line)
        if csv_dir:
            write_csv(dataset(t), os.path.join(csv_dir, os.path.splitext(os.path.basename(path))[0] + '.csv'))


if __name__ == '__main__':
    main(sys.argv[1:])
