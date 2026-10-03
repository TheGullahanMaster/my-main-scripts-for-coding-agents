"""Safe JSON encoding for AFPO checkpoint data."""
import base64
import math
import zlib

import numpy as np
import pandas as pd


# Plain numeric dtypes only: object, string and structured arrays never take
# the binary path, so decoding cannot build anything but numbers.
BINARY_KINDS = "biuf"


_COMPRESS_BLOCK = 1 << 24


def _binary_value(array):
    data = np.ascontiguousarray(array)
    flat = data.reshape(-1).view(np.uint8)
    # Fast deflate, fed in blocks so no second full copy of the array is made.
    # One-hot and integer-valued columns shrink several-fold; dense floats ~20%.
    compressor = zlib.compressobj(1)
    packed = b"".join([*(compressor.compress(flat[start:start + _COMPRESS_BLOCK]) for start in range(0, len(flat), _COMPRESS_BLOCK)),
                       compressor.flush()])
    return {"__afpo_type__": "ndarray_zlib", "dtype": data.dtype.str, "shape": list(array.shape),
            "data": base64.b64encode(packed).decode("ascii")}


def _binary_array(value):
    dtype = np.dtype(value["dtype"])
    if dtype.kind not in BINARY_KINDS:
        raise ValueError(f"Checkpoint binary array has unsupported dtype {dtype}")
    shape = [int(size) for size in value["shape"]]
    size = math.prod(shape) * dtype.itemsize
    decompressor = zlib.decompressobj()
    # Bounded: a corrupt or hostile entry cannot inflate past its stated shape.
    raw = decompressor.decompress(base64.b64decode(value["data"], validate=True), size + 1)
    if len(raw) != size or not decompressor.eof:
        raise ValueError("Checkpoint binary array size does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def to_json_value(value):
    """Encode supported state without executable deserialization hooks."""
    if isinstance(value, np.ndarray) and value.dtype.kind in BINARY_KINDS:
        # Raw bytes (the dtype string records their byte order), compressed and
        # base64-wrapped: bit-exact (NaN and inf included), far smaller than
        # JSON number lists, and no per-element Python floats while saving or
        # loading, which is what made large-dataset checkpoints need gigabytes.
        return _binary_value(value)
    if isinstance(value, np.ndarray):
        return {"__afpo_type__": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return to_json_value(value.item())
    if isinstance(value, tuple):
        return {"__afpo_type__": "tuple", "items": [to_json_value(item) for item in value]}
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {"__afpo_type__": "dict", "items": [[to_json_value(key), to_json_value(item)] for key, item in value.items()]}
    if isinstance(value, pd.DataFrame):
        return {"__afpo_type__": "dataframe", "columns": list(value.columns), "data": to_json_value(value.to_dict(orient="list"))}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__afpo_type__": "float", "value": "inf" if value > 0 else "-inf" if value < 0 else "nan"}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Checkpoint cannot safely encode {type(value).__name__}")


def from_json_value(value):
    """Restore values produced by :func:`to_json_value`."""
    if isinstance(value, list):
        return [from_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__afpo_type__")
    if kind == "ndarray_zlib":
        return _binary_array(value)
    if kind == "ndarray":
        return np.asarray(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if kind == "tuple":
        return tuple(from_json_value(item) for item in value["items"])
    if kind == "dict":
        return {from_json_value(key): from_json_value(item) for key, item in value["items"]}
    if kind == "dataframe":
        return pd.DataFrame(from_json_value(value["data"]), columns=value["columns"])
    if kind == "float":
        return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[value["value"]]
    return {key: from_json_value(item) for key, item in value.items()}
