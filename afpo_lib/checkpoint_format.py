"""Safe JSON encoding for AFPO checkpoint data."""
import math

import numpy as np
import pandas as pd


def to_json_value(value):
    """Encode supported state without executable deserialization hooks."""
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
