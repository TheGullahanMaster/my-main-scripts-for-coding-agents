"""Declared, data-only soft constraints for AFPO scoring."""
from dataclasses import dataclass, field

import numpy as np


def _as_bounds(value):
    if value is None:
        return (None, None)
    if isinstance(value, dict):
        return value.get("min"), value.get("max")
    return tuple(value) if isinstance(value, (list, tuple)) and len(value) == 2 else (None, None)


@dataclass
class ConstraintEvaluator:
    profile: str = "general"
    metadata: dict = field(default_factory=dict)

    @property
    def active(self):
        return self.profile != "general" and bool(self.metadata)

    def validate(self, n_features, cats, output_names):
        outputs = self.metadata.get("outputs", self.metadata)
        if not isinstance(outputs, dict):
            raise ValueError("Constraint metadata must be an object keyed by output")
        names = set(output_names)
        for name, spec in outputs.items():
            if name not in names and not (isinstance(name, str) and name.isdigit() and int(name) < len(output_names)):
                raise ValueError(f"Constraint metadata names unknown output {name!r}")
            if not isinstance(spec, dict):
                raise ValueError(f"Constraint for {name!r} must be an object")
            output_index = output_names.index(name) if name in names else int(name)
            mono = spec.get("monotonic", {})
            if mono and cats[output_index] is not None:
                raise ValueError("Monotonic constraints require a numeric output")
            if mono and not isinstance(mono, dict):
                raise ValueError("monotonic must map feature indices to directions")
            for feature, direction in mono.items():
                try:
                    index = int(feature)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid monotonic feature {feature!r}") from None
                if not 0 <= index < n_features:
                    raise ValueError(f"Monotonic feature {index} is outside encoded inputs")
                if str(direction).lower() not in ("increasing", "+", "1", "decreasing", "-", "-1"):
                    raise ValueError(f"Invalid monotonic direction {direction!r}")
            for pair in spec.get("permutation_symmetry", spec.get("symmetry", [])):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("Symmetry entries must be feature-index pairs")
                try:
                    a, b = map(int, pair)
                except (TypeError, ValueError):
                    raise ValueError("Symmetry feature indices must be integers") from None
                if not 0 <= a < n_features or not 0 <= b < n_features:
                    raise ValueError("Symmetry feature is outside encoded inputs")

    def violations(self, prediction, X=None, output_names=(), predict=None, cats=None):
        result = np.zeros(prediction.shape[1], dtype=float)
        outputs = self.metadata.get("outputs", self.metadata)
        for j in range(prediction.shape[1]):
            spec = outputs.get(output_names[j], outputs.get(str(j), {})) if isinstance(outputs, dict) else {}
            if not isinstance(spec, dict):
                continue
            p = prediction[:, j]
            terms = []
            low, high = _as_bounds(spec.get("bounds"))
            if spec.get("nonnegative"):
                low = 0. if low is None else max(0., float(low))
            if low is not None:
                terms.append(np.mean(np.maximum(float(low) - p, 0.) / (1 + abs(float(low)))))
            if high is not None:
                terms.append(np.mean(np.maximum(p - float(high), 0.) / (1 + abs(float(high)))))
            for feature, direction in (spec.get("monotonic") or {}).items():
                if predict is None:
                    raise ValueError("Monotonic constraints require a prediction callback")
                index = int(feature)
                values = np.unique(X[:, index])
                successors = np.searchsorted(values, X[:, index], side="right")
                comparable = successors < len(values)
                if not np.any(comparable):
                    terms.append(0.)
                    continue
                alternate_inputs = X[comparable].copy()
                alternate_inputs[:, index] = values[successors[comparable]]
                # Change only this feature; correlations between observed
                # covariates are not partial monotonicity violations.
                delta = predict(alternate_inputs)[:, j] - p[comparable]
                terms.append(float(np.mean(np.maximum(-delta if str(direction).lower() in ("increasing", "+", "1") else delta, 0.))))
            for pair in spec.get("permutation_symmetry", spec.get("symmetry", [])):
                if predict is None:
                    raise ValueError("Symmetry constraints require a prediction callback")
                a, b = map(int, pair)
                swapped = X.copy()
                swapped[:, [a, b]] = swapped[:, [b, a]]
                alternate = predict(swapped)[:, j]
                terms.append(float(np.mean(p != alternate)) if cats is not None and cats[j] is not None
                             else float(np.mean(np.abs(p - alternate) / (1 + np.abs(p)))))
            result[j] = float(np.mean(terms)) if terms else 0.
        return result

    def describe(self):
        return {"profile": self.profile, "metadata": self.metadata, "soft_objective": self.active}
