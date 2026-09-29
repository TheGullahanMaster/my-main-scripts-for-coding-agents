#!/usr/bin/env python3
"""Interactive AFPO/NSGA-II symbolic regression.

Only numpy and pandas are required; matplotlib is optional and only used by
the exported model.  Start it with ``python afpo_symbolic_regression.py``.
Use ``--max-generations N`` for a bounded unattended run (useful in CI).
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from html import escape as xml_escape
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# All linear algebra here is tiny (two-column least squares, vector dots), so
# a BLAS thread pool only adds spin/sync overhead -- and with one pool per
# worker process it oversubscribes every core (measured ~38x slower serially
# at 20k rows).  Parallelism comes from the worker processes instead.  Must be
# set before numpy loads; an explicit user setting still wins.
for _blas_threads in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_threads,"1")

import numpy as np
import pandas as pd
from afpo_lib.checkpoint_format import from_json_value as _from_json_checkpoint_value
from afpo_lib.checkpoint_format import to_json_value as _json_checkpoint_value
from afpo_lib.constraints import ConstraintEvaluator

EPS, CLIP = 1e-12, 1e12
rng = random.Random()
_NEXT_LINEAGE_ID = 0
INVALID_DIAGNOSTICS = {}
REWRITE_CACHE = {}
PROFILES=("general","physics","chemistry","biology","mathematics")
INTERACTION_DISCOVERY_POLICY={
    "version":1,
    "feature_limit_per_output":12,
    "operators":["+","-","*","/"],
    "unary_operators":["square","sqrt","inv","log1p","abs"],
    "minimum_relative_loss_reduction":.10,
    "max_fragments_per_output":8,
    "max_fragments_per_family":2,
    "admission":"train-fitted affine or ratio-multiplier readout must improve both training and held-out validation",
}
_WORKER_EVALUATION_CONTEXT = None
_WORKER_EPOCH = None

def compile_constraints(profile="general", metadata=None):
    if profile not in PROFILES: raise ValueError("Unknown profile: %s" % profile)
    return ConstraintEvaluator(profile,metadata or {})

def discover_hypotheses(X, Y, feature_names=(), output_names=()):
    """Training-only, conservative advisory structure candidates.

    Nothing returned here alters a search unless a caller explicitly elects to
    use it; the record is primarily model-card evidence.
    """
    findings=[]
    for j in range(Y.shape[1]):
        y=Y[:,j]; name=output_names[j] if j<len(output_names) else str(j)
        findings.append({"output":name,"kind":"bounds","evidence":{"min":float(np.min(y)),"max":float(np.max(y))},"validated":False})
        for i in range(X.shape[1]):
            corr=np.corrcoef(X[:,i],y)[0,1] if len(X)>2 and np.std(X[:,i])>EPS and np.std(y)>EPS else 0.
            if np.isfinite(corr) and abs(corr)>=.9:
                findings.append({"output":name,"kind":"monotonic_candidate","feature":feature_names[i] if i<len(feature_names) else i,"evidence":{"pearson":float(corr)},"validated":False})
    return findings

def _fit_piecewise_multiplier(values, target, thresholds, minimum_support):
    """Fit an ordered, multiplicative regime around a nonzero ratio fragment."""
    regions=np.searchsorted(thresholds,values,side="right")
    factors=[]
    for region in range(len(thresholds)+1):
        mask=regions==region
        if int(np.sum(mask))<minimum_support or np.any(np.abs(values[mask])<EPS): return None
        factor=float(np.median(target[mask]/values[mask]))
        if not np.isfinite(factor): return None
        factors.append(factor)
    return tuple(factors)

def _predict_piecewise_multiplier(values, thresholds, factors):
    regions=np.searchsorted(thresholds,values,side="right")
    result=np.empty(len(values),float)
    for region,factor in enumerate(factors):
        mask=regions==region; result[mask]=clean(factor*values[mask])
    return result

def _ratio_threshold_sets(values, target, minimum_support, limit=6):
    """Locate the strongest ordered changes in a target-to-ratio multiplier."""
    if np.any(np.abs(values)<EPS): return []
    order=np.argsort(values,kind="stable"); ordered_values=values[order]
    multipliers=target[order]/ordered_values
    jumps=np.abs(np.diff(np.log(np.abs(multipliers)+EPS)))
    choices=[]
    for index in np.argsort(-jumps,kind="stable"):
        left=int(index)+1; right=len(values)-left
        if left<minimum_support or right<minimum_support: continue
        if ordered_values[index]==ordered_values[index+1]: continue
        choices.append(float((ordered_values[index]+ordered_values[index+1])/2))
        if len(choices)>=limit: break
    choices=sorted(set(choices)); result=[(value,) for value in choices]
    result += [(left,right) for index,left in enumerate(choices) for right in choices[index+1:]]
    return result

def discover_interaction_fragments(X, Y, feature_names=(), output_names=(), operators=(), Xv=None, Yv=None):
    """Screen bounded pairwise arithmetic fragments and admit only held-out wins.

    The probe fits only affine readouts, except that ratios may earn a bounded
    multiplicative-regime check.  It proposes fragments, never a complete
    equation, so evolution remains responsible for composition.
    """
    policy=dict(INTERACTION_DISCOVERY_POLICY)
    report={"policy":policy,"status":"validation_required","screened":0,"accepted":[]}
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim!=2 or Y.ndim!=2 or len(X)<4 or X.shape[1]<2:
        report["status"]="insufficient_training_data"; return report
    if Xv is None or Yv is None:
        return report
    Xv=np.asarray(Xv,float); Yv=np.asarray(Yv,float)
    if Xv.ndim!=2 or Yv.ndim!=2 or len(Xv)<3 or Xv.shape[1]!=X.shape[1] or Yv.shape[1]!=Y.shape[1]:
        report["status"]="insufficient_validation_data"; return report
    available=set(operators); enabled=available.intersection(policy["operators"])
    if not enabled:
        report["status"]="no_supported_operators"; return report
    accepted=[]
    for output in range(Y.shape[1]):
        y=Y[:,output]; yv=Yv[:,output]
        baseline_value=float(np.median(y))
        baseline_train=robust_loss(np.full(len(y),baseline_value),y)
        baseline_validation=robust_loss(np.full(len(yv),baseline_value),yv)
        if baseline_train<EPS or baseline_validation<EPS:
            continue
        ranked=[]
        for feature in range(X.shape[1]):
            values=X[:,feature]
            corr=(np.corrcoef(values,y)[0,1] if np.std(values)>EPS and np.std(y)>EPS else 0.)
            ranked.append((-abs(float(corr)) if np.isfinite(corr) else 0.,feature))
        features=[feature for _,feature in sorted(ranked)[:policy["feature_limit_per_output"]]]
        candidates=[]
        for feature in features:
            leaf=("x",feature)
            for operator,kind,family in (("square","square","power"),("sqrt","square_root","root"),
                                         ("inv","reciprocal","reciprocal"),("log1p","log1p","log"),
                                         ("abs","absolute","absolute")):
                if operator in available and operator in policy["unary_operators"]:
                    candidates.append((kind,(operator,leaf),family))
        for position,left in enumerate(features):
            for right in features[position+1:]:
                x_left=("x",left); x_right=("x",right)
                if "+" in enabled: candidates.append(("sum",("+",x_left,x_right),"additive"))
                if "-" in enabled: candidates.append(("difference",("-",x_left,x_right),"difference"))
                if "*" in enabled: candidates.append(("product",("*",x_left,x_right),"product"))
                if "/" in enabled:
                    candidates.append(("ratio",("/",x_left,x_right),"ratio"))
                    candidates.append(("inverse_ratio",("/",x_right,x_left),"ratio"))
        for kind,tree,family in candidates:
            report["screened"]+=1
            try:
                values=evaluate(tree,X); values_v=evaluate(tree,Xv)
            except (ArithmeticError, IndexError, ValueError):
                continue
            if not np.isfinite(values).all() or np.std(values)<EPS:
                continue
            scale,offset=affine(values,y)
            train_loss=robust_loss(clean(scale*values+offset),y)
            validation_loss=robust_loss(clean(scale*values_v+offset),yv)
            readout="affine"; thresholds=()
            if kind in {"ratio","inverse_ratio"}:
                minimum_support=max(4,len(values)//8)
                for candidate_thresholds in _ratio_threshold_sets(values,y,minimum_support):
                    factors=_fit_piecewise_multiplier(values,y,candidate_thresholds,minimum_support)
                    if factors is None: continue
                    candidate_train=robust_loss(_predict_piecewise_multiplier(values,candidate_thresholds,factors),y)
                    if candidate_train>=train_loss: continue
                    candidate_validation=robust_loss(_predict_piecewise_multiplier(values_v,candidate_thresholds,factors),yv)
                    train_loss,validation_loss=candidate_train,candidate_validation
                    readout="piecewise_multiplier"; thresholds=candidate_thresholds
            train_gain=(baseline_train-train_loss)/baseline_train
            validation_gain=(baseline_validation-validation_loss)/baseline_validation
            if min(train_gain,validation_gain)<policy["minimum_relative_loss_reduction"]:
                continue
            used=sorted({left for left in range(X.shape[1]) if ("x",left) in walk_tree(tree)})
            record={"tree":tree,"output":output_names[output] if output<len(output_names) else str(output),
                             "kind":kind,"family":family,"features":[feature_names[index] if index<len(feature_names) else str(index) for index in used],
                             "train_loss":float(train_loss),"validation_loss":float(validation_loss),
                             "train_gain":float(train_gain),"validation_gain":float(validation_gain),
                             "readout":readout,"thresholds":list(thresholds),
                             "contribution":float(min(baseline_train-train_loss,baseline_validation-validation_loss))}
            accepted.append(record)
            if thresholds and {"gt","lt"}.issubset(available):
                for threshold in thresholds:
                    predicate=dict(record); predicate["tree"]=("gt",tree,("c",float(threshold)))
                    predicate["kind"]=f"{kind}_threshold"; predicate["family"]="predicate"; predicate["parent_family"]=family; predicate["threshold"] = float(threshold)
                    accepted.append(predicate)
    accepted.sort(key=lambda item:(-min(item["train_gain"],item["validation_gain"]),item["validation_loss"],repr(item["tree"])))
    limited=[]; counts={}; family_counts={}
    for item in accepted:
        if counts.get(item["output"],0)>=policy["max_fragments_per_output"]: continue
        family_key=(item["output"],item["family"])
        if family_counts.get(family_key,0)>=policy["max_fragments_per_family"]: continue
        limited.append(item); counts[item["output"]]=counts.get(item["output"],0)+1; family_counts[family_key]=family_counts.get(family_key,0)+1
    report["status"]="ok"; report["accepted"]=limited
    return report

def safe_rewrite(tree):
    """Bounded equivalence cache for semantics-preserving canonical rewrites."""
    key=repr(tree)
    if key not in REWRITE_CACHE:
        if len(REWRITE_CACHE)>=4096: REWRITE_CACHE.clear()
        REWRITE_CACHE[key]=simplify_tree(tree)
    return REWRITE_CACHE[key]

def next_lineage_id():
    global _NEXT_LINEAGE_ID
    value=_NEXT_LINEAGE_ID; _NEXT_LINEAGE_ID+=1
    return value

# Every selectable operator from OPS.md.  All operators use guarded,
# vectorised semantics.
OPS: dict[str, tuple[int, int]] = {
    "+": (2, 1), "-": (2, 1), "*": (2, 1), "delta": (2, 1), "/": (2, 2),
    "max": (2, 1), "min": (2, 1), "pow": (2, 3), "hypot": (2, 2), "distance_2": (2, 2),
    "log_base": (2, 3), "exp_decay": (2, 3), "rbf": (2, 3), "atan2": (2, 2),
    "harmonic": (2, 2), "geometric": (2, 2), "mod": (2, 2), "copysign": (2, 1),
    "floordiv": (2, 2), "gt": (2, 2), "lt": (2, 2), "gte": (2, 2), "lte": (2, 2),
    "eq": (2, 2), "ne": (2, 2), "round2": (2, 2), "floor2": (2, 2), "ceil2": (2, 2),
    "quantize": (2, 2), "perceptronSigma2": (2, 2), "perceptronReLU2": (2, 2), "perceptronCustom2": (2, 2),
    "bitwise_and": (2, 3), "bitwise_or": (2, 3), "bitwise_xor": (2, 3), "lshift": (2, 3), "rshift": (2, 3),
    "gcd": (2, 3), "lcm": (2, 3), "cat": (2, 4), "x_at_pos_y": (2, 4),
    "sin": (1, 1), "cos": (1, 1), "tan": (1, 2), "exp": (1, 3), "expm1": (1, 3), "10^x": (1, 3),
    "log": (1, 2), "log1p": (1, 2), "log10": (1, 2), "sqrt": (1, 2), "abs": (1, 1),
    "neg": (1, 1), "square": (1, 1), "cube": (1, 1), "pow4": (1, 2), "pow5": (1, 2),
    "pow6": (1, 2), "root3": (1, 2), "root4": (1, 2), "frac": (1, 1), "round": (1, 1),
    "floor": (1, 1), "ceil": (1, 1), "int": (1, 1), "sigmoid": (1, 2), "tanh": (1, 1),
    "sinh": (1, 3), "cosh": (1, 3), "relu": (1, 1), "leaky_relu": (1, 1), "atan": (1, 1),
    "gaussian": (1, 2), "softplus": (1, 2), "sign": (1, 1), "signbit": (1, 1), "sinc": (1, 2),
    "xlogx": (1, 2), "erf": (1, 2), "inv": (1, 2), "deg2rad": (1, 1), "rad2deg": (1, 1),
    "perceptronSigma1": (1, 2), "perceptronReLU1": (1, 1), "perceptronCustom1": (1, 2),
    "pow7": (1, 2), "pow8": (1, 2), "pow9": (1, 2), "pow10": (1, 2),
    "root5": (1, 2), "root6": (1, 2), "root7": (1, 2), "root8": (1, 2), "root9": (1, 2), "root10": (1, 2),
    "oom": (1, 2), "bitwise_not": (1, 3),
    "if_else": (3, 3), "if_in_range": (3, 2), "if_out_of_range": (3, 2), "lerp": (3, 2), "distance_3": (3, 2),
    "python_rng": (1, 5), "perlin_noise": (1, 5),
    "seqsum": (1, 3), "seqprod": (1, 3),
}
OPERATOR_GROUPS = {
    "1": ("Arithmetic", ("+","-","*","/","delta","max","min","mod","floordiv","copysign","abs","neg")),
    "2": ("Power laws", ("pow","square","cube","pow4","pow5","pow6","pow7","pow8","pow9","pow10","sqrt","root3","root4","root5","root6","root7","root8","root9","root10","inv")),
    "3": ("Exponential and logarithmic", ("exp","expm1","10^x","log","log1p","log10","log_base","exp_decay","rbf","gaussian","xlogx","oom")),
    "4": ("Trigonometric and angle", ("sin","cos","tan","atan","atan2","sinc","sinh","cosh","deg2rad","rad2deg")),
    "5": ("Smooth nonlinear", ("sigmoid","tanh","relu","leaky_relu","softplus","erf")),
    "6": ("Geometry and interpolation", ("hypot","distance_2","distance_3","harmonic","geometric","lerp")),
    "7": ("Comparisons and conditionals", ("gt","lt","gte","lte","eq","ne","if_else","if_in_range","if_out_of_range")),
    "8": ("Rounding and discrete transforms", ("round","floor","ceil","int","frac","round2","floor2","ceil2","quantize","sign","signbit")),
    "9": ("Integer, bitwise, and digits", ("bitwise_and","bitwise_or","bitwise_xor","bitwise_not","lshift","rshift","gcd","lcm","cat","x_at_pos_y")),
    "10": ("Neural perceptrons", ("perceptronSigma1","perceptronReLU1","perceptronCustom1","perceptronSigma2","perceptronReLU2","perceptronCustom2")),
    "11": ("Stochastic noise", ("python_rng","perlin_noise")),
    "12": ("Sequence aggregates (needs --sequence-group)", ("seqsum","seqprod")),
}
DEFAULT_GROUP_IDS = tuple(str(index) for index in range(1,10))
DEFAULT_OPS = [operator for group in DEFAULT_GROUP_IDS for operator in OPERATOR_GROUPS[group][1]]
if set(DEFAULT_OPS).union(*[set(group[1]) for group in OPERATOR_GROUPS.values() if group[0] in {"Neural perceptrons","Stochastic noise"} or group[1]==("seqsum","seqprod")]) != set(OPS):
    raise RuntimeError("Operator groups must cover every operator exactly once")

# Extra structural cost beyond the one node every operator already contributes.
# Arithmetic stays cheap; domain guards, discontinuities, integer/digit tricks,
# and stochastic/noise-like primitives pay progressively more to reduce their
# ability to win merely by constructing a high-capacity nested expression.
OP_COMPLEXITY_BONUS = {
    # cheap algebra / elementary transforms
    "+":0, "-":0, "*":0, "max":0, "min":0, "abs":0, "neg":0, "square":0,
    "cube":0, "sqrt":0, "sign":0, "relu":0, "tanh":0,
    # smooth nonlinear or guarded arithmetic
    "/":1, "delta":1, "hypot":1, "distance_2":1, "sin":1, "cos":1, "atan":1,
    "exp":1, "expm1":1, "log":1, "log1p":1, "log10":1, "inv":1, "sigmoid":1,
    "gaussian":1, "softplus":1, "root3":1, "root4":1, "deg2rad":1, "rad2deg":1,
    # higher-order powers, branches, comparisons, and composite operations
    "pow":2, "log_base":2, "exp_decay":2, "rbf":2, "atan2":2, "harmonic":2,
    "geometric":2, "mod":2, "floordiv":2, "lerp":2, "distance_3":2, "if_else":2,
    "if_in_range":2, "if_out_of_range":2, "tan":2, "sinh":2, "cosh":2, "erf":2,
    "sinc":2, "xlogx":2,
    "pow4":2, "pow5":2, "pow6":2, "pow7":3, "pow8":3, "pow9":3, "pow10":3,
    "root5":2, "root6":2, "root7":2, "root8":2, "root9":2, "root10":2,
    # discrete/integer and representation-level operators are high capacity
    "gt":3, "lt":3, "gte":3, "lte":3, "eq":3, "ne":3, "round":3, "floor":3,
    "ceil":3, "int":3, "frac":3, "signbit":3, "oom":3, "round2":3, "floor2":3,
    "ceil2":3, "quantize":3, "bitwise_and":4, "bitwise_or":4, "bitwise_xor":4,
    "bitwise_not":4, "lshift":4, "rshift":4, "gcd":4, "lcm":4, "cat":5, "x_at_pos_y":5,
    # Neural-style perceptron primitives are individually expressive enough
    # to overfit small data, so make their structural prior deliberately high.
    "perceptronSigma1":5, "perceptronReLU1":5, "perceptronCustom1":5,
    "perceptronSigma2":5, "perceptronReLU2":5, "perceptronCustom2":5,
    "python_rng":6, "perlin_noise":6,
    # A sequence aggregate replaces an unrolled chain of per-position terms.
    "seqsum":2, "seqprod":2,
}

MDL_POLICY={
    "version":2,
    "grammar":"uniform_fixed_width",
    "tree_node_tag_bits":2,
    "constant_encoding":"fixed_parameter_bits:16",
    "special_constants":[-1.0,0.0,1.0],
    "affine_coefficients":"included",
}

def fixed_width_bits(count):
    """Bits for one member of a fixed uniform finite codebook."""
    return 0 if count<=1 else int(math.ceil(math.log2(count)))

def universal_integer_bits(value):
    """Elias-gamma length for a positive integer payload."""
    if value < 1: raise ValueError("Universal integer codes require a positive value")
    return 2*int(math.floor(math.log2(value)))+1

CONSTANT_PARAMETER_BITS = 16
def exact_float_code(value):
    """Code-length breakdown for one fitted real constant.

    Every non-special constant costs the same fixed parameter budget.  Pricing
    by decimal length made fitted constants (``2.0665311091502883``) cost as
    much as nine operators, so evolution rebuilt numbers from structure such
    as ``x/x`` or ``sqrt(sqrt(exp(x)))`` instead of using a constant.
    """
    number=float(value)
    if not np.isfinite(number): raise ValueError("MDL constants must be finite")
    if number==0.0: return {"kind":"zero","text":"0.0","bits":2}
    if number==1.0: return {"kind":"one","text":"1.0","bits":2}
    if number==-1.0: return {"kind":"minus_one","text":"-1.0","bits":2}
    return {"kind":"real","text":repr(number),"bits":2+CONSTANT_PARAMETER_BITS}

def tree_description(tree, n_features, operators, adfs=None, argument_count=0):
    """Fixed-grammar MDL code for a prefix-encoded expression tree."""
    operators=tuple(operators)
    if tree[0]=="x":
        if not isinstance(tree[1],int) or not 0<=tree[1]<n_features: raise ValueError("Invalid feature in MDL tree")
        return {"bits":2+fixed_width_bits(n_features),"nodes":[{"kind":"feature","index":tree[1],"bits":2+fixed_width_bits(n_features)}]}
    if tree[0]=="c":
        payload=exact_float_code(tree[1]); bits=2+payload["bits"]
        return {"bits":bits,"nodes":[{"kind":"constant","bits":bits,"payload":payload}]}
    if tree[0]=="arg":
        if not isinstance(tree[1],int) or not 0<=tree[1]<argument_count: raise ValueError("Invalid ADF argument in MDL tree")
        bits=2+fixed_width_bits(argument_count)
        return {"bits":bits,"nodes":[{"kind":"argument","index":tree[1],"bits":bits}]}
    if tree[0] not in operators: raise ValueError(f"Operator {tree[0]!r} is outside the active MDL grammar")
    children=[tree_description(child,n_features,operators,adfs,argument_count) for child in tree[1:]]
    own=2+fixed_width_bits(len(operators)); return {"bits":own+sum(child["bits"] for child in children),"nodes":[{"kind":"operator","operator":tree[0],"bits":own}]+[node for child in children for node in child["nodes"]]}

def model_description(model, n_features=None, operators=None, adfs=None):
    """Auditable structural MDL bits for trees and fitted affine coefficients."""
    n_features=model.mdl_feature_count if n_features is None else n_features
    operators=tuple(model.mdl_operators if operators is None else operators)
    if not n_features: raise ValueError("MDL requires the active feature count")
    if not operators: operators=tuple(OPS)
    adfs=dict(getattr(model,"adfs",{}) if adfs is None else adfs)
    used=[]; visiting=set()
    def visit(tree):
        if tree[0] in ("x","c","arg"): return
        if tree[0].startswith("adf_"):
            name=tree[0]
            if name in visiting: raise ValueError(f"Cyclic ADF definition {name!r}")
            if name not in used:
                item=adfs.get(name)
                if item is None: raise ValueError(f"Missing ADF definition {name!r}")
                visiting.add(name); visit(item["tree"]); visiting.remove(name); used.append(name)
        for child in tree[1:]: visit(child)
    for tree in model.trees: visit(tree)
    operators=tuple(dict.fromkeys([*operators,*used]))
    trees=[tree_description(tree,n_features,operators,adfs) for tree in model.trees]
    definitions=[]
    for name in used:
        item=adfs.get(name)
        if item is None: raise ValueError(f"Missing ADF definition {name!r}")
        definition=tree_description(item["tree"],n_features,operators,adfs,int(item["arity"]))
        definitions.append({"name":name,"arity":int(item["arity"]),"definition":definition})
    affine=[]
    for scale,intercept in model.scales:
        scale_code=exact_float_code(scale); intercept_code=exact_float_code(intercept)
        affine.append({"scale":scale_code,"intercept":intercept_code,"bits":scale_code["bits"]+intercept_code["bits"]})
    definition_bits=sum(item["definition"]["bits"] for item in definitions)
    return {"policy":MDL_POLICY,"grammar":{"operators":list(operators),"feature_count":n_features},"adfs":definitions,
            "trees":trees,"affine":affine,"tree_bits":sum(item["bits"] for item in trees),
            "affine_bits":sum(item["bits"] for item in affine),
            "adf_definition_bits":definition_bits,
            "total_bits":sum(item["bits"] for item in trees)+sum(item["bits"] for item in affine)+definition_bits}

def model_description_bits(model, n_features=None, operators=None, adfs=None):
    return float(model_description(model,n_features,operators,adfs)["total_bits"])

def clean(x: Any) -> np.ndarray:
    # Same result as nan_to_num(nan=0, posinf=CLIP, neginf=-CLIP) then clip
    # (clip already maps +/-inf to +/-CLIP), without nan_to_num's overhead.
    # np.minimum/np.maximum are the ufuncs np.clip dispatches to, minus its
    # Python-level wrapper; NaN propagates through both exactly as in clip.
    z=np.minimum(np.maximum(np.asarray(x,dtype=float),-CLIP),CLIP)
    nan=np.isnan(z)
    if nan.any(): z=np.where(nan,0.,z)
    return z

def op_eval(op: str, a: list[np.ndarray]) -> np.ndarray:
    x = a[0]
    with np.errstate(all="ignore"):
        if op == "+": z = x+a[1]
        elif op == "-": z = x-a[1]
        elif op == "*": z = x*a[1]
        elif op == "delta": z = np.abs(x-a[1])
        elif op == "/": z = x/np.where(np.abs(a[1]) < EPS, EPS, a[1])
        elif op == "max": z = np.maximum(x, a[1])
        elif op == "min": z = np.minimum(x, a[1])
        elif op == "pow": z = np.sign(x)*np.abs(x)**np.clip(a[1], -12, 12)
        elif op in ("hypot", "distance_2"): z = np.hypot(x, a[1])
        elif op == "log_base": z = np.log(np.abs(x)+EPS)/np.log(np.abs(a[1])+1.000001)
        elif op == "exp_decay": z = np.exp(-np.clip(x*a[1], -50, 50))
        elif op == "rbf": z = np.exp(-np.clip((x-a[1])**2, 0, 50))
        elif op == "atan2": z = np.arctan2(x, a[1])
        elif op == "harmonic": z = 2*x*a[1]/(np.abs(x+a[1])+EPS)
        elif op == "geometric": z = np.sqrt(np.abs(x*a[1]))
        elif op == "mod": z = np.mod(x, np.where(np.abs(a[1]) < EPS, 1., a[1]))
        elif op == "copysign": z = np.abs(x)*np.sign(a[1])
        elif op == "floordiv": z = np.floor(x/np.where(np.abs(a[1]) < EPS, 1., a[1]))
        elif op in ("gt","lt","gte","lte","eq","ne"):
            z = getattr(np, {"gt":"greater","lt":"less","gte":"greater_equal","lte":"less_equal","eq":"equal","ne":"not_equal"}[op])(x,a[1]).astype(float)
        elif op in ("round2","floor2","ceil2"):
            f=np.power(10.,np.clip(np.rint(a[1]),-10,10)); z=getattr(np,op[:-1])(x*f)/f
        elif op == "quantize": z=np.round(x/np.where(np.abs(a[1])<EPS,1.,a[1]))*a[1]
        elif op in ("bitwise_and","bitwise_or","bitwise_xor"):
            z=getattr(np,op)(np.clip(np.trunc(x),-1e12,1e12).astype(np.int64),np.clip(np.trunc(a[1]),-1e12,1e12).astype(np.int64)).astype(float)
        elif op == "lshift": z=np.left_shift(np.clip(np.trunc(x),-1e12,1e12).astype(np.int64),np.clip(np.trunc(a[1]),0,62).astype(np.int64)).astype(float)
        elif op == "rshift": z=np.right_shift(np.clip(np.trunc(x),-1e12,1e12).astype(np.int64),np.clip(np.trunc(a[1]),0,62).astype(np.int64)).astype(float)
        elif op in ("gcd","lcm"):
            u=np.abs(np.trunc(x).astype(np.int64)); v=np.abs(np.trunc(a[1]).astype(np.int64)); g=np.gcd(u,v); z=(np.gcd(u,v) if op=="gcd" else np.abs((u//np.where(g==0,1,g))*v)).astype(float)
        elif op == "cat":
            u=np.abs(np.trunc(x)); v=np.abs(np.trunc(a[1])); z=u*10**np.where(v==0,1,np.minimum(9,np.floor(np.log10(v+EPS))+1))+v
        elif op == "x_at_pos_y": z=np.mod(np.floor(np.abs(x))/10**np.clip(np.trunc(a[1]),0,12),10).astype(int)
        elif op == "perceptronSigma2": z=1/(1+np.exp(-np.clip(x+a[1],-50,50)))
        elif op == "perceptronReLU2": z=np.maximum(0,x+a[1])
        elif op == "perceptronCustom2": z=(x+a[1])/(1+np.exp(-np.clip(x+a[1],-50,50)))
        elif op == "sin": z=np.sin(x)
        elif op == "cos": z=np.cos(x)
        elif op == "tan": z=np.tan(np.clip(x,-1.55,1.55))
        elif op in ("exp","expm1","10^x"): z=(np.exp(np.clip(x,-50,50)) if op=="exp" else np.expm1(np.clip(x,-50,50)) if op=="expm1" else 10**np.clip(x,-12,12))
        elif op == "log": z=np.log(np.abs(x)+EPS)
        elif op == "log1p": z=np.log1p(np.abs(x))
        elif op == "log10": z=np.log10(np.abs(x)+EPS)
        elif op == "sqrt": z=np.sign(x)*np.sqrt(np.abs(x))
        elif op == "abs": z=np.abs(x)
        elif op == "neg": z=-x
        elif op in ("square","cube","pow4","pow5","pow6","pow7","pow8","pow9","pow10"): z=x**{"square":2,"cube":3,"pow4":4,"pow5":5,"pow6":6,"pow7":7,"pow8":8,"pow9":9,"pow10":10}[op]
        elif op in ("root3","root4","root5","root6","root7","root8","root9","root10"): z=np.sign(x)*np.abs(x)**(1/int(op[4:]))
        elif op == "oom": z=np.floor(np.log10(np.abs(x)+EPS))
        elif op == "frac": z=x-np.floor(x)
        elif op in ("round","floor","ceil","int"): z=(np.rint(x) if op=="round" else np.floor(x) if op=="floor" else np.ceil(x) if op=="ceil" else np.trunc(x))
        elif op == "sigmoid": z=1/(1+np.exp(-np.clip(x,-50,50)))
        elif op == "tanh": z=np.tanh(x)
        elif op in ("sinh","cosh"): z=getattr(np,op)(np.clip(x,-20,20))
        elif op == "relu": z=np.maximum(0,x)
        elif op == "leaky_relu": z=np.where(x>=0,x,.01*x)
        elif op == "atan": z=np.arctan(x)
        elif op == "gaussian": z=np.exp(-np.clip(x*x,0,50))
        elif op == "softplus": z=np.logaddexp(0,x)
        elif op == "sign": z=np.sign(x)
        elif op == "signbit": z=np.signbit(x).astype(float)
        elif op == "sinc": z=np.sinc(x/np.pi)
        elif op == "xlogx": z=x*np.log(np.abs(x)+EPS)
        elif op == "erf": z=np.vectorize(math.erf)(x)
        elif op == "inv": z=1/(np.abs(x)+EPS)
        elif op == "bitwise_not": z=np.bitwise_not(np.clip(np.trunc(x),-1e12,1e12).astype(np.int64)).astype(float)
        elif op == "deg2rad": z=np.deg2rad(x)
        elif op == "rad2deg": z=np.rad2deg(x)
        elif op == "perceptronSigma1": z=1/(1+np.exp(-np.clip(x,-50,50)))
        elif op == "perceptronReLU1": z=np.maximum(0,x)
        elif op == "perceptronCustom1": z=x/(1+np.exp(-np.clip(x,-50,50)))
        elif op == "if_else": z=np.where(x>.5,a[1],a[2])
        elif op == "if_in_range": z=np.clip(x,np.minimum(a[1],a[2]),np.maximum(a[1],a[2]))
        elif op == "if_out_of_range":
            lo,hi=np.minimum(a[1],a[2]),np.maximum(a[1],a[2]); z=np.where((x<lo)|(x>hi),x,np.where(x<(lo+hi)/2,lo,hi))
        elif op == "lerp": z=x+(a[1]-x)*a[2]
        elif op == "distance_3": z=np.sqrt(x*x+a[1]*a[1]+a[2]*a[2])
        elif op == "python_rng": z=np.sin(np.floor(x)*12.9898)*43758.5453 % 1
        elif op == "perlin_noise": z=np.sin(x*12.9898)*.5+np.sin(x*78.233)*.25
        else: raise ValueError(op)
    return clean(z)

# Search-time dispatch table: the same formulas as op_eval (which stays the
# reference, and is embedded verbatim in exported models), looked up by name
# instead of walking the elif chain on every node evaluation.
def _guard_zero(v, fill): return np.where(np.abs(v) < EPS, fill, v)
def _int_pair(x, y, lo=-1e12): return np.clip(np.trunc(x),-1e12,1e12).astype(np.int64),np.clip(np.trunc(y),lo,1e12 if lo<0 else 62).astype(np.int64)
def _round_places(fn):
    def apply(x, a): f=np.power(10.,np.clip(np.rint(a[1]),-10,10)); return fn(x*f)/f
    return apply
def _gcd_lcm(x, a, gcd):
    u=np.abs(np.trunc(x).astype(np.int64)); v=np.abs(np.trunc(a[1]).astype(np.int64)); g=np.gcd(u,v)
    return (np.gcd(u,v) if gcd else np.abs((u//np.where(g==0,1,g))*v)).astype(float)
def _out_of_range(x, a):
    lo,hi=np.minimum(a[1],a[2]),np.maximum(a[1],a[2]); return np.where((x<lo)|(x>hi),x,np.where(x<(lo+hi)/2,lo,hi))
def _power(n): return lambda x,a: x**n
def _root(n): return lambda x,a: np.sign(x)*np.abs(x)**(1/n)
_COMPARE={"gt":np.greater,"lt":np.less,"gte":np.greater_equal,"lte":np.less_equal,"eq":np.equal,"ne":np.not_equal}
# math.erf through frompyfunc is a Python call per row.  SciPy's ufunc agrees
# with it to 1 ulp; it is optional, so a missing SciPy keeps the slow path.
try: from scipy.special import erf as _vector_erf
except ImportError: _vector_erf=None
_OP_TABLE={
    "+":lambda x,a: x+a[1], "-":lambda x,a: x-a[1], "*":lambda x,a: x*a[1],
    "delta":lambda x,a: np.abs(x-a[1]),
    "/":lambda x,a: x/_guard_zero(a[1],EPS),
    "max":lambda x,a: np.maximum(x,a[1]), "min":lambda x,a: np.minimum(x,a[1]),
    "pow":lambda x,a: np.sign(x)*np.abs(x)**np.clip(a[1],-12,12),
    "hypot":lambda x,a: np.hypot(x,a[1]), "distance_2":lambda x,a: np.hypot(x,a[1]),
    "log_base":lambda x,a: np.log(np.abs(x)+EPS)/np.log(np.abs(a[1])+1.000001),
    "exp_decay":lambda x,a: np.exp(-np.clip(x*a[1],-50,50)),
    "rbf":lambda x,a: np.exp(-np.clip((x-a[1])**2,0,50)),
    "atan2":lambda x,a: np.arctan2(x,a[1]),
    "harmonic":lambda x,a: 2*x*a[1]/(np.abs(x+a[1])+EPS),
    "geometric":lambda x,a: np.sqrt(np.abs(x*a[1])),
    "mod":lambda x,a: np.mod(x,_guard_zero(a[1],1.)),
    "copysign":lambda x,a: np.abs(x)*np.sign(a[1]),
    "floordiv":lambda x,a: np.floor(x/_guard_zero(a[1],1.)),
    **{name:(lambda fn: lambda x,a: fn(x,a[1]).astype(float))(fn) for name,fn in _COMPARE.items()},
    "round2":_round_places(np.round), "floor2":_round_places(np.floor), "ceil2":_round_places(np.ceil),
    "quantize":lambda x,a: np.round(x/_guard_zero(a[1],1.))*a[1],
    **{name:(lambda fn: lambda x,a: fn(*_int_pair(x,a[1])).astype(float))(getattr(np,name)) for name in ("bitwise_and","bitwise_or","bitwise_xor")},
    "lshift":lambda x,a: np.left_shift(*_int_pair(x,a[1],0)).astype(float),
    "rshift":lambda x,a: np.right_shift(*_int_pair(x,a[1],0)).astype(float),
    "gcd":lambda x,a: _gcd_lcm(x,a,True), "lcm":lambda x,a: _gcd_lcm(x,a,False),
    "cat":lambda x,a: (lambda u,v: u*10**np.where(v==0,1,np.minimum(9,np.floor(np.log10(v+EPS))+1))+v)(np.abs(np.trunc(x)),np.abs(np.trunc(a[1]))),
    "x_at_pos_y":lambda x,a: np.mod(np.floor(np.abs(x))/10**np.clip(np.trunc(a[1]),0,12),10).astype(int),
    "perceptronSigma2":lambda x,a: 1/(1+np.exp(-np.clip(x+a[1],-50,50))),
    "perceptronReLU2":lambda x,a: np.maximum(0,x+a[1]),
    "perceptronCustom2":lambda x,a: (x+a[1])/(1+np.exp(-np.clip(x+a[1],-50,50))),
    "sin":lambda x,a: np.sin(x), "cos":lambda x,a: np.cos(x),
    "tan":lambda x,a: np.tan(np.clip(x,-1.55,1.55)),
    "exp":lambda x,a: np.exp(np.clip(x,-50,50)), "expm1":lambda x,a: np.expm1(np.clip(x,-50,50)),
    "10^x":lambda x,a: 10**np.clip(x,-12,12),
    "log":lambda x,a: np.log(np.abs(x)+EPS), "log1p":lambda x,a: np.log1p(np.abs(x)),
    "log10":lambda x,a: np.log10(np.abs(x)+EPS), "sqrt":lambda x,a: np.sign(x)*np.sqrt(np.abs(x)),
    "abs":lambda x,a: np.abs(x), "neg":lambda x,a: -x,
    **{name:_power(n) for name,n in {"square":2,"cube":3,"pow4":4,"pow5":5,"pow6":6,"pow7":7,"pow8":8,"pow9":9,"pow10":10}.items()},
    **{f"root{n}":_root(n) for n in range(3,11)},
    "oom":lambda x,a: np.floor(np.log10(np.abs(x)+EPS)),
    "frac":lambda x,a: x-np.floor(x),
    "round":lambda x,a: np.rint(x), "floor":lambda x,a: np.floor(x), "ceil":lambda x,a: np.ceil(x), "int":lambda x,a: np.trunc(x),
    "sigmoid":lambda x,a: 1/(1+np.exp(-np.clip(x,-50,50))),
    "tanh":lambda x,a: np.tanh(x),
    "sinh":lambda x,a: np.sinh(np.clip(x,-20,20)), "cosh":lambda x,a: np.cosh(np.clip(x,-20,20)),
    "relu":lambda x,a: np.maximum(0,x), "leaky_relu":lambda x,a: np.where(x>=0,x,.01*x),
    "atan":lambda x,a: np.arctan(x),
    "gaussian":lambda x,a: np.exp(-np.clip(x*x,0,50)),
    "softplus":lambda x,a: np.logaddexp(0,x),
    "sign":lambda x,a: np.sign(x), "signbit":lambda x,a: np.signbit(x).astype(float),
    "sinc":lambda x,a: np.sinc(x/np.pi),
    "xlogx":lambda x,a: x*np.log(np.abs(x)+EPS),
    "erf":(lambda x,a: _vector_erf(np.asarray(x,float))) if _vector_erf is not None else (lambda x,a: np.frompyfunc(math.erf,1,1)(x).astype(float)),
    "inv":lambda x,a: 1/(np.abs(x)+EPS),
    "bitwise_not":lambda x,a: np.bitwise_not(np.clip(np.trunc(x),-1e12,1e12).astype(np.int64)).astype(float),
    "deg2rad":lambda x,a: np.deg2rad(x), "rad2deg":lambda x,a: np.rad2deg(x),
    "perceptronSigma1":lambda x,a: 1/(1+np.exp(-np.clip(x,-50,50))),
    "perceptronReLU1":lambda x,a: np.maximum(0,x),
    "perceptronCustom1":lambda x,a: x/(1+np.exp(-np.clip(x,-50,50))),
    "if_else":lambda x,a: np.where(x>.5,a[1],a[2]),
    "if_in_range":lambda x,a: np.clip(x,np.minimum(a[1],a[2]),np.maximum(a[1],a[2])),
    "if_out_of_range":_out_of_range,
    "lerp":lambda x,a: x+(a[1]-x)*a[2],
    "distance_3":lambda x,a: np.sqrt(x*x+a[1]*a[1]+a[2]*a[2]),
    "python_rng":lambda x,a: np.sin(np.floor(x)*12.9898)*43758.5453 % 1,
    "perlin_noise":lambda x,a: np.sin(x*12.9898)*.5+np.sin(x*78.233)*.25,
}
def fast_op_eval(op: str, a: list[np.ndarray]) -> np.ndarray:
    fn=_OP_TABLE.get(op)
    if fn is None: return op_eval(op,a)
    with np.errstate(all="ignore"): z=fn(a[0],a)
    return clean(z)

# Nodes: ('x', feature), ('c', constant), or (operator, child, ...).
def node_size(t): return 1 if t[0] in ("x","c","arg") else 1+sum(node_size(q) for q in t[1:])
COMMUTATIVE_OPS={"+","*","max","min","hypot","distance_2"}
def simplify_tree(t):
    if t[0] in ("x","c","arg"): return t
    op=t[0]; children=[simplify_tree(c) for c in t[1:]]
    if op in COMMUTATIVE_OPS: children.sort(key=repr)
    if all(child[0]=="c" for child in children):
        try:
            values=[np.array([child[1]],dtype=float) for child in children]
            return ("c",float(op_eval(op,values)[0]))
        except Exception:
            pass
    if op=="neg" and children[0][0]=="neg": return children[0][1]
    if op=="+" and (children[0]==("c",0.) or children[1]==("c",0.)): return children[1] if children[0]==("c",0.) else children[0]
    if op=="*" and (children[0]==("c",1.) or children[1]==("c",1.)): return children[1] if children[0]==("c",1.) else children[0]
    if op=="*" and (children[0]==("c",0.) or children[1]==("c",0.)): return ("c",0.)
    if op=="/" and children[1]==("c",1.): return children[0]
    if op in {"min","max"} and children[0]==children[1]: return children[0]
    if op=="delta" and children[0]==children[1]: return ("c",0.)
    if op=="-" and children[0]==children[1]: return ("c",0.)
    folded=_fold_constant_offsets(op,children)
    if folded is not None: return folded
    return tuple([op]+children)

def _fold_constant_offsets(op, children):
    """Merge two constants separated by one +, - or * step: (c1-f)-c2 -> (c1-c2)-f.

    Such pairs are one parameter written twice; they cost bits and give the
    constant fitter a redundant, unidentifiable direction.
    """
    if op not in ("+","-","*") or len(children)!=2: return None
    left,right=children
    def split(node):
        """(constant, rest, constant_sign, rest_sign) for c+f, f+c, c-f, f-c."""
        if node[0] not in ("+","-") or len(node)!=3: return None
        a,b=node[1],node[2]
        if a[0]=="c" and b[0]!="c": return a[1],b,1.,(1. if node[0]=="+" else -1.)
        if b[0]=="c" and a[0]!="c": return (b[1] if node[0]=="+" else -b[1]),a,1.,1.
        return None
    def build(constant, rest, rest_sign):
        return simplify_tree(("+",("c",float(constant)),rest) if rest_sign>0 else ("-",("c",float(constant)),rest))
    if op=="*":
        for c,other in ((left,right),(right,left)):
            if c[0]=="c" and other[0]=="*" and len(other)==3:
                inner=[q for q in other[1:] if q[0]=="c"]; rest=[q for q in other[1:] if q[0]!="c"]
                if len(inner)==1 and len(rest)==1: return simplify_tree(("*",("c",float(c[1]*inner[0][1])),rest[0]))
        return None
    sign=1. if op=="+" else -1.
    if left[0]=="c" and split(right):
        c2,rest,_,rest_sign=split(right); return build(left[1]+sign*c2,rest,sign*rest_sign)
    if right[0]=="c" and split(left):
        c1,rest,_,rest_sign=split(left); return build(c1+sign*right[1],rest,rest_sign)
    return None
# Equivalence keys: one identity for algebraically equal equations, so x+y and
# y+x, (a+b)+c and a+(b+c), x+x and 2*x, x*x and square(x), or x-x and 0 are
# treated as one candidate.  Sums and products are flattened into sorted
# coefficient/term and factor multisets (associative-commutative normal form);
# other commutative operators sort their operands.  The key is an identity,
# never a rewrite: trees keep evolving in their own syntax and scoring stays
# exact.  Coefficients are rounded to 12 significant digits so reassociated
# float sums of the same constants still match.
EQUIVALENCE_COLLAPSE = True
_EQUIVALENCE_CACHE = {}
EQUIVALENCE_STATS = {"children_redrawn":0}
_POWER_FACTORS = {"square":2,"cube":3}
def _key_number(value):
    value=float(value)
    return 0. if value==0 else float(f"{value:.12g}")
def _sum_parts(t, sign, parts):
    """Accumulate sign*t into parts = [constant, {term: coefficient}]."""
    op=t[0]
    if op=="c": parts[0]+=sign*float(t[1]); return
    if op=="+" and len(t)==3: _sum_parts(t[1],sign,parts); _sum_parts(t[2],sign,parts); return
    if op=="-" and len(t)==3: _sum_parts(t[1],sign,parts); _sum_parts(t[2],-sign,parts); return
    if op=="neg" and len(t)==2: _sum_parts(t[1],-sign,parts); return
    coefficient,term=_product_key(t)
    if term is None: parts[0]+=sign*coefficient; return
    parts[1][term]=parts[1].get(term,0.)+sign*coefficient
def _product_parts(t, parts):
    """Accumulate t into parts = [coefficient, [factor keys]]."""
    op=t[0]
    if op=="c": parts[0]*=float(t[1]); return
    if op=="*" and len(t)==3: _product_parts(t[1],parts); _product_parts(t[2],parts); return
    if op=="neg" and len(t)==2: parts[0]=-parts[0]; _product_parts(t[1],parts); return
    if op in _POWER_FACTORS and len(t)==2:
        for _ in range(_POWER_FACTORS[op]): _product_parts(t[1],parts)
        return
    key=_equivalence_key(t)
    if key[0]=="sum" and not key[1] and len(key[2])==1:
        # A single scaled term, e.g. neg(x) or 2*x reached through a sum.
        coefficient,term=key[2][0]; parts[0]*=coefficient; parts[1].append(term); return
    if key[0]=="const": parts[0]*=key[1]; return
    parts[1].append(key)
def _product_key(t):
    """(coefficient, factor-multiset key or None for a pure constant)."""
    parts=[1.,[]]; _product_parts(t,parts)
    if parts[0]==0 or not parts[1]: return parts[0],None
    factors=tuple(sorted(parts[1],key=repr))
    return parts[0],(factors[0] if len(factors)==1 else ("prod",factors))
def _equivalence_key(t):
    op=t[0]
    if op=="c": return ("const",_key_number(t[1]))
    if op in ("x","arg"): return (op,t[1])
    if op in ("+","-","neg","*") or op in _POWER_FACTORS:
        parts=[0.,{}]; _sum_parts(t,1.,parts)
        terms=tuple(sorted(((_key_number(coefficient),term) for term,coefficient in parts[1].items() if _key_number(coefficient)!=0),key=repr))
        constant=_key_number(parts[0])
        if not terms: return ("const",constant)
        if constant==0 and len(terms)==1 and terms[0][0]==1.: return terms[0][1]
        return ("sum",constant,terms)
    children=[_equivalence_key(child) for child in t[1:]]
    if op in COMMUTATIVE_OPS: children.sort(key=repr)
    return (op,*children)
def equivalence_key(tree):
    """Hashable identity shared by algebraically equal trees (see above)."""
    if not EQUIVALENCE_COLLAPSE: return repr(tree)
    raw=repr(tree); hit=_EQUIVALENCE_CACHE.get(raw)
    if hit is None:
        if len(_EQUIVALENCE_CACHE)>=16384: _EQUIVALENCE_CACHE.clear()
        hit=_EQUIVALENCE_CACHE[raw]=repr(_equivalence_key(tree))
    return hit
def model_equivalence_key(model_or_trees):
    trees=model_or_trees.trees if hasattr(model_or_trees,"trees") else model_or_trees
    return tuple(equivalence_key(tree) for tree in trees)
def weighted_complexity(t):
    """Node count plus the selected operator's anti-overfitting cost bonus."""
    if t[0] in ("x","c","arg"): return 1
    return 1+OP_COMPLEXITY_BONUS.get(t[0],2)+sum(weighted_complexity(q) for q in t[1:])
def constant_regularization(t):
    return sum(math.log1p(abs(v)) for v in constant_vector(t))
def node_depth(t): return 0 if t[0] in ("x","c","arg") or len(t)==1 else 1+max(node_depth(q) for q in t[1:])
# Ordered column groups (--sequence-group NAME=col1,col2,...).  Each group adds
# virtual features: NAME[i] (element at position i), prod(NAME[<i]) and
# sum(NAME[<i]) (exclusive prefix product/sum), plus one shared index i.
# seqsum/seqprod evaluate their child at every position i and reduce, so
#   1 + sum_i (k_i-1) * prod_{j<i} s_j  is  seqsum((K[i]-1)*prod(S[<i])) + 1.
# Outside seqsum/seqprod a virtual feature takes its position-0 value.
SEQUENCE_LAYOUT = None
SEQUENCE_GROUP_REQUEST = ()
SEQUENCE_LAYOUT_KEY = "__afpo_sequence_layout__"
def build_sequence_layout(names, groups):
    """groups: [(name, [column names in order]), ...] over numeric feature names."""
    if not groups: return None
    index={name:i for i,name in enumerate(names)}; built=[]; lengths=set()
    for group_name,columns in groups:
        missing=[column for column in columns if column not in index]
        if missing: raise ValueError(f"Sequence group {group_name!r} names unknown numeric input column(s): {', '.join(missing)}")
        if len(columns)<2: raise ValueError(f"Sequence group {group_name!r} needs at least two columns")
        built.append({"name":group_name,"columns":[index[column] for column in columns]}); lengths.add(len(columns))
    if len(lengths)!=1: raise ValueError("All sequence groups must have the same number of columns")
    virtual=[[kind,g] for g in range(len(built)) for kind in ("elem","prefix_prod","prefix_sum")]+[["index",-1]]
    return {"real":len(names),"length":lengths.pop(),"groups":built,"virtual":virtual}
def sequence_feature_names(layout):
    labels={"elem":"{}[i]","prefix_prod":"prod({}[<i])","prefix_sum":"sum({}[<i])"}
    return [("i" if kind=="index" else labels[kind].format(layout["groups"][g]["name"])) for kind,g in layout["virtual"]]
def sequence_value(X, layout, v, position):
    kind,g=layout["virtual"][v]
    if kind=="index": return np.full(len(X),float(position))
    columns=layout["groups"][g]["columns"]
    if kind=="elem": return X[:,columns[position]]
    before=X[:,columns[:position]]
    return np.prod(before,axis=1) if kind=="prefix_prod" else np.sum(before,axis=1)
def sequence_augment(X, layout):
    if layout is None: return X
    return np.column_stack([X[:,:layout["real"]]]+[sequence_value(X,layout,v,0) for v in range(len(layout["virtual"]))])
def evaluate(t, X, adfs=None, arguments=None, position=None):
    if t[0] == "x":
        layout=SEQUENCE_LAYOUT
        if position is not None and layout is not None and t[1]>=layout["real"]: return sequence_value(X,layout,t[1]-layout["real"],position)
        return X[:,t[1]]
    if t[0] == "c": return np.full(len(X),t[1])
    if t[0] == "arg":
        if arguments is None or not isinstance(t[1],int) or not 0<=t[1]<len(arguments): raise ValueError("Invalid ADF argument")
        return arguments[t[1]]
    if t[0].startswith("adf_"):
        item=(adfs or {}).get(t[0])
        if item is None or len(t)-1 != int(item["arity"]): raise ValueError(f"Unknown or malformed ADF {t[0]!r}")
        values=[evaluate(q,X,adfs,arguments,position) for q in t[1:]]
        return evaluate(item["tree"],X,adfs,values,position)
    if t[0] in ("seqsum","seqprod"):
        if SEQUENCE_LAYOUT is None: raise ValueError(f"{t[0]} needs --sequence-group")
        parts=[evaluate(t[1],X,adfs,arguments,i) for i in range(SEQUENCE_LAYOUT["length"])]
        with np.errstate(all="ignore"): return clean(np.sum(parts,axis=0) if t[0]=="seqsum" else np.prod(parts,axis=0))
    return fast_op_eval(t[0],[evaluate(q,X,adfs,arguments,position) for q in t[1:]])

# Deterministic top-level tree outputs, reused across the many places one
# generation re-predicts the same trees on the same rows (mutation baselines,
# behavioral dedup, lexicase, QD cells, serial scoring).  Entries hold the
# data array itself, so its buffer address cannot be recycled while cached;
# results are read-only so no caller can corrupt a shared entry.
_EVALUATION_CACHE={}; _EVALUATION_CACHE_SIZE=[0]
EVALUATION_CACHE_ELEMENTS=16_000_000
def referenced_adfs(trees, adfs):
    """The ADF definitions the trees call, transitively (name order)."""
    if not adfs: return {}
    found={}; pending=list(trees)
    while pending:
        node=pending.pop()
        if node[0] in ("x","c","arg"): continue
        if node[0].startswith("adf_") and node[0] not in found and node[0] in adfs:
            found[node[0]]=adfs[node[0]]; pending.append(adfs[node[0]]["tree"])
        pending.extend(node[1:])
    return {name:found[name] for name in sorted(found)}
def checkpoint_adfs(model):
    """Persist only the called definitions' evaluation data.  The registry
    snapshot holds the full catalog; per-model copies of every definition's
    bookkeeping made checkpoints grow with population x definitions."""
    return {name:{key:item[key] for key in ("tree","arity","dependencies") if key in item}
            for name,item in referenced_adfs(model.trees,model.adfs).items()}
def adf_signature(trees, adfs):
    """Cache identity of the ADFs that affect evaluation.  Registry
    bookkeeping (usage counts, validation history) and definitions the
    trees never call change every generation and must not change keys."""
    if not adfs: return None
    return tuple((name,repr(item["tree"]),int(item["arity"])) for name,item in referenced_adfs(trees,adfs).items()) or None
_ROW_SUBSETS={}
def row_subset(X, rows):
    """X[rows], returning the same array object for the same rows of the same
    array.  Fancy indexing copies, and the evaluation caches key on buffer
    addresses: a fresh copy per call never hit and pinned its own rows."""
    if isinstance(rows,slice): return X[rows]
    rows=np.asarray(rows)
    interface=X.__array_interface__
    key=(interface["data"][0],X.shape,X.strides,interface["typestr"],rows.dtype.str,rows.tobytes())
    hit=_ROW_SUBSETS.get(key)
    if hit is not None: return hit[0]
    if len(_ROW_SUBSETS)>=64: _ROW_SUBSETS.clear()
    subset=X[rows]; subset.setflags(write=False)
    # Holding X keeps its buffer address from being recycled while cached.
    _ROW_SUBSETS[key]=(subset,X)
    return subset
def evaluate_cached(t, X, adfs=None):
    if not isinstance(X,np.ndarray) or X.ndim!=2: return evaluate(t,X,adfs)
    interface=X.__array_interface__
    # repr, not the tuple: equal-hashing constants such as 0.0/-0.0 or 1/1.0
    # can evaluate differently.
    key=(repr(t),adf_signature((t,),adfs),interface["data"][0],X.shape,X.strides,interface["typestr"])
    hit=_EVALUATION_CACHE.get(key)
    if hit is not None: return hit[0]
    value=np.asarray(evaluate(t,X,adfs))
    if value.ndim!=1 or len(value)!=len(X): return value
    value.setflags(write=False)
    _EVALUATION_CACHE[key]=(value,X); _EVALUATION_CACHE_SIZE[0]+=value.size
    while _EVALUATION_CACHE_SIZE[0]>EVALUATION_CACHE_ELEMENTS and _EVALUATION_CACHE:
        oldest=next(iter(_EVALUATION_CACHE)); _EVALUATION_CACHE_SIZE[0]-=_EVALUATION_CACHE.pop(oldest)[0].size
    return value

def tree_contains_adf(tree):
    return tree[0].startswith("adf_") or (tree[0] not in ("x","c","arg") and any(tree_contains_adf(child) for child in tree[1:]))

def grammar_for_trees(base, trees, adfs=None):
    """Keep retired ADF calls decodable while charging the current active grammar."""
    found=[]
    def visit(tree):
        if tree[0] not in ("x","c","arg") and tree[0] not in found:
            found.append(tree[0])
            if tree[0].startswith("adf_") and tree[0] in (adfs or {}): visit((adfs or {})[tree[0]]["tree"])
        if tree[0] not in ("x","c","arg"):
            for child in tree[1:]: visit(child)
    for tree in trees: visit(tree)
    return tuple(dict.fromkeys([*base,*found]))

def canonical_adf_template(tree):
    """Replace up to three distinct feature leaves with ordered ADF arguments."""
    features=[]
    def convert(node):
        if node[0]=="x":
            if node[1] not in features: features.append(node[1])
            return ("arg",features.index(node[1]))
        if node[0] in ("c","arg"): return node
        return tuple([node[0]]+[convert(child) for child in node[1:]])
    result=convert(simplify_tree(tree))
    return (result,len(features)) if 1<=len(features)<=3 else (None,0)

class ADFRegistry:
    """V2 ADF catalog: founder-supported, bounded, and dependency-aware."""
    def __init__(self, enabled=False, capacity=8, allow_nested=True):
        self.enabled=bool(enabled); self.capacity=int(capacity); self.allow_nested=bool(allow_nested); self.definitions={}; self.active=[]; self.history={}; self.last_used={}; self.schema_version=2; self.invalid_calls=0
    def operators(self, base): return list(base)+[name for name in self.active if name not in base]
    def import_models(self, models):
        """Retain inherited definitions without activating foreign proposal operators."""
        for model in models:
            if model is None: continue
            for name,item in model.adfs.items():
                previous=self.definitions.get(name)
                if previous is not None and (previous["tree"]!=item["tree"] or previous["arity"]!=item["arity"]):
                    raise ValueError(f"Conflicting inherited ADF definition {name!r}")
                if previous is None:
                    # Checkpointed models carry only tree/arity; restore the
                    # bookkeeping fields the registry updates in place.
                    self.definitions[name]={"dependencies":self._dependencies(item["tree"]),"supporting_founders":[],"activated_generation":None,
                                            "retired_generation":None,"elite_uses":0,"validation":[],**deepcopy(item)}

    def attach(self, models):
        """Give every live/archive/particle model the permanent definition catalog.

        Retiring an ADF changes only the proposal grammar; a surviving model
        must always carry enough data to evaluate its historical call sites.
        """
        for model in models:
            if model is not None: model.adfs.update(self.definitions)
    def snapshot(self): return {"schema_version":self.schema_version,"enabled":self.enabled,"capacity":self.capacity,"allow_nested":self.allow_nested,"definitions":self.definitions,"active":self.active,"history":self.history,"last_used":self.last_used,"invalid_calls":self.invalid_calls}
    @classmethod
    def from_snapshot(cls,data):
        result=cls(data.get("enabled",False),data.get("capacity",8),data.get("allow_nested",False)); result.schema_version=int(data.get("schema_version",1)); result.definitions=dict(data.get("definitions",{})); result.active=list(data.get("active",[])); result.history=dict(data.get("history",{})); result.last_used=dict(data.get("last_used",{})); result.invalid_calls=int(data.get("invalid_calls",0)); return result
    def _dependencies(self, tree):
        found=[]
        def visit(node):
            if node[0].startswith("adf_") and node[0] not in found: found.append(node[0])
            if node[0] not in ("x","c","arg"):
                for child in node[1:]: visit(child)
        visit(tree); return found
    def _valid_definition(self, tree, arity):
        def visit(node):
            if node[0]=="arg": return isinstance(node[1],int) and 0<=node[1]<arity
            if node[0] in ("x","c"): return True
            expected=self.definitions.get(node[0],{}).get("arity") if node[0].startswith("adf_") else OPS.get(node[0],(-1,))[0]
            return expected==len(node)-1 and all(visit(child) for child in node[1:])
        return visit(tree)
    def _acyclic(self, name, tree):
        graph={key:self._dependencies(item["tree"]) for key,item in self.definitions.items()}; graph[name]=self._dependencies(tree)
        visiting=set(); visited=set()
        def visit(node):
            if node in visiting: return False
            if node in visited: return True
            if node not in graph: return False
            visiting.add(node); valid=all(visit(child) for child in graph[node]); visiting.remove(node); visited.add(node); return valid
        return visit(name)
    def observe(self, elite, generation):
        if not self.enabled or generation%10: return False
        seen={}
        for model in elite:
            for root in model.trees:
                for path in subtree_paths(root):
                    subtree=subtree_at(root,path)
                    if node_size(subtree)<3 or node_size(subtree)>12 or (not self.allow_nested and tree_contains_adf(subtree)): continue
                    template,arity=canonical_adf_template(subtree)
                    if template is None: continue
                    key=repr(template); seen.setdefault(key,{"tree":template,"arity":arity,"founders":set()})["founders"].update(model.founder_ids)
        for key,item in seen.items():
            record=self.history.setdefault(key,{"tree":item["tree"],"arity":item["arity"],"founders":[],"last":generation})
            record["founders"]=sorted(set(record["founders"]).union(item["founders"]))[-64:]; record["last"]=generation
        for key in list(self.history):
            if generation-self.history[key]["last"]>50: del self.history[key]
        choices=[(key,item) for key,item in self.history.items() if len(item["founders"])>=3 and self._valid_definition(item["tree"],item["arity"]) and all(existing["tree"]!=item["tree"] for existing in self.definitions.values())]
        if not choices: self._retire(generation); return False
        key,item=max(choices,key=lambda pair:(len(pair[1]["founders"]),-node_size(pair[1]["tree"]),pair[0]))
        name="adf_"+hashlib.sha256(key.encode()).hexdigest()[:10]
        if not self._acyclic(name,item["tree"]): return False
        self.definitions[name]={"tree":item["tree"],"arity":item["arity"],"dependencies":self._dependencies(item["tree"]),"supporting_founders":list(item["founders"]),"activated_generation":generation,"retired_generation":None,"elite_uses":0,"validation":[]}; self.active.append(name); self.last_used[name]=generation
        self._retire(generation); return True
    def mark_usage(self, models, generation, elite=()):
        used=set(); elite_used=set()
        def visit(tree):
            if tree[0] in ("x","c","arg"): return
            if tree[0].startswith("adf_"): used.add(tree[0])
            for child in tree[1:]: visit(child)
        for model in models:
            for tree in model.trees: visit(tree)
        for model in elite:
            for tree in model.trees: visit(tree)
        elite_used.update(used.intersection(self._dependencies_from_models(elite)))
        for name in used:
            self.last_used[name]=generation
        for name in elite_used:
            if name in self.definitions: self.definitions[name]["elite_uses"]+=1
        self._retire(generation,used)
    def record_validation(self, model, loss, generation):
        for name in self._dependencies_from_models([model]):
            if name in self.definitions: self.definitions[name]["validation"].append({"generation":generation,"loss":float(loss)})
    def _dependencies_from_models(self, models):
        found=set()
        for model in models:
            if model is None: continue
            for tree in model.trees: found.update(self._dependencies(tree))
        return found
    def _retire(self,generation,protected=()):
        protected=set(protected)
        pending=list(protected)
        while pending:
            name=pending.pop()
            for dependency in self._dependencies(self.definitions[name]["tree"]) if name in self.definitions else ():
                if dependency not in protected: protected.add(dependency); pending.append(dependency)
        retained=[]
        for name in self.active:
            if name in protected or generation-self.last_used.get(name,generation)<=30: retained.append(name)
            elif name in self.definitions: self.definitions[name]["retired_generation"]=generation
        self.active=retained
        if len(self.active)>self.capacity:
            evicted=sorted(self.active,key=lambda name:(self.last_used.get(name,-1),name))[:-self.capacity]
            self.active=[name for name in self.active if name not in evicted]
            for name in evicted: self.definitions[name]["retired_generation"]=generation
    def diagnostics(self):
        invalid=sum(value for key,value in INVALID_DIAGNOSTICS.items() if str(key).startswith("adf_") or ":adf_" in str(key))
        return {"schema_version":self.schema_version,"nested":self.allow_nested,"promotions":len(self.definitions),"definitions":len(self.definitions),"active":list(self.active),"retired":sum(item.get("retired_generation") is not None for item in self.definitions.values()),"elite_uses":sum(item.get("elite_uses",0) for item in self.definitions.values()),"invalid_calls":invalid,"definitions_detail":self.definitions}

def constant_paths(tree, prefix=()):
    if tree[0]=="c": return [prefix]
    if tree[0] in ("x","arg"): return []
    paths=[]
    for i,child in enumerate(tree[1:]): paths.extend(constant_paths(child,prefix+(i,)))
    return paths
def constant_vector(tree): return tuple(float(subtree_at(tree,p)[1]) for p in constant_paths(tree))
def with_constants(tree, values):
    out=tree
    for path,value in zip(constant_paths(tree),values): out=replace_subtree(out,path,("c",float(value)))
    return out

# Fitted constants may be as large as physical constants (c**2 ~ 9e16); the
# affine readout stays bounded at 1e9, so large scales live in the tree.
CONSTANT_FIT_ROWS, CONSTANT_FIT_ITERATIONS, CONSTANT_LIMIT = 256, 12, 1e30
# Guarded operators replace a near-zero divisor (x/0 -> x/1e-12, mod(x,0) ->
# mod(x,1)).  With a constant divisor that guard always fires, so the shown
# equation (m/0) is not what is computed (m*1e12) and it smuggles in a free
# huge constant.  Such trees are infeasible; data-driven divisors keep the guard.
GUARDED_DIVISOR_OPS = {"/":1, "mod":1, "floordiv":1, "quantize":1,
                       # log(|x|+EPS) guards: log(0) would show 0 but compute log(1e-12)
                       "inv":0, "log":0, "log10":0, "oom":0, "log_base":0}
def guarded_constant_divisor(tree):
    for node in walk_tree(tree):
        position=GUARDED_DIVISOR_OPS.get(node[0])
        if position is not None and len(node)>position+1:
            divisor=node[position+1]
            if divisor[0]=="c" and abs(float(divisor[1]))<EPS: return True
    return False
class _FlatTree:
    """Pre-order program of one tree that keeps every node's value.

    Nudging one constant only changes the nodes on its path to the root, so a
    finite-difference Jacobian column costs the path, not the whole tree.
    Trees with ADFs, ADF arguments or sequence aggregates use evaluate()."""
    UNSUPPORTED=("seqsum","seqprod","arg")
    def __init__(self, tree):
        self.kind=[]; self.payload=[]; self.parent=[]; self.children=[]; self.constants=[]
        def visit(node, parent):
            if node[0] in self.UNSUPPORTED or node[0].startswith("adf_"): raise TypeError(node[0])
            index=len(self.kind); self.kind.append(node[0]); self.parent.append(parent); self.children.append([])
            self.payload.append(node[1] if node[0] in ("x","c") else None)
            if parent>=0: self.children[parent].append(index)
            if node[0]=="c": self.constants.append(index)
            elif node[0]!="x":
                for child in node[1:]: visit(child,index)
        visit(tree,-1)
    @classmethod
    def build(cls, tree):
        try: return cls(tree)
        except (TypeError, RecursionError): return None
    def values(self, constants, X):
        """All node values (index 0 is the root) with constants in constant_paths order."""
        payload=list(self.payload)
        for index,value in zip(self.constants,constants): payload[index]=float(value)
        values=[None]*len(self.kind)
        for index in range(len(self.kind)-1,-1,-1):
            kind=self.kind[index]
            if kind=="x": values[index]=X[:,payload[index]]
            elif kind=="c": values[index]=np.full(len(X),payload[index])
            else: values[index]=fast_op_eval(kind,[values[child] for child in self.children[index]])
        return values
    def nudged_root(self, values, constant, value, X):
        """Root output when only one constant changes, reusing every off-path value."""
        changed=self.constants[constant]; current=np.full(len(X),float(value)); index=self.parent[changed]
        while index>=0:
            current=fast_op_eval(self.kind[index],[current if child==changed else values[child] for child in self.children[index]])
            changed,index=index,self.parent[index]
        return current

def fit_tree_constants(tree, X, y, adfs=None, fit_readout=True, iterations=CONSTANT_FIT_ITERATIONS, robust=True):
    """Levenberg-Marquardt on a tree's inner constants (variable projection).

    The affine readout a*f+b is re-solved in closed form for every constant
    vector, so only the constants inside the tree are searched.  Returns the
    tree unchanged unless its residual cost strictly improves.  ``robust``
    fits what assess() scores: a Huber-reweighted readout and robust_loss's
    Huber cost.  Plain least squares let a few outliers drag correct
    constants away, and the result is written back into the model.
    """
    start=np.asarray(constant_vector(tree),float)
    if not len(start) or not len(y): return tree
    scale=target_scale(y); ones=np.ones(len(y)); flat=_FlatTree.build(tree); huber=ROBUST_LOSS_DELTA
    def readout(pred):
        if fit_readout:
            # Mirror affine()'s coefficient bound, or the fitter would treat any
            # overall scale as free and never move a large constant into the tree.
            bound=AFFINE_COEFFICIENT_BOUND; weights=ones; fitted=None
            for _ in range(3 if robust else 1):
                line=_weighted_line(pred,y,weights)
                if line is None: fitted=np.full(len(y),float(np.mean(y))); break
                slope,intercept=line
                if abs(slope)>bound: slope=float(np.sign(slope))*bound; intercept=float(np.average(y-slope*pred,weights=weights))
                fitted=slope*pred+float(np.clip(intercept,-bound,bound))
                if robust:
                    weights=np.minimum(1.,huber*scale/np.maximum(np.abs(fitted-y),EPS))
                    if np.all(weights>=1.): break  # no point is in the Huber tail: least squares is exact
            pred=fitted
        r=(pred-y)/scale
        if robust:
            # sum(rho**2)/2 equals the Huber loss, so least squares on rho is Huber.
            magnitude=np.abs(r)
            r=np.where(magnitude<=huber,r,np.sign(r)*np.sqrt(np.maximum(2*huber*magnitude-huber*huber,0.)))
        return r if np.all(np.isfinite(r)) else None
    def residual(values):
        """Residual plus the node values that produced it (None without the flat program)."""
        try:
            if flat is None: return readout(evaluate(with_constants(tree,values),X,adfs)),None
            nodes=flat.values(values,X); return readout(nodes[0]),nodes
        except (ArithmeticError, IndexError, RecursionError, ValueError): return None,None
    current=start.copy(); r,nodes=residual(current)
    if r is None: return tree
    cost=initial=float(r@r); damping=1e-3
    for _ in range(iterations):
        if cost<=1e-24: break
        J=np.empty((len(r),len(current)))
        for k in range(len(current)):
            step=1e-6*max(1.,abs(current[k]))
            if nodes is None:
                probe=current.copy(); probe[k]+=step; rk,_=residual(probe)
            else:
                try: rk=readout(flat.nudged_root(nodes,k,current[k]+step,X))
                except (ArithmeticError, IndexError, RecursionError, ValueError): rk=None
            # Stop, but through the exit below: it applies the zero-divisor guard.
            if rk is None: J=None; break
            J[:,k]=(rk-r)/step
        if J is None: break
        # Constants the output ignores (absorbed by the readout, behind a step)
        # leave J all zero; no damping can move them, so stop instead of trying.
        if not np.any(J): break
        g=J.T@r; H=J.T@J; improved=False
        for _ in range(8):
            try: delta=np.linalg.solve(H+damping*(np.diag(np.diag(H))+1e-12*np.eye(len(current))),-g)
            except np.linalg.LinAlgError: damping*=10; continue
            candidate=np.clip(current+delta,-CONSTANT_LIMIT,CONSTANT_LIMIT); rc,candidate_nodes=residual(candidate)
            if rc is not None and float(rc@rc)<cost:
                gain=cost-float(rc@rc); current,r,nodes,cost=candidate,rc,candidate_nodes,float(rc@rc); damping=max(damping/3,1e-9); improved=True; break
            damping*=4
        if not improved or gain<=1e-10*max(cost,1e-30): break
    if cost>=initial: return tree
    tuned=with_constants(tree,current)
    return tree if guarded_constant_divisor(tuned) else tuned

def _classifier_log_loss(raw, truth, class_count):
    scales=fit_classifier_affine(raw,truth,class_count)
    scores=np.column_stack([a*raw[:,k]+b for k,(a,b) in enumerate(scales)])
    probabilities=binary_probabilities(scores[:,0]) if raw.shape[1]==1 else stable_softmax(scores)
    truth=np.asarray(np.rint(truth),int); valid=(truth>=0)&(truth<probabilities.shape[1])
    return float(np.mean(-np.log(np.maximum(probabilities[np.arange(len(truth))[valid],truth[valid]],EPS)))) if np.any(valid) else 0.

def _tune_classifier_heads(trees, heads, X, truth, class_count, adfs):
    """Tune each class head's constants against its 0/1 indicator (least squares surrogate),
    keeping the result only when the calibrated log loss of the whole target improves."""
    labels=np.asarray(np.rint(truth),int); changed=False
    try: raw=np.column_stack([evaluate(trees[head],X,adfs) for head in heads])
    except (ArithmeticError, IndexError, RecursionError, ValueError): return False
    best=_classifier_log_loss(raw,truth,class_count)
    for position,head in enumerate(heads):
        indicator=(labels==(1 if len(heads)==1 else position)).astype(float)
        tuned=fit_tree_constants(trees[head],X,indicator,adfs,robust=False)
        if tuned is trees[head]: continue
        try: candidate=raw.copy(); candidate[:,position]=clean(evaluate(tuned,X,adfs))
        except (ArithmeticError, IndexError, RecursionError, ValueError): continue
        loss=_classifier_log_loss(candidate,truth,class_count)
        if loss<best-1e-9: trees[head],raw,best,changed=tuned,candidate,loss,True
    return changed

def tune_model_constants(model, X, Y, affine_on, cats):
    """Refit inner constants of every regression head in place (Lamarckian)."""
    targets,_=classification_layout(cats)
    rows=stratified_probe_indices(X,CONSTANT_FIT_ROWS) if len(X)>CONSTANT_FIT_ROWS else slice(None)
    Xs,Ys=row_subset(X,rows),row_subset(Y,rows); trees=list(model.trees); changed=False
    for j,heads in enumerate(targets):
        if cats[j] is not None:
            if len(cats[j])>=2 and affine_on and _tune_classifier_heads(trees,heads,Xs,Ys[:,j],len(cats[j]),model.adfs): changed=True
            continue
        for head in heads:
            tuned=fit_tree_constants(trees[head],Xs,Ys[:,j],model.adfs,fit_readout=affine_on)
            if tuned is not trees[head]: trees[head]=tuned; changed=True
    if changed: model.trees=trees
    return changed
class PosteriorParticlePopulation:
    """Adaptive finite-catalogue search distribution, not an exact SMC posterior.

    Catalogue entries have a uniform base mass and an explicit energy weight.
    Resampling affects draws only; it never erases the catalogue or reuses an
    empirical draw's weight as another likelihood factor.
    """
    def __init__(self, capacity=96, complexity_prior=.016, ess_ratio=.60):
        self.capacity=capacity; self.complexity_prior=complexity_prior; self.ess_ratio=ess_ratio
        self.particles=[]; self.weights=np.empty(0); self.inverse_temperature=1.0
        self.catalog=[]; self.catalog_weights=np.empty(0)
        self.last_ess=0.0; self.log_evidence=0.0; self.resample_count=0; self.rejuvenation_accepts=0; self.rejuvenation_tries=0
        self.post_resample_ess=0.0; self.unique_particles=0
        self.history=[]; self.predictive_history=[]; self.last_predictive={}
        self.likelihood_kind="legacy"; self.student_t_df=4.; self.noise_scale=None
        self.injection_rate=.65; self.injection_trials=0.; self.injection_successes=0.

    @staticmethod
    def _model_data(model):
        return {"trees":model.trees,"scales":model.scales,"age":model.age,"objectives":model.objectives,
                "lineage_id":model.lineage_id,"origin":model.origin,"parent_ids":model.parent_ids,
                "feasible":model.feasible,"invalid_reason":model.invalid_reason,"constraint_count":model.constraint_count,
                "mdl_operators":model.mdl_operators,"mdl_feature_count":model.mdl_feature_count,"adfs":checkpoint_adfs(model),"founder_ids":model.founder_ids,"birth_generation":model.birth_generation}

    def snapshot(self):
        return {"capacity":self.capacity,"complexity_prior":self.complexity_prior,"ess_ratio":self.ess_ratio,
                "particles":[self._model_data(model) for model in self.particles],"weights":self.weights,
                "catalog":[self._model_data(model) for model in self.catalog],"catalog_weights":self.catalog_weights,
                "inverse_temperature":self.inverse_temperature,"last_ess":self.last_ess,"log_evidence":self.log_evidence,
                "resample_count":self.resample_count,"rejuvenation_accepts":self.rejuvenation_accepts,
                "rejuvenation_tries":self.rejuvenation_tries,"post_resample_ess":self.post_resample_ess,
                "unique_particles":self.unique_particles,"history":self.history,
                "predictive_history":self.predictive_history,"last_predictive":self.last_predictive,
                "likelihood_kind":self.likelihood_kind,"student_t_df":self.student_t_df,"noise_scale":self.noise_scale,
                "injection_rate":self.injection_rate,"injection_trials":self.injection_trials,"injection_successes":self.injection_successes}

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data["capacity"],data["complexity_prior"],data["ess_ratio"])
        for key in ("inverse_temperature","last_ess","log_evidence","resample_count","rejuvenation_accepts","rejuvenation_tries","post_resample_ess","unique_particles"):
            setattr(result,key,data.get(key,getattr(result,key)))
        for key in ("history","predictive_history","last_predictive"):
            setattr(result,key,data.get(key,getattr(result,key)))
        for key in ("likelihood_kind","student_t_df","noise_scale","injection_rate","injection_trials","injection_successes"):
            setattr(result,key,data.get(key,getattr(result,key)))
        result.particles=[Model(**item) for item in data.get("particles",[])]
        result.weights=np.asarray(data.get("weights",[]),dtype=float)
        result.catalog=[Model(**item) for item in data.get("catalog",[])]
        result.catalog_weights=np.asarray(data.get("catalog_weights",[]),dtype=float)
        if "source" not in result.last_predictive: result.last_predictive={}
        return result

    @staticmethod
    def _softmax(log_values):
        shifted=log_values-np.max(log_values); values=np.exp(shifted); return values/values.sum()

    def configure_likelihood(self, kind, noise_scale=None):
        if kind not in {"legacy","student_t","bernoulli","categorical"}: raise ValueError(f"Unknown posterior likelihood: {kind}")
        self.likelihood_kind=kind; self.noise_scale=None if noise_scale is None else float(max(EPS,noise_scale))

    def _likelihood_energy(self, model, X, Y, cats):
        raw=predict_model(model,X)
        if self.likelihood_kind=="student_t":
            prediction=raw[:,0]; residual=np.asarray(Y[:,0]-prediction,float)
            scale=self.noise_scale or max(EPS,float(np.median(np.abs(residual-np.median(residual)))*1.4826))
            return float(np.mean(np.log(scale)+((self.student_t_df+1)/2)*np.log1p((residual/scale)**2/self.student_t_df)))
        if self.likelihood_kind=="bernoulli":
            probability=binary_probabilities(raw[:,0])[:,1]
            truth=np.asarray(np.rint(Y[:,0]),int); valid=(truth>=0)&(truth<=1)
            return float(np.mean(np.where(valid,-np.log(np.where(truth==1,probability,1-probability)+EPS),-np.log(EPS))))
        if self.likelihood_kind=="categorical":
            probability=stable_softmax(raw); truth=np.asarray(np.rint(Y[:,0]),int); valid=(truth>=0)&(truth<probability.shape[1]); row=np.arange(len(truth))
            return float(np.mean(np.where(valid,-np.log(np.maximum(probability[row,np.clip(truth,0,probability.shape[1]-1)],EPS)),-np.log(EPS))))
        return float("inf")

    def _energies(self, models, X=None, Y=None, cats=None):
        if X is not None and self.likelihood_kind!="legacy":
            return np.asarray([self._likelihood_energy(model,X,Y,cats)+self.complexity_prior*model_complexity(model) for model in models])
        losses=np.asarray([max(0.0,aggregate_loss(m)) for m in models])
        finite=losses[np.isfinite(losses)]
        scale=max(EPS,float(np.median(finite))) if len(finite) else 1.0
        # log1p makes the likelihood robust to a small number of extreme-loss equations.
        return np.log1p(np.minimum(losses,CLIP)/scale)+self.complexity_prior*np.asarray([model_complexity(m) for m in models])

    @staticmethod
    def _ess(weights): return float(1.0/np.sum(np.square(weights)))

    def _adaptive_weights(self, energies, log_prior=None):
        target=max(1.0,self.ess_ratio*len(energies))
        maximum=4.0
        prior=np.zeros(len(energies)) if log_prior is None else log_prior
        if self._ess(self._softmax(prior-maximum*energies)) >= target: beta=maximum
        else:
            low,high=0.0,maximum
            for _ in range(24):
                middle=(low+high)/2
                if self._ess(self._softmax(prior-middle*energies)) >= target: low=middle
                else: high=middle
            beta=low
        weights=self._softmax(prior-beta*energies)
        return beta,weights

    def update(self, candidates, X=None, Y=None, cats=None, diverse=()):
        """Reweight useful structures while reserving capacity for distinct families."""
        unique={}
        diversity_keys={(repr(model.trees),tuple(model.scales)) for model in diverse}
        for model in [*(self.catalog or self.particles),*candidates,*diverse]:
            if not model.feasible or not np.all(np.isfinite(model.objectives)): continue
            key=(repr(model.trees),tuple(model.scales))
            if key not in unique or secondary_key(model) < secondary_key(unique[key]): unique[key]=model
        ranked=sorted(unique.values(),key=secondary_key)
        # A pure quality trim makes the particle catalog an echo of one local
        # basin.  Keep a small structural reserve so later injections can join
        # partial discoveries that are not yet globally competitive.
        reserve=min(max(1,self.capacity//4),len(ranked))
        chosen=[]; signatures=set()
        for model in ranked:
            key=(repr(model.trees),tuple(model.scales))
            signature=tuple(sorted({node[0] for tree in model.trees for node in walk_tree(tree)}))
            if key in diversity_keys and signature not in signatures:
                chosen.append(model); signatures.add(signature)
                if len(chosen)>=reserve: break
        for model in ranked:
            if len(chosen)>=reserve: break
            signature=tuple(sorted({node[0] for tree in model.trees for node in walk_tree(tree)}))
            if signature not in signatures:
                chosen.append(model); signatures.add(signature)
                if len(chosen)>=reserve: break
        chosen_ids={id(model) for model in chosen}
        for model in ranked:
            if len(chosen)>=self.capacity: break
            if id(model) not in chosen_ids:
                chosen.append(model)
                chosen_ids.add(id(model))
            if len(chosen)>=self.capacity: break
        models=chosen
        if not models: return
        energies=self._energies(models,X,Y,cats); beta,weights=self._adaptive_weights(energies)
        self.inverse_temperature=beta; self.last_ess=self._ess(weights)
        self.log_evidence=float(np.log(np.mean(np.exp(np.clip(-beta*energies,-700,700)))))
        self.catalog=[model.clone() for model in models]; self.catalog_weights=weights.copy()
        self.particles=[model.clone() for model in models]; self.weights=weights.copy()
        resampled=self.last_ess <= self.ess_ratio*len(self.particles)+1e-6
        if resampled: self._resample()
        self.post_resample_ess=self._ess(self.weights)
        self.unique_particles=len({repr(model.trees) for model in self.particles})
        self.history.append({"pre_resample_ess":self.last_ess,"post_resample_ess":self.post_resample_ess,
                             "unique_particles":self.unique_particles,"temperature":self.inverse_temperature,
                             "log_evidence_proxy":self.log_evidence,"particles":len(self.particles),"resampled":resampled})
        self.history=self.history[-200:]

    def _resample(self):
        positions=(np.arange(len(self.particles))+rng.random())/len(self.particles)
        indices=np.searchsorted(np.cumsum(self.weights),positions,side="right")
        self.particles=[self.particles[int(index)].clone() for index in indices]
        self.weights=np.full(len(self.particles),1.0/len(self.particles)); self.resample_count+=1

    def sample(self):
        if not self.particles: return None
        return self.particles[int(np.random.choice(len(self.particles),p=self.weights))].clone()

    def record_injection(self, success):
        self.injection_trials=.98*self.injection_trials+1.
        self.injection_successes=.98*self.injection_successes+float(bool(success))
        rate=(self.injection_successes+1.)/(self.injection_trials+2.)
        self.injection_rate=float(np.clip(.35+.45*rate,.35,.80))

    def posterior_predictive_check(self, X, Y, cats, source="validation"):
        """Record held-out diagnostics appropriate for regression and classification."""
        predictions=[]; weights=[]; raws=[]
        for model,weight in zip(self.particles,self.weights):
            try:
                prediction=predict_targets(model,X,cats); raw=predict_model(model,X)
                if prediction.shape==Y.shape and np.all(np.isfinite(prediction)) and np.all(np.isfinite(raw)):
                    predictions.append(prediction); weights.append(weight); raws.append(raw)
            except (ArithmeticError, IndexError, RecursionError, ValueError):
                continue
        if not predictions: return {}
        weights=np.asarray(weights,dtype=float); weights/=weights.sum()
        stack=np.asarray(predictions)
        result={"source":source,"models":len(predictions)}
        numeric=[index for index, labels in enumerate(cats) if labels is None]
        if numeric:
            numeric_stack=stack[:,:,numeric]; numeric_y=Y[:,numeric]
            mean=np.tensordot(weights,numeric_stack,axes=(0,0))
            spread=np.sqrt(np.tensordot(weights,(numeric_stack-mean)**2,axes=(0,0)))
            residual=numeric_y-mean
            # Particle disagreement is epistemic only.  A residual component
            # is needed before interpreting a band as predictive rather than
            # merely structural spread.
            noise_sd=float(np.sqrt(np.mean(residual**2)))
            predictive_sd=np.sqrt(spread**2+noise_sd**2)
            result["numeric"]={
                "rmse":noise_sd,
                "mean_structural_spread":float(np.mean(spread)),
                "residual_sd":noise_sd,
                "coverage_95":float(np.mean(np.abs(residual)<=1.96*np.maximum(predictive_sd,EPS))),
            }
        categorical=[index for index, labels in enumerate(cats) if labels is not None]
        if categorical:
            correct=total=0; log_losses=[]; confidences=[]; briers=[]
            for index in categorical:
                class_count=len(cats[index])
                truth=np.asarray(np.rint(Y[:,index]),dtype=int)
                valid=(truth>=0)&(truth<class_count)
                if not np.any(valid):
                    continue
                # Only the particles that predicted cleanly above carry weights;
                # re-predicting every particle here misaligned (or raised).
                if class_count>2:
                    particle_probabilities=[stable_softmax(raw) for raw in raws]
                else:
                    particle_probabilities=[binary_probabilities(raw[:,0]) if class_count==2 else np.ones((len(X),1)) for raw in raws]
                probabilities=np.tensordot(weights,np.asarray(particle_probabilities),axes=(0,0))
                chosen=np.argmax(probabilities,axis=1)
                correct+=int(np.sum(chosen[valid]==truth[valid])); total+=int(np.sum(valid))
                row=np.arange(len(truth))[valid]
                log_losses.extend(-np.log(np.maximum(probabilities[row,truth[valid]],EPS)))
                confidences.extend(np.max(probabilities[valid],axis=1))
                one_hot=np.eye(class_count)[truth[valid]]; briers.extend(np.sum((probabilities[valid]-one_hot)**2,axis=1))
            if total:
                result["classification"]={
                    "accuracy":correct/total,
                    "log_loss":float(np.mean(log_losses)),
                    "mean_confidence":float(np.mean(confidences)),"brier_score":float(np.mean(briers)),
                    "observations":total,
                }
        self.last_predictive=result; self.predictive_history.append(result); self.predictive_history=self.predictive_history[-200:]
        return result

    def rejuvenate(self, proposal, X, Y, affine_on, cats, max_nodes, max_depth, attempts=4):
        """Extend and reweight the finite catalogue with scored structural moves.

        Tree mutation is asymmetric and can be irreversible after rewriting;
        it is a discovery proposal, not a Metropolis transition kernel.
        """
        if not self.particles: return
        count=min(attempts,len(self.particles))
        candidates=[]
        for index in rng.sample(range(len(self.particles)),count):
            current=self.particles[index]
            candidate=current.clone(); candidate.trees=[mutate(tree,X.shape[1],proposal.ops,max_nodes,max_depth,proposal,current.adfs) for tree in current.trees]
            candidate.mdl_operators=grammar_for_trees(proposal.ops,candidate.trees,candidate.adfs)
            assess(candidate,X,Y,affine_on,cats)
            self.rejuvenation_tries+=1
            if candidate.feasible:
                candidate.origin="catalogue_rejuvenation"; candidates.append(candidate)
        self.update(candidates,X,Y,cats)
        admitted={(repr(model.trees),tuple(model.scales)) for model in self.catalog}
        self.rejuvenation_accepts+=sum((repr(model.trees),tuple(model.scales)) in admitted for model in candidates)
class BayesianEquationGenerator:
    """Smoothed posterior over useful operators/features for new equations.

    Updates use elite rank rather than raw loss, making a single extreme
    outlier unable to dominate the posterior.  Dirichlet floors, temperature,
    and a uniform sampling mixture preserve exploration and prevent posterior
    collapse onto one operator.
    """
    def __init__(self, ops, n_features, exploration=.25, decay=.96, floor=.35, temperature=1.35):
        self.ops=list(ops); self.n_features=n_features; self.base_exploration=exploration; self.exploration=exploration
        self.pressure_exploration=0.
        self.decay=decay; self.floor=floor; self.temperature=temperature
        self.op_alpha={op:1.0 for op in self.ops}
        self.feature_alpha=np.ones(n_features, dtype=float)
        self.op_draw=None; self.feature_draw=None; self.entropy=1.0
        self.depth_alpha=np.ones(12)
        # A weak half-normal-like magnitude prior for ephemeral constants.
        # The running sufficient statistics are decayed with the rest of the
        # posterior so stale equations cannot lock in an unsuitable scale.
        self.constant_abs_sum=3.0; self.constant_weight=1.0; self.constant_scale=3.0
        self.particles=PosteriorParticlePopulation()

    def sync_operators(self, ops):
        """Refresh a changing grammar without discarding shared evidence."""
        old_alpha=dict(self.op_alpha); old_draw={} if self.op_draw is None else dict(zip(self.ops,self.op_draw))
        self.ops=list(ops); self.op_alpha={op:old_alpha.get(op,1.) for op in self.ops}
        self.op_draw=np.asarray([old_draw.get(op,1.) for op in self.ops],float) if old_draw else None

    def begin_equation(self):
        """Thompson-sample one posterior for a coherent candidate equation."""
        op_alpha=np.maximum(np.array([self.op_alpha[o] for o in self.ops]),self.floor)**(1/self.temperature)
        feature_alpha=np.maximum(self.feature_alpha,self.floor)**(1/self.temperature)
        self.op_draw=np.random.dirichlet(op_alpha)
        self.feature_draw=np.random.dirichlet(feature_alpha)

    def _pick(self, items, weights, prior=None):
        weights=np.asarray(weights,dtype=float)
        if prior is not None: weights=weights*prior
        weights/=weights.sum()
        if rng.random() < self.exploration:
            return rng.choice(items) if prior is None else rng.choices(items,weights=prior)[0]
        return items[int(np.random.choice(len(items),p=weights))]

    def operator(self):
        if self.op_draw is None: self.begin_equation()
        return self._pick(self.ops,self.op_draw,np.asarray([operator_prior(op) for op in self.ops]))
    def feature(self):
        if self.feature_draw is None: self.begin_equation()
        return self._pick(list(range(self.n_features)),self.feature_draw)

    def update(self, elite, X=None, Y=None, cats=None, diverse=()):
        if not elite: return
        self.op_alpha={op:max(self.floor,self.decay*v) for op,v in self.op_alpha.items()}
        self.feature_alpha=np.maximum(self.floor,self.feature_alpha*self.decay)
        self.depth_alpha=np.maximum(self.floor,self.depth_alpha*self.decay)
        self.constant_abs_sum=max(self.floor,self.constant_abs_sum*self.decay)
        self.constant_weight=max(self.floor,self.constant_weight*self.decay)
        # Rank likelihood is robust: exceptionally bad/good numeric scales
        # do not produce disproportionately certain posterior updates.
        ordered=sorted(elite,key=secondary_key)
        for rank,model in enumerate(ordered):
            # Bayesian structural prior: equally accurate simpler equations
            # should contribute more posterior evidence than large trees.
            weight=1.0/((1.0+rank)*math.sqrt(max(1,model_complexity(model))))
            for tree in model.trees: self._observe(tree,weight)
        self.constant_scale=float(np.clip(self.constant_abs_sum/self.constant_weight,.05,100.0))
        probs=np.array([self.op_alpha[o] for o in self.ops],dtype=float); probs/=probs.sum()
        self.entropy=float(-np.sum(probs*np.log(probs+EPS))/math.log(len(probs))) if len(probs)>1 else 1.0
        # When the posterior becomes concentrated, restore exploration rather
        # than repeatedly generating near-identical equations.
        self.exploration=min(.60,self.base_exploration+self.pressure_exploration+max(0.,.65-self.entropy)*.7)
        self.begin_equation()
        self.particles.update(elite,X,Y,cats,diverse)

    def sample_particle(self): return self.particles.sample()

    def rejuvenate_particles(self, X, Y, affine_on, cats, max_nodes, max_depth):
        self.particles.rejuvenate(self,X,Y,affine_on,cats,max_nodes,max_depth)

    def record_predictive_check(self, X, Y, cats, source="validation"):
        return self.particles.posterior_predictive_check(X,Y,cats,source)

    def summary(self):
        top=sorted(self.op_alpha,key=self.op_alpha.get,reverse=True)[:4]
        predictive=self.particles.last_predictive
        check=", PPC=unavailable (no validation set)"
        if predictive and "source" in predictive:
            parts=[]
            if "numeric" in predictive:
                numeric=predictive["numeric"]
                parts.append(f"RMSE={numeric['rmse']:.4g}, structural spread={numeric['mean_structural_spread']:.4g}, "
                             f"95% band={numeric['coverage_95']:.0%}")
            if "classification" in predictive:
                classification=predictive["classification"]
                parts.append(f"class accuracy={classification['accuracy']:.1%}, log loss={classification['log_loss']:.4g}, "
                             f"confidence={classification['mean_confidence']:.1%}")
            check=f", {predictive['source']} PPC " + "; ".join(parts) if parts else ", PPC=no valid held-out targets"
        return (f"Adaptive Bayesian proposals: entropy={self.entropy:.2f}, exploration={self.exploration:.0%}, injection={self.particles.injection_rate:.0%}, "
                f"catalogue draws={len(self.particles.particles)} slots/{self.particles.unique_particles} unique, "
                f"ESS={self.particles.last_ess:.1f}->{self.particles.post_resample_ess:.1f}, "
                f"temperature={self.particles.inverse_temperature:.3g}, logZ-proxy={self.particles.log_evidence:.3g}" + check +
                ", top ops=" + ", ".join(top))

    def stop_probability(self, depth):
        index=min(depth,len(self.depth_alpha)-1)
        relative_evidence=self.depth_alpha[index]/max(EPS,float(np.mean(self.depth_alpha)))
        base=.18+depth*.06
        # Frequently observed depths make a continuation less surprising;
        # unsupported depths retain the conservative shallow-tree bias.
        return min(.75,max(.08,base-.06*math.tanh(math.log(relative_evidence))))
    def constant(self): return float(np.clip(rng.gauss(0,self.constant_scale),-100,100))

    def _observe(self, tree, weight, depth=0):
        self.depth_alpha[min(depth,len(self.depth_alpha)-1)]+=weight
        if tree[0]=="x": self.feature_alpha[tree[1]]+=weight; return
        if tree[0]=="c":
            magnitude=abs(float(tree[1]))
            self.constant_abs_sum+=weight*min(100.0,magnitude)
            self.constant_weight+=weight
            return
        if tree[0] in self.op_alpha: self.op_alpha[tree[0]]+=weight
        for child in tree[1:]: self._observe(child,weight,depth+1)

# Bank catalogues are mostly unchanged between generations, yet every entry was
# rescored on all training rows each generation.  Scoring is deterministic in
# the trees, called ADFs, grammar and data, so reuse it (age is live state).
_PARTICLE_SCORE_CACHE={}
def assess_particle_cached(particle, X, Y, affine_on, cats):
    key=(repr(particle.trees),adf_signature(particle.trees,particle.adfs),tuple(particle.mdl_operators),particle.mdl_feature_count,
         array_digest(X),array_digest(Y),bool(affine_on),repr(cats))
    hit=_PARTICLE_SCORE_CACHE.get(key)
    if hit is None:
        assess(particle,X,Y,affine_on,cats)
        if len(_PARTICLE_SCORE_CACHE)>=20000: _PARTICLE_SCORE_CACHE.clear()
        _PARTICLE_SCORE_CACHE[key]=(list(particle.scales),tuple(particle.objectives[:-1]),particle.feasible,particle.invalid_reason,particle.constraint_count)
        return
    scales,objectives,particle.feasible,particle.invalid_reason,particle.constraint_count=hit
    particle.scales=list(scales); particle.objectives=(*objectives,particle.age)

class PerOutputBayesianBanks:
    """One adaptive catalogue per logical output; multiclass entries own all logits."""
    def __init__(self, ops, n_features, n_outputs=None, particles=96, cats=None):
        self.head_groups=classification_layout(cats)[0] if cats is not None else tuple((index,) for index in range(n_outputs))
        self.head_to_bank={head:index for index,heads in enumerate(self.head_groups) for head in heads}
        self.banks=[BayesianEquationGenerator(ops,n_features) for _ in self.head_groups]
        for bank in self.banks: bank.particles=PosteriorParticlePopulation(capacity=particles)
    def __len__(self): return sum(len(heads) for heads in self.head_groups)
    def __getitem__(self,index): return self.banks[self.head_to_bank[index]]
    def begin_equation(self):
        for bank in self.banks: bank.begin_equation()
    def sync_operators(self, ops):
        for bank in self.banks: bank.sync_operators(ops)
    def update(self, elite, cats=None, X=None, Y=None, diverse=(), affine_on=True):
        labels=cats if cats is not None else [None]*len(self.banks)
        for target,(heads,bank) in enumerate(zip(self.head_groups,self.banks)):
            projected=[]; projected_diverse=[]
            for model, destination in ((model,projected) for model in elite):
                if max(heads)>=len(model.trees): continue
                q=_quality_objectives(model); score=q[2*target:2*target+2] if len(q)>=2*target+2 else (float("inf"),)*2
                particle=Model([model.trees[head] for head in heads],[model.scales[head] for head in heads],model.age,tuple(score)+(0.,model.age),lineage_id=model.lineage_id,feasible=model.feasible,mdl_operators=model.mdl_operators,mdl_feature_count=model.mdl_feature_count,adfs=dict(model.adfs),founder_ids=model.founder_ids,birth_generation=model.birth_generation)
                particle.objectives=tuple(score)+(model_description_bits(particle,X.shape[1] if X is not None else None),model.age)
                destination.append(particle)
            for model, destination in ((model,projected_diverse) for model in diverse):
                if max(heads)>=len(model.trees): continue
                q=_quality_objectives(model); score=q[2*target:2*target+2] if len(q)>=2*target+2 else (float("inf"),)*2
                particle=Model([model.trees[head] for head in heads],[model.scales[head] for head in heads],model.age,tuple(score)+(0.,model.age),lineage_id=model.lineage_id,feasible=model.feasible,mdl_operators=model.mdl_operators,mdl_feature_count=model.mdl_feature_count,adfs=dict(model.adfs),founder_ids=model.founder_ids,birth_generation=model.birth_generation)
                particle.objectives=tuple(score)+(model_description_bits(particle,X.shape[1] if X is not None else None),model.age)
                destination.append(particle)
            label=labels[target] if target<len(labels) else None
            if X is not None and Y is not None:
                for particle in [*bank.particles.catalog,*bank.particles.particles]:
                    assess_particle_cached(particle,X,Y[:,target:target+1],affine_on,[label])
            if label is None: kind="student_t"
            elif len(label)==2: kind="bernoulli"
            else: kind="categorical"
            if X is not None and Y is not None and projected:
                if kind=="student_t":
                    prediction=predict_model(projected[0],X)[:,0]
                    scale=max(EPS,float(np.median(np.abs(Y[:,target]-prediction-np.median(Y[:,target]-prediction)))*1.4826))
                    bank.particles.configure_likelihood(kind,scale)
                else: bank.particles.configure_likelihood(kind)
            bank.update(projected,X,None if Y is None else Y[:,target:target+1],[label],projected_diverse)
    def rejuvenate_particles(self,X,Y,affine_on,cats,max_nodes,max_depth,constraints=None,output_names=()):
        for target,(heads,bank) in enumerate(zip(self.head_groups,self.banks)):
            if not bank.particles.particles: continue
            bank.rejuvenate_particles(X,Y[:,target:target+1],affine_on,[cats[target]],max_nodes,max_depth)
    def record_predictive_check(self,X,Y,cats,source="validation"):
        return [bank.record_predictive_check(X,Y[:,target:target+1],[cats[target]],source)
                for target,bank in enumerate(self.banks)]
    def record_injection(self, success):
        for bank in self.banks: bank.particles.record_injection(success)
    def summary(self): return " | ".join(f"output {j}: {bank.summary()}" for j,bank in enumerate(self.banks))

def bayesian_banks_snapshot(banks):
    return {"head_groups":banks.head_groups,"banks":[{"ops":b.ops,"n_features":b.n_features,"base_exploration":b.base_exploration,"exploration":b.exploration,"pressure_exploration":b.pressure_exploration,"decay":b.decay,"floor":b.floor,"temperature":b.temperature,"op_alpha":b.op_alpha,"feature_alpha":b.feature_alpha,"op_draw":b.op_draw,"feature_draw":b.feature_draw,"entropy":b.entropy,"depth_alpha":b.depth_alpha,"constant_abs_sum":b.constant_abs_sum,"constant_weight":b.constant_weight,"constant_scale":b.constant_scale,"particles":b.particles.snapshot()} for b in banks.banks]}

def bayesian_banks_from_snapshot(items):
    if not items: raise ValueError("Checkpoint lacks per-output Bayesian banks; start a new run")
    if isinstance(items,list):
        result=PerOutputBayesianBanks(items[0]["ops"],items[0]["n_features"],len(items),items[0]["particles"]["capacity"]); result.legacy_head_banks=True; payload=items
    else:
        payload=items["banks"]; result=PerOutputBayesianBanks(payload[0]["ops"],payload[0]["n_features"],len(payload),payload[0]["particles"]["capacity"])
        result.head_groups=tuple(tuple(group) for group in items["head_groups"]); result.head_to_bank={head:index for index,group in enumerate(result.head_groups) for head in group}; result.banks=[BayesianEquationGenerator(data["ops"],data["n_features"]) for data in payload]
    for target,data in zip(result.banks,payload):
        for key in ("exploration","pressure_exploration","op_alpha","feature_alpha","op_draw","feature_draw","entropy","depth_alpha","constant_abs_sum","constant_weight","constant_scale"):
            if key in data: setattr(target,key,data[key])
        target.particles=PosteriorParticlePopulation.from_snapshot(data["particles"])
    return result

# Operators are proposed with probability proportional to 2**(-strength*bonus):
# every operator stays reachable, but simple algebra is tried far more often
# than branches, digit tricks, or noise.  0 restores uniform proposals.
OPERATOR_PRIOR_STRENGTH = 1.0
def operator_prior(op):
    return 2.0**(-OPERATOR_PRIOR_STRENGTH*OP_COMPLEXITY_BONUS.get(op,2))
def choose_operator(options):
    options=list(options)
    return options[0] if len(options)==1 else rng.choices(options,weights=[operator_prior(op) for op in options])[0]
def operator_arity(op, adfs=None):
    return int((adfs or {}).get(op,{}).get("arity",OPS.get(op,(-1,))[0]))

def random_tree(n_features, ops, max_nodes, max_depth, depth=0, proposal=None, adfs=None):
    if proposal is not None and depth==0: proposal.begin_equation()
    stop=proposal.stop_probability(depth) if proposal else .28
    if depth >= max_depth or max_nodes <= 1 or rng.random()<stop:
        return ("x",proposal.feature() if proposal else rng.randrange(n_features)) if rng.random()<.65 else ("c",proposal.constant() if proposal else rng.uniform(-3,3))
    # A Bayesian proposal can be one promotion ahead of a particle's local
    # catalog.  Never turn that transient mismatch into a zero-argument ADF
    # node: use only operators whose arity is known by this tree's catalog.
    available=[candidate for candidate in ops if 0<operator_arity(candidate,adfs)<max_nodes]
    if not available:
        return ("x",proposal.feature() if proposal else rng.randrange(n_features)) if rng.random()<.65 else ("c",proposal.constant() if proposal else rng.uniform(-3,3))
    op=proposal.operator() if proposal else choose_operator(available)
    if op not in available: op=choose_operator(available)
    arity=operator_arity(op,adfs)
    budget=max(1,(max_nodes-1)//arity)
    return tuple([op]+[random_tree(n_features,ops,budget,max_depth,depth+1,proposal,adfs) for _ in range(arity)])
def mutate(t, n_features, ops, max_nodes, max_depth, proposal=None, adfs=None):
    """Regrow or perturb one node.  max_nodes/max_depth are the room left at
    this position, so a regrown subtree fits instead of failing the size
    check (which discarded the whole move), and a leaf always changes."""
    if rng.random()<.18: return random_tree(n_features,ops,max(1,max_nodes),max(0,max_depth),proposal=proposal,adfs=adfs)
    if t[0]=="x":
        if rng.random()<.5:
            for _ in range(4):
                feature=proposal.feature() if proposal else rng.randrange(n_features)
                if feature!=t[1]: return ("x",feature)
        return ("c",proposal.constant() if proposal else rng.uniform(-3,3))
    if t[0]=="c":
        if rng.random()<.5: return ("c",float(np.clip(t[1]+rng.gauss(0,.5),-100,100)))
        return ("x",proposal.feature() if proposal else rng.randrange(n_features))
    children=list(t[1:]); i=rng.randrange(len(children))
    room=max_nodes-(node_size(t)-node_size(children[i]))
    children[i]=mutate(children[i],n_features,ops,max(1,room),max_depth-1,proposal,adfs)
    candidate=simplify_tree(tuple([t[0]]+children))
    return candidate if node_size(candidate)<=max_nodes and node_depth(candidate)<=max_depth else t

def bayesian_injection_trees(bayes, n_outputs, n_features, ops, max_nodes, max_depth, particle_mode="adaptive", adfs=None, return_sources=False):
    """Draw structures while retaining the ancestry of reused catalogue entries."""
    sources=[]
    if isinstance(bayes,PerOutputBayesianBanks):
        trees=[None]*n_outputs; scales=[(1.,0.)]*n_outputs
        for heads,bank in zip(bayes.head_groups,bayes.banks):
            particle=bank.sample_particle()
            particle_rate=.65 if particle_mode=="fixed" else bank.particles.injection_rate
            if particle_mode!="grammar" and particle is not None and particle.trees and rng.random()<particle_rate:
                sources.append(particle)
                for local,head in enumerate(heads):
                    trees[head]=mutate(particle.trees[local],n_features,ops,max_nodes,max_depth,proposal=bank,adfs=adfs); scales[head]=particle.scales[local]
            else:
                for head in heads: trees[head]=random_tree(n_features,ops,max_nodes,max_depth,proposal=bank,adfs=adfs)
        return (trees,scales,sources) if return_sources else (trees,scales)
    particle=bayes.sample_particle()
    if particle_mode!="grammar" and particle is not None and len(particle.trees)==n_outputs and rng.random()<.65:
        trees=[mutate(tree,n_features,ops,max_nodes,max_depth,proposal=bayes,adfs=adfs) for tree in particle.trees]
        return (trees,list(particle.scales),[particle]) if return_sources else (trees,list(particle.scales))
    trees=[random_tree(n_features,ops,max_nodes,max_depth,proposal=bayes,adfs=adfs) for _ in range(n_outputs)]
    scales=[(1.,0.)]*n_outputs
    return (trees,scales,[]) if return_sources else (trees,scales)

def point_mutate(t, ops, adfs=None):
    if t[0] in ("x","c"): return t
    # Exclude the current operator: redrawing it is a no-op move.
    options=[op for op in ops if op!=t[0] and operator_arity(op,adfs)==operator_arity(t[0],adfs)]
    return tuple([choose_operator(options)]+list(t[1:])) if options else t

def hoist_mutate(t):
    # The root path would return the tree itself; hoist a proper subtree.
    paths=subtree_paths(t)[1:]
    return subtree_at(t,rng.choice(paths)) if paths else t
def constant_mutate(t):
    """Nudge one existing constant, scaled to its magnitude; None without constants."""
    paths=constant_paths(t)
    if not paths: return None
    path=rng.choice(paths); value=float(subtree_at(t,path)[1])
    nudged=float(np.clip(value+rng.gauss(0,.25*max(1.,abs(value))),-CONSTANT_LIMIT,CONSTANT_LIMIT))
    return replace_subtree(t,path,("c",nudged))
def bilinear_mutate(t, n_features, ops, max_nodes, max_depth):
    """Insert xa*xb or xa*xb - xc*xd (a 2x2 determinant / interaction term).

    Cross products, determinants and Cramer-style ratios only pay off once a
    whole product-difference exists; building it one node at a time gets no
    partial credit, so offer the block as one move.
    """
    if "*" not in ops: return t
    feature=lambda: ("x",rng.randrange(n_features))
    block=("*",feature(),feature())
    if "-" in ops and rng.random()<.6: block=("-",block,("*",feature(),feature()))
    path=rng.choice(subtree_paths(t)); child=replace_subtree(t,path,block)
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t
def parametrize_mutate(t, ops, max_nodes, max_depth):
    """Give the constant fitter a handle: s -> c*s or s -> s+c at an inner node.

    Turns sin(x) into sin(c*x) or x/(x) into x/(x+c); fitting then moves c to
    its optimum.  The start value is perturbed so the child differs from its
    parent (identity moves are rejected by the semantic guard).
    """
    forms=[form for form in ("*","+") if form in ops]
    paths=[path for path in subtree_paths(t) if path and subtree_at(t,path)[0]!="c"]
    if not forms or not paths: return t
    path=rng.choice(paths); target=subtree_at(t,path); form=rng.choice(forms)
    wrapped=("*",("c",float(np.clip(1.+rng.gauss(0,.5),-10,10))),target) if form=="*" else ("+",target,("c",float(rng.gauss(0,1.))))
    child=replace_subtree(t,path,wrapped)
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t
def shrink_mutate(t): return rng.choice(t[1:]) if t[0] not in ("x","c") else t

# Off by default: it did not solve the linear-system benchmark and was neutral
# to harmful elsewhere (4-way ablation).  Set > 0 to re-enable.
BILINEAR_MUTATION_WEIGHT = 0.
class MutationPortfolio:
    def __init__(self):
        self.weights={"subtree":1.,"point":1.,"constant":1.,"hoist":.7,"shrink":.7,"parametrize":1.,"bilinear":BILINEAR_MUTATION_WEIGHT}
        self.tries={k:0 for k in self.weights}; self.wins={k:0 for k in self.weights}
    def choose(self): return rng.choices(list(self.weights),weights=list(self.weights.values()))[0]
    def record(self, kind, improved):
        self.tries[kind]+=1; self.wins[kind]+=int(improved)
        if self.tries[kind]%8==0: self.weights[kind]=max(.15,min(4.,.5+3*self.wins[kind]/self.tries[kind]))
    def apply(self,t,n_features,ops,max_nodes,max_depth,proposal=None,adfs=None):
        kind=self.choose()
        if kind=="point": return point_mutate(t,ops,adfs),kind
        if kind=="hoist": return hoist_mutate(t),kind
        if kind=="shrink": return shrink_mutate(t),kind
        if kind=="parametrize": return parametrize_mutate(t,ops,max_nodes,max_depth),kind
        if kind=="bilinear": return bilinear_mutate(t,n_features,ops,max_nodes,max_depth),kind
        if kind=="constant":
            # Without constants to move, give the tree one (credited as such).
            child=constant_mutate(t)
            return (child,kind) if child is not None else (parametrize_mutate(t,ops,max_nodes,max_depth),"parametrize")
        return mutate(t,n_features,ops,max_nodes,max_depth,proposal,adfs),kind
    def snapshot(self): return {"weights":self.weights,"tries":self.tries,"wins":self.wins}
    def restore(self,data):
        # Merge so checkpoints from before a mutation kind existed still load.
        for name in ("weights","tries","wins"):
            if name in data: getattr(self,name).update(data[name])

# The semantic guard discards moves that change a tree's output by more than
# max_delta (17% of attempts); a "macro" lane lets such jumps through.  It
# opens by MACRO_STAGNATION_STEP per generation without a new best model.
# Off by default: in a 4-seed benchmark (.02) it lost to the fixed rate in 10
# of 16 matched runs and cancelled most of the age cap's gains.
MACRO_STAGNATION_STEP, MACRO_MAX_RATE = 0., .50
_FRAGMENT_FIT_CACHE={}; _FRAGMENT_FIT_FAILED=object()
FRAGMENT_MIN_CONTRIBUTION = 5e-3   # held-out share of the residual loss a fragment must remove
FRAGMENT_CONTRIBUTION_DECAY = .95
FRAGMENT_MIGRANT_PROBATION = 10   # observations a migrated fragment may stay unproven
def fragment_key(tree):
    """Identity of a fragment up to the affine fit it always receives: c*f,
    f*c, f+c, c+f, f-c and neg(f) all key as f (then algebraic equivalence)."""
    while True:
        op=tree[0]
        if op=="neg" and len(tree)==2: tree=tree[1]; continue
        if op in ("*","+") and len(tree)==3 and (tree[1][0]=="c")!=(tree[2][0]=="c"):
            tree=tree[2] if tree[1][0]=="c" else tree[1]; continue
        if op=="-" and len(tree)==3 and tree[2][0]=="c" and tree[1][0]!="c": tree=tree[1]; continue
        return equivalence_key(tree)
def _fragment_halves(n):
    """Deterministic interleaved row halves for cross-fitting (None when too few rows)."""
    if n<8: return None
    rows=np.arange(n); return rows[0::2],rows[1::2]
def cross_fitted_reduction(values, residual, halves):
    """Held-out residual reduction of an affinely fitted fragment beyond the best constant.

    Fit on one half, score on the other, both ways, and average.  A fragment
    that is constant on the rows earns nothing (it can only move the mean,
    which the model's own affine readout already does)."""
    values=np.asarray(values,float)
    if not np.all(np.isfinite(values)) or float(np.std(values))<=1e-12*(1.+float(np.mean(np.abs(values)))): return 0.
    if halves is None: return 0.
    gains=[]
    for fit,score in (halves,halves[::-1]):
        scale,offset=affine(values[fit],residual[fit])
        _,level=affine(np.zeros(len(fit)),residual[fit])
        target=residual[score]
        gains.append(robust_loss(np.full(len(score),level),target)-robust_loss(clean(scale*values[score]+offset),target))
    return float(np.mean(gains))
class FragmentLibrary:
    """Small, checkpointable store of partial symbolic discoveries."""
    def __init__(self, capacity=96, fragment_rate=.15, macro_rate=.10, probe_reserve=24):
        self.capacity=int(capacity); self.fragment_rate=float(fragment_rate); self.macro_rate=float(macro_rate); self.probe_reserve=int(probe_reserve)
        self.items={}; self.fragment_attempts=0; self.fragment_survivors=0
        self.macro_attempts=0; self.macro_survivors=0; self.admissions=0; self.rejections=0
        self.probe_candidates=0; self.probe_accepted=0; self.scaffold_attempts=0
        self.base_macro_rate=self.macro_rate; self.last_improvement=0

    def adapt_macro_rate(self, generation, improved):
        """Let more large semantic moves through the longer the best model stalls."""
        if improved: self.last_improvement=int(generation)
        stalled=max(0,int(generation)-self.last_improvement)
        self.macro_rate=min(MACRO_MAX_RATE,self.base_macro_rate+MACRO_STAGNATION_STEP*stalled)

    @staticmethod
    def _item_score(item):
        return float(item.get("support",0))+float(item.get("contribution",0.))

    def _trim_items(self):
        """Reserve bounded capacity for independently validated discoveries."""
        probes=[item for item in self.items.values() if item.get("source")=="interaction_probe"]
        probes=sorted(probes,key=lambda item:(-self._item_score(item),repr(item["tree"])))[:min(self.probe_reserve,self.capacity)]
        retained={fragment_key(item["tree"]):item for item in probes}
        ordinary=[item for item in self.items.values() if fragment_key(item["tree"]) not in retained]
        ordinary.sort(key=lambda item:(-self._item_score(item),repr(item["tree"])))
        for item in ordinary[:max(0,self.capacity-len(retained))]: retained[fragment_key(item["tree"])]=item
        self.items=retained

    def admit_discoveries(self, report):
        """Add evidence-backed pre-evolution fragments without creating models."""
        self.probe_candidates+=int(report.get("screened",0))
        for discovery in report.get("accepted",[]):
            tree=simplify_tree(discovery["tree"]); key=fragment_key(tree)
            item=self.items.get(key)
            if item is None:
                item={"tree":tree,"support":2,"contribution":max(0.,float(discovery.get("contribution",0.))),
                      "uses":0,"rejections":0,"source":"interaction_probe","family":discovery.get("family","unknown"),"evidence":discovery}
                self.items[key]=item; self.probe_accepted+=1
            else:
                item["support"]=max(int(item.get("support",0)),2)
                item["contribution"]=max(float(item.get("contribution",0.)),max(0.,float(discovery.get("contribution",0.))))
        self._trim_items()

    def observe(self, models, X, Y, cats):
        """Collect independently supported or held-out-helpful subtrees.

        A fragment is admitted when it recurs in models from at least two
        independent lineages (disjoint founder sets), or when, affinely fitted
        to a model's residual on one half of the rows, it reduces that residual
        on the other half beyond what the best constant does (cross-fitted, so
        chance correlations earn nothing).  Constant-valued fragments are
        rejected, affine rescalings of one fragment (c*f, f+c) share a single
        entry, and evolved contributions decay so stale credit fades."""
        targets,_=classification_layout(cats)
        for item in self.items.values():
            if item.get("source")!="interaction_probe": item["contribution"]=float(item["contribution"])*FRAGMENT_CONTRIBUTION_DECAY
            if item.get("probation"): item["probation"]=int(item["probation"])-1
        halves=_fragment_halves(len(X))
        for model in models[:32]:
            founders=set(model.founder_ids)
            try: prediction=predict_targets(model,X,cats)
            except (ArithmeticError, IndexError, ValueError): continue
            for output,(labels,heads) in enumerate(zip(cats,targets)):
                if labels is not None or not heads: continue
                residual=np.asarray(Y[:,output]-prediction[:,output],float)
                residual_key=(hashlib.blake2b(np.ascontiguousarray(residual).view(np.uint8),digest_size=16).digest(),array_digest(X),adf_signature(model.trees,model.adfs))
                tree=model.trees[heads[0]]
                for path in subtree_paths(tree):
                    fragment=subtree_at(tree,path); size=node_size(fragment)
                    if not 2<=size<=12: continue
                    key=fragment_key(fragment); item=self.items.get(key)
                    if item is None:
                        item={"tree":fragment,"support":0,"contribution":0.,"uses":0,"rejections":0,"source":"evolved","family":"evolved"}
                        self.items[key]=item
                    elif size<node_size(item["tree"]): item["tree"]=fragment      # keep the simplest spelling
                    lineage=set(item.setdefault("founders",[]))
                    if not founders&lineage:
                        item["support"]+=1; item["founders"]=sorted(lineage|founders)[-256:]
                    # Elite/archive/QD models recur across generations, so the
                    # same fragment is refitted to the same residual; the fit
                    # is deterministic, so reuse it (failures included).
                    fit_key=(key,residual_key); reduction=_FRAGMENT_FIT_CACHE.get(fit_key)
                    if reduction is None:
                        try: reduction=cross_fitted_reduction(evaluate_cached(fragment,X,model.adfs),residual,halves)
                        except (ArithmeticError, IndexError, ValueError): reduction=_FRAGMENT_FIT_FAILED
                        if len(_FRAGMENT_FIT_CACHE)>=100_000: _FRAGMENT_FIT_CACHE.clear()
                        _FRAGMENT_FIT_CACHE[fit_key]=reduction
                    if reduction is _FRAGMENT_FIT_FAILED:
                        item["rejections"]+=1; self.rejections+=1; continue
                    item["contribution"]=max(float(item["contribution"]),float(reduction))
        admitted=[item for item in self.items.values()
                  if item.get("source")=="interaction_probe" or item.get("probation",0)>0
                  or item["support"]>=2 or item["contribution"]>FRAGMENT_MIN_CONTRIBUTION]
        self.rejections+=len(self.items)-len(admitted)
        self.admissions=len(admitted)
        admitted.sort(key=lambda item:(-(item["support"]+item["contribution"]),repr(item["tree"])))
        self.items={fragment_key(item["tree"]):item for item in admitted}
        self._trim_items()

    def sample(self, family=None, exclude_families=()):
        choices=[item for item in self.items.values() if (family is None or item.get("family")==family) and item.get("family") not in set(exclude_families)]
        if not choices: return None
        weights=[item["support"]+item["contribution"]+1e-3 for item in choices]
        return rng.choices(choices,weights=weights)[0]

    def compose(self, tree, ops, max_nodes, max_depth):
        item=self.sample()
        if item is None: return None
        fragment=item["tree"]
        candidates=[]
        if "+" in ops: candidates.append((("+",tree,fragment),False))
        if "*" in ops: candidates.append((("*",tree,fragment),False))
        paths=subtree_paths(tree)
        if len(paths)>1: candidates.append((replace_subtree(tree,rng.choice(paths[1:]),fragment),False))
        predicate=self.sample(family="predicate") if "if_else" in ops else None
        branch=self.sample(exclude_families=("predicate",)) if predicate is not None else None
        if predicate is not None and branch is not None:
            candidates.extend([(("if_else",predicate["tree"],tree,branch["tree"]),True),
                               (("if_else",predicate["tree"],branch["tree"],tree),True)])
        rng.shuffle(candidates)
        for candidate,is_scaffold in candidates:
            candidate=simplify_tree(candidate)
            if node_size(candidate)<=max_nodes and node_depth(candidate)<=max_depth:
                item["uses"]+=1; self.fragment_attempts+=1; self.scaffold_attempts+=int(is_scaffold)
                return candidate
        item["rejections"]+=1; self.rejections+=1
        return None

    def record(self, origin, survived):
        if origin=="fragment": self.fragment_survivors+=int(survived)
        elif origin=="macro_mutation": self.macro_survivors+=int(survived)

    def snapshot(self):
        return {"capacity":self.capacity,"fragment_rate":self.fragment_rate,"macro_rate":self.macro_rate,"probe_reserve":self.probe_reserve,
                "base_macro_rate":self.base_macro_rate,"last_improvement":self.last_improvement,
                "items":list(self.items.values()),"fragment_attempts":self.fragment_attempts,
                "fragment_survivors":self.fragment_survivors,"macro_attempts":self.macro_attempts,
                "macro_survivors":self.macro_survivors,"admissions":self.admissions,"rejections":self.rejections,
                "probe_candidates":self.probe_candidates,"probe_accepted":self.probe_accepted,"scaffold_attempts":self.scaffold_attempts}

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data.get("capacity",96),data.get("fragment_rate",.15),data.get("macro_rate",.10),data.get("probe_reserve",24))
        for key in ("fragment_attempts","fragment_survivors","macro_attempts","macro_survivors","admissions","rejections","probe_candidates","probe_accepted","scaffold_attempts"):
            setattr(result,key,int(data.get(key,0)))
        result.base_macro_rate=float(data.get("base_macro_rate",result.macro_rate)); result.last_improvement=int(data.get("last_improvement",0))
        result.items={fragment_key(item["tree"]):item for item in data.get("items",[]) if "tree" in item}; result._trim_items()
        return result

    def stats(self):
        return (f"Fragments={len(self.items)}/{self.capacity}; admitted={self.admissions}; "
                f"probe candidates/accepted={self.probe_candidates}/{self.probe_accepted}; "
                f"probe reserve/scaffolds={self.probe_reserve}/{self.scaffold_attempts}; "
                f"uses/survivors={self.fragment_attempts}/{self.fragment_survivors}; "
                f"macro uses/survivors={self.macro_attempts}/{self.macro_survivors}")

class CasePopulation:
    """Difficulty/diversity/coverage weighted training-case sampler with uniform guard."""
    def __init__(self,n):
        self.weights=np.ones(n); self.last_sample=np.arange(n); self.visits=np.zeros(n,dtype=np.int64)
    def update(self,models,X,Y,cats):
        if not models: return
        preds=np.array([predict_targets(m,X,cats) for m in models])
        errors=np.empty((len(models),len(X),Y.shape[1]))
        for j,labels in enumerate(cats):
            errors[:,:,j]=np.abs(preds[:,:,j]-Y[:,j]) if labels is None else (np.rint(preds[:,:,j])!=Y[:,j])
        residual=np.median(np.mean(errors,axis=2),axis=0)
        disagreement=np.mean(np.std(preds,axis=0),axis=1)
        coverage=1/np.sqrt(1+self.visits)
        score=(residual/(np.mean(residual)+EPS)+disagreement/(np.mean(disagreement)+EPS)+
               coverage/(np.mean(coverage)+EPS))
        self.weights=.8*self.weights+.2*np.maximum(score,EPS); self.weights/=self.weights.sum()
    def sample(self,count):
        count=min(count,len(self.weights)); uniform=max(1,count//4)
        u=np.random.choice(len(self.weights),uniform,replace=False)
        remaining=np.setdiff1d(np.arange(len(self.weights)),u,assume_unique=True)
        p=self.weights[remaining]; p/=p.sum()
        w=np.random.choice(remaining,count-uniform,replace=False,p=p) if count>uniform else np.array([],int)
        self.last_sample=np.r_[u,w]; self.visits[self.last_sample]+=1; return self.last_sample
    def diagnostics(self,X):
        q=np.quantile(self.weights,[.1,.5,.9])
        sampled=X[self.last_sample] if len(self.last_sample) else X
        shift=float(np.mean(np.abs(np.mean(sampled,axis=0)-np.mean(X,axis=0))/(np.std(X,axis=0)+EPS)))
        return f"cases={len(self.last_sample)}/{len(X)}; weight q10/50/90={q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}; feature-shift={shift:.3g}"
    def snapshot(self):
        return {"weights":self.weights.copy(),"last_sample":self.last_sample.copy(),"visits":self.visits.copy()}
    @classmethod
    def from_snapshot(cls, data, n):
        result=cls(n)
        weights=np.asarray(data["weights"],float)
        last_sample=np.asarray(data["last_sample"],int)
        if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
            raise ValueError("Checkpoint has invalid case-population weights")
        if np.any(last_sample<0) or np.any(last_sample>=n):
            raise ValueError("Checkpoint has invalid case-population sample")
        visits=np.asarray(data.get("visits",np.zeros(n)),dtype=np.int64)
        if visits.shape != (n,) or np.any(visits<0): raise ValueError("Checkpoint has invalid case-population visits")
        # Sampling normalizes its local probabilities. Preserve the exact
        # saved adaptation state; rescaling here changes the next EMA update.
        result.weights=weights.copy(); result.last_sample=last_sample; result.visits=visits
        return result

def semantic_distance(baseline, candidate):
    """Normalize moves without trapping constant parents behind a zero scale."""
    difference=candidate-baseline
    if not np.all(np.isfinite(difference)): return float("inf")
    scale=float(np.std(baseline))
    if scale<EPS:
        scale=max(float(np.std(candidate)),float(np.sqrt(np.mean(difference**2))),EPS)
    return float(np.sqrt(np.mean((difference/scale)**2)))

def semantic_mutate(tree, X, portfolio, n_features, ops, max_nodes, max_depth, proposal=None, adfs=None, library=None, min_delta=1e-8, max_delta=5.0):
    # A returned kind of None means "no move was applied": the unchanged
    # parent must not be credited (or blamed) to any mutation kind.
    try: baseline=evaluate_cached(tree,X,adfs)
    except ValueError: return tree,None,False
    for _ in range(6):
        child,kind=portfolio.apply(tree,n_features,ops,max_nodes,max_depth,proposal,adfs)
        try: delta=semantic_distance(baseline,evaluate_cached(child,X,adfs))
        except ValueError: continue
        if min_delta < delta <= max_delta: return child,kind,False
    # A rare macro lane keeps the normal semantic guard as the default while
    # allowing a finite, genuinely different step across distant basins.
    if library is not None and rng.random()<library.macro_rate:
        for _ in range(3):
            child,kind=portfolio.apply(tree,n_features,ops,max_nodes,max_depth,proposal,adfs)
            try: delta=semantic_distance(baseline,evaluate_cached(child,X,adfs))
            except ValueError: continue
            if delta>max_delta and np.isfinite(delta):
                library.macro_attempts+=1
                return child,kind,True
    return tree,None,False

def subtree_paths(tree, prefix=()):
    paths=[prefix]
    if tree[0] not in ("x","c","arg"):
        for i,child in enumerate(tree[1:]): paths.extend(subtree_paths(child,prefix+(i,)))
    return paths

def walk_tree(tree):
    """Yield every node once; used for bounded structural diversity summaries."""
    yield tree
    if tree[0] not in ("x","c","arg"):
        for child in tree[1:]:
            yield from walk_tree(child)
def subtree_at(tree, path):
    for i in path: tree=tree[i+1]
    return tree
def replace_subtree(tree, path, replacement):
    if not path: return replacement
    children=list(tree[1:]); i=path[0]; children[i]=replace_subtree(children[i],path[1:],replacement)
    return tuple([tree[0]]+children)
def node_type(tree):
    """Current grammar is numeric-scalar only; keep this gate for extensions."""
    return "scalar"
def crossover(left, right, max_nodes, max_depth):
    """Typed, bounded subtree exchange; retries avoid needless no-op children."""
    for _ in range(8):
        left_path=rng.choice(subtree_paths(left)); right_path=rng.choice(subtree_paths(right))
        donor=subtree_at(right,right_path)
        if node_type(subtree_at(left,left_path)) != node_type(donor): continue
        child=simplify_tree(replace_subtree(left,left_path,donor))
        if node_size(child)<=max_nodes and node_depth(child)<=max_depth and child!=left: return child
    return left

def semantic_crossover(left, right, X, max_nodes, max_depth, adfs=None, min_delta=1e-8, max_delta=5.0):
    """Prefer bounded semantic moves over syntactically random exchanges."""
    try: baseline=evaluate_cached(left,X,adfs)
    except ValueError: return left
    for _ in range(8):
        child=crossover(left,right,max_nodes,max_depth)
        try: delta=semantic_distance(baseline,evaluate_cached(child,X,adfs))
        except ValueError: continue
        if min_delta < delta <= max_delta: return child
    return left
def expr(t, names, argument_names=None):
    if t[0]=="x": return names[t[1]]
    if t[0]=="c": return f"{t[1]:.6g}"
    if t[0]=="arg":
        labels=argument_names or ()
        return labels[t[1]] if 0<=t[1]<len(labels) else f"u{t[1]}"
    args=[expr(q,names,argument_names) for q in t[1:]]
    if t[0] in {"+", "-", "*", "/", "pow", "max", "min"}:
        return f"({args[0]} {t[0]} {args[1]})"
    return f"{t[0]}({', '.join(args)})"

def referenced_adf_names(model):
    """Return direct/transitive ADFs in definition-before-use order."""
    ordered=[]; visiting=set(); visited=set()
    def visit(tree):
        if tree[0] in ("x","c","arg"): return
        if tree[0].startswith("adf_"):
            name=tree[0]
            if name in visiting: raise ValueError(f"Cyclic ADF definition {name!r}")
            if name not in visited:
                item=model.adfs.get(name)
                if item is None: raise ValueError(f"Missing ADF definition {name!r}")
                visiting.add(name); visit(item["tree"]); visiting.remove(name)
                visited.add(name); ordered.append(name)
        for child in tree[1:]: visit(child)
    for tree in model.trees: visit(tree)
    return ordered

def adf_display_definitions(model, feature_names):
    """Readable definitions for the ADFs actually used by one model."""
    result=[]
    for name in referenced_adf_names(model):
        item=model.adfs[name]; arity=int(item["arity"]); arguments=[f"u{index}" for index in range(arity)]
        dependencies=[]
        def collect(tree):
            if tree[0].startswith("adf_") and tree[0] not in dependencies: dependencies.append(tree[0])
            if tree[0] not in ("x","c","arg"):
                for child in tree[1:]: collect(child)
        collect(item["tree"])
        result.append({"name":name,"arity":arity,"arguments":arguments,"expression":expr(item["tree"],feature_names,arguments),"dependencies":dependencies})
    return result

def classification_layout(cats):
    """Map logical targets to equation heads; 3+ classes receive one head each."""
    targets=[]; head_targets=[]
    start=0
    for target,labels in enumerate(cats):
        count=len(labels) if labels is not None and len(labels)>2 else 1
        heads=tuple(range(start,start+count)); targets.append(heads); head_targets.extend([target]*count); start+=count
    return tuple(targets),tuple(head_targets)

def stable_softmax(scores):
    shifted=scores-np.max(scores,axis=1,keepdims=True)
    weights=np.exp(np.clip(shifted,-50,50))
    return weights/np.maximum(weights.sum(axis=1,keepdims=True),EPS)

def binary_labels(scores, class_count=2):
    """Binary heads represent class indices, including in standalone exports."""
    return np.clip(np.rint(scores),0,class_count-1)

def binary_probabilities(scores):
    """A binary head's score minus 0.5 is its log-odds, so the 0.5 label threshold is p=0.5."""
    positive=1./(1.+np.exp(-np.clip(np.asarray(scores,float)-.5,-50,50)))
    return np.column_stack((1.-positive,positive))

CLASSIFIER_RIDGE=1e-3
def fit_classifier_affine(raw, truth, class_count):
    """Fit per-head (scale, offset) minimizing ridge-penalized log loss by damped Newton.

    raw has one column per head; binary targets use one head whose log-odds are
    score-0.5, so the returned offset already includes that +0.5 shift."""
    raw=np.asarray(raw,float); n,heads=raw.shape
    truth=np.asarray(np.rint(truth),int); valid=(truth>=0)&(truth<class_count)
    if not np.any(valid): return [(1.,0.)]*heads
    raw=raw[valid]; truth=truth[valid]; n=len(truth)
    centre=raw.mean(axis=0); spread=raw.std(axis=0); live=spread>=EPS*np.maximum(1.,np.abs(centre))
    spread=np.where(live,spread,1.)
    u=(raw-centre)/spread
    # Class features: binary pins class 0's logit at zero; a constant head keeps only its offset.
    features=np.zeros((n,class_count,2))
    columns=slice(1,2) if heads==1 else slice(0,class_count)
    features[:,columns,0]=u*live; features[:,columns,1]=1.
    one_hot=np.eye(class_count)[truth]; theta=np.zeros(2*class_count); flat=features.reshape(n,-1)
    same_class=np.kron(np.eye(class_count),np.ones((2,2))); ridge=CLASSIFIER_RIDGE*np.eye(len(theta))
    def objective(t):
        logits=np.einsum("nkd,kd->nk",features,t.reshape(class_count,2))
        logits-=logits.max(axis=1,keepdims=True)
        log_probabilities=logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))
        return float(-np.mean(log_probabilities[np.arange(n),truth])+.5*CLASSIFIER_RIDGE*t@t),np.exp(log_probabilities)
    value,probabilities=objective(theta)
    for _ in range(30):
        weighted=(probabilities-one_hot)[:,:,None]*features
        gradient=weighted.reshape(n,-1).mean(axis=0)+CLASSIFIER_RIDGE*theta
        # Softmax Hessian block (k,l) = sum_n (p_k[k=l] - p_k p_l) f_k f_l^T.
        weighted_features=(probabilities[:,:,None]*features).reshape(n,-1)
        hessian=(same_class*((flat*np.repeat(probabilities,2,axis=1)).T@flat)-weighted_features.T@weighted_features)/n+ridge
        try: step=np.linalg.solve(hessian,gradient)
        except np.linalg.LinAlgError: step=gradient
        rate=1.
        while rate>1e-4:
            candidate=theta-rate*step; new_value,new_probabilities=objective(candidate)
            if new_value<=value-1e-4*rate*float(gradient@step): break
            rate*=.5
        else: break
        improvement=value-new_value; theta,value,probabilities=candidate,new_value,new_probabilities
        if improvement<1e-10: break
    theta=theta.reshape(class_count,2); bound=AFFINE_COEFFICIENT_BOUND; result=[]
    for head in range(heads):
        cls=head+1 if heads==1 else head
        a=theta[cls,0]/spread[head] if live[head] else 0.; b=theta[cls,1]-a*centre[head]+(.5 if heads==1 else 0.)
        result.append((float(np.clip(a,-bound,bound)),float(np.clip(b,-bound,bound))))
    return result

def predict_targets(m, X, cats, probabilities=False):
    """Decode raw equation heads into one numeric value or class index per target."""
    raw=predict_model(m,X); targets,_=classification_layout(cats)
    if raw.shape[1] != sum(len(heads) for heads in targets):
        raise ValueError("Model equation-head count does not match the categorical output schema")
    values=[]; distributions=[]
    for labels,heads in zip(cats,targets):
        if labels is not None and len(labels)>2:
            distribution=stable_softmax(raw[:,heads]); values.append(np.argmax(distribution,axis=1)); distributions.append(distribution)
        elif labels is not None:
            values.append(binary_labels(raw[:,heads[0]],len(labels)))
            distributions.append(binary_probabilities(raw[:,heads[0]]) if len(labels)==2 else np.ones((len(X),1)))
        else:
            values.append(raw[:,heads[0]]); distributions.append(None)
    result=np.column_stack(values)
    return (result,distributions) if probabilities else result

def equations(m, feature_names, output_names, cats=None):
    """Human-readable equations, including fitted affine output scaling."""
    cats=[None]*len(output_names) if cats is None else cats
    targets,_=classification_layout(cats)
    rendered=[]
    for name,labels,heads in zip(output_names,cats,targets):
        for class_index,head in enumerate(heads):
            tree,(a,b)=m.trees[head],m.scales[head]; formula=expr(tree,feature_names)
            label=f"{name}[{labels[class_index]!r}] score" if labels is not None and len(labels)>2 else name
            if abs(a-1.0)<EPS and abs(b)<EPS: rendered.append(f"{label} = {formula}")
            elif abs(b)<EPS: rendered.append(f"{label} = {a:.6g} * ({formula})")
            else: rendered.append(f"{label} = {a:.6g} * ({formula}) + {b:.6g}")
    return "; ".join(rendered)

def unique_models(models):
    """Collapse equivalent equations (equivalence_key) for concise reporting/choices."""
    kept = {}
    for m in models:
        key = model_equivalence_key(m)
        if key not in kept or secondary_key(m) < secondary_key(kept[key]):
            kept[key] = m
    return list(kept.values())

def structural_skeleton(tree):
    """Constant-agnostic expression identity used only for concise reporting."""
    if tree[0]=="c": return ("c",)
    if tree[0]=="x": return tree
    return tuple([tree[0]]+[structural_skeleton(child) for child in tree[1:]])

def frontier_representatives(models):
    """Keep the best model per constant-agnostic structural family for display."""
    kept={}
    for model in models:
        key=repr([structural_skeleton(tree) for tree in model.trees])
        if key not in kept or secondary_key(model)<secondary_key(kept[key]): kept[key]=model
    return list(kept.values())

def print_model_options(labels, choices, feature_names, output_names, cats=None, evaluation=None):
    """Print the same labeled candidates available at final model selection."""
    evaluation=evaluation or selection_evaluation(choices,cats=cats)
    scored={selection_identity(entry[0]):entry[1] for entry in evaluation[1]}
    print(f"\nRecommended models ({evaluation[0]} scores):")
    for index,(label,model) in enumerate(zip(labels,choices),1):
        model=scored[selection_identity(model)]
        print(f"  [{index:>2}] {label}: mean loss={aggregate_loss(model):.6g}, losses={output_loss_summary(model_losses(model),output_names)}, mean shape={np.mean(model_shapes(model)):.6g}, MDL bits={model_complexity(model):.6g}, age={model_age(model)}")
        print(f"       {equations(model,feature_names,output_names,cats)}")
        definitions=adf_display_definitions(model,feature_names)
        if definitions:
            print("       ADFs used:")
            for definition in definitions:
                print(f"         {definition['name']}({', '.join(definition['arguments'])}) = {definition['expression']}")

def print_frontier(models, feature_names, output_names, cats=None, limit=12, recommendations=None, evaluation=None):
    evaluation=evaluation or selection_evaluation(models,cats=cats)
    if recommendations is None:
        labels,choices,_=model_options(models,cats=cats,evaluation=evaluation)
    else: labels,choices=recommendations
    print_model_options(labels,choices,feature_names,output_names,cats,evaluation)
    frontier=[entry[1] for entry in selection_frontier(evaluation[1])]
    distinct=sorted(frontier_representatives(frontier),key=lambda m:(aggregate_loss(m),model_complexity(m),float(np.mean(model_shapes(m))),model_age(m)))
    shown=distinct[:limit]
    family_label="family" if len(distinct)==1 else "families"
    print(f"\nPareto frontier ({evaluation[0]} loss/MDL bits) — {len(distinct)} structural {family_label}; showing {len(shown)}:")
    for i,m in enumerate(shown, 1):
        parts=[]
        for name,loss,shape in zip(output_names,model_losses(m),model_shapes(m)):
            parts.append(f"{name}: loss={loss:.6g}, shape={shape:.6g}")
        metrics=" | ".join(parts+[f"MDL bits={model_complexity(m):.6g}",f"age={model_age(m)}"])
        print(f"  [{i:>2}] {metrics}")
        print(f"       {equations(m, feature_names, output_names, cats)}")
    if len(distinct)>len(shown):
        print(f"       … {len(distinct)-len(shown)} additional nondominated model(s) omitted")

def shape_error(pred, y):
    """Scale- and sign-invariant error: centred, normed vectors have identical shape.

    A mirrored prediction is one negative scale away from the target (the
    affine readout fits that for free), so it counts as the same shape.
    """
    p=pred-np.mean(pred); q=y-np.mean(y)
    pn=np.linalg.norm(p); qn=np.linalg.norm(q)
    if pn<EPS or qn<EPS: return float(pn>=EPS or qn>=EPS)
    p=p/pn; q=q/qn
    return float(min(np.mean((p-q)**2),np.mean((p+q)**2)))
_ARRAY_DIGESTS={}
def array_digest(values):
    """Content digest, memoized by buffer identity.  An entry keeps its array
    alive, so the buffer address cannot be recycled while the entry exists;
    views of one training array (Xt[slice(None)], Y[:,j]) all hit."""
    interface=values.__array_interface__
    identity=(interface["data"][0],values.shape,values.strides,interface["typestr"])
    hit=_ARRAY_DIGESTS.get(identity)
    if hit is not None: return hit[0]
    if len(_ARRAY_DIGESTS)>=128: _ARRAY_DIGESTS.clear()
    digest=(values.shape,interface["typestr"],hashlib.blake2b(np.ascontiguousarray(values).view(np.uint8),digest_size=16).digest())
    _ARRAY_DIGESTS[identity]=(digest,values)
    return digest
_TARGET_SCALE_CACHE={}
def target_scale(y):
    y=np.asarray(y,dtype=float)
    if y.ndim==1:
        key=y.tobytes() if len(y)<=4096 else array_digest(y); cached=_TARGET_SCALE_CACHE.get(key)
        if cached is not None: return cached
        if len(_TARGET_SCALE_CACHE)>256: _TARGET_SCALE_CACHE.clear()
        cached=_target_scale(y); _TARGET_SCALE_CACHE[key]=cached; return cached
    return _target_scale(y)
def _target_scale(y):
    scale=1.4826*np.median(np.abs(y-np.median(y)))
    if not np.isfinite(scale) or scale < EPS: scale=max(float(np.std(y)),1.0)
    return scale

ROBUST_LOSS_DELTA=1.5
def robust_loss(pred, y, delta=ROBUST_LOSS_DELTA):
    """MAD-scaled Huber loss; a few extreme target values cannot dominate."""
    scale=target_scale(y)
    r=np.abs((pred-y)/scale)
    return float(np.mean(np.where(r<=delta,.5*r*r,delta*(r-.5*delta))))
AFFINE_COEFFICIENT_BOUND = 1e9
def _weighted_line(u, y, w):
    """Closed-form weighted least-squares line y ~ c0*u+c1; None when degenerate."""
    s=float(np.sum(w))
    if not s>0: return None
    mu=float(np.dot(w,u))/s; my=float(np.dot(w,y))/s; du=u-mu
    suu=float(np.dot(w,du*du))
    if not np.isfinite(suu) or suu<=1e-12*max(float(np.dot(w,u*u)),EPS): return None
    c0=float(np.dot(w,du*(y-my)))/suu
    return np.asarray((c0,my-c0*mu)) if np.isfinite(c0) else None
_CONSTANT_AFFINE_CACHE={}
def affine(pred, y):
    """Fit Huber loss with coefficient bounds enforced during weighted least squares."""
    # A prediction exactly equal to its own mean makes the centred input u
    # identically zero, so every solve sees the same matrix whatever the
    # constant is, and least squares returns slope exactly 0: the fit is the
    # robust location of y alone.  (A mean off by rounding leaves a tiny
    # nonzero u and a constant-dependent answer, so that case is not cached.)
    # The bound branch reads pred directly, so only cache when it cannot fire.
    if np.ndim(y)==1 and len(y) and len(pred)==len(y) and np.all(pred==float(np.mean(pred))):
        y=np.asarray(y,dtype=float)
        key=y.tobytes() if len(y)<=4096 else array_digest(y)
        cached=_CONSTANT_AFFINE_CACHE.get(key)
        if cached is not None: return cached
        result=_affine(pred,y)
        if result[0]==0. and float(np.max(np.abs(y)))<=AFFINE_COEFFICIENT_BOUND/2:
            if len(_CONSTANT_AFFINE_CACHE)>256: _CONSTANT_AFFINE_CACHE.clear()
            _CONSTANT_AFFINE_CACHE[key]=result
        return result
    return _affine(pred,y)
def _shorter_values(value):
    """Candidate replacements for a coefficient, simplest first."""
    yield 0.
    if value: yield float(np.sign(value))
    for digits in range(1,16): yield float(f"{value:.{digits}g}")
def simplify_affine(pred, y, a, b):
    """Snap fitted (a, b) to shorter values when predictions move only by solver noise.

    A fit like 0.5625000000000002*x - 8.5e-14 otherwise pays for two real
    constants and loses duplicate tie-breaks to bloated twins on noise-level loss."""
    pred=np.asarray(pred,float)
    magnitude=float(np.max(np.abs(pred))) if len(pred) else 0.
    # The Huber IRLS solve converges to ~sqrt(eps), so noise below that is not information.
    tolerance=np.sqrt(np.finfo(float).eps)*max(abs(a)*magnitude+abs(b),EPS)
    def simplest(current, weight):
        return next((candidate for candidate in _shorter_values(current) if abs(candidate-current)*weight<=tolerance),current)
    short_a,short_b=simplest(a,magnitude),simplest(b,1.)
    if short_a==a and short_b==b: return float(a),float(b)
    original=robust_loss(a*pred+b,y)
    for candidate_a,candidate_b in ((short_a,short_b),(short_a,b),(a,short_b)):
        if (candidate_a,candidate_b)==(a,b) or abs(candidate_a-a)*magnitude+abs(candidate_b-b)>tolerance: continue
        if robust_loss(candidate_a*pred+candidate_b,y)<=original: return float(candidate_a),float(candidate_b)
    return float(a),float(b)
def _affine(pred, y):
    centre=float(np.mean(pred)); spread=float(np.std(pred))
    spread=spread if spread>=EPS else 1.
    u=(pred-centre)/spread; A=None
    bound=AFFINE_COEFFICIENT_BOUND
    def weighted_fit(weights):
        nonlocal A
        coefficients=_weighted_line(u,y,weights*weights)
        if coefficients is None:
            if A is None: A=np.column_stack((u,np.ones(len(u))))
            coefficients=np.linalg.lstsq(A*weights[:,None],y*weights,rcond=None)[0]
        a=coefficients[0]/spread; b=coefficients[1]-a*centre
        if abs(a)<=bound and abs(b)<=bound: return np.asarray((a,b))
        # A convex two-variable quadratic reaches its box-constrained minimum
        # either at the unconstrained solution or on one of these four edges.
        w=weights*weights; total=float(np.sum(w))
        candidates=[(0.,float(np.clip(np.dot(w,y)/total,-bound,bound)))]
        for a in (-bound,bound):
            b=float(np.clip(np.dot(w,y-a*pred)/total,-bound,bound))
            candidates.append((a,b))
        magnitude=max(float(np.max(np.abs(pred))),1.)
        scaled=pred/magnitude; denominator=float(np.dot(w,scaled*scaled))
        for b in (-bound,bound):
            a=float(np.clip(np.dot(w*scaled,y-b)/denominator/magnitude,-bound,bound)) if denominator>0 else 0.
            candidates.append((a,b))
        return np.asarray(min(candidates,key=lambda ab:float(np.sum(w*(ab[0]*pred+ab[1]-y)**2))))
    try:
        coefficients=weighted_fit(np.ones(len(pred)))
        cutoff=1.5*target_scale(y)
        # Solver termination limits are numerical safeguards, not search settings.
        for _ in range(200):
            residual=coefficients[0]*pred+coefficients[1]-y
            weights=np.sqrt(np.minimum(1.,cutoff/np.maximum(np.abs(residual),EPS)))
            updated=weighted_fit(weights)
            change=(updated[0]-coefficients[0])*pred+updated[1]-coefficients[1]
            converged=np.linalg.norm(change)<=np.sqrt(np.finfo(float).eps)*(1.+np.linalg.norm(residual+y))
            coefficients=updated
            if converged: break
        a,b=coefficients
    except np.linalg.LinAlgError: a,b=0.,float(np.clip(np.median(y),-bound,bound))
    return float(a),float(b)

@dataclass
class Model:
    trees:list; scales:list; age:int=0; objectives:tuple=field(default_factory=lambda:(float("inf"),)*4)
    lineage_id:int=field(default_factory=next_lineage_id)
    origin:str="seed"
    parent_ids:tuple=field(default_factory=tuple)
    feasible:bool=True
    invalid_reason:str=""
    constraint_count:int=0
    mdl_operators:tuple=field(default_factory=tuple)
    mdl_feature_count:int=0
    adfs:dict=field(default_factory=dict)
    founder_ids:tuple=field(default_factory=tuple)
    birth_generation:int|None=None
    def __post_init__(self):
        if not self.founder_ids: self.founder_ids=(self.lineage_id,)
        else: self.founder_ids=tuple(sorted(set(self.founder_ids)))
    def clone(self): return Model(trees=list(self.trees),scales=list(self.scales),age=self.age,objectives=tuple(self.objectives),lineage_id=self.lineage_id,origin=self.origin,parent_ids=tuple(self.parent_ids),feasible=self.feasible,invalid_reason=self.invalid_reason,constraint_count=self.constraint_count,mdl_operators=tuple(self.mdl_operators),mdl_feature_count=self.mdl_feature_count,adfs=dict(self.adfs),founder_ids=tuple(self.founder_ids),birth_generation=self.birth_generation)

def predict_model(m, X): return np.column_stack([clean(a*evaluate_cached(t,X,m.adfs)+b) for t,(a,b) in zip(m.trees,m.scales)])

def _quality_objectives(model_or_objectives):
    objectives=model_or_objectives.objectives if isinstance(model_or_objectives,Model) else model_or_objectives
    count=getattr(model_or_objectives,"constraint_count",0)
    return objectives[:len(objectives)-2-count]
def model_losses(model): return tuple(_quality_objectives(model)[::2])
def model_shapes(model): return tuple(_quality_objectives(model)[1::2])
def model_violations(model): return tuple(model.objectives[-2-model.constraint_count:-2]) if model.constraint_count else ()
def model_complexity(model): return float(model.objectives[-2])
def model_age(model): return int(model.age)
def output_loss_summary(losses, output_names):
    """Format one named loss for every configured output."""
    if len(losses)!=len(output_names): raise ValueError("Loss count does not match output names")
    return " | ".join(f"{name}={loss:.6g}" for name,loss in zip(output_names,losses))
def aggregate_loss(model_or_objectives):
    values=model_losses(model_or_objectives) if isinstance(model_or_objectives,Model) else tuple(model_or_objectives[:-2:2])
    return float(np.mean(values)) if values else float("inf")
def secondary_key(model):
    """Deterministic non-fitness key for duplicate handling and diagnostics."""
    # Losses that differ only by round-off are ties, so the shorter model wins.
    return (_noise_rounded(aggregate_loss(model)),_noise_rounded(float(np.mean(model_shapes(model)))),model_complexity(model),repr(model.trees))
def _noise_rounded(value):
    return 0. if abs(value)<1e-20 else float(f"{value:.10g}")
def objective_labels(output_names, include_violations=False):
    labels=tuple(label for name in output_names for label in (f"{name}:loss",f"{name}:shape"))
    if include_violations: labels+=tuple(f"{name}:constraint_violation" for name in output_names)
    return labels+("mdl_bits","age")

# Behavioural heuristics -- duplicate detection, fragment mining, the Bayesian
# proposal model and the mutation step-size guard -- only need to tell models
# apart, not to score them.  They ran on every training row in the parent
# process, which no worker can share: ~60% of a generation at 100k rows, so 15
# workers ran no faster than one.  Above this many rows they use one fixed
# random subset; survival and selection scores still use every row.
BEHAVIOUR_PROBE_ROWS = 1024
_BEHAVIOUR_INDICES={}
def behaviour_rows(X, Y=None):
    """X (and Y) themselves up to BEHAVIOUR_PROBE_ROWS rows, else the same fixed random rows of both."""
    if not isinstance(X,np.ndarray) or X.ndim!=2 or len(X)<=BEHAVIOUR_PROBE_ROWS:
        return X if Y is None else (X,Y)
    rows=_BEHAVIOUR_INDICES.get(len(X))
    if rows is None:
        # Seeded by the row count alone: deterministic, and the run's own RNG
        # streams are untouched, so seeded runs stay reproducible.
        rows=np.sort(np.random.default_rng(len(X)).choice(len(X),BEHAVIOUR_PROBE_ROWS,replace=False))
        _BEHAVIOUR_INDICES[len(X)]=rows
    return row_subset(X,rows) if Y is None else (row_subset(X,rows),row_subset(Y,rows))

def semantic_key(model, X, decimals=8):
    """Stable behavioral identity on the active evaluation sample (a fixed subset of it on large data)."""
    return np.round(predict_model(model,behaviour_rows(X)),decimals).tobytes()
def novelty_pool(models, X):
    """One best representative per equivalence (algebraic) and behavioral identity."""
    structural={}
    for model in models:
        key=model_equivalence_key(model)
        if key not in structural or secondary_key(model)<secondary_key(structural[key]): structural[key]=model
    semantic={}
    for model in structural.values():
        key=semantic_key(model,X)
        if key not in semantic or secondary_key(model)<secondary_key(semantic[key]): semantic[key]=model
    return list(semantic.values())
# AFPO's age objective protects new lineages.  Uncapped, every lineage's age
# keeps differing, so ~30% of the population survived on age alone.  Past this
# many generations, models compete on quality and size only.  None = uncapped.
AGE_PROTECTION_LIMIT = 10
def selection_objectives(model):
    objectives=model.objectives
    if AGE_PROTECTION_LIMIT is None or not objectives: return objectives
    return (*objectives[:-1],min(objectives[-1],AGE_PROTECTION_LIMIT))
def dominates(a,b):
    if a.feasible != b.feasible: return a.feasible
    x_values,y_values=selection_objectives(a),selection_objectives(b)
    return all(x<=y for x,y in zip(x_values,y_values)) and any(x<y for x,y in zip(x_values,y_values))

def variation_improved(child, parent):
    """Compare variation quality without AFPO's mandatory age increment."""
    if child.feasible != parent.feasible: return child.feasible
    # MDL is excluded: a move that adds structure (parametrize, bilinear,
    # subtree growth) always costs bits, so full-objective dominance never
    # credited it however much it improved the fit.  Survival still pays for
    # size; a pure simplification at equal quality still counts as a win.
    child_quality=(*_quality_objectives(child),*model_violations(child))
    parent_quality=(*_quality_objectives(parent),*model_violations(parent))
    if not all(x<=y for x,y in zip(child_quality,parent_quality)): return False
    return any(x<y for x,y in zip(child_quality,parent_quality)) or model_complexity(child)<model_complexity(parent)
def fronts(pop):
    """Deb-style O(n²) domination-count nondominated sorting."""
    n=len(pop)
    if not n: return []
    objectives=[selection_objectives(model) for model in pop]
    if n>2 and len({len(values) for values in objectives})==1:
        return _fronts_vectorized(pop,objectives)
    dominated=[[] for _ in range(n)]; count=[0]*n
    for i in range(n):
        for j in range(i+1,n):
            if dominates(pop[i],pop[j]): dominated[i].append(j); count[j]+=1
            elif dominates(pop[j],pop[i]): dominated[j].append(i); count[i]+=1
    first=[i for i in range(n) if count[i]==0]
    result=[]; current=first
    while current:
        result.append([pop[i] for i in current]); following=[]
        for i in current:
            for j in dominated[i]:
                count[j]-=1
                if count[j]==0: following.append(j)
        current=following
    return result
def _fronts_vectorized(pop, objectives):
    """fronts() on a domination matrix; reproduces its exact front order.

    The loop version appends each newly freed model when its last dominator
    in the current front is visited, so the next front is ordered by
    (position of that last dominator, model index)."""
    values=np.asarray(objectives,dtype=float); feasible=np.fromiter((bool(model.feasible) for model in pop),bool,len(pop))
    with np.errstate(invalid="ignore"):
        weak=np.all(values[:,None,:]<=values[None,:,:],axis=2)
        strict=np.any(values[:,None,:]<values[None,:,:],axis=2)
    same=feasible[:,None]==feasible[None,:]
    dominates_matrix=np.where(same,weak&strict,feasible[:,None])
    count=dominates_matrix.sum(axis=0)
    current=np.flatnonzero(count==0); result=[]
    while len(current):
        result.append([pop[i] for i in current])
        rows=dominates_matrix[current]
        count=count-rows.sum(axis=0)
        freed=np.flatnonzero((count==0)&rows.any(axis=0))
        if not len(freed): break
        last=len(current)-1-np.argmax(rows[::-1][:,freed],axis=0)
        current=freed[np.lexsort((freed,last))]
    return result
def reference_directions(dimensions, target):
    """Deterministic Das-Dennis simplex directions for NSGA-III niching."""
    if dimensions < 2: return np.ones((1,1))
    divisions=1
    while math.comb(dimensions+divisions-1,divisions)<target: divisions+=1
    directions=[]
    def compose(remaining, slots, prefix):
        if slots==1: directions.append(prefix+[remaining]); return
        for value in range(remaining+1): compose(remaining-value,slots-1,prefix+[value])
    compose(divisions,dimensions,[])
    return np.asarray(directions,float)/divisions

NSGA_NORMALIZATIONS=("intercept","legacy")

def finite_objective_values(models):
    """Return a finite objective matrix without making invalid models competitive."""
    values=np.asarray([selection_objectives(m) for m in models],float)
    if not len(values): return values
    for column in range(values.shape[1]):
        finite=values[np.isfinite(values[:,column]),column]
        if len(finite):
            span=max(float(np.ptp(finite)),1.)
            fill=float(np.max(finite))+span
        else: fill=1.
        values[~np.isfinite(values[:,column]),column]=fill
    return values

def normalize_objectives(models, mode="intercept"):
    """NSGA-III objective normalization with a deterministic degenerate fallback."""
    if mode not in NSGA_NORMALIZATIONS: raise ValueError(f"Unknown NSGA normalization: {mode}")
    values=finite_objective_values(models)
    if not len(values): return values
    ideal=values.min(axis=0); translated=values-ideal
    if mode=="legacy":
        return translated/np.maximum(translated.max(axis=0),EPS)
    dimensions=translated.shape[1]
    weights=np.full((dimensions,dimensions),1e-6); np.fill_diagonal(weights,1.)
    extreme=np.asarray([translated[np.argmin(np.max(translated/weight,axis=1))] for weight in weights])
    fallback=np.maximum(translated.max(axis=0),EPS)
    try:
        intercepts=1./np.linalg.solve(extreme,np.ones(dimensions))
        if not np.all(np.isfinite(intercepts)) or np.any(intercepts<=EPS): raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts=fallback
    return translated/np.maximum(intercepts,EPS)

def associate_normalized_points(points, directions):
    norms=np.maximum(np.linalg.norm(directions,axis=1),EPS)
    unit=directions/norms[:,None]
    projections=points@unit.T
    distances=np.linalg.norm(points[:,None,:]-projections[:,:,None]*unit[None,:,:],axis=2)
    niches=np.argmin(distances,axis=1)
    return niches,distances[np.arange(len(points)),niches]

def associate_directions(models, directions, normalization="intercept"):
    """Compatibility wrapper for associating one population to directions."""
    points=normalize_objectives(models,normalization)
    niches,distances=associate_normalized_points(points,directions)
    return niches,distances

def parsimony_preferred(shorter, longer, tolerance):
    """Whether a lower-MDL model is quality-equivalent enough to replace a peer."""
    if tolerance<=0 or not shorter.feasible or not longer.feasible: return False
    if not model_complexity(shorter)<model_complexity(longer): return False
    for left,right in zip(model_violations(shorter),model_violations(longer)):
        if left>right: return False
    for left,right in zip(_quality_objectives(shorter),_quality_objectives(longer)):
        if left>right+tolerance*max(abs(right),EPS): return False
    return True

def parsimony_candidates(candidates, tolerance):
    """Keep candidates not near-dominated by a quality-equivalent shorter model."""
    if tolerance<=0: return list(candidates)
    return [candidate for candidate in candidates
            if not any(parsimony_preferred(other,candidate,tolerance) for other in candidates if other is not candidate)]

def materially_better_quality(candidate, incumbent, tolerance):
    """Whether candidate improves quality without giving back another quality objective."""
    if not candidate.feasible or not incumbent.feasible: return candidate.feasible and not incumbent.feasible
    if any(left>right for left,right in zip(model_violations(candidate),model_violations(incumbent))): return False
    comparisons=[(left,right,tolerance*max(abs(right),EPS)) for left,right in zip(_quality_objectives(candidate),_quality_objectives(incumbent))]
    return all(left<=right+band for left,right,band in comparisons) and any(left<right-band for left,right,band in comparisons)

class BestModelArchive:
    """One persistent model that cannot be lost to population or Pareto turnover."""
    def __init__(self, tolerance=.01, model=None): self.tolerance=float(tolerance); self.model=None if model is None else model.clone()
    def update(self, candidates):
        improved=False
        for candidate in sorted((m for m in candidates if m.feasible),key=secondary_key):
            if self.model is None:
                self.model=candidate.clone(); improved=True; continue
            if parsimony_preferred(candidate,self.model,self.tolerance):
                self.model=candidate.clone(); continue
            if materially_better_quality(candidate,self.model,self.tolerance):
                self.model=candidate.clone(); improved=True
        return improved
    def snapshot(self): return {"tolerance":self.tolerance,"model":None if self.model is None else PosteriorParticlePopulation._model_data(self.model)}
    @classmethod
    def from_snapshot(cls, data, tolerance=.01):
        return cls(data.get("tolerance",tolerance),None if data.get("model") is None else Model(**data["model"]))

class DynamicPressureController:
    """Opt-in, bounded pressure after simultaneous quality and QD stagnation."""
    def __init__(self, enabled=False, base_parsimony=.01, base_uniform=.25, window=100):
        self.enabled=bool(enabled); self.base_parsimony=float(base_parsimony); self.base_uniform=float(base_uniform); self.window=int(window)
        self.level=0; self.last_quality_generation=0; self.last_check_generation=0; self.last_qd_activity=((0,0),(0,0))
    def observe(self, generation, quality_improved, archives):
        if not self.enabled: return False
        if quality_improved:
            self.level=0; self.last_quality_generation=int(generation); return True
        if generation-self.last_check_generation<self.window: return False
        self.last_check_generation=int(generation); activity=tuple((len(archive.cells),archive.replacements) for archive in archives)
        changed=False
        if generation-self.last_quality_generation>=self.window and activity==self.last_qd_activity:
            self.level=min(4,self.level+1); changed=True
        self.last_qd_activity=activity
        return changed
    def effective_parsimony(self): return min(.05,self.base_parsimony+.01*self.level) if self.enabled else self.base_parsimony
    def exploration_boost(self): return min(.25,.05*self.level) if self.enabled else 0.
    def uniform_rate(self): return min(.50,self.base_uniform+.05*self.level) if self.enabled else self.base_uniform
    def novelty_rate(self): return min(.20,.04*self.level) if self.enabled else 0.
    def apply(self, bayes, qd_controller):
        banks=bayes.banks if isinstance(bayes,PerOutputBayesianBanks) else [bayes]
        for bank in banks: bank.pressure_exploration=self.exploration_boost()
        qd_controller.uniform_rate=self.uniform_rate()
    def snapshot(self): return {key:getattr(self,key) for key in ("enabled","base_parsimony","base_uniform","window","level","last_quality_generation","last_check_generation","last_qd_activity")}
    @classmethod
    def from_snapshot(cls, data):
        result=cls(data.get("enabled",False),data.get("base_parsimony",.01),data.get("base_uniform",.25),data.get("window",100))
        for key in ("level","last_quality_generation","last_check_generation"): setattr(result,key,int(data.get(key,0)))
        result.last_qd_activity=tuple(tuple(item) for item in data.get("last_qd_activity",((0,0),(0,0)))); return result
    def stats(self):
        return f"Dynamic pressure={'on' if self.enabled else 'off'}; level={self.level}; parsimony={self.effective_parsimony():.0%}; Bayesian boost={self.exploration_boost():.0%}; QD uniform={self.uniform_rate():.0%}; novelty={self.novelty_rate():.0%}"

class EvaluationBudget:
    """Checkpointable evaluation policy; adaptive mode only changes large-data co-evolution."""
    def __init__(self, mode="baseline", refresh=25):
        if mode not in ("baseline","adaptive"): raise ValueError("Unknown evaluation budget mode")
        self.mode=mode; self.refresh=max(1,int(refresh)); self.last_full_refresh=-1
        self.screen_rows=0; self.anchor_rows=0; self.full_rows=0
    def screen_count(self, total):
        baseline=max(128,total//8)
        return min(total, max(64,total//16) if self.mode=="adaptive" else baseline)
    def anchor_indices(self, X): return stratified_probe_indices(X,min(128,len(X)))
    def refresh_due(self, generation): return self.mode=="adaptive" and generation>0 and generation%self.refresh==0
    def record(self, kind, rows):
        setattr(self,f"{kind}_rows",getattr(self,f"{kind}_rows")+int(rows))
    def snapshot(self): return {key:getattr(self,key) for key in ("mode","refresh","last_full_refresh","screen_rows","anchor_rows","full_rows")}
    @classmethod
    def from_snapshot(cls, data):
        result=cls(data.get("mode","baseline"),data.get("refresh",25))
        for key in ("last_full_refresh","screen_rows","anchor_rows","full_rows"): setattr(result,key,int(data.get(key,getattr(result,key))))
        return result
    def stats(self): return f"Evaluation budget={self.mode}; rows screen/anchor/full={self.screen_rows}/{self.anchor_rows}/{self.full_rows}; refresh={self.refresh}"

def reference_niching(selected, candidates, count, direction_target=None, normalization="intercept", parsimony_quality_tolerance=0.):
    if count<=0: return []
    directions=reference_directions(len(candidates[0].objectives),direction_target or max(len(selected)+count,count))
    if normalization=="legacy":
        selected_niches,_=associate_directions(selected,directions,"legacy") if selected else (np.empty(0,dtype=int),np.empty(0))
        niches,distances=associate_directions(candidates,directions,"legacy")
    else:
        points=normalize_objectives([*selected,*candidates],normalization)
        selected_niches,_=associate_normalized_points(points[:len(selected)],directions) if selected else (np.empty(0,dtype=int),np.empty(0))
        niches,distances=associate_normalized_points(points[len(selected):],directions)
    preferred=parsimony_candidates(candidates,parsimony_quality_tolerance)
    if len(preferred)<len(candidates):
        preferred_ids={id(model) for model in preferred}
        preferred_indices=[i for i,model in enumerate(candidates) if id(model) in preferred_ids]
        if len(preferred_indices)<count:
            chosen=[candidates[i] for i in preferred_indices]
            remainder=[model for model in candidates if id(model) not in preferred_ids]
            return chosen+reference_niching([*selected,*chosen],remainder,count-len(chosen),direction_target,normalization,0.)
        candidates=[candidates[i] for i in preferred_indices]
        niches=niches[preferred_indices]; distances=distances[preferred_indices]
    niche_counts=np.bincount(selected_niches,minlength=len(directions)); remaining=set(range(len(candidates))); chosen=[]
    while remaining and len(chosen)<count:
        available=sorted({niches[index] for index in remaining},key=lambda niche:(niche_counts[niche],niche))
        niche=available[0]; options=[index for index in remaining if niches[index]==niche]
        index=min(options,key=lambda item:(distances[item],secondary_key(candidates[item]))) if niche_counts[niche]==0 else min(options,key=lambda item:secondary_key(candidates[item]))
        chosen.append(candidates[index]); remaining.remove(index); niche_counts[niche]+=1
    return chosen

def select_nsga(pop, n, normalization="intercept", parsimony_quality_tolerance=0.):
    """NSGA-III-style reference-direction survival after nondominated sorting."""
    ans=[]
    for f in fronts(pop):
        if len(ans)+len(f)<=n: ans+=f
        else:
            target=n if normalization=="intercept" else max(len(ans)+len(f),n-len(ans))
            ans+=reference_niching(ans,f,n-len(ans),target,normalization,parsimony_quality_tolerance); break
    return ans

class ParetoArchive:
    """Persistent structural Pareto archive, capped with reference niching."""
    def __init__(self, capacity=256, normalization="intercept", parsimony_quality_tolerance=.01):
        if normalization not in NSGA_NORMALIZATIONS: raise ValueError(f"Unknown NSGA normalization: {normalization}")
        if parsimony_quality_tolerance<0: raise ValueError("Parsimony quality tolerance must be non-negative")
        self.capacity=capacity; self.normalization=normalization; self.parsimony_quality_tolerance=parsimony_quality_tolerance
        self.items=[]; self.semantic_keys=set(); self.last_change=0; self.generation=0; self.history=[]
    def update(self, candidates, X=None):
        self.generation+=1; before={repr(m.trees) for m in self.items}
        by_tree={}
        for model in [*self.items,*candidates]:
            key=model_equivalence_key(model)
            if key not in by_tree or secondary_key(model) < secondary_key(by_tree[key]):
                by_tree[key]=model.clone()
                # Genotypic age belongs to live AFPO survival, not a permanent
                # quality archive. Keep the actual age/birth metadata intact.
                by_tree[key].objectives=(*model.objectives[:-1],0)
        self.items=fronts(list(by_tree.values()))[0] if by_tree else []
        if X is not None:
            # Keep only the best representative of behaviorally identical
            # equations; different syntax alone should not fill the archive.
            by_semantics={}
            for model in self.items:
                key=semantic_key(model,X)
                if key not in by_semantics or secondary_key(model)<secondary_key(by_semantics[key]):
                    by_semantics[key]=model
            self.items=list(by_semantics.values())
        if len(self.items)>self.capacity: self.items=select_nsga(self.items,self.capacity,self.normalization,self.parsimony_quality_tolerance)
        self.semantic_keys=set()
        if X is not None:
            for m in self.items: self.semantic_keys.add(semantic_key(m,X))
        if {repr(m.trees) for m in self.items} != before: self.last_change=self.generation
        vals=np.array([m.objectives[:-1] for m in self.items],float)
        # A bounded, comparable convergence proxy (not an exact hypervolume).
        proxy=float(np.mean(1/(1+np.maximum(vals,0)))) if len(vals) else 0.
        self.history.append({"generation":self.generation,"models":len(self.items),"semantic":len(self.semantic_keys),"proxy":proxy})
    def stats(self):
        values=np.array([m.objectives[:-1] for m in self.items],float)
        spread=float(np.mean(np.std(values,axis=0))) if len(values)>1 else 0.
        proxy=self.history[-1]["proxy"] if self.history else 0.
        return f"archive={len(self.items)} models; semantic={len(self.semantic_keys)}; spread={spread:.3g}; HV-proxy={proxy:.3g}; stagnant={self.generation-self.last_change} gen"

_PROBE_INDEX_CACHE={}
def stratified_probe_indices(X, maximum=256):
    """Deterministic maximin input coverage for a fixed, training-only QD probe."""
    values=np.asarray(X,float); count=min(int(maximum),len(values))
    if count<=0: return np.array([],dtype=int)
    if count==len(values): return np.arange(len(values),dtype=int)
    # Pure function of the data: constant tuning asks for the same probe once
    # per model, so memoize on a digest of the rows (O(n) versus O(count*n)).
    key=(count,array_digest(values))
    cached=_PROBE_INDEX_CACHE.get(key)
    if cached is not None: return cached.copy()
    if len(_PROBE_INDEX_CACHE)>=64: _PROBE_INDEX_CACHE.clear()
    selected=_stratified_probe_indices(values,count)
    _PROBE_INDEX_CACHE[key]=selected
    return selected.copy()

def _stratified_probe_indices(values, count):
    low=np.min(values,axis=0); span=np.maximum(np.ptp(values,axis=0),EPS)
    scaled=(values-low)/span
    first=int(np.argmin(np.sum(scaled,axis=1))); selected=[first]
    distance=np.sum((scaled-scaled[first])**2,axis=1)
    while len(selected)<count:
        distance[selected]=-1.
        index=int(np.argmax(distance)); selected.append(index)
        distance=np.minimum(distance,np.sum((scaled-scaled[index])**2,axis=1))
    return np.asarray(selected,dtype=int)

def cvt_landmarks(descriptors, capacity=144, iterations=20):
    """Deterministic maximin-seeded k-means landmarks for a frozen CVT repertoire."""
    values=np.unique(np.asarray(descriptors,float),axis=0)
    if not len(values): return np.empty((0,0),float)
    count=min(int(capacity),len(values))
    first=int(np.argmin(np.sum(values*values,axis=1))); selected=[first]
    distance=np.sum((values-values[first])**2,axis=1)
    while len(selected)<count:
        distance[selected]=-1.
        index=int(np.argmax(distance)); selected.append(index)
        distance=np.minimum(distance,np.sum((values-values[index])**2,axis=1))
    centers=values[selected].copy()
    for _ in range(iterations):
        assignment=np.argmin(np.sum((values[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        updated=centers.copy()
        for index in range(len(centers)):
            members=values[assignment==index]
            if len(members): updated[index]=np.mean(members,axis=0)
        if np.allclose(updated,centers,rtol=0.,atol=1e-12): break
        centers=updated
    return centers

def qd_eligible_candidates(candidates):
    """Keep feasible stepping stones no worse than the current median loss."""
    feasible=[model for model in candidates if model.feasible]
    if not feasible: return [],float("inf")
    threshold=float(np.median([aggregate_loss(model) for model in feasible]))
    return [model for model in feasible if aggregate_loss(model)<=threshold],threshold

# How QD archives pick parent cells beyond their protected uniform share:
# "quality_coverage" weights each cell by its success rate, a bounded quality
# rank and a coverage bonus for rarely tried cells; "legacy" uses success only.
QD_PARENT_CHOICES=("quality_coverage","legacy")
QD_PARENT_CHOICE="quality_coverage"
class QualityDiversityArchive:
    """Frozen-CVT semantic repertoire used only as a bounded parent source."""
    policy="stratified_standardized_prediction_cvt"
    def __init__(self, probe, cats, seed, capacity=144, landmarks=None):
        self.probe=np.asarray(probe,float); self.cats=list(cats); self.seed=int(seed); self.capacity=int(capacity)
        if self.capacity<1: raise ValueError("QD capacity must be positive")
        self.landmarks=None if landmarks is None else np.asarray(landmarks,float)
        self.cells={}; self.updates=0; self.replacements=0; self.last_loss_threshold=None
        self.cell_trials={}; self.cell_successes={}

    @property
    def initialized(self): return self.landmarks is not None and len(self.landmarks)>0

    def descriptor(self, model):
        prediction=np.asarray(predict_targets(model,self.probe,self.cats),float)
        centred=prediction-np.mean(prediction,axis=0,keepdims=True)
        scale=np.std(centred,axis=0,keepdims=True)
        return np.divide(centred,scale,out=np.zeros_like(centred),where=scale>=EPS).reshape(-1)

    def initialize(self, candidates):
        descriptors=[self.descriptor(model) for model in candidates if model.feasible]
        if descriptors and not self.initialized:
            self.landmarks=cvt_landmarks(np.asarray(descriptors),self.capacity)
        elif descriptors and len(self.landmarks)<self.capacity:
            self._grow_landmarks(np.asarray(descriptors))
        return self.initialized

    def _grow_landmarks(self, descriptors):
        """Keep maximin seeding until the map is full.  The first eligible
        generation can hold far fewer distinct behaviours than cells; freezing
        then left most of the capacity unusable for the whole run."""
        if descriptors.shape[1]!=self.landmarks.shape[1]: return
        distance=np.min(np.sum((descriptors[:,None,:]-self.landmarks[None,:,:])**2,axis=2),axis=1)
        added=[]
        while len(self.landmarks)+len(added)<self.capacity:
            index=int(np.argmax(distance))
            if distance[index]<=1e-12: break
            added.append(descriptors[index])
            distance=np.minimum(distance,np.sum((descriptors-descriptors[index])**2,axis=1))
        if added: self.landmarks=np.vstack([self.landmarks,np.asarray(added)])

    def cell(self, model):
        if not self.initialized: raise ValueError("QD landmarks have not been initialized")
        descriptor=self.descriptor(model)
        return int(np.argmin(np.sum((self.landmarks-descriptor)**2,axis=1)))

    def update(self, candidates, loss_threshold=None):
        candidates=[model for model in candidates if model.feasible]
        self.updates+=len(candidates); self.last_loss_threshold=loss_threshold
        if not candidates or not self.initialize(candidates): return
        for model in candidates:
            key=self.cell(model); previous=self.cells.get(key)
            if previous is None or secondary_key(model)<secondary_key(previous):
                self.cells[key]=model.clone()
                if previous is not None: self.replacements+=1

    def sample(self, count):
        if count<=0 or not self.cells: return []
        items=[self.cells[key] for key in sorted(self.cells)]
        return [rng.choice(items).clone() for _ in range(count)]

    def sample_tagged(self, count, source, uniform_rate=.25):
        """Sample a protected uniform share, then favor productive cells."""
        if count<=0 or not self.cells: return []
        keys=sorted(self.cells); uniform=min(count,int(math.ceil(count*uniform_rate)))
        chosen=[rng.choice(keys) for _ in range(uniform)]
        weights=[(self.cell_successes.get(key,0.)+1.)/(self.cell_trials.get(key,0.)+2.) for key in keys]
        if QD_PARENT_CHOICE=="quality_coverage" and len(keys)>1:
            # Quality: a bounded rank weight (best cell at most e^2 ~ 7.4x the
            # worst), so a strong cell is preferred but can never take over.
            # Coverage: an optimism bonus for cells rarely tried as parents.
            losses={key:aggregate_loss(self.cells[key]) for key in keys}
            rank={key:sum(other<losses[key] for other in losses.values()) for key in keys}    # ties share a rank
            total=sum(self.cell_trials.get(key,0.) for key in keys)
            weights=[weight*math.exp(-2.*rank[key]/(len(keys)-1))*(1.+math.sqrt(math.log(total+2.)/(self.cell_trials.get(key,0.)+1.)))
                     for weight,key in zip(weights,keys)]
        chosen+=rng.choices(keys,weights=weights,k=count-uniform)
        return [ParentChoice(self.cells[key].clone(),source,key) for key in chosen]

    def begin_generation(self, decay=.98):
        for key in self.cell_trials:
            self.cell_trials[key]*=decay; self.cell_successes[key]*=decay

    def record_outcome(self, cell, success):
        if cell is None: return
        self.cell_trials[cell]=self.cell_trials.get(cell,0.)+1.
        self.cell_successes[cell]=self.cell_successes.get(cell,0.)+float(bool(success))

    def snapshot(self):
        return {"probe":self.probe,"cats":self.cats,"seed":self.seed,"capacity":self.capacity,"landmarks":self.landmarks,
                "cells":{key:PosteriorParticlePopulation._model_data(model) for key,model in self.cells.items()},
                "updates":self.updates,"replacements":self.replacements,"last_loss_threshold":self.last_loss_threshold,
                "cell_trials":self.cell_trials,"cell_successes":self.cell_successes}

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data["probe"],data["cats"],data["seed"],data["capacity"],data.get("landmarks"))
        result.cells={int(key):Model(**model) for key,model in data.get("cells",{}).items()}
        result.updates=int(data.get("updates",0)); result.replacements=int(data.get("replacements",0)); result.last_loss_threshold=data.get("last_loss_threshold")
        result.cell_trials={int(key):float(value) for key,value in data.get("cell_trials",{}).items()}
        result.cell_successes={int(key):float(value) for key,value in data.get("cell_successes",{}).items()}
        return result

    def diagnostics(self):
        capacity=len(self.landmarks) if self.initialized else self.capacity
        return {"policy":self.policy,"cells":len(self.cells),"capacity":capacity,
                "coverage":len(self.cells)/capacity if capacity else 0.,"updates":self.updates,"replacements":self.replacements,
                "probe_rows":len(self.probe),"initialized":self.initialized,"last_loss_threshold":self.last_loss_threshold,
                "cell_success":float(sum(self.cell_successes.values())/(sum(self.cell_trials.values())+EPS))}

    def stats(self):
        info=self.diagnostics()
        state="ready" if info["initialized"] else "warming"
        return f"QD CVT={info['cells']}/{info['capacity']} cells ({info['coverage']:.0%}); {state}; replacements={info['replacements']}; probe={info['probe_rows']}"

STRUCTURAL_OPERATOR_FAMILIES=(
    {"+","-","*","/","delta","max","min","hypot","distance_2","harmonic","geometric","lerp"},
    {"sin","cos","tan","exp","expm1","10^x","log","log1p","log10","sqrt","abs","neg","atan","sinh","cosh","erf","sinc","xlogx","inv","deg2rad","rad2deg","gaussian","softplus","sigmoid","tanh","relu","leaky_relu"},
    {"pow","square","cube","pow4","pow5","pow6","pow7","pow8","pow9","pow10","root3","root4","root5","root6","root7","root8","root9","root10"},
    {"if_else","if_in_range","if_out_of_range","rbf","exp_decay","log_base","atan2","mod","copysign","quantize"},
    {"gt","lt","gte","lte","eq","ne","round","floor","ceil","int","frac","oom","round2","floor2","ceil2","floordiv","bitwise_and","bitwise_or","bitwise_xor","bitwise_not","lshift","rshift","gcd","lcm","cat","x_at_pos_y"},
    {"perceptronSigma1","perceptronReLU1","perceptronCustom1","perceptronSigma2","perceptronReLU2","perceptronCustom2","python_rng","perlin_noise"},
)

def structural_descriptor(model, n_features):
    """Normalized operator families, shape, and feature use for structural QD."""
    counts=np.zeros(len(STRUCTURAL_OPERATOR_FAMILIES)+1,float); features=np.zeros(n_features,float); nodes=[]
    def visit(tree):
        nodes.append(tree)
        if tree[0]=="x":
            if 0<=tree[1]<n_features: features[tree[1]]=1.
        elif tree[0] not in ("c","x"):
            family=next((index for index,group in enumerate(STRUCTURAL_OPERATOR_FAMILIES) if tree[0] in group),len(STRUCTURAL_OPERATOR_FAMILIES))
            counts[family]+=1
            for child in tree[1:]: visit(child)
    for tree in model.trees: visit(tree)
    total=max(1.,float(len(nodes)))
    depth=max((node_depth(tree) for tree in model.trees),default=0)
    return np.r_[counts/total,min(1.,depth/10.),min(1.,math.log1p(total)/math.log(65.)),features]

class StructuralQualityDiversityArchive(QualityDiversityArchive):
    """Frozen-CVT map of expression construction, complementary to semantics."""
    policy="operator_family_depth_feature_cvt"
    def __init__(self, n_features, seed, capacity=144, landmarks=None):
        self.n_features=int(n_features)
        super().__init__(np.empty((0,self.n_features)),(),seed,capacity,landmarks)

    def descriptor(self, model): return structural_descriptor(model,self.n_features)

    def snapshot(self):
        data=super().snapshot(); data["n_features"]=self.n_features; return data

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data["n_features"],data["seed"],data["capacity"],data.get("landmarks"))
        result.cells={int(key):Model(**model) for key,model in data.get("cells",{}).items()}
        result.updates=int(data.get("updates",0)); result.replacements=int(data.get("replacements",0)); result.last_loss_threshold=data.get("last_loss_threshold")
        result.cell_trials={int(key):float(value) for key,value in data.get("cell_trials",{}).items()}
        result.cell_successes={int(key):float(value) for key,value in data.get("cell_successes",{}).items()}
        return result

RESIDUAL_ARCHIVE = True       # --residual-archive
RESIDUAL_TARGET_BINS = 3      # low / middle / high target values, per numeric output
RESIDUAL_REGION_BINS = 4      # quartiles of the probe inputs' first principal component
def residual_regions(probe_X, probe_Y, cats):
    """Fixed row groups for residual signatures: target-magnitude bins per
    numeric output and input-region bins, each a list of row-index arrays."""
    X=np.asarray(probe_X,float); groups=[]
    for j,labels in enumerate(cats):
        if labels is not None: continue
        y=np.asarray(probe_Y[:,j],float)
        edges=np.quantile(y,np.linspace(0,1,RESIDUAL_TARGET_BINS+1)[1:-1]) if len(y) else []
        bins=np.searchsorted(edges,y,side="left")
        groups.append([np.flatnonzero(bins==b) for b in range(RESIDUAL_TARGET_BINS)])
    if len(X)>=RESIDUAL_REGION_BINS and X.shape[1]:
        centred=X-X.mean(axis=0); scale=X.std(axis=0); centred=np.divide(centred,scale,out=np.zeros_like(centred),where=scale>EPS)
        try: component=np.linalg.svd(centred,full_matrices=False)[2][0]
        except np.linalg.LinAlgError: component=np.ones(X.shape[1])/math.sqrt(X.shape[1])
        score=centred@component
        edges=np.quantile(score,np.linspace(0,1,RESIDUAL_REGION_BINS+1)[1:-1])
        bins=np.searchsorted(edges,score,side="left")
        groups.append([np.flatnonzero(bins==b) for b in range(RESIDUAL_REGION_BINS)])
    return [[rows for rows in group if len(rows)] for group in groups]

class ResidualQualityDiversityArchive(QualityDiversityArchive):
    """Frozen-CVT map of *where* a model errs, so complementary partial models
    survive: one that nails small targets and one that nails large targets, or
    one per input regime, land in different cells even when neither is the
    overall best.  The descriptor is each row group's share of the model's
    error (target-magnitude bins per numeric output, then input-region bins),
    so it describes the error's location, not its size; each cell still keeps
    its lowest-loss model."""
    policy="residual_signature_cvt"
    def __init__(self, probe, probe_targets, cats, seed, capacity=64, landmarks=None):
        super().__init__(probe,cats,seed,capacity,landmarks)
        self.probe_targets=np.asarray(probe_targets,float)
        self.groups=residual_regions(self.probe,self.probe_targets,self.cats)

    def descriptor(self, model):
        prediction=np.asarray(predict_targets(model,self.probe,self.cats),float)
        errors=np.zeros(len(self.probe))
        for j,labels in enumerate(self.cats):
            y=self.probe_targets[:,j]
            errors+=(np.abs(prediction[:,j]-y)/max(target_scale(y),EPS) if labels is None else (np.rint(prediction[:,j])!=y).astype(float))
        errors=np.nan_to_num(errors,nan=CLIP,posinf=CLIP)
        parts=[]
        for group in self.groups:
            means=np.array([float(np.mean(errors[rows])) for rows in group])
            total=float(np.sum(means)); parts.append(means/total if total>EPS else np.full(len(means),1./len(means)))
        return np.concatenate(parts) if parts else np.zeros(1)

    def snapshot(self):
        data=super().snapshot(); data["probe_targets"]=self.probe_targets; return data

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data["probe"],data["probe_targets"],data["cats"],data["seed"],data["capacity"],data.get("landmarks"))
        result.cells={int(key):Model(**model) for key,model in data.get("cells",{}).items()}
        result.updates=int(data.get("updates",0)); result.replacements=int(data.get("replacements",0)); result.last_loss_threshold=data.get("last_loss_threshold")
        result.cell_trials={int(key):float(value) for key,value in data.get("cell_trials",{}).items()}
        result.cell_successes={int(key):float(value) for key,value in data.get("cell_successes",{}).items()}
        return result

    def stats(self):
        return "Residual "+super().stats()

@dataclass
class ParentChoice:
    model:Model
    source:str="ordinary"
    cell:int|None=None

class QDOutcomeController:
    """Decayed archive-vs-ordinary success evidence controlling the next QD rate."""
    def __init__(self, rate=.20, minimum=.10, maximum=.30, decay=.98, uniform_rate=.25):
        self.rate=float(rate); self.minimum=float(minimum); self.maximum=float(maximum); self.decay=float(decay); self.uniform_rate=float(uniform_rate)
        self.archive_trials=0.; self.archive_successes=0.; self.ordinary_trials=0.; self.ordinary_successes=0.

    def begin_generation(self, archives):
        self.archive_trials*=self.decay; self.archive_successes*=self.decay; self.ordinary_trials*=self.decay; self.ordinary_successes*=self.decay
        for archive in archives: archive.begin_generation(self.decay)

    def record(self, child, parents, archives):
        by_source=dict(zip(("semantic","structural","residual"),archives))
        for parent in parents:
            success=variation_improved(child,parent.model)
            if parent.source=="ordinary":
                self.ordinary_trials+=1.; self.ordinary_successes+=float(success)
            elif parent.source in by_source:
                self.archive_trials+=1.; self.archive_successes+=float(success); by_source[parent.source].record_outcome(parent.cell,success)

    def update_rate(self):
        if not self.archive_trials: return self.rate
        archive=(self.archive_successes+1.)/(self.archive_trials+2.)
        ordinary=(self.ordinary_successes+1.)/(self.ordinary_trials+2.)
        self.rate=float(np.clip(.20+.20*(archive-ordinary),self.minimum,self.maximum))
        return self.rate

    def snapshot(self):
        return {key:getattr(self,key) for key in ("rate","minimum","maximum","decay","uniform_rate","archive_trials","archive_successes","ordinary_trials","ordinary_successes")}

    @classmethod
    def from_snapshot(cls, data):
        result=cls(data.get("rate",.20),data.get("minimum",.10),data.get("maximum",.30),data.get("decay",.98),data.get("uniform_rate",.25))
        for key in ("archive_trials","archive_successes","ordinary_trials","ordinary_successes"): setattr(result,key,float(data.get(key,0.)))
        return result

    def diagnostics(self):
        return {"rate":self.rate,"bounds":[self.minimum,self.maximum],"uniform_cell_rate":self.uniform_rate,
                "archive_success":self.archive_successes/(self.archive_trials+EPS),"ordinary_success":self.ordinary_successes/(self.ordinary_trials+EPS)}

    def stats(self):
        info=self.diagnostics()
        return f"QD adaptive parents={info['rate']:.0%} (bounds {info['bounds'][0]:.0%}-{info['bounds'][1]:.0%}); archive/ordinary success={info['archive_success']:.0%}/{info['ordinary_success']:.0%}"

def qd_snapshot(semantic, structural, controller, residual=None):
    data={"semantic":semantic.snapshot(),"structural":structural.snapshot(),"controller":controller.snapshot()}
    if residual is not None: data["residual"]=residual.snapshot()
    return data
def residual_qd_from_snapshot(data):
    """The optional residual-signature repertoire (absent in older checkpoints)."""
    residual=(data or {}).get("residual")
    return None if residual is None else ResidualQualityDiversityArchive.from_snapshot(residual)
def qd_archives(semantic, structural, residual=None):
    """The active QD repertoires, in parent-source order."""
    return (semantic,structural) if residual is None else (semantic,structural,residual)
def qd_cell_models(semantic, structural, residual=None):
    return [model for archive in qd_archives(semantic,structural,residual) for model in archive.cells.values()]

def qd_from_snapshot(data):
    if not {"semantic","structural","controller"} <= set(data):
        raise ValueError("Checkpoint lacks adaptive dual-repertoire Quality-Diversity state and cannot resume")
    return (QualityDiversityArchive.from_snapshot(data["semantic"]),
            StructuralQualityDiversityArchive.from_snapshot(data["structural"]),
            QDOutcomeController.from_snapshot(data["controller"]))

def blend_qd_parents(base_parents, qd_archive, count, rate):
    """Replace a bounded fraction of ordinary parents with uniformly sampled QD elites."""
    if not 0<=rate<=1: raise ValueError("QD parent rate must be between zero and one")
    qd_count=min(count,int(round(count*rate))) if qd_archive.cells else 0
    parents=list(base_parents[:count-qd_count])+qd_archive.sample(qd_count)
    if len(parents)<count: parents+=list(base_parents[len(parents):count])
    rng.shuffle(parents)
    return parents

def dual_qd_parent_count(count, controller, semantic, structural, residual=None):
    return min(count,int(round(count*controller.rate))) if any(archive.cells for archive in qd_archives(semantic,structural,residual)) else 0

def blend_dual_qd_parents(base_parents, semantic, structural, count, qd_count, uniform_rate=.25, residual=None):
    """Tag ordinary/semantic/structural/residual parent origins and split QD equally."""
    active=[(source,archive) for source,archive in zip(("semantic","structural","residual"),qd_archives(semantic,structural,residual)) if archive.cells]
    if not active: return [ParentChoice(model,"ordinary") for model in base_parents[:count]]
    qd_count=min(count,qd_count); quota=[qd_count//len(active)]*len(active)
    for index in range(qd_count%len(active)): quota[index]+=1
    parents=[ParentChoice(model,"ordinary") for model in base_parents[:count-qd_count]]
    for (source,archive),amount in zip(active,quota): parents+=archive.sample_tagged(amount,source,uniform_rate)
    if len(parents)<count: parents += [ParentChoice(model,"ordinary") for model in base_parents[len(parents):count]]
    rng.shuffle(parents)
    return parents

def blend_fixed_semantic_parents(base_parents, semantic, count, qd_count):
    """Semantic-only fixed-rate control matching the pre-adaptive QD policy."""
    qd_count=min(count,qd_count) if semantic.cells else 0
    parents=[ParentChoice(model,"ordinary") for model in base_parents[:count-qd_count]]
    parents += [ParentChoice(model,"semantic",semantic.cell(model)) for model in semantic.sample(qd_count)]
    if len(parents)<count: parents += [ParentChoice(model,"ordinary") for model in base_parents[len(parents):count]]
    rng.shuffle(parents)
    return parents

def _lazy_shuffle(items):
    """rng.sample(items,len(items)) drawn on demand: the same pool algorithm
    and draws, so lexicase stopping after a few cases costs a few draws
    instead of one per training row."""
    pool=list(items); n=len(pool)
    for i in range(n):
        j=rng.randrange(n-i); value=pool[j]; pool[j]=pool[n-i-1]
        yield value

SCALE_BALANCED_SELECTION = False
SCALE_BALANCE_BINS = 5
def scale_balanced_row_weights(Y, cats, bins=SCALE_BALANCE_BINS):
    """Selection-only row weights giving every target-magnitude quantile bin
    of every numeric output the same total weight (mean weight 1)."""
    numeric=[j for j,labels in enumerate(cats) if labels is None]
    if not numeric or len(Y)<2: return None
    weights=np.zeros(len(Y))
    for j in numeric:
        magnitude=np.abs(np.asarray(Y[:,j],float))
        edges=np.unique(np.quantile(magnitude,np.linspace(0,1,bins+1)[1:-1]))
        # side="left": a tie mass sitting on an edge stays in one lower bin instead of swallowing the next.
        bin_index=np.searchsorted(edges,magnitude,side="left")
        counts=np.bincount(bin_index,minlength=len(edges)+1).astype(float)
        weights+=1./counts[bin_index]
    return weights*len(weights)/float(np.sum(weights))
def selection_errors(prediction, y, labels, scale_balanced=False):
    """Per-row errors that parent selection compares.  Scale-balanced mode
    compares asinh-compressed values, so an error on a small target counts
    like the same relative error on a large one."""
    if labels is not None: return (np.rint(prediction)!=y).astype(float)
    if not scale_balanced: return np.abs(prediction-y)
    unit=max(.05*target_scale(y),EPS)
    return np.abs(np.arcsinh(prediction/unit)-np.arcsinh(y/unit))
def lexicase_parents(pop, count, X, Y, cats, max_cases=0, case_weights=None, scale_balanced=False):
    """Epsilon-lexicase parents.  case_weights (one per row of X, selection
    only) make heavily weighted rows tend to be examined first, which is how a
    self-organised island role steers its parents; None keeps uniform order."""
    predictions=[predict_targets(m,X,cats) for m in pop]; errors=np.empty((len(pop),len(X)*Y.shape[1]))
    for i,pred in enumerate(predictions):
        parts=[selection_errors(pred[:,j],Y[:,j],cats[j],scale_balanced) for j in range(Y.shape[1])]
        errors[i]=np.concatenate(parts)
    if scale_balanced:
        balance=scale_balanced_row_weights(Y,cats)
        if balance is not None: case_weights=balance if case_weights is None else np.asarray(case_weights,float)*balance
    cases=list(range(errors.shape[1]))
    if max_cases and max_cases<len(cases):
        # Informed down-sampling prioritizes cases where the population still
        # disagrees most, preserving selection pressure under large datasets.
        cases=np.argsort(np.median(errors,axis=0))[-max_cases:].tolist()
    weights=None
    if case_weights is not None:
        weights=np.maximum(np.tile(np.asarray(case_weights,float),Y.shape[1])[cases],EPS)
    selected=[]
    for _ in range(count):
        pool=np.arange(len(pop))
        # Exponential race: sorting E/w gives a weight-proportional random order.
        order=_lazy_shuffle(cases) if weights is None else (cases[k] for k in np.argsort(np.random.exponential(size=len(cases))/weights))
        for case in order:
            values=errors[pool,case]; epsilon=np.median(np.abs(values-np.median(values)))
            pool=pool[values<=values.min()+epsilon+EPS]
            if len(pool)<=1: break
        selected.append(pop[int(rng.choice(pool))])
    return selected

def ask(prompt, default=""):
    x=input(f"{prompt}{' ['+str(default)+']' if default != '' else ''}: ").strip()
    return x if x else str(default)
def yes(x): return str(x).strip().lower() in ("1","y","yes","true")
def resolve_operator_groups(group_ids):
    """Expand ordered operator-group IDs while keeping each operator once."""
    selected=[]
    for group_id in group_ids:
        if group_id=="0": groups=DEFAULT_GROUP_IDS
        elif group_id in OPERATOR_GROUPS: groups=(group_id,)
        else: raise ValueError(f"Unknown operator group: {group_id}")
        for group in groups:
            for operator in OPERATOR_GROUPS[group][1]:
                if operator not in selected: selected.append(operator)
    if not selected: raise ValueError("Select at least one operator group")
    return selected
def choose_operator_groups():
    """Prompt for a default group followed by optional line-delimited additions."""
    print("Operator groups:")
    print("  0 = All standard deterministic operators (groups 1-9; excludes neural and stochastic)")
    for group_id,(name,operators) in OPERATOR_GROUPS.items(): print(f"  {group_id} = {name}: {', '.join(operators)}")
    groups=[ask("Operator group", "0")]
    while True:
        group=ask("Additional operator group (blank=done)")
        if not group: break
        groups.append(group)
    return resolve_operator_groups(groups)
def parse_delimiter(choice):
    standard={"0":",","1":";","2":" ","3":"\t"}
    return standard[choice] if choice in standard else ask("Custom delimiter")

def holdout_split_indices(n_rows, validation_rows, seed):
    """Deterministic disjoint train/validation split used by every run."""
    if not 0 < validation_rows < n_rows: raise ValueError("Validation rows must be between 1 and n_rows - 1")
    indices=np.random.default_rng(seed).permutation(n_rows)
    return indices[:-validation_rows],indices[-validation_rows:]

def kfold_split_indices(n_rows, folds, seed):
    """Deterministic disjoint K-fold partitions for external CV orchestration."""
    if folds < 2 or folds > n_rows: raise ValueError("folds must be between 2 and row count")
    shuffled=np.random.default_rng(seed).permutation(n_rows)
    return [np.asarray(part,dtype=int) for part in np.array_split(shuffled,folds)]

def dataset_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def write_run_manifest(path, seed, config, df, train_indices, validation_indices, external_validation=None):
    """Persist enough provenance to reproduce a run's data and split."""
    stamp=time.strftime("%Y%m%d-%H%M%S")
    run_dir=Path("afpo_runs") / f"{stamp}-seed{seed}"
    suffix=1
    while run_dir.exists():
        suffix+=1; run_dir=Path("afpo_runs") / f"{stamp}-seed{seed}-{suffix}"
    run_dir.mkdir(parents=True)
    manifest={
        "format_version":2, "seed":seed, "dataset":{"path":str(Path(path).resolve()), "sha256":dataset_sha256(path),
        "rows":len(df), "columns":list(df.columns)}, "configuration":config,
        "split":{"train_indices":train_indices.tolist(), "validation_indices":validation_indices.tolist(),
                 "external_validation":external_validation},
    }
    manifest_path=run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest_path

def record_selection_manifest(path, selection):
    """Append final selection evidence without changing immutable run inputs."""
    manifest=json.loads(Path(path).read_text())
    manifest["selection"]=selection
    Path(path).write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")

def write_model_card(path, model, feature_names, output_names, constraints, hypotheses, bayes, split, cats=None, bootstrap=None, selection=None, quality_diversity=None, survival=None, adf_diagnostics=None, evaluation=None, interaction_discovery=None, island_diagnostics=None):
    """Write auditable, machine-readable evidence beside a run manifest."""
    card={"format_version":1,"formulae":equations(model,feature_names,output_names,cats),
          "adf_definitions":adf_display_definitions(model,feature_names),
          "profile":constraints.profile,"constraints":constraints.describe(),
          "mdl":model_description(model,len(feature_names)),
          "per_output_constraint_violations":list(model_violations(model)),
          "hypotheses":hypotheses,"interaction_discovery":interaction_discovery,"posterior_diagnostics":([b.particles.last_predictive for b in bayes.banks] if isinstance(bayes,PerOutputBayesianBanks) else bayes.particles.last_predictive),
          "split_provenance":split,"coefficient_intervals":"not estimated; bounded point fitting only",
          "feature_operator_stability":"not estimated unless bootstrap audits are enabled",
          "bootstrap":bootstrap,"selection":selection,
          "quality_diversity":quality_diversity,
          "islands":island_diagnostics,
          "survival":survival,
          "evaluation":evaluation,
          "adf_registry":adf_diagnostics,
          "caveats":["Constraints are advisory soft objectives.","Hypotheses are training-only and are not inferred domain labels."]}
    target=Path(path).with_name("model_card.json"); target.write_text(json.dumps(card,indent=2,sort_keys=True)+"\n")
    return target

SAFE_CHECKPOINT_FORMAT=15

def save_checkpoint(path, generation, population, bayes, archive, state):
    """Atomically persist all stochastic/evolutionary state as safe JSON."""
    population_data=[PosteriorParticlePopulation._model_data(m) for m in population]
    if isinstance(bayes,PerOutputBayesianBanks):
        payload={"format_version":14,"generation":generation,"population":population_data,"bayesian_banks":bayesian_banks_snapshot(bayes),"archive":{"capacity":archive.capacity,"items":[PosteriorParticlePopulation._model_data(m) for m in archive.items]},"next_lineage_id":_NEXT_LINEAGE_ID,"python_rng_state":rng.getstate(),"numpy_rng_state":np.random.get_state(),"state":state}
    else:
        bayes_data={"ops":bayes.ops,"n_features":bayes.n_features,"base_exploration":bayes.base_exploration,
            "exploration":bayes.exploration,"pressure_exploration":bayes.pressure_exploration,"decay":bayes.decay,"floor":bayes.floor,"temperature":bayes.temperature,
            "op_alpha":bayes.op_alpha,"feature_alpha":bayes.feature_alpha,"op_draw":bayes.op_draw,
            "feature_draw":bayes.feature_draw,"entropy":bayes.entropy,"depth_alpha":bayes.depth_alpha,
            "constant_abs_sum":bayes.constant_abs_sum,"constant_weight":bayes.constant_weight,
            "constant_scale":bayes.constant_scale,"particles":bayes.particles.snapshot()}
        archive_data={"capacity":archive.capacity,"items":[PosteriorParticlePopulation._model_data(m) for m in archive.items]}
        payload={"format_version":14,"generation":generation,"population":population_data,"bayes":bayes_data,"archive":archive_data,"next_lineage_id":_NEXT_LINEAGE_ID,
                 "python_rng_state":rng.getstate(),"numpy_rng_state":np.random.get_state(),"state":state}
    encoded=_json_checkpoint_value(payload)
    body=json.dumps(encoded,sort_keys=True,separators=(",",":"),allow_nan=False)
    wrapper={"format_version":SAFE_CHECKPOINT_FORMAT,"checksum":hashlib.sha256(body.encode()).hexdigest(),"payload":encoded}
    temporary=Path(str(path)+".tmp")
    temporary.write_text(json.dumps(wrapper,sort_keys=True,separators=(",",":"))+"\n")
    temporary.replace(path)

def load_checkpoint(path, allow_unsafe_pickle=True):
    global _NEXT_LINEAGE_ID
    source=Path(path)
    try:
        wrapper=json.loads(source.read_text())
    except (UnicodeDecodeError,json.JSONDecodeError):
        if not allow_unsafe_pickle: raise ValueError("Refusing legacy pickle checkpoint; rerun with --allow-unsafe-pickle only for a trusted local file")
        with source.open("rb") as handle: payload=pickle.load(handle)
    else:
        if wrapper.get("format_version")!=SAFE_CHECKPOINT_FORMAT: raise ValueError("Unknown safe checkpoint format")
        body=json.dumps(wrapper["payload"],sort_keys=True,separators=(",",":"),allow_nan=False)
        if hashlib.sha256(body.encode()).hexdigest()!=wrapper.get("checksum"): raise ValueError("Checkpoint checksum does not match")
        payload=_from_json_checkpoint_value(wrapper["payload"])
    version=payload.get("format_version")
    if version in (10,11,12,13) and payload.get("state",{}).get("adf_registry",{}).get("enabled"):
        raise ValueError("Checkpoint contains v1 ADF state and cannot resume under v2; start a new ADF run")
    if version in (10,11,12,13,14):
        pop=[Model(**item) for item in payload["population"]]
        if "bayesian_banks" not in payload:
            b=payload["bayes"]; bayes=BayesianEquationGenerator(b["ops"],b["n_features"],b["base_exploration"],b["decay"],b["floor"],b["temperature"])
            for key in ("exploration","pressure_exploration","op_alpha","feature_alpha","op_draw","feature_draw","entropy","depth_alpha","constant_abs_sum","constant_weight","constant_scale"):
                if key in b: setattr(bayes,key,b[key])
            if "particles" in b: bayes.particles=PosteriorParticlePopulation.from_snapshot(b["particles"])
            state=payload["state"]
            archive=ParetoArchive(payload["archive"]["capacity"],state.get("nsga_normalization","legacy"),state.get("parsimony_quality_tolerance",0.)); archive.items=[Model(**item) for item in payload["archive"]["items"]]
            _NEXT_LINEAGE_ID=int(payload.get("next_lineage_id",max([m.lineage_id for m in [*pop,*archive.items]]+[-1])+1))
            rng.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"])
            return payload["generation"],pop,bayes,archive,state
        bayes=bayesian_banks_from_snapshot(payload.get("bayesian_banks"))
        state=payload["state"]
        archive=ParetoArchive(payload["archive"]["capacity"],state.get("nsga_normalization","legacy"),state.get("parsimony_quality_tolerance",0.)); archive.items=[Model(**item) for item in payload["archive"]["items"]]
        _NEXT_LINEAGE_ID=int(payload.get("next_lineage_id",max([m.lineage_id for m in [*pop,*archive.items]]+[-1])+1))
        rng.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"])
        return payload["generation"],pop,bayes,archive,state
    if payload.get("format_version") in (4,5,6):
        raise ValueError("Checkpoint predates the MDL-bits objective and cannot resume; start a new run")
    if payload.get("format_version") in (7,8,9):
        raise ValueError("Checkpoint predates the adaptive dual-repertoire Quality-Diversity state and cannot resume; start a new run")
    raise ValueError("Checkpoint uses aggregate objectives and cannot resume under the QD/MDL objective schema; start a new run")
def column_types(df):
    """Collect a deliberate per-column schema and remove constant columns."""
    print("Column types: 0=ignore, 1=numerical input, 2=categorical input, "
          "5=numerical output, 6=categorical output.")
    print("Example: choose 1 for a temperature feature, 2 for a city label, "
          "and 5 for the continuous value to predict.")
    print("At any prompt, type `type*count` (for example `1*3`) to assign that "
          "type to this column and the next count-1 columns, skipping their prompts.")
    print("Classification: choose 6 for class labels (text or integer codes such as 0/1/2); "
          "3+ classes then get one score equation per class. Type 5 fits a single regression equation.")
    usable=[i for i,c in enumerate(df.columns) if df[c].dropna().nunique() > 1]
    suggested_output=usable[-1] if usable else -1
    def class_like(values):
        """Few distinct integer codes usually mean class labels, not a quantity."""
        return (pd.api.types.is_numeric_dtype(values) and values.nunique()<=10
                and bool(np.all(np.isclose(values.to_numpy(float),np.round(values.to_numpy(float))))))
    types=[0]*len(df.columns)
    i=0
    while i < len(df.columns):
        col=df.columns[i]
        values=df[col].dropna()
        unique=int(values.nunique())
        examples=", ".join(repr(v) for v in values.iloc[:3].tolist()) or "(all missing)"
        if unique <= 1:
            print(f"[{i}] {col!r}: examples {examples}; {unique} unique value — automatically ignored.")
            types[i]=0
            i += 1
            continue
        numeric=pd.api.types.is_numeric_dtype(df[col])
        default=(5 if numeric else 6) if i==suggested_output else (1 if numeric else 2)
        if i==suggested_output and numeric and class_like(values):
            print(f"[{i}] {col!r} holds {unique} integer codes; if these are classes, choose 6 for per-class equations.")
        while True:
            answer=ask(f"[{i}] {col!r}: examples {examples}; {unique} unique — type or type*count", default)
            try:
                pieces=answer.replace(" ","").split("*",1)
                typ=int(pieces[0]); count=int(pieces[1]) if len(pieces)==2 else 1
            except ValueError:
                typ,count=-1,0
            if typ in (0,1,2,5,6) and count >= 1:
                end=min(len(df.columns),i+count)
                for j in range(i,end):
                    if df.iloc[:,j].dropna().nunique() <= 1:
                        types[j]=0
                        print(f"[{j}] {df.columns[j]!r} is constant and remains ignored.")
                    else:
                        types[j]=typ
                i=end
                break
            print("Enter 0, 1, 2, 5, or 6; optionally append *count (for example 1*3).")
    return types
def configure(path):
    delim=parse_delimiter(ask("Delimiter: 0=comma, 1=semicolon, 2=space, 3=tab, 4=custom", "0"))
    df=pd.read_csv(path,sep=delim,engine="python")
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns: {list(df.columns)}")
    types=column_types(df)
    if not any(t in (5,6) for t in types) or not any(t in (1,2) for t in types): raise ValueError("Select at least one input and one output")
    return df,types,delim
def parse_sequence_group(text):
    name,sep,columns=str(text).partition("=")
    if not sep or not name.strip(): raise ValueError(f"--sequence-group expects NAME=COL1,COL2,...; got {text!r}")
    return name.strip(),[column.strip() for column in columns.split(",") if column.strip()]
def encode(df, types, fitted_maps=None):
    features=[]; names=[]; outputs=[]; output_names=[]; categorical=[]; maps={}
    numeric_fills=dict((fitted_maps or {}).get("__afpo_numeric_fills__",{}))
    for col,t in zip(df.columns,types):
        s=df[col]
        if t==1:
            z=pd.to_numeric(s,errors="coerce").to_numpy(float)
            fill=float(numeric_fills.get(col,np.nanmedian(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.))
            numeric_fills[col]=fill; z=np.where(np.isfinite(z),z,fill); features.append(z); names.append(col)
        elif t==2:
            vals=s.fillna("__MISSING__").astype(str)
            classes=(fitted_maps or {}).get(col, sorted(vals.unique())); maps[col]=classes
            for cl in classes: features.append((vals==cl).to_numpy(float)); names.append(f"{col}={cl}")
        elif t in (5,6):
            if t==5:
                z=pd.to_numeric(s,errors="coerce").to_numpy(float); mask=np.isfinite(z); fill=np.nanmedian(z[mask]) if mask.any() else 0.; outputs.append(np.where(mask,z,fill)); categorical.append(None)
            else:
                vals=s.fillna("__MISSING__").astype(str)
                classes=(fitted_maps or {}).get(col, sorted(vals.unique())); maps[col]=classes
                # Unseen validation labels are deliberately treated as a loss,
                # not silently added as a new output dimension.
                outputs.append(np.array([classes.index(v) if v in classes else -1 for v in vals],float)); categorical.append(classes)
            output_names.append(col)
    maps["__afpo_numeric_fills__"]=numeric_fills
    X=np.column_stack(features)
    layout=(fitted_maps or {}).get(SEQUENCE_LAYOUT_KEY) or build_sequence_layout(names,SEQUENCE_GROUP_REQUEST)
    if layout is not None:
        global SEQUENCE_LAYOUT
        SEQUENCE_LAYOUT=layout; maps[SEQUENCE_LAYOUT_KEY]=layout
        X=sequence_augment(X,layout); names=names+sequence_feature_names(layout)
    return X,np.column_stack(outputs),names,output_names,categorical,maps

INTERPOLATION_PROBES=64
_INTERPOLATION_PROBE_CACHE={}
def interpolation_probes(X):
    """Fixed in-between inputs: midpoints of row pairs, keeping discrete columns from one row.

    Scoring only at the data rows lets a model pass through every point while
    spiking in between (x/x at 0, poles of inv); these probes expose that."""
    if not isinstance(X,np.ndarray) or X.ndim!=2 or len(X)<4: return None
    key=(X.__array_interface__["data"][0],X.shape,X.strides)
    cached=_INTERPOLATION_PROBE_CACHE.get(key)
    if cached is not None and cached[1] is X: return cached[0]
    generator=np.random.default_rng(len(X))
    count=min(INTERPOLATION_PROBES,len(X)); left=generator.integers(0,len(X),count); right=generator.integers(0,len(X),count)
    discrete=np.all(np.isclose(X,np.round(X)),axis=0)
    probes=np.where(discrete,X[left],(X[left]+X[right])/2)
    if len(_INTERPOLATION_PROBE_CACHE)>32: _INTERPOLATION_PROBE_CACHE.clear()
    _INTERPOLATION_PROBE_CACHE[key]=(probes,X)
    return probes

def interpolation_excursion(prediction, y):
    """Mean distance (in target spans) by which in-between predictions leave the target envelope."""
    low,high=float(np.min(y)),float(np.max(y)); span=max(high-low,EPS)
    outside=np.maximum(0.,np.maximum(low-span-prediction,prediction-high-span))/span
    return float(np.mean(np.minimum(outside,1e6)))

def assess(m, X, Y, affine_on, cats, fit_affine=True, constraints=None, output_names=(), adfs=None):
    """Score a model, optionally reusing train-fitted affine coefficients."""
    def valid(t):
        if t[0]=="x": return "" if isinstance(t[1],int) and t[1]>=0 else "invalid_feature"
        if t[0]=="c": return "" if np.isfinite(t[1]) else "nonfinite_constant"
        if t[0]=="arg": return "invalid_adf_argument"
        arity=(adfs or m.adfs).get(t[0],{}).get("arity") if t[0].startswith("adf_") else OPS.get(t[0],(-1,))[0]
        if arity is None: return f"unknown_operator:{t[0]}"
        if len(t)-1 != arity: return f"arity:{t[0]}"
        position=GUARDED_DIVISOR_OPS.get(t[0])
        if position is not None and t[position+1][0]=="c" and abs(float(t[position+1][1]))<EPS: return f"constant_zero_divisor:{t[0]}"
        for child in t[1:]:
            child_reason=valid(child)
            if child_reason: return child_reason
        return ""
    reason=""
    for tree in m.trees:
        reason=valid(tree)
        if reason: break
    if reason:
        m.feasible=False; m.invalid_reason=reason; INVALID_DIAGNOSTICS[reason]=INVALID_DIAGNOSTICS.get(reason,0)+1
        extra=len(cats) if constraints is not None and constraints.active else 0
        m.constraint_count=extra; m.objectives=(*(float("inf"),)*(2*len(cats)+extra),float("inf"),m.age); return
    m.feasible=True; m.invalid_reason=""
    if adfs is not None: m.adfs=dict(adfs)
    try:
        raw=np.column_stack([evaluate_cached(t,X,m.adfs) for t in m.trees])
    except (ArithmeticError, IndexError, RecursionError, ValueError) as error:
        m.feasible=False; m.invalid_reason=f"adf_evaluation:{error}"; INVALID_DIAGNOSTICS[m.invalid_reason]=INVALID_DIAGNOSTICS.get(m.invalid_reason,0)+1
        extra=len(cats) if constraints is not None and constraints.active else 0
        m.constraint_count=extra; m.objectives=(*(float("inf"),)*(2*len(cats)+extra),float("inf"),m.age); return
    targets,_=classification_layout(cats)
    if raw.shape[1] != sum(len(heads) for heads in targets):
        # Same objective layout as the other infeasible paths, or fronts()
        # compares misaligned slots against constrained models.
        m.feasible=False; m.invalid_reason="head_layout_mismatch"; INVALID_DIAGNOSTICS[m.invalid_reason]=INVALID_DIAGNOSTICS.get(m.invalid_reason,0)+1
        extra=len(cats) if constraints is not None and constraints.active else 0
        m.constraint_count=extra; m.objectives=(*(float("inf"),)*(2*len(cats)+extra),float("inf"),m.age); return
    scales=[]; losses=[]; shapes=[]; decoded=[]
    for j in range(Y.shape[1]):
        heads=targets[j]
        if fit_affine and affine_on and cats[j] is not None and len(cats[j])>=2:
            scales.extend(fit_classifier_affine(raw[:,heads],Y[:,j],len(cats[j])))
        else:
            for head in heads:
                if fit_affine: a,b=simplify_affine(raw[:,head],Y[:,j],*affine(raw[:,head],Y[:,j])) if affine_on and cats[j] is None else (1.,0.)
                else:
                    if head >= len(m.scales): raise ValueError("Validation was scored before affine scaling was fitted on training data")
                    a,b=m.scales[head]
                scales.append((a,b))
        p=clean(scales[heads[0]][0]*raw[:,heads[0]]+scales[heads[0]][1])
        if cats[j] is None:
            loss=robust_loss(p,Y[:,j]); probes=interpolation_probes(X)
            if probes is not None:
                try: between=clean(scales[heads[0]][0]*evaluate_cached(m.trees[heads[0]],probes,m.adfs)+scales[heads[0]][1])
                except (ArithmeticError, IndexError, RecursionError, ValueError): between=np.full(len(probes),CLIP)
                loss+=interpolation_excursion(between,Y[:,j])
            losses.append(loss); shapes.append(shape_error(p,Y[:,j]))
            decoded.append(p)
        elif len(cats[j])>2:
            scores=np.column_stack([clean(scales[head][0]*raw[:,head]+scales[head][1]) for head in heads]); probabilities=stable_softmax(scores)
            truth=np.asarray(np.rint(Y[:,j]),dtype=int); valid=(truth>=0)&(truth<len(cats[j])); row=np.arange(len(truth))
            cross_entropy=np.where(valid,-np.log(np.maximum(probabilities[row,np.clip(truth,0,len(cats[j])-1)],EPS)),-np.log(EPS))
            labels=np.argmax(probabilities,axis=1); losses.append(float(np.mean(cross_entropy))); shapes.append(float(np.mean(labels!=truth))); decoded.append(labels)
        else:
            labels=binary_labels(p,len(cats[j])); error_rate=float(np.mean(labels!=Y[:,j]))
            if len(cats[j])==2:
                # Log loss rewards moving the decision score toward the right side of 0.5, which a 0/1 error cannot.
                truth=np.asarray(np.rint(Y[:,j]),dtype=int); positive=binary_probabilities(p)[:,1]
                chosen=np.where(truth==1,positive,np.where(truth==0,1.-positive,0.))
                losses.append(float(np.mean(-np.log(np.maximum(chosen,EPS)))))
            else: losses.append(error_rate)
            shapes.append(error_rate)
            decoded.append(labels)
    if fit_affine:
        m.scales=scales
    complexity=model_description_bits(m,X.shape[1],adfs=m.adfs)
    pred=np.column_stack(decoded)
    def predict_for_constraints(inputs):
        alternate_raw=np.column_stack([evaluate(tree,inputs,m.adfs) for tree in m.trees])
        alternate=[]
        for output_index,heads in enumerate(targets):
            first=clean(scales[heads[0]][0]*alternate_raw[:,heads[0]]+scales[heads[0]][1])
            if cats[output_index] is None: alternate.append(first)
            elif len(cats[output_index])>2:
                scores=np.column_stack([clean(scales[head][0]*alternate_raw[:,head]+scales[head][1]) for head in heads])
                alternate.append(np.argmax(stable_softmax(scores),axis=1))
            else: alternate.append(binary_labels(first,len(cats[output_index])))
        return np.column_stack(alternate)
    violations=(constraints.violations(pred,X,output_names,predict_for_constraints,cats)
                if constraints is not None and constraints.active else ())
    m.constraint_count=len(violations)
    m.objectives=tuple(value for pair in zip(losses,shapes) for value in pair)+tuple(violations)+(complexity,m.age)
    if not np.all(np.isfinite(m.objectives)):
        m.feasible=False; m.invalid_reason="nonfinite_objectives"
        INVALID_DIAGNOSTICS[m.invalid_reason]=INVALID_DIAGNOSTICS.get(m.invalid_reason,0)+1
        m.objectives=(*(float("inf"),)*(len(m.objectives)-1),m.age)

def resolve_worker_count(requested, population):
    """Return a bounded process count; parallel evaluation is Linux-only."""
    if requested < 0: raise ValueError("--workers must be non-negative")
    if requested == 1: return 1
    if not sys.platform.startswith("linux"): return 1
    available=max(1,(os.cpu_count() or 1)-1)
    return max(1,min(population,requested if requested else available))

def _worker_init():
    signal.signal(signal.SIGINT,signal.SIG_IGN)

def _worker_assess_batch(models, dataset, indices, fit_affine, epoch, tune=False):
    """Score a batch using read-only arrays inherited by forked workers."""
    context=_WORKER_EVALUATION_CONTEXT
    X,Y=context["datasets"][dataset]
    if indices is not None: X,Y=row_subset(X,indices),row_subset(Y,indices)
    for model in models:
        if tune: tune_model_constants(model,X,Y,context["affine_on"],context["cats"])
        assess(model,X,Y,context["affine_on"],context["cats"],fit_affine,context["constraints"],context["output_names"])
    return models

def _copy_scored_model(target, scored):
    target.trees=scored.trees; target.scales=scored.scales; target.objectives=scored.objectives
    target.feasible=scored.feasible; target.invalid_reason=scored.invalid_reason
    target.constraint_count=scored.constraint_count
    if not target.feasible:
        INVALID_DIAGNOSTICS[target.invalid_reason]=INVALID_DIAGNOSTICS.get(target.invalid_reason,0)+1

class ModelEvaluator:
    """Parent-owned serial/process evaluator with deterministic result order."""
    def __init__(self, workers, datasets, affine_on, cats, constraints, output_names):
        self.workers=workers; self.epoch=0; self.executor=None; self.datasets=datasets
        self.affine_on=affine_on; self.cats=cats; self.constraints=constraints; self.output_names=output_names
        self._score_cache={}; self.cache_hits=0; self.cache_misses=0; self.row_model_evaluations=0
        if workers > 1:
            global _WORKER_EVALUATION_CONTEXT
            _WORKER_EVALUATION_CONTEXT={"datasets":datasets,"affine_on":affine_on,"cats":cats,
                                        "constraints":constraints,"output_names":output_names}
            self._open_pool()
    def _open_pool(self):
        self.executor=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context("fork"),initializer=_worker_init)
    def reopen(self):
        """Restart worker processes an interrupt shut down, so final scoring stays parallel."""
        if self.workers>1 and self.executor is None: self._open_pool()
    SCORE_CACHE_LIMIT=50000
    def begin_generation(self):
        """Bound the score cache.  A key fixes the tree, grammar, ADFs, rows and
        readout, and scoring is deterministic, so entries stay valid across
        generations; archives and QD cells are no longer rescored each time."""
        if len(self._score_cache)>self.SCORE_CACHE_LIMIT: self._score_cache.clear()
    def _cache_key(self, model, dataset, indices, fit_affine, tune=False):
        sample=None if indices is None else np.asarray(indices,dtype=np.int64).tobytes()
        scales=() if fit_affine else tuple((float(a),float(b)) for a,b in model.scales)
        return (dataset,sample,fit_affine,tune,repr(model.trees),adf_signature(model.trees,model.adfs),scales,tuple(model.mdl_operators),model.mdl_feature_count)
    @staticmethod
    def _score_data(model):
        return (list(model.trees),list(model.scales),tuple(model.objectives),model.feasible,model.invalid_reason,model.constraint_count)
    @staticmethod
    def _restore_score(model, data):
        model.trees,model.scales,model.objectives,model.feasible,model.invalid_reason,model.constraint_count=data
        model.trees=list(model.trees)
        model.scales=list(model.scales)
        model.objectives=(*model.objectives[:-1],model.age)
    def assess(self, models, dataset="train", indices=None, fit_affine=True, tune=False):
        """Score models; ``tune`` first refits their inner constants (training only)."""
        if not models: return
        self.epoch+=1
        pending=[]; duplicates=[]; pending_keys=set()
        for model in models:
            key=self._cache_key(model,dataset,indices,fit_affine,tune)
            if key in self._score_cache:
                self.cache_hits+=1; self._restore_score(model,self._score_cache[key])
            elif key in pending_keys:
                self.cache_hits+=1; duplicates.append((model,key))
            else:
                self.cache_misses+=1; pending.append((model,key)); pending_keys.add(key)
        if not pending: return
        models=[model for model,_ in pending]
        rows=len(self.datasets[dataset][0]) if indices is None else len(indices)
        self.row_model_evaluations+=len(models)*rows
        if self.executor is None:
            X,Y=self.datasets[dataset]
            if indices is not None: X,Y=row_subset(X,indices),row_subset(Y,indices)
            for model in models:
                if tune: tune_model_constants(model,X,Y,self.affine_on,self.cats)
                assess(model,X,Y,self.affine_on,self.cats,fit_affine,self.constraints,self.output_names)
        else:
            chunks=[list(models[index:index+max(1,math.ceil(len(models)/(self.workers*2)))]) for index in range(0,len(models),max(1,math.ceil(len(models)/(self.workers*2))))]
            futures=[self.executor.submit(_worker_assess_batch,chunk,dataset,indices,fit_affine,self.epoch,tune) for chunk in chunks]
            try:
                scored=[model for future in futures for model in future.result()]
            except KeyboardInterrupt:
                for future in futures: future.cancel()
                self.close()
                raise
            for target,result in zip(models,scored): _copy_scored_model(target,result)
        for model,key in pending:
            data=self._score_data(model); self._score_cache[key]=data
            # The tuned tree is itself a finished model: its later untuned
            # rescoring (stable copies, archives) is the same computation.
            if tune: self._score_cache[self._cache_key(model,dataset,indices,fit_affine)]=data
        for model,key in duplicates: self._restore_score(model,self._score_cache[key])
    def diagnostics(self):
        return {"cache_hits":self.cache_hits,"cache_misses":self.cache_misses,"row_model_evaluations":self.row_model_evaluations}
    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True,cancel_futures=True); self.executor=None

def frozen_objectives(m, X, Y, cats, constraints=None, output_names=()):
    """Evaluate held-out data without altering a selected model or its scales."""
    scored=m.clone()
    assess(scored,X,Y,False,cats,fit_affine=False,constraints=constraints,output_names=output_names)
    return tuple(scored.objectives)

def frozen_metrics(m, X, Y, cats, constraints=None, output_names=()):
    objectives=frozen_objectives(m,X,Y,cats,constraints,output_names)
    quality=objectives[:2*Y.shape[1]]
    return {"loss":aggregate_loss(quality+(0.,0.)),"shape":float(np.mean(quality[1::2])),
            "losses":tuple(quality[::2]),"shapes":tuple(quality[1::2])}

def _selection_metrics(model, objectives, cats):
    """Extract final-selection metrics without treating age as model quality."""
    target_count=len(cats) if cats is not None else len(_quality_objectives(model))//2
    quality=objectives[:2*target_count]
    return {"loss":float(np.mean(quality[::2])),"shape":float(np.mean(quality[1::2])),
            "mdl_bits":float(objectives[-2])}

def selection_identity(model):
    """Different fitted coefficients or ADF definitions are different predictors."""
    return repr(model.trees),tuple(model.scales),adf_signature(model.trees,model.adfs),tuple(model.mdl_operators),model.mdl_feature_count

def selection_evaluation(models, X=None, Y=None, cats=None, constraints=None, output_names=()):
    """One immutable score set for recommendations, rendering, and selection."""
    if (X is None)!=(Y is None): raise ValueError("Selection requires both X and Y, or neither")
    source="training" if X is None else "validation"
    entries=[]; seen=set()
    for model in models:
        if model is None or not model.feasible: continue
        if not np.all(np.isfinite(model.scales)): continue
        key=selection_identity(model)
        if key in seen: continue
        scored=model.clone()
        if X is not None:
            assess(scored,X,Y,False,cats if cats is not None else [None]*Y.shape[1],fit_affine=False,constraints=constraints,output_names=output_names)
        if not scored.feasible or not np.all(np.isfinite(scored.objectives)): continue
        metrics=_selection_metrics(scored,scored.objectives,cats)
        if not all(np.isfinite(value) for value in metrics.values()): continue
        seen.add(key); entries.append((model,scored,metrics))
    if not entries: raise ValueError(f"No feasible model with finite {source} scores is available for selection")
    return source,entries

def selection_frontier(entries):
    """Loss/MDL nondominance; shape breaks ties instead of excusing prediction error."""
    ordered=sorted(entries,key=lambda e:(e[2]["mdl_bits"],e[2]["loss"],e[2]["shape"],model_age(e[0]),repr(e[0].trees)))
    frontier=[]; best_loss=float("inf")
    for entry in ordered:
        if entry[2]["loss"] < best_loss:
            frontier.append(entry); best_loss=entry[2]["loss"]
    return frontier

def _select_knee(evaluation):
    source,entries=evaluation
    frontier=selection_frontier(entries)
    # On the nondominated loss/complexity curve, find the largest improvement
    # below the chord between its endpoints. No bend means no interior knee.
    values=np.asarray([[e[2]["mdl_bits"],e[2]["loss"]] for e in frontier])
    span=np.ptp(values,axis=0)
    normalized=(values-values.min(axis=0))/np.where(span>0,span,1.)
    bend=1.-normalized.sum(axis=1)
    interior=[i for i in range(1,len(frontier)-1) if bend[i]>np.finfo(float).eps]
    if interior:
        index=min(interior,key=lambda i:(-bend[i],frontier[i][2]["loss"],frontier[i][2]["shape"]))
    else:
        index=min(range(len(frontier)),key=lambda i:(frontier[i][2]["loss"],frontier[i][2]["mdl_bits"],frontier[i][2]["shape"]))
    model,scored,metrics=frontier[index]
    return model,{"source":source,"objectives":tuple(scored.objectives),"metrics":metrics,"policy":"pareto_knee",
                  "axes":["loss","mdl_bits"],"interior_knee":bool(interior),"bend":float(bend[index])}

def select_pareto_knee(models, X=None, Y=None, cats=None, constraints=None, output_names=()):
    """Return the bend in the nondominated loss/MDL curve, or lowest loss if absent."""
    return _select_knee(selection_evaluation(models,X,Y,cats,constraints,output_names))

def select_best_model(models, X=None, Y=None, cats=None, loss_tolerance=.01, constraints=None, output_names=()):
    """Choose the shortest-MDL model within a relative validation-loss tolerance."""
    return _select_best(selection_evaluation(models,X,Y,cats,constraints,output_names),loss_tolerance)

def _select_best(evaluation, loss_tolerance):
    if not np.isfinite(loss_tolerance) or loss_tolerance < 0: raise ValueError("selection loss tolerance must be finite and non-negative")
    source,entries=evaluation
    candidates=[e[0] for e in entries]; vectors=[tuple(e[1].objectives) for e in entries]; metrics=[e[2] for e in entries]
    best_loss=min(item["loss"] for item in metrics)
    allowed_loss=best_loss+loss_tolerance*max(abs(best_loss),EPS)
    eligible=[i for i,item in enumerate(metrics) if item["loss"]<=allowed_loss]
    index=min(eligible,key=lambda i:(metrics[i]["mdl_bits"],metrics[i]["shape"],metrics[i]["loss"],repr(candidates[i].trees)))
    return candidates[index],{"source":source,"objectives":vectors[index],"metrics":metrics[index],
        "policy":"loss_tolerance_shortest_mdl","loss_tolerance":loss_tolerance,"best_loss":best_loss,
        "allowed_loss":allowed_loss,"eligible_candidates":len(eligible)}

def model_options(models, X=None, Y=None, cats=None, loss_tolerance=.01, constraints=None, output_names=(), best_so_far=None, evaluation=None):
    """Return the deduplicated candidates shown when the user saves a model."""
    models=[*models]+([best_so_far] if best_so_far is not None else [])
    evaluation=evaluation or selection_evaluation(models,X,Y,cats,constraints,output_names)
    best,selection=_select_best(evaluation,loss_tolerance)
    lowest_loss,_=_select_best(evaluation,0.)
    knee,knee_info=_select_knee(evaluation)
    entries=evaluation[1]
    candidates=[
        (f"Best Score ({selection['source']} equivalent shortest MDL; ≤{selection['loss_tolerance']:.1%} loss tolerance)",best),
        *(((("Best-so-far retained",best_so_far),) if best_so_far is not None and any(e[0] is best_so_far for e in entries) else ())),
        (f"Lowest {selection['source'].title()} Loss",lowest_loss),
        (("Pareto Knee (loss/MDL bits)" if knee_info['interior_knee'] else "Pareto Knee fallback (no interior bend; lowest loss)"),knee),
        ("Shortest MDL Model",min(entries,key=lambda e:(e[2]["mdl_bits"],e[2]["loss"],e[2]["shape"]))[0]),
        ("Most Correct Shape",min(entries,key=lambda e:(e[2]["shape"],e[2]["loss"],e[2]["mdl_bits"]))[0]),
        ("Youngest Model",min(entries,key=lambda e:(model_age(e[0]),e[2]["loss"],e[2]["mdl_bits"]))[0]),
    ]
    labels=[]; choices=[]; seen=set()
    for label,model in candidates:
        key=selection_identity(model)
        if key not in seen:
            labels.append(label); choices.append(model); seen.add(key)
    return labels,choices,selection

def used_feature_indices(model):
    """Return encoded feature indices referenced by any output tree."""
    used=set()
    def visit(tree):
        if tree[0]=="x": used.add(int(tree[1])); return
        if tree[0]=="c": return
        for child in tree[1:]: visit(child)
    for tree in model.trees: visit(tree)
    return tuple(sorted(used))

def tree_map_svg(model, feature_names, output_names, cats=None):
    """Render selected symbolic trees as a standalone, dependency-free SVG."""
    cats=[None]*len(output_names) if cats is None else cats
    targets,_=classification_layout(cats); nodes={}; edges=[]; leaf=0; next_id=0; max_depth=0
    def label(tree):
        if tree[0]=="x": return feature_names[tree[1]]
        if tree[0]=="c": return f"{tree[1]:.6g}"
        return tree[0]
    def place(tree, depth):
        nonlocal leaf,next_id,max_depth
        max_depth=max(max_depth,depth); node_id=next_id; next_id+=1
        if tree[0] in {"x","c"}:
            x=leaf; leaf+=1; kind=tree[0]
        else:
            children=[place(child,depth+1) for child in tree[1:]]
            x=sum(nodes[child]["x"] for child in children)/len(children); kind="op"
            edges.extend((node_id,child) for child in children)
        nodes[node_id]={"id":node_id,"x":x,"depth":depth,"label":label(tree),"kind":kind}
        return node_id
    roots=[place(tree,0) for tree in model.trees]
    width=max(440,120*(leaf+1)); height=100+110*(max_depth+1)
    for node in nodes.values(): node["px"]=70+120*node["x"]; node["py"]=60+110*node["depth"]
    headers=[]
    for name,labels,heads in zip(output_names,cats,targets):
        for class_index,head in enumerate(heads):
            root=nodes[roots[head]]; a,b=model.scales[head]
            title=f"{name}[{labels[class_index]!r}] score" if labels is not None and len(labels)>2 else name
            affine="" if abs(a-1.)<EPS and abs(b)<EPS else f"  affine: {a:.6g}·y + {b:.6g}"
            headers.append(f'<text x="{root["px"]:.1f}" y="22" class="output">{xml_escape(title+affine)}</text>')
    palette={"x":"#d9f2e6","c":"#fff1c9","op":"#dbeafe"}
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.edge{stroke:#64748b;stroke-width:1.5}.node{stroke:#334155;stroke-width:1.2}.label{font:13px sans-serif;text-anchor:middle;dominant-baseline:middle}.output{font:600 13px sans-serif;text-anchor:middle;fill:#0f172a}</style>',
           '<rect width="100%" height="100%" fill="white"/>',*headers]
    for parent,child in edges:
        left,right=nodes[parent],nodes[child]
        lines.append(f'<line class="edge" x1="{left["px"]:.1f}" y1="{left["py"]+21:.1f}" x2="{right["px"]:.1f}" y2="{right["py"]-21:.1f}"/>')
    for node in nodes.values():
        text=xml_escape(node["label"][:30]+("…" if len(node["label"])>30 else ""))
        lines.extend((f'<rect class="node" x="{node["px"]-48:.1f}" y="{node["py"]-21:.1f}" width="96" height="42" rx="7" fill="{palette[node["kind"]]}"/>',
                      f'<text class="label" x="{node["px"]:.1f}" y="{node["py"]:.1f}">{text}</text>'))
    return "\n".join([*lines,"</svg>"])+"\n"

def training_input_ranges(frame, source_columns, types):
    """Observed numeric input bounds embedded in an exported model."""
    ranges={}
    for column,kind in zip(source_columns,types):
        if kind!=1 or column not in frame: continue
        values=pd.to_numeric(frame[column],errors="coerce").to_numpy(float); values=values[np.isfinite(values)]
        ranges[column]=[float(values.min()),float(values.max())] if len(values) else [0.,1.]
    return ranges

def export_model(m, feature_names, output_names, cats, maps, source_columns, types, fixture_df=None, input_ranges=None):
    used_indices=set(used_feature_indices(m)); input_columns=[]; offset=0
    layout=maps.get(SEQUENCE_LAYOUT_KEY)
    if layout is not None and (any(index>=layout["real"] for index in used_indices) or any(node[0] in ("seqsum","seqprod") for tree in m.trees for node in walk_tree(tree))):
        used_indices|={column for group in layout["groups"] for column in group["columns"]}
    for column,kind in zip(source_columns,types):
        width=1 if kind==1 else len(maps[column]) if kind==2 else 0
        if kind in (1,2) and used_indices.intersection(range(offset,offset+width)): input_columns.append(column)
        offset+=width
    if any(index<0 or index>=len(feature_names) for index in used_indices): raise ValueError("Model references an unavailable feature")
    ranges=training_input_ranges(fixture_df,source_columns,types) if input_ranges is None and fixture_df is not None else (input_ranges or {})
    payload=repr({"trees":m.trees,"scales":m.scales,"adfs":m.adfs,"features":feature_names,"outputs":output_names,"cats":cats,"maps":maps,"source_columns":source_columns,"types":types,"input_ranges":ranges,
                  "contract":{"version":3,"input_columns":input_columns,"feature_indices":sorted(used_indices),"feature_count":len(feature_names),"fixture":"best_model_fixture.csv","expected":"best_model_fixture_predictions.csv"}})
    # Keep exports genuinely standalone: embed exactly the guarded evaluator
    # used during search, rather than importing this training script.
    operator_source=inspect.getsource(op_eval).replace("def op_eval", "def op", 1)
    sequence_source=inspect.getsource(sequence_value)+inspect.getsource(sequence_augment)
    code=f'''#!/usr/bin/env python3
"""Exported AFPO symbolic model.  Run directly for sampling, plots, CSV generation, or CSV evaluation."""
import argparse, math, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
MODEL = {payload}
CONTRACT = MODEL['contract']
EPS=1e-12; CLIP=1e12
def clean(x): return np.clip(np.nan_to_num(x,nan=0.,posinf=CLIP,neginf=-CLIP),-CLIP,CLIP)
{operator_source}
{sequence_source}
SEQUENCE_LAYOUT=MODEL['maps'].get('{SEQUENCE_LAYOUT_KEY}')
def ev(t,X,args=None,pos=None):
    if t[0]=='x':
        if pos is not None and SEQUENCE_LAYOUT is not None and t[1]>=SEQUENCE_LAYOUT['real']: return sequence_value(X,SEQUENCE_LAYOUT,t[1]-SEQUENCE_LAYOUT['real'],pos)
        return X[:,t[1]]
    if t[0]=='c': return np.full(len(X),t[1])
    if t[0] in ('seqsum','seqprod'):
        parts=[ev(t[1],X,args,i) for i in range(SEQUENCE_LAYOUT['length'])]
        with np.errstate(all='ignore'): return clean(np.sum(parts,axis=0) if t[0]=='seqsum' else np.prod(parts,axis=0))
    if t[0]=='arg':
        if args is None: raise ValueError('ADF argument outside definition')
        return args[t[1]]
    if t[0].startswith('adf_'):
        item=MODEL['adfs'].get(t[0])
        if item is None or len(t)-1!=item['arity']: raise ValueError('Unknown or malformed ADF: '+t[0])
        return ev(item['tree'],X,[ev(q,X,args,pos) for q in t[1:]],pos)
    return op(t[0],[ev(q,X,args,pos) for q in t[1:]])
def transform(df):
    X=np.zeros((len(df),CONTRACT['feature_count'])); offset=0
    for col,t in zip(MODEL['source_columns'],MODEL['types']):
        width=1 if t==1 else len(MODEL['maps'][col]) if t==2 else 0
        if col not in CONTRACT['input_columns']:
            offset+=width; continue
        if col not in df: raise ValueError('Input misses required column: '+col)
        if t==1:
            x=pd.to_numeric(df[col],errors='coerce').to_numpy(float); fill=MODEL['maps'].get('__afpo_numeric_fills__',{{}}).get(col,0.); X[:,offset]=np.where(np.isfinite(x),x,fill)
        elif t==2:
            v=df[col].fillna('__MISSING__').astype(str)
            for index,c in enumerate(MODEL['maps'][col]): X[:,offset+index]=(v==c).to_numpy(float)
        offset+=width
    return X if SEQUENCE_LAYOUT is None else sequence_augment(X,SEQUENCE_LAYOUT)
def predict_frame(df):
    X=transform(df); raw=np.column_stack([a*ev(t,X)+b for t,(a,b) in zip(MODEL['trees'],MODEL['scales'])]); out=pd.DataFrame(index=df.index); head=0
    for i,n in enumerate(MODEL['outputs']):
        labels=MODEL['cats'][i]; count=len(labels) if labels is not None and len(labels)>2 else 1
        if labels is not None and count>1:
            scores=raw[:,head:head+count]; shifted=scores-np.max(scores,axis=1,keepdims=True); probabilities=np.exp(np.clip(shifted,-50,50)); chosen=np.argmax(probabilities/np.maximum(probabilities.sum(axis=1,keepdims=True),EPS),axis=1); out[n]=[labels[int(value)] for value in chosen]
        elif labels:
            out[n]=[labels[int(np.clip(round(v),0,len(labels)-1))] for v in raw[:,head]]
        else: out[n]=raw[:,head]
        head+=count
    return out
def write_new_csv(rows, destination='new_formula_predictions.csv'):
    """Save model inputs and their learned-formula predictions."""
    inputs=pd.DataFrame(rows).reindex(columns=CONTRACT['input_columns'])
    if inputs.empty: raise ValueError('At least one input row is required')
    destination=Path(destination); inputs.join(predict_frame(inputs)).to_csv(destination,index=False)
    return destination
def random_prediction_csv():
    count=int(input('Rows to generate [1]: ') or 1)
    if count<1: raise ValueError('Rows to generate must be positive')
    generator=np.random.default_rng(); data={{}}
    for column in CONTRACT['input_columns']:
        kind=MODEL['types'][MODEL['source_columns'].index(column)]
        if kind==1:
            lower,upper=MODEL.get('input_ranges',{{}}).get(column,[0.,1.])
            low=float(input(f'{{column}} minimum [{{lower:.6g}}]: ') or lower)
            high=float(input(f'{{column}} maximum [{{upper:.6g}}]: ') or upper)
            if low>high: raise ValueError(f'{{column}} minimum cannot exceed maximum')
            data[column]=generator.uniform(low,high,count)
        else:
            values=MODEL['maps'][column]
            if not values: raise ValueError(f'{{column}} has no known categories')
            data[column]=generator.choice(values,count)
    destination=input('Output CSV [new_formula_predictions.csv]: ') or 'new_formula_predictions.csv'
    return write_new_csv(pd.DataFrame(data),destination)
def verify_fixture(directory=None):
    """Check the saved preprocessing/prediction contract without the trainer."""
    root=Path(directory) if directory is not None else Path(__file__).resolve().parent
    fixture=root/CONTRACT['fixture']; expected=root/CONTRACT['expected']
    if not fixture.is_file() or not expected.is_file(): raise FileNotFoundError('Export fixture or expected predictions is missing')
    frame=pd.read_csv(fixture)
    missing=[column for column in CONTRACT['input_columns'] if column not in frame]
    if missing: raise ValueError('Fixture misses required input columns: '+', '.join(missing))
    actual=predict_frame(frame); reference=pd.read_csv(expected)
    if list(actual.columns)!=list(reference.columns): raise AssertionError('Output-column contract mismatch')
    for column in actual:
        if MODEL['cats'][MODEL['outputs'].index(column)] is None:
            if not np.allclose(actual[column].to_numpy(float),reference[column].to_numpy(float),rtol=1e-9,atol=1e-9): raise AssertionError('Prediction mismatch for '+column)
        elif not actual[column].astype(str).equals(reference[column].astype(str)): raise AssertionError('Prediction mismatch for '+column)
    return True
def numeric_output_names():
    return [name for index,name in enumerate(MODEL['outputs']) if MODEL['cats'][index] is None]
def parse_tolerances(specifications):
    """Parse repeated OUTPUT=TOLERANCE command-line values."""
    parsed={{}}
    for specification in specifications:
        if '=' not in specification: raise ValueError('Tolerance must use OUTPUT=VALUE: '+specification)
        output,value=specification.split('=',1)
        if not output or output in parsed: raise ValueError('Tolerance output is missing or repeated: '+output)
        try: tolerance=float(value)
        except ValueError: raise ValueError('Tolerance must be numeric for '+output)
        if not np.isfinite(tolerance) or tolerance<0: raise ValueError('Tolerance must be finite and non-negative for '+output)
        parsed[output]=tolerance
    expected=set(numeric_output_names()); supplied=set(parsed)
    if supplied!=expected:
        missing=sorted(expected-supplied); unexpected=sorted(supplied-expected); details=[]
        if missing: details.append('missing '+', '.join(missing))
        if unexpected: details.append('unexpected '+', '.join(unexpected))
        raise ValueError('Supply one tolerance for each numeric output ('+'; '.join(details)+')')
    return parsed
def evaluate_csv(path, tolerances):
    """Print ground-truth accuracy metrics for a CSV and return them as a dictionary."""
    frame=pd.read_csv(path)
    if frame.empty: raise ValueError('Test CSV must contain at least one row')
    required=list(CONTRACT['input_columns'])+list(MODEL['outputs'])
    missing=[column for column in required if column not in frame]
    if missing: raise ValueError('Test CSV misses required columns: '+', '.join(missing))
    predicted=predict_frame(frame); report={{}}
    for index,output in enumerate(MODEL['outputs']):
        actual=frame[output]
        if MODEL['cats'][index] is not None:
            if actual.isna().any(): raise ValueError('Ground truth has missing values for '+output)
            matches=predicted[output].astype(str).to_numpy()==actual.astype(str).to_numpy()
            accuracy=100.*float(matches.mean())
            report[output]={{'type':'categorical','rows':len(frame),'accuracy':accuracy}}
            print(f'{{output}}: categorical accuracy={{accuracy:.6g}}% ({{int(matches.sum())}}/{{len(frame)}} exact matches)')
            continue
        truth=pd.to_numeric(actual,errors='coerce').to_numpy(float)
        if not np.isfinite(truth).all(): raise ValueError('Ground truth must be finite numeric values for '+output)
        values=predicted[output].to_numpy(float)
        errors=np.abs(values-truth); tolerance=tolerances[output]; within=100.*float(np.mean(errors<=tolerance))
        order=np.argsort(-errors,kind='stable')[:10]
        report[output]={{'type':'numeric','rows':len(frame),'tolerance':tolerance,'average_absolute_error':float(errors.mean()),'minimum_absolute_error':float(errors.min()),'maximum_absolute_error':float(errors.max()),'within_tolerance_percent':within,'top_errors':[{{'row':int(row)+1,'actual':float(truth[row]),'predicted':float(values[row]),'absolute_error':float(errors[row])}} for row in order]}}
        print(f'{{output}}: rows={{len(frame)}}, tolerance=±{{tolerance:.6g}}, average absolute error={{errors.mean():.6g}}, minimum absolute error={{errors.min():.6g}}, maximum absolute error={{errors.max():.6g}}, within tolerance={{within:.6g}}%')
        print('  Top absolute errors:')
        for row in order: print(f'  row {{row+1}}: actual={{truth[row]:.12g}}, predicted={{values[row]:.12g}}, absolute error={{errors[row]:.12g}}')
    return report
def safe_name(value): return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in str(value))
def prepare_plot_dir():
    out=Path('evo_plots')
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    return out
def plots():
    import itertools
    import matplotlib.pyplot as plt
    inputs=[(c,t) for c,t in zip(MODEL['source_columns'],MODEL['types']) if t in (1,2) and c in CONTRACT['input_columns']]
    numeric=[c for c,t in inputs if t==1]; categorical=[c for c,t in inputs if t==2]
    bounds={{}}
    for col in numeric:
        lo=float(input(f'{{col}} minimum [0]: ') or 0); hi=float(input(f'{{col}} maximum [1]: ') or 1)
        if not lo < hi: raise ValueError(f'{{col}} maximum must exceed minimum')
        bounds[col]=(lo,hi)
    base={{col:(lo+hi)/2 for col,(lo,hi) in bounds.items()}}
    levels={{}}
    for col in categorical:
        levels[col]=list(MODEL['maps'][col])
        default=levels[col][0] if levels[col] else ''
        base[col]=input(f'Freeze {{col}} [{{default}}]: ') or default
    out=prepare_plot_dir(); numeric_outputs=[name for i,name in enumerate(MODEL['outputs']) if MODEL['cats'][i] is None]
    skipped=[name for i,name in enumerate(MODEL['outputs']) if MODEL['cats'][i] is not None]
    if skipped: print('Skipping categorical outputs for plots:', ', '.join(skipped))
    def frame(count): return pd.DataFrame([base.copy() for _ in range(count)])
    for col in numeric:
        grid=np.linspace(*bounds[col],250); d=frame(len(grid)); d[col]=grid; p=predict_frame(d)
        for output in numeric_outputs:
            plt.figure(); plt.plot(grid,p[output]); plt.xlabel(col); plt.ylabel(output); plt.tight_layout(); plt.savefig(out/f'1d_{{safe_name(col)}}__{{safe_name(output)}}.png',dpi=150); plt.close()
    for col in categorical:
        values=levels[col]; d=frame(len(values)); d[col]=values; p=predict_frame(d)
        for output in numeric_outputs:
            plt.figure(); plt.bar([str(v) for v in values],p[output]); plt.xlabel(col); plt.ylabel(output); plt.tight_layout(); plt.savefig(out/f'1d_{{safe_name(col)}}__{{safe_name(output)}}.png',dpi=150); plt.close()
    for (left,left_type),(right,right_type) in itertools.combinations(inputs,2):
        pair=f'{{safe_name(left)}}__{{safe_name(right)}}'
        if left_type==right_type==1:
            gx,gy=np.meshgrid(np.linspace(*bounds[left],100),np.linspace(*bounds[right],100)); d=frame(gx.size); d[left]=gx.ravel(); d[right]=gy.ravel(); p=predict_frame(d)
            for output in numeric_outputs:
                plt.figure(); plt.contourf(gx,gy,p[output].to_numpy().reshape(gx.shape),levels=30); plt.colorbar(label=output); plt.xlabel(left); plt.ylabel(right); plt.tight_layout(); plt.savefig(out/f'2d_{{pair}}__{{safe_name(output)}}.png',dpi=150); plt.close()
        elif left_type!=right_type:
            number,category=(left,right) if left_type==1 else (right,left); grid=np.linspace(*bounds[number],250)
            for output in numeric_outputs:
                plt.figure()
                for value in levels[category]:
                    d=frame(len(grid)); d[number]=grid; d[category]=value; plt.plot(grid,predict_frame(d)[output],label=str(value))
                plt.legend(title=category); plt.xlabel(number); plt.ylabel(output); plt.tight_layout(); plt.savefig(out/f'2d_{{pair}}__{{safe_name(output)}}.png',dpi=150); plt.close()
        else:
            rows,columns=levels[left],levels[right]; d=frame(len(rows)*len(columns)); d[left]=np.repeat(rows,len(columns)); d[right]=np.tile(columns,len(rows)); p=predict_frame(d)
            for output in numeric_outputs:
                plt.figure(); image=plt.imshow(p[output].to_numpy().reshape(len(rows),len(columns)),aspect='auto'); plt.colorbar(image,label=output); plt.xticks(range(len(columns)),[str(v) for v in columns]); plt.yticks(range(len(rows)),[str(v) for v in rows]); plt.xlabel(right); plt.ylabel(left); plt.tight_layout(); plt.savefig(out/f'2d_{{pair}}__{{safe_name(output)}}.png',dpi=150); plt.close()
    print('Saved plots to',out)
def main():
    parser=argparse.ArgumentParser(description='Use or evaluate an exported AFPO model.')
    parser.add_argument('--verify-fixture',action='store_true',help='Verify the exported prediction fixture')
    parser.add_argument('--test-csv',metavar='PATH',help='CSV containing model inputs and ground-truth output columns')
    parser.add_argument('--tolerance',action='append',default=[],metavar='OUTPUT=VALUE',help='Absolute tolerance for one numeric output; repeat for each numeric output')
    args=parser.parse_args()
    if args.verify_fixture:
        if args.test_csv: parser.error('--verify-fixture cannot be combined with --test-csv')
        verify_fixture(); print('Export contract verified.'); return
    if args.test_csv:
        try: evaluate_csv(args.test_csv,parse_tolerances(args.tolerance))
        except ValueError as error: parser.error(str(error))
        return
    if args.tolerance: parser.error('--tolerance requires --test-csv')
    mode=input('Mode: 0=use model, 1=test export contract, 2=test CSV [0]: ') or '0'
    if mode=='1': verify_fixture(); print('Export contract verified.'); return
    if mode=='2':
        path=input('Test CSV: ')
        if not path: raise ValueError('Test CSV path is required')
        tolerances={{}}
        for output in numeric_output_names():
            value=input(f'{{output}} absolute tolerance: ')
            try: tolerance=float(value)
            except ValueError: raise ValueError('Tolerance must be numeric for '+output)
            if not np.isfinite(tolerance) or tolerance<0: raise ValueError('Tolerance must be finite and non-negative for '+output)
            tolerances[output]=tolerance
        evaluate_csv(path,tolerances); return
    if mode!='0': raise ValueError('Mode must be 0, 1, or 2')
    print('Model outputs:', ', '.join(MODEL['outputs']))
    if (input('Plot model? [y/N]: ').lower()=='y'): plots()
    if input('Generate predicted CSV? [y/N]: ').lower()=='y':
        src=Path(input('Input CSV: ')); d=pd.read_csv(src); predict_frame(d).join(d).to_csv('generated_predictions.csv',index=False); print('Saved generated_predictions.csv')
    if input('Generate a randomized prediction CSV from training ranges? [y/N]: ').lower()=='y': print('Saved',random_prediction_csv())
    if input('Prediction scatterplot from CSV? [y/N]: ').lower()=='y':
        import matplotlib.pyplot as plt
        d=pd.read_csv(input('CSV: ')); p=predict_frame(d); out=prepare_plot_dir()
        for n in MODEL['outputs']:
            if n in d and MODEL['cats'][MODEL['outputs'].index(n)] is None: plt.figure(); plt.scatter(d[n],p[n],s=12); q=np.r_[d[n],p[n]]; plt.plot([q.min(),q.max()],[q.min(),q.max()],'k--'); plt.savefig(out/f'scatter_{{safe_name(n)}}.png',dpi=150); plt.close()
    while True:
        try:
            row={{c:input(f'{{c}}: ') for c,t in zip(MODEL['source_columns'],MODEL['types']) if t in (1,2) and c in CONTRACT['input_columns']}}; print(predict_frame(pd.DataFrame([row])).iloc[0].to_dict())
        except KeyboardInterrupt: break
if __name__=='__main__': main()
'''
    Path("best_model.py").write_text(code)
    Path("model_tree.svg").write_text(tree_map_svg(m,feature_names,output_names,cats),encoding="utf-8")
    if fixture_df is not None:
        fixture=fixture_df.head(16).copy()
        X,_,_,_,_,_=encode(fixture,types,maps)
        decoded=predict_targets(m,X,cats); expected=pd.DataFrame(index=fixture.index)
        for index,name in enumerate(output_names):
            expected[name]=([cats[index][int(value)] for value in decoded[:,index]] if cats[index] else decoded[:,index])
        fixture.loc[:,input_columns].to_csv("best_model_fixture.csv",index=False)
        expected.to_csv("best_model_fixture_predictions.csv",index=False)

def snapshot_evolution_state(state, portfolio, cases, semantic_qd, structural_qd, qd_controller, best_models, pressure, library=None, adf_registry=None, budget=None):
    state["mutation_portfolio"]=portfolio.snapshot()
    state["case_population"]=cases.snapshot()
    state["quality_diversity"]=qd_snapshot(semantic_qd,structural_qd,qd_controller)
    state["best_model"]=best_models.snapshot()
    state["dynamic_pressure"]=pressure.snapshot()
    if library is not None: state["fragment_library"]=library.snapshot()
    if budget is not None: state["evaluation_budget"]=budget.snapshot()
    if adf_registry is not None: state["adf_registry"]=adf_registry.snapshot()

@dataclass
class IslandRuntime:
    """One independently adapting AFPO population and its local memories."""
    population:list
    bayes:Any
    archive:ParetoArchive
    portfolio:MutationPortfolio
    cases:CasePopulation
    semantic_qd:QualityDiversityArchive
    structural_qd:StructuralQualityDiversityArchive
    qd_controller:QDOutcomeController
    best_models:BestModelArchive
    pressure:DynamicPressureController
    library:FragmentLibrary
    adf_registry:ADFRegistry
    budget:EvaluationBudget
    population_size:int=0
    island_index:int=0
    stage:int=0
    role:dict=field(default_factory=dict)
    residual_qd:Any=None
    def __post_init__(self):
        if not self.population_size: self.population_size=len(self.population)

def island_snapshot(island):
    """Serialize an island without sharing mutable evolutionary state."""
    return {
        "population_size":island.population_size,
        "island":island.island_index,"stage":island.stage,"role":dict(island.role),
        "population":[PosteriorParticlePopulation._model_data(model) for model in island.population],
        "bayesian_banks":bayesian_banks_snapshot(island.bayes),
        "archive":{"capacity":island.archive.capacity,"normalization":island.archive.normalization,"parsimony_quality_tolerance":island.archive.parsimony_quality_tolerance,"items":[PosteriorParticlePopulation._model_data(model) for model in island.archive.items]},
        "runtime":{
            "mutation_portfolio":island.portfolio.snapshot(),
            "case_population":island.cases.snapshot(),
            "quality_diversity":qd_snapshot(island.semantic_qd,island.structural_qd,island.qd_controller,island.residual_qd),
            "best_model":island.best_models.snapshot(),
            "dynamic_pressure":island.pressure.snapshot(),
            "fragment_library":island.library.snapshot(),
            "adf_registry":island.adf_registry.snapshot(),
            "evaluation_budget":island.budget.snapshot(),
        },
    }

def island_from_snapshot(data, n_rows, parsimony_quality_tolerance):
    """Restore an independently checkpointed island."""
    runtime=data["runtime"]
    archive=ParetoArchive(data["archive"]["capacity"],data["archive"].get("normalization","intercept"),data["archive"].get("parsimony_quality_tolerance",parsimony_quality_tolerance))
    archive.items=[Model(**model) for model in data["archive"]["items"]]
    semantic_qd,structural_qd,qd_controller=qd_from_snapshot(runtime["quality_diversity"])
    portfolio=MutationPortfolio(); portfolio.restore(runtime["mutation_portfolio"])
    best_models=BestModelArchive.from_snapshot(runtime.get("best_model",{}),parsimony_quality_tolerance)
    population=[Model(**model) for model in data["population"]]
    if best_models.model is None: best_models.update([*archive.items,*population])
    return IslandRuntime(
        population,
        bayesian_banks_from_snapshot(data["bayesian_banks"]),
        archive,portfolio,CasePopulation.from_snapshot(runtime["case_population"],n_rows),semantic_qd,structural_qd,qd_controller,
        best_models,
        DynamicPressureController.from_snapshot(runtime["dynamic_pressure"]),
        FragmentLibrary.from_snapshot(runtime.get("fragment_library",{})),
        ADFRegistry.from_snapshot(runtime.get("adf_registry",{})),
        EvaluationBudget.from_snapshot(runtime.get("evaluation_budget",{})),
        data.get("population_size",len(population)),
        int(data.get("island",0)),int(data.get("stage",0)),dict(data.get("role") or {}),
        residual_qd_from_snapshot(runtime["quality_diversity"]),
    )

def snapshot_islands(state, islands, island_config):
    """Persist every island while retaining island zero's legacy state fields."""
    snapshots=[island_snapshot(island) for island in islands]
    state["island_config"]={**island_config,"migration_events":int(island_config.get("migration_events",0))}
    state["island_states"]=snapshots
    runtime=snapshots[0]["runtime"]
    state.update(runtime)

def migrate_fragments(islands, count, generation=None):
    """Send each island's best fragments to the next island on the ring.

    Migrants arrive on probation (support 1, no contribution, provenance
    kept) and must earn their place through the receiver's own support and
    contribution accounting; fragments calling ADFs the receiver lacks stay home."""
    if len(islands)<2 or count<1: return 0
    outgoing=[]
    for island in islands:
        ranked=sorted(island.library.items.values(),key=lambda item:(-FragmentLibrary._item_score(item),repr(item["tree"])))
        outgoing.append([dict(item) for item in ranked[:count]])
    moved=0
    for index,island in enumerate(islands):
        source=(index-1)%len(islands)
        for item in outgoing[source]:
            key=fragment_key(item["tree"])
            calls={node[0] for node in walk_tree(item["tree"]) if node[0].startswith("adf_")}
            if key in island.library.items or not calls<=set(island.adf_registry.definitions): continue
            island.library.items[key]={**item,"support":1,"contribution":0.,"uses":0,"rejections":0,"source":"island_migrant","probation":FRAGMENT_MIGRANT_PROBATION,
                                       "provenance":{"island":islands[source].island_index,"stage":islands[source].stage,"generation":generation,"origin_source":item.get("source")}}
            moved+=1
        island.library._trim_items()
    return moved

def migrate_islands(islands, migrant_count, *, X, nsga_normalization, parsimony_quality_tolerance, evaluator=None):
    """Send local Pareto elites around a ring, keeping each island's size fixed."""
    if len(islands)<2 or migrant_count<1: return 0
    outgoing=[]
    for island in islands:
        if evaluator is not None:
            refresh_persistent_scores(island.archive,island.best_models,island.semantic_qd,island.structural_qd,evaluator,island.residual_qd)
            evaluator.assess(island.population,"train")
        pool=[model for model in island.population if model.feasible]
        if not pool: pool=island.population
        outgoing.append([model.clone() for model in select_nsga(pool,min(migrant_count,len(pool)),nsga_normalization,parsimony_quality_tolerance)])
    for index,island in enumerate(islands):
        incoming=outgoing[(index-1)%len(islands)]
        island.adf_registry.import_models(incoming)
        for migrant in incoming: migrant.origin="island_migrant"
        island.population=select_nsga([*island.population,*incoming],len(island.population),nsga_normalization,parsimony_quality_tolerance)
        island.archive.update(incoming,X)
        island.best_models.update(incoming)
    return sum(map(len,outgoing))

# AFPO stages ("vertical islands").  Each island is split into a ladder of
# stages, each a full, isolated IslandRuntime (its archives, QD cells and
# fragment library are parent sources, so sharing them would leak top-stage
# models back down).  Only stage 0 receives brand-new random models; upper
# stages are fed by promotion.  Modes:
#   fitness  HFC-style: a model moves up once its full-training loss beats the
#            admission threshold of the stage above (a quantile of that stage's
#            losses, never loosened).
#   age      ALPS-style: stage k holds models up to an age limit; older ones
#            leave and are admitted above only if they survive its NSGA
#            selection; stage 0 is reseeded every age_gap generations.
#   both     fitness promotion plus age eviction; an evicted model is admitted
#            above only if it also beats the admission threshold.
# Cells are stored island-major: cells[island*stages+stage].  With one stage
# this is exactly the plain island model.
STAGE_MODES=("off","fitness","age","both")
STAGE_AGE_SCHEDULES=("linear","polynomial","exponential")
def stage_config(mode="off", count=1, interval=5, age_gap=10, schedule="polynomial", threshold_quantile=.5, **state):
    """Validated stage configuration plus its persistent bookkeeping."""
    mode=str(mode)
    if mode not in STAGE_MODES: raise ValueError(f"Unknown stage mode {mode!r}; choose one of {', '.join(STAGE_MODES)}")
    count=1 if mode=="off" else int(count)
    if mode!="off" and count<2: raise ValueError("Stages need at least two stages")
    if int(interval)<1 or int(age_gap)<1: raise ValueError("Stage promotion interval and age gap must be positive")
    if schedule not in STAGE_AGE_SCHEDULES: raise ValueError(f"Unknown age schedule {schedule!r}")
    if not 0<float(threshold_quantile)<=1: raise ValueError("Stage threshold quantile must be in (0, 1]")
    return {"mode":mode,"count":count,"interval":int(interval),"age_gap":int(age_gap),"schedule":schedule,
            "threshold_quantile":float(threshold_quantile),"thresholds":dict(state.get("thresholds",{})),
            "promotion_events":int(state.get("promotion_events",0)),"promoted":int(state.get("promoted",0)),
            "evicted":int(state.get("evicted",0)),"reseeds":int(state.get("reseeds",0))}
def stage_age_limit(stage, config):
    """Oldest age stage `stage` may hold; None for the unlimited top stage."""
    if config["mode"] not in ("age","both") or stage>=config["count"]-1: return None
    gap=config["age_gap"]; n=stage+1
    return gap*{"linear":n,"polynomial":n*n,"exponential":2**stage}[config["schedule"]]
def cell_population_sizes(total, cells):
    return [total//cells+(index<total%cells) for index in range(cells)]
def fresh_stage_models(count, cell, generation, *, n_features, ops, nodes, depth):
    """Brand-new random models for stage 0, in the cell's current grammar."""
    registry=cell.adf_registry; definitions=registry.definitions
    active_ops=registry.operators(ops) if registry.enabled else list(ops)
    head_count=len(cell.population[0].trees) if cell.population else 1
    models=[]
    for _ in range(max(0,count)):
        trees=[random_tree(n_features,active_ops,nodes,depth,adfs=definitions) for _ in range(head_count)]
        models.append(Model(trees,[(1.,0.)]*head_count,0,origin="stage_seed",mdl_operators=grammar_for_trees(active_ops,trees,definitions),
                            mdl_feature_count=n_features,adfs=dict(definitions),birth_generation=generation))
    return models
def _stage_threshold(config, island_index, stage, receiver):
    """Admission loss for `stage` (monotone: a stage's bar is never lowered)."""
    losses=[aggregate_loss(model) for model in receiver.population if model.feasible and np.isfinite(aggregate_loss(model))]
    key=f"{island_index}:{stage}"; previous=config["thresholds"].get(key)
    if not losses: return previous
    current=float(np.quantile(losses,config["threshold_quantile"]))
    value=current if previous is None else min(float(previous),current)
    config["thresholds"][key]=value
    return value
def promote_stages(cells, island_count, config, generation, *, X, n_features, ops, nodes, depth, nsga_normalization,
                   parsimony_quality_tolerance, evaluator=None):
    """Move models up each island's stage ladder; returns (promoted, evicted, reseeded)."""
    stages=config["count"]; mode=config["mode"]
    if mode=="off" or stages<2: return 0,0,0
    promoted=evicted=reseeded=0
    for island_index in range(island_count):
        ladder=cells[island_index*stages:(island_index+1)*stages]
        if evaluator is not None:
            for cell in ladder:
                refresh_persistent_scores(cell.archive,cell.best_models,cell.semantic_qd,cell.structural_qd,evaluator,cell.residual_qd)
                evaluator.assess(cell.population,"train")
        # Top-down, so one event moves a model at most one rung.
        for stage in range(stages-2,-1,-1):
            source,receiver=ladder[stage],ladder[stage+1]
            threshold=_stage_threshold(config,island_index,stage+1,receiver) if mode in ("fitness","both") else None
            limit=stage_age_limit(stage,config)
            good=[m for m in source.population if m.feasible and threshold is not None and aggregate_loss(m)<threshold]
            if good:
                good=select_nsga(good,min(len(good),max(1,len(source.population)//4)),nsga_normalization,parsimony_quality_tolerance)
            old=[m for m in source.population if limit is not None and m.age>limit]
            leaving={id(m):m for m in [*good,*old]}
            if not leaving: continue
            staying=[m for m in source.population if id(m) not in leaving]
            if len(staying)<2:
                # Never empty a stage: keep its best leavers until offspring refill it.
                keep=select_nsga(list(leaving.values()),min(len(leaving),2-len(staying)),nsga_normalization,parsimony_quality_tolerance)
                for m in keep: leaving.pop(id(m))
                staying+=keep
            good_ids={id(m) for m in good}
            offered=[m for m in leaving.values() if id(m) in good_ids or (mode=="age" and m.feasible)
                     or (mode=="both" and m.feasible and threshold is not None and aggregate_loss(m)<threshold)]
            incoming=[m.clone() for m in offered]
            for m in incoming: m.origin="stage_promotion"
            if incoming:
                receiver.adf_registry.import_models(incoming)
                incoming_ids={id(m) for m in incoming}
                receiver.population=select_nsga([*receiver.population,*incoming],receiver.population_size,nsga_normalization,parsimony_quality_tolerance)
                admitted=[m for m in receiver.population if id(m) in incoming_ids]
                if admitted:
                    receiver.archive.update(admitted,X); receiver.best_models.update(admitted)
                promoted+=len(admitted)
                evicted+=len(leaving)-len(admitted)
            else: evicted+=len(leaving)
            source.population=staying
        bottom=ladder[0]
        if mode in ("age","both") and generation%config["age_gap"]==0:
            # ALPS: stage 0 restarts from scratch once its best had the chance to move up.
            bottom.population=fresh_stage_models(bottom.population_size,bottom,generation,n_features=n_features,ops=ops,nodes=nodes,depth=depth)
            reseeded+=1
        elif len(bottom.population)<bottom.population_size:
            # HFC: vacated bottom slots are the continuous supply of fresh models.
            bottom.population+=fresh_stage_models(bottom.population_size-len(bottom.population),bottom,generation,n_features=n_features,ops=ops,nodes=nodes,depth=depth)
    config["promotion_events"]+=1; config["promoted"]+=promoted; config["evicted"]+=evicted; config["reseeds"]+=reseeded
    return promoted,evicted,reseeded
def island_config_stages(island_config):
    """Stage configuration of a (possibly pre-stage) island configuration."""
    return stage_config(**island_config.get("stages",{}))
def advance_topology(cells, island_config, generation, *, X, n_features, ops, nodes, depth, nsga_normalization,
                     parsimony_quality_tolerance, evaluator=None, Y=None, cats=None, Xv=None, Yv=None,
                     crossover_rate=.35, bayesian_proposal_rate=.25):
    """After a generation: horizontal ring migration (per stage level),
    vertical stage promotion, and self-organising role updates."""
    stages=island_config["stages"]; stage_count=stages["count"]; island_count=island_config["count"]
    roles=island_config.get("roles") or role_config()
    if island_count>1 and island_config.get("migration_interval") and generation%island_config["migration_interval"]==0:
        migrated=0
        for stage in range(stage_count):
            level=[cells[index*stage_count+stage] for index in range(island_count)]
            migrated+=migrate_islands(level,island_config["migrants_per_island"],X=X,nsga_normalization=nsga_normalization,
                                      parsimony_quality_tolerance=parsimony_quality_tolerance,evaluator=evaluator)
            if roles["enabled"]:
                roles["fragment_migrants"]+=migrate_fragments(level,ROLE_FRAGMENT_MIGRANTS,generation)
        island_config["migration_events"]=int(island_config.get("migration_events",0))+1
        per_level=" per stage level" if stage_count>1 else ""
        print(f"Island migration {island_config['migration_events']}: moved {migrated} local Pareto elite(s) around the ring{per_level}.",flush=True)
    if stage_count>1 and generation%stages["interval"]==0:
        promoted,evicted,reseeded=promote_stages(cells,island_count,stages,generation,X=X,n_features=n_features,ops=ops,nodes=nodes,depth=depth,
                                                 nsga_normalization=nsga_normalization,parsimony_quality_tolerance=parsimony_quality_tolerance,evaluator=evaluator)
        note=f", stage 0 reseeded on {reseeded} island(s)" if reseeded else ""
        print(f"Stage promotion {stages['promotion_events']} ({stages['mode']}): {promoted} model(s) climbed, {evicted} left without admission{note}.",flush=True)
    if roles["enabled"] and island_count>1 and Y is not None and generation%roles["interval"]==0:
        retired,collapsed=update_roles(cells,island_config,generation,X=X,Y=Y,cats=cats,Xv=Xv,Yv=Yv,crossover_rate=crossover_rate,
                                       bayesian_proposal_rate=bayesian_proposal_rate,nodes=nodes)
        contributions=", ".join(f"{cell_label(cell,island_count,stage_count)}={cell.role.get('contribution',0.):.3g}" for cell in cells if cell.island_index>0)
        extra="".join([f"; retired {retired}" if retired else "",f"; split {collapsed} collapsed pair(s)" if collapsed else ""])
        print(f"Island roles {roles['updates']}: held-out contribution {contributions}{extra}.",flush=True)
# Self-organising island roles.  No role is named in advance: each specialist
# island's parent selection weights the training rows it already handles
# better than the other islands (soft responsibilities, as in a mixture of
# experts), so the positive feedback splits the data into niches for any
# island count.  Island 0 of each stage level stays a generalist.  Specialists
# also get a deterministic spread of search settings (tree size, crossover,
# Bayesian proposals), a held-out complementarity check retires specialists
# that stop contributing, and fragment libraries migrate with the ring.  All
# of it is selection-only: reported losses, archives, and the final pick use
# the true unweighted objective.
ROLE_FRAGMENT_MIGRANTS = 4
ROLE_COLLAPSE_CORRELATION = .95
ROLE_MIN_CONTRIBUTION = 1e-3
def role_config(enabled=False, interval=10, mix=.5, retire_after=5, **state):
    if int(interval)<1 or int(retire_after)<1: raise ValueError("Role interval and retirement window must be positive")
    if not 0<=float(mix)<=.9: raise ValueError("Role mix must be in [0, 0.9] so every island still sees all rows")
    return {"enabled":bool(enabled),"interval":int(interval),"mix":float(mix),"retire_after":int(retire_after),
            "updates":int(state.get("updates",0)),"retirements":int(state.get("retirements",0)),
            "collapses":int(state.get("collapses",0)),"fragment_migrants":int(state.get("fragment_migrants",0))}
def role_parameters(t, crossover_rate, bayesian_proposal_rate, nodes):
    """Search settings along one axis t in [0,1]: small, proposal-driven trees
    at 0; large, crossover-driven trees at 1."""
    t=float(np.clip(t,0.,1.))
    return {"t":t,"crossover_rate":float(np.clip(crossover_rate*(.5+t),0.,.9)),
            "bayesian_proposal_rate":float(np.clip(bayesian_proposal_rate*(1.5-t),0.,1.)),
            "nodes":max(3,int(round(nodes*(.5+.5*t))))}
def assign_role_parameters(cells, island_count, crossover_rate, bayesian_proposal_rate, nodes):
    """Spread specialist settings evenly over [0,1] by island index; island 0 keeps the defaults."""
    for cell in cells:
        if cell.island_index==0: cell.role={}; continue
        t=(cell.island_index-1)/max(1,island_count-2)
        cell.role={"params":role_parameters(t,crossover_rate,bayesian_proposal_rate,nodes),"stale":0}
def cell_search_settings(cell, crossover_rate, bayesian_proposal_rate, nodes):
    """(crossover_rate, bayesian_proposal_rate, nodes, case_weights) for one cell's generation."""
    params=cell.role.get("params") or {}
    weights=cell.role.get("case_weights")
    return (params.get("crossover_rate",crossover_rate),params.get("bayesian_proposal_rate",bayesian_proposal_rate),
            params.get("nodes",nodes),None if weights is None else np.asarray(weights,float))
def row_errors(model, X, Y, cats):
    """Scale-free per-row error (mean over outputs) used for responsibilities."""
    pred=predict_targets(model,X,cats); parts=[]
    for j,labels in enumerate(cats):
        if labels is None: parts.append(np.abs(pred[:,j]-Y[:,j])/max(target_scale(Y[:,j]),EPS))
        else: parts.append((np.rint(pred[:,j])!=Y[:,j]).astype(float))
    return np.nan_to_num(np.mean(parts,axis=0),nan=CLIP,posinf=CLIP)
def _cell_representative(cell):
    if cell.best_models.model is not None: return cell.best_models.model
    feasible=[m for m in cell.population if m.feasible]
    return min(feasible or cell.population,key=secondary_key)
def _random_affinity(n):
    return np.random.exponential(size=n)
def _normalized(weights):
    weights=np.maximum(np.asarray(weights,float),0.); mean=float(np.mean(weights))
    return np.ones_like(weights) if not np.isfinite(mean) or mean<=EPS else weights/mean
def update_roles(cells, island_config, generation, *, X, Y, cats, Xv=None, Yv=None, crossover_rate, bayesian_proposal_rate, nodes):
    """Recompute responsibilities per stage level; returns (retired, collapsed)."""
    roles=island_config["roles"]; island_count=island_config["count"]; stage_count=island_config["stages"]["count"]
    if not roles["enabled"] or island_count<2: return 0,0
    retired=collapsed=0
    Xc,Yc=(Xv,Yv) if Xv is not None and len(Xv) else (X,Y)
    for stage in range(stage_count):
        level=[cells[index*stage_count+stage] for index in range(island_count)]
        representatives=[_cell_representative(cell) for cell in level]
        errors=np.array([row_errors(m,X,Y,cats) for m in representatives])
        temperature=float(np.median(errors))+EPS
        logits=-errors/temperature; logits-=logits.max(axis=0)
        responsibility=np.exp(logits); responsibility/=responsibility.sum(axis=0)
        # Held-out complementarity: how much worse the best-of-islands error gets without this island.
        held=np.array([row_errors(m,Xc,Yc,cats) for m in representatives])
        best=held.min(axis=0); scale=float(np.mean(best))+EPS
        for index,cell in enumerate(level):
            if index==0: cell.role.pop("case_weights",None); continue
            target=responsibility[index]*island_count
            previous=cell.role.get("specialization")
            if previous is None or len(previous)!=len(target):
                specialization=_normalized(.5*_normalized(target)+.5*_normalized(_random_affinity(len(target))))
            else: specialization=_normalized(.5*np.asarray(previous,float)+.5*_normalized(target))
            others=np.delete(held,index,axis=0).min(axis=0)
            contribution=float(np.mean(others-best))/scale
            cell.role["contribution"]=contribution
            cell.role["stale"]=0 if contribution>ROLE_MIN_CONTRIBUTION else int(cell.role.get("stale",0))+1
            if cell.role["stale"]>=roles["retire_after"]:
                # A niche that no longer helps anywhere is abandoned for a fresh random one.
                specialization=_normalized(_random_affinity(len(target)))
                cell.role["params"]=role_parameters(rng.random(),crossover_rate,bayesian_proposal_rate,nodes)
                cell.role["stale"]=0; retired+=1
            cell.role["specialization"]=specialization
        specialists=level[1:]
        for i,left in enumerate(specialists):
            for right in specialists[i+1:]:
                a,b=left.role["specialization"],right.role["specialization"]
                if np.std(a)>EPS and np.std(b)>EPS and np.corrcoef(a,b)[0,1]>ROLE_COLLAPSE_CORRELATION:
                    right.role["specialization"]=_normalized(_random_affinity(len(b))); collapsed+=1
        for cell in specialists:
            cell.role["case_weights"]=_normalized(roles["mix"]*cell.role["specialization"]+(1-roles["mix"]))
    roles["updates"]+=1; roles["retirements"]+=retired; roles["collapses"]+=collapsed
    return retired,collapsed
def cell_label(cell, island_count, stage_count):
    parts=[]
    if island_count>1: parts.append(f"Island {cell.island_index+1}/{island_count}")
    if stage_count>1: parts.append(f"stage {cell.stage+1}/{stage_count}")
    return " ".join(parts)

def direct_feature_baselines(X, Y, cats, ops, affine_on):
    """One model per input feature whose every head reads that feature directly.

    Under affine scaling these are the one-feature linear (or linear-logit)
    baselines, so the search starts with them instead of hoping random trees
    rediscover them.  Models are scored when Y is given."""
    head_count=sum(len(heads) for heads in classification_layout(cats)[0])
    models=[Model([("x",index)]*head_count,[(1.,0.)]*head_count,mdl_operators=tuple(ops),mdl_feature_count=X.shape[1])
            for index in range(X.shape[1])]
    if Y is not None:
        for model in models: assess(model,X,Y,affine_on,cats)
    return models

def new_island_runtime(population_size, *, X, Xt, cats, ops, nodes, depth, head_count, bayesian_particles,
                       run_seed, island_index, qd_parent_rate, nsga_normalization, parsimony_quality_tolerance,
                       dynamic_pressure_on, stagnation_window, adf_enabled, adf_mode, evaluation_budget,
                       evaluation_refresh, interaction_discovery, cell=None, Yt=None):
    """Create one isolated AFPO/Bayesian search state for an island run.

    island_index seeds the cell's archives; cell=(island, stage) labels it
    (defaults to (island_index, 0) for stage-free runs).  Yt enables the
    residual-signature QD repertoire (when RESIDUAL_ARCHIVE is on)."""
    adf_registry=ADFRegistry(adf_enabled,allow_nested=adf_mode=="nested")
    seeds=direct_feature_baselines(X,None,cats,ops,True)[:population_size//4]
    for seed in seeds: seed.adfs=dict(adf_registry.definitions)
    population=seeds+[Model([random_tree(X.shape[1],ops,nodes,depth) for _ in range(head_count)],[(1.,0.)]*head_count,
                      mdl_operators=tuple(ops),mdl_feature_count=X.shape[1],adfs=dict(adf_registry.definitions))
                for _ in range(population_size-len(seeds))]
    probe_indices=stratified_probe_indices(Xt,256)
    qd_controller=QDOutcomeController(rate=float(np.clip(qd_parent_rate,.10,.30)))
    library=FragmentLibrary(); library.admit_discoveries(interaction_discovery)
    return IslandRuntime(
        population,PerOutputBayesianBanks(ops,X.shape[1],head_count,bayesian_particles,cats),
        ParetoArchive(normalization=nsga_normalization,parsimony_quality_tolerance=parsimony_quality_tolerance),
        MutationPortfolio(),CasePopulation(len(Xt)),
        QualityDiversityArchive(Xt[probe_indices],cats,run_seed ^ 0x5144 ^ island_index),
        StructuralQualityDiversityArchive(X.shape[1],run_seed ^ 0x5354 ^ island_index),qd_controller,
        BestModelArchive(parsimony_quality_tolerance),
        DynamicPressureController(dynamic_pressure_on,parsimony_quality_tolerance,qd_controller.uniform_rate,stagnation_window),
        library,adf_registry,EvaluationBudget(evaluation_budget,evaluation_refresh),
        island_index=(island_index if cell is None else cell[0]),stage=(0 if cell is None else cell[1]),
        residual_qd=(ResidualQualityDiversityArchive(Xt[probe_indices],np.asarray(Yt)[probe_indices],cats,run_seed ^ 0x5245 ^ island_index)
                     if RESIDUAL_ARCHIVE and Yt is not None else None),
    )

# Called once per island per generation with that island's live state (the
# browser GUI streams it); None keeps the terminal-only behaviour.
PROGRESS_HOOK = None
def evolution_progress(generation, elite, sample, *, started, Xt, Yt, Xv, Yv, names, out_names, cats, constraints, coev, cases, archive, semantic_qd, structural_qd, qd_controller, pressure, bayes, library=None, budget=None, evaluator=None, adf_registry=None, population=(), best_so_far=None, loss_tolerance=.01):
    """Emit the same bounded search telemetry for fresh and resumed runs."""
    if PROGRESS_HOOK is not None:
        PROGRESS_HOOK(generation=generation,elite=elite,population=population,archive=archive,best_so_far=best_so_far,started=started,
                      Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,pressure=pressure,
                      semantic_qd=semantic_qd,structural_qd=structural_qd,qd_controller=qd_controller,evaluator=evaluator,library=library,bayes=bayes)
    if generation%10==0:
        candidates=[m.clone() for m in [*population,*elite,*archive.items,best_so_far] if m is not None]
        if evaluator is not None: evaluator.assess(candidates,"train")
        else:
            for m in candidates: assess(m,Xt,Yt,False,cats,fit_affine=False,constraints=constraints,output_names=out_names)
        try:
            evaluation=selection_evaluation(candidates,Xv,Yv,cats,constraints,out_names)
        except ValueError as error:
            print(f"generation {generation:6d} | {error}",flush=True)
            return
        best,selection=_select_best(evaluation,loss_tolerance)
        scored=next(e[1] for e in evaluation[1] if e[0] is best)
        print(f"generation {generation:6d} | best {selection['source']} mean loss {aggregate_loss(scored):.6g} | losses {output_loss_summary(model_losses(scored),out_names)} | mean shape {np.mean(model_shapes(scored)):.6g} | MDL bits {model_complexity(scored):.6g} | {time.time()-started:.0f}s",flush=True)
        if Xv is not None:
            print(f"same model, full training | mean loss {aggregate_loss(best):.6g} | losses {output_loss_summary(model_losses(best),out_names)} | shape {np.mean(model_shapes(best)):.6g}",flush=True)
        if coev: print("case diagnostics | "+cases.diagnostics(Xt),flush=True)
        print(bayes.summary(),flush=True); print(semantic_qd.stats(),flush=True); print(structural_qd.stats(),flush=True); print(qd_controller.stats(),flush=True); print(pressure.stats(),flush=True)
        if library is not None: print(library.stats(),flush=True)
        if budget is not None: print(budget.stats(),flush=True)
        if evaluator is not None: print("Evaluator:",evaluator.diagnostics(),flush=True)
        if adf_registry is not None and adf_registry.enabled:
            info=adf_registry.diagnostics(); print(f"ADF v2 | promotions={info['promotions']}; definitions={info['definitions']}; active={','.join(info['active']) or 'none'}; retired={info['retired']}; elite_uses={info['elite_uses']}; invalid={info['invalid_calls']}",flush=True)
    if generation and generation%100==0:
        labels,choices,_=model_options(candidates,cats=cats,loss_tolerance=loss_tolerance,evaluation=evaluation)
        print_frontier(candidates,names,out_names,cats,recommendations=(labels,choices),evaluation=evaluation); print(archive.stats()); print(semantic_qd.stats()); print(structural_qd.stats()); print(qd_controller.stats()); print(bayes.summary())

class GracefulStop:
    """First Ctrl-C finishes the current generation, so the saved checkpoint is
    a whole generation (an interrupt inside one left archives, banks and RNG
    advanced but the population not) and worker processes are not torn down.
    A second Ctrl-C interrupts immediately."""
    def __init__(self): self.requested=False; self.previous=None
    def __enter__(self):
        # signal handlers can only be set from the main thread; elsewhere
        # Ctrl-C keeps its default immediate behaviour.
        try: self.previous=signal.getsignal(signal.SIGINT); signal.signal(signal.SIGINT,self._handle)
        except ValueError: self.previous=None
        return self
    def _handle(self, signum, frame):
        if self.requested: raise KeyboardInterrupt
        self.requested=True
        print("\nStopping after the current generation; press Ctrl-C again to interrupt now.",flush=True)
    def __exit__(self, *exc):
        if self.previous is not None: signal.signal(signal.SIGINT,self.previous)
        return False

def synchronize_model_ages(models, generation):
    """Age archived copies by elapsed generations, not by how often they are drawn."""
    for model in models:
        if model is None: continue
        if model.birth_generation is None: model.birth_generation=generation-model.age
        model.age=generation-model.birth_generation
        model.objectives=(*model.objectives[:-1],model.age)

def refresh_persistent_scores(archive, best_models, semantic_qd, structural_qd, evaluator, residual_qd=None):
    """Every persistent comparison uses a full-training fit, including after resume."""
    models=[*archive.items,*qd_cell_models(semantic_qd,structural_qd,residual_qd)]
    if best_models.model is not None: models.append(best_models.model)
    evaluator.assess(models,"train")
    for model in archive.items: model.objectives=(*model.objectives[:-1],0)
    for repertoire in qd_archives(semantic_qd,structural_qd,residual_qd):
        previous=list(repertoire.cells.values()); repertoire.cells={}
        for model in previous:
            if not model.feasible: continue
            cell=repertoire.cell(model); incumbent=repertoire.cells.get(cell)
            if incumbent is None or secondary_key(model)<secondary_key(incumbent): repertoire.cells[cell]=model

def evolve_generation(pop, generation, *, X, Xt, Yt, Xv, Yv, cats, constraints, out_names, ops, nodes, depth, affine_on, coev,
                      bayes, archive, semantic_qd, structural_qd, qd_controller, best_models, pressure, cases, portfolio, library, evaluator,
                      bayesian_proposal_rate, crossover_rate, qd_mode, lexicase_cases, nsga_normalization, progress=None, bayesian_mode="adaptive", adf_registry=None, budget=None, population_size=None, case_weights=None, residual_qd=None):
    """Advance one generation; fresh and resumed runs share this exact path.

    case_weights: optional selection-only per-training-row weights (island roles)."""
    population_size=len(pop) if population_size is None else population_size
    if population_size<1 or not pop: raise ValueError("Evolution requires a nonempty population")
    particle_models=[particle for bank in (bayes.banks if isinstance(bayes,PerOutputBayesianBanks) else [bayes]) for particle in [*bank.particles.catalog,*bank.particles.particles]]
    owners=[*pop,*archive.items,*qd_cell_models(semantic_qd,structural_qd,residual_qd),best_models.model,*particle_models]
    synchronize_model_ages(owners,generation)
    if adf_registry is not None:
        adf_registry.import_models(owners); adf_registry.attach(owners)
    evaluator.begin_generation()
    refresh_persistent_scores(archive,best_models,semantic_qd,structural_qd,evaluator,residual_qd)
    budget=EvaluationBudget() if budget is None else budget
    pressure.apply(bayes,qd_controller); effective_tolerance=pressure.effective_parsimony()
    repertoires=qd_archives(semantic_qd,structural_qd,residual_qd)
    qd_controller.begin_generation(repertoires)
    # Co-evolution only subsamples above 512 rows; below that it is inert.
    coev_active=coev and len(Xt)>512
    sample=cases.sample(budget.screen_count(len(Xt))) if coev_active else slice(None)
    # Index once: every Xt[sample] is a new array, and the evaluation caches
    # key on the array's buffer, so per-call indexing never hit (and each
    # entry pinned its own copy of the rows).
    Xs,Ys=row_subset(Xt,sample),row_subset(Yt,sample)
    budget.record("screen",len(Xt) if isinstance(sample,slice) else len(sample))
    evaluator.assess(pop,"train",None if isinstance(sample,slice) else sample,tune=generation==0)
    elite=select_nsga(pop,max(8,len(pop)//8),nsga_normalization,effective_tolerance)
    if coev_active: cases.update(elite,Xt,Yt,cats)
    if adf_registry is not None and adf_registry.enabled:
        adf_registry.observe(novelty_pool(elite,Xs),generation)
        # A promotion changes the grammar immediately.  Refresh every live
        # owner before Bayesian updating/rejuvenation can draw from it; those
        # paths otherwise pair a new ADF operator with a stale particle catalog.
        particle_models=[particle for bank in (bayes.banks if isinstance(bayes,PerOutputBayesianBanks) else [bayes]) for particle in [*bank.particles.catalog,*bank.particles.particles]]
        adf_registry.attach([*pop,*archive.items,*qd_cell_models(semantic_qd,structural_qd,residual_qd),best_models.model,*particle_models])
    stable_pop=[model.clone() for model in pop]; evaluator.assess(stable_pop,"train")
    stable_by_id={id(original):scored for original,scored in zip(pop,stable_pop)}
    stable_elite=[stable_by_id[id(model)] for model in elite]
    quality_improved=best_models.update(stable_pop)
    library.adapt_macro_rate(generation,quality_improved)
    archive.update(stable_elite,Xt)
    if adf_registry is not None and adf_registry.enabled:
        particle_models=[particle for bank in (bayes.banks if isinstance(bayes,PerOutputBayesianBanks) else [bayes]) for particle in [*bank.particles.catalog,*bank.particles.particles]]
        adf_registry.mark_usage([*pop,*archive.items,*qd_cell_models(semantic_qd,structural_qd,residual_qd),*particle_models],generation,elite)
        active_ops=adf_registry.operators(ops); bayes.sync_operators(active_ops)
        for particle in particle_models: particle.adfs.update(adf_registry.definitions)
    else: active_ops=list(ops)
    pressure.observe(generation,quality_improved,repertoires); pressure.apply(bayes,qd_controller)
    diverse=qd_cell_models(semantic_qd,structural_qd,residual_qd)
    Xb,Yb=behaviour_rows(Xt,Yt)
    if isinstance(bayes,PerOutputBayesianBanks): bayes.update(stable_elite,cats,Xb,Yb,diverse,affine_on=affine_on)
    else: bayes.update(stable_elite,Xb,Yb,cats,diverse)
    if generation%5==0: bayes.rejuvenate_particles(Xb,Yb,affine_on,cats,nodes,depth)
    if generation%10==0 and Xv is not None:
        bayes.record_predictive_check(Xv,Yv,cats,"validation")
        if adf_registry is not None and adf_registry.enabled and elite:
            diagnostic=min(elite,key=secondary_key)
            adf_registry.record_validation(diagnostic,frozen_metrics(diagnostic,Xv,Yv,cats,constraints,out_names)["loss"],generation)
    if progress is not None: progress(generation,elite,sample)
    library.observe(unique_models([*stable_elite,*archive.items,*diverse]),Xb,Yb,cats)
    Xsb=behaviour_rows(Xs)
    parent_pool=novelty_pool(pop,Xs)
    parent_count=max(1,population_size//2); qd_count=dual_qd_parent_count(parent_count,qd_controller,semantic_qd,structural_qd,residual_qd)
    row_weights=None if case_weights is None else (np.asarray(case_weights) if isinstance(sample,slice) else np.asarray(case_weights)[sample])
    ordinary=lexicase_parents(parent_pool,parent_count-qd_count,Xs,Ys,cats,lexicase_cases,row_weights,SCALE_BALANCED_SELECTION)
    parents=(blend_dual_qd_parents(ordinary,semantic_qd,structural_qd,parent_count,qd_count,qd_controller.uniform_rate,residual_qd)
             if qd_mode=="adaptive_dual" else blend_fixed_semantic_parents(ordinary,semantic_qd,parent_count,qd_count)); children=[]; feedback=[]; credits=[]; injections=[]; discovery_children=[]
    # Archive parents need the same screen fit as their children for feedback.
    evaluator.assess([parent.model for parent in parents if parent.source!="ordinary"],"train",None if isinstance(sample,slice) else sample)
    # A child identical to its parent is a wasted slot and a wasted tuning
    # pass; redraw it, with a bound so tiny grammars cannot loop forever.
    unchanged=0; unchanged_limit=4*population_size
    # Equivalent offspring (y+x beside x+y) would only be deduplicated after
    # paying for tuning and scoring; redraw them under the same bound instead.
    seen={model_equivalence_key(model) for model in pop}
    def duplicate(trees):
        nonlocal unchanged
        key=model_equivalence_key(trees)
        if EQUIVALENCE_COLLAPSE and key in seen and unchanged<unchanged_limit:
            unchanged+=1; EQUIVALENCE_STATS["children_redrawn"]+=1; return True
        seen.add(key); return False
    while len(children)<population_size:
        sources=[]
        if pressure.novelty_rate() and rng.random()<pressure.novelty_rate():
            trees=[random_tree(X.shape[1],active_ops,nodes,depth,adfs=adf_registry.definitions if adf_registry else None) for _ in range(len(pop[0].trees))]; scales=[(1.,0.)]*len(trees); child_age=0; child_origin="novelty_injection"
        elif bayesian_mode!="off" and rng.random() < bayesian_proposal_rate:
            trees,scales,sources=bayesian_injection_trees(bayes,len(pop[0].trees),X.shape[1],active_ops,nodes,depth,"grammar" if bayesian_mode=="grammar" else bayesian_mode,adf_registry.definitions if adf_registry else None,return_sources=True)
            child_age=max((model.age+1 for model in sources),default=0); child_origin="bayesian_injection"
        elif library.items and rng.random()<library.fragment_rate:
            p=rng.choice(parents); trees=[]; composed=False
            for tree in p.model.trees:
                candidate=library.compose(tree,active_ops,nodes,depth)
                trees.append(candidate if candidate is not None else tree); composed|=candidate is not None
            if not composed and unchanged<unchanged_limit: unchanged+=1; continue
            scales=list(p.model.scales); child_age=p.model.age+1; child_origin="fragment"
            sources=[p.model]
        elif rng.random() < crossover_rate and len(parents)>=2:
            p,q=rng.sample(parents,2); trees=[semantic_crossover(a,b,Xsb,nodes,depth,adf_registry.definitions if adf_registry else None) for a,b in zip(p.model.trees,q.model.trees)]; scales=list(p.model.scales); child_age=max(p.model.age,q.model.age)+1
            if trees==list(p.model.trees) and unchanged<unchanged_limit: unchanged+=1; continue
            if duplicate(trees): continue
            child=Model(trees,scales,child_age,origin="crossover",parent_ids=(p.model.lineage_id,q.model.lineage_id),mdl_operators=grammar_for_trees(active_ops,trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),founder_ids=tuple(sorted(set(p.model.founder_ids).union(q.model.founder_ids))),birth_generation=generation+1-child_age)
            children.append(child); credits.append((child,(p,q))); continue
        else:
            p=rng.choice(parents)
            if bayesian_mode!="off": bayes.begin_equation()
            trees=[]; kinds=[]; macro_used=False
            for index,tree in enumerate(p.model.trees):
                proposal=None if bayesian_mode=="off" else (bayes[index] if isinstance(bayes,PerOutputBayesianBanks) else bayes)
                child_tree,kind,macro=semantic_mutate(tree,Xsb,portfolio,X.shape[1],active_ops,nodes,depth,proposal,adf_registry.definitions if adf_registry else None,library)
                trees.append(child_tree); kinds.append(kind); macro_used|=macro
            if trees==list(p.model.trees) and unchanged<unchanged_limit: unchanged+=1; continue
            scales=list(p.model.scales); child_age=p.model.age+1; child_origin="macro_mutation" if macro_used else "mutation"
            sources=[p.model]
        if duplicate(trees): continue
        child=Model(trees,scales,child_age,origin=child_origin,parent_ids=tuple(model.lineage_id for model in sources),mdl_operators=grammar_for_trees(active_ops,trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),founder_ids=tuple(sorted({founder for model in sources for founder in model.founder_ids})),birth_generation=generation+1-child_age)
        children.append(child)
        if child.origin=="bayesian_injection": injections.append(child)
        if child.origin in {"fragment","macro_mutation"}: discovery_children.append(child)
        if child.origin=="mutation": feedback.append((child,p.model,kinds)); credits.append((child,(p,)))
    # A correct structure with untuned constants otherwise scores like a wrong
    # one and is lost; fit every offspring's inner constants before scoring.
    evaluator.assess(children,"train",None if isinstance(sample,slice) else sample,tune=True)
    for child,parent_choices in credits: qd_controller.record(child,parent_choices,repertoires)
    for child,parent,kinds in feedback:
        for kind in kinds:
            if kind is not None: portfolio.record(kind,variation_improved(child,parent))
    stable_children=[model.clone() for model in children]; evaluator.assess(stable_children,"train")
    if best_models.update(stable_children):
        # Offspring are where new bests appear; the earlier signal only
        # rescored already-seen survivors, so the macro lane never saw one.
        pressure.observe(generation,True,repertoires); library.adapt_macro_rate(generation,True)
    if coev_active:
        promoted=[model.clone() for model in select_nsga(children,max(8,len(pop)//4),nsga_normalization,effective_tolerance)]
        anchor=budget.anchor_indices(Xt); evaluator.assess(promoted,"train",anchor); budget.record("anchor",len(anchor))
        # Anchor scores are screening evidence only; persistence always uses full training.
        evaluator.assess(promoted,"train"); best_models.update(promoted); archive.update(promoted,Xt)
    qd_candidates,qd_threshold=qd_eligible_candidates([*stable_pop,*stable_children]); semantic_qd.update(qd_candidates,qd_threshold)
    if qd_mode=="adaptive_dual":
        structural_qd.update(qd_candidates,qd_threshold)
        if residual_qd is not None: residual_qd.update(qd_candidates,qd_threshold)
    if qd_mode!="fixed_semantic": qd_controller.update_rate()
    for model in pop:
        model.age += 1; model.objectives=(*model.objectives[:-1],model.age)
    survivor_pool=novelty_pool(pop+children,Xs); attempts=0
    # Top up missing behaviours in batches: one tuned, parallel, cached
    # evaluator call and one deduplication pass per batch, not per model.
    while len(survivor_pool)<population_size and attempts<population_size*8:
        batch=[]
        for _ in range(min(population_size-len(survivor_pool),population_size*8-attempts)):
            fresh_trees=[random_tree(X.shape[1],active_ops,nodes,depth,proposal=None if bayesian_mode=="off" else (bayes[index] if isinstance(bayes,PerOutputBayesianBanks) else bayes),adfs=adf_registry.definitions if adf_registry else None) for index in range(len(pop[0].trees))]
            batch.append(Model(fresh_trees,[(1.,0.)]*len(pop[0].trees),0,origin="novelty_injection",mdl_operators=grammar_for_trees(active_ops,fresh_trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),birth_generation=generation+1))
        evaluator.assess(batch,"train",None if isinstance(sample,slice) else sample,tune=True)
        survivor_pool=novelty_pool([*survivor_pool,*batch],Xs); attempts+=len(batch)
    survivors=select_nsga(survivor_pool,min(population_size,len(survivor_pool)),nsga_normalization,effective_tolerance)
    # Some datasets admit fewer distinct behaviors than population slots.
    # Keep population capacity even when behavioral deduplication is exhausted.
    distinct=list(survivors)
    while len(survivors)<population_size: survivors.append(distinct[(len(survivors)-len(distinct))%len(distinct)].clone())
    if budget.refresh_due(generation):
        refresh=[model.clone() for model in survivors]
        evaluator.assess(refresh,"train"); budget.record("full",len(Xt)); budget.last_full_refresh=generation
        best_models.update(refresh); archive.update(refresh,Xt)
    if injections:
        survivor_ids={id(model) for model in survivors}
        for child in injections:
            if bayesian_mode=="adaptive":
                if isinstance(bayes,PerOutputBayesianBanks): bayes.record_injection(id(child) in survivor_ids)
                else: bayes.particles.record_injection(id(child) in survivor_ids)
    if discovery_children:
        survivor_ids={id(model) for model in survivors}
        for child in discovery_children: library.record(child.origin,id(child) in survivor_ids)
    return survivors

def resume_main(args):
    generation,pop,bayes,archive,state=load_checkpoint(args.resume,args.allow_unsafe_pickle)
    # Island snapshots, top-up proposals and final ADF refresh all need
    # per-output banks; a single shared generator failed at the first save.
    if not isinstance(bayes,PerOutputBayesianBanks): raise ValueError("Checkpoint predates per-output Bayesian banks and cannot resume; start a new run")
    X,Y,Xt,Yt,Xv,Yv=(state[k] for k in ("X","Y","Xt","Yt","Xv","Yv"))
    names,out_names,cats,maps=(state[k] for k in ("names","out_names","cats","maps"))
    global SEQUENCE_LAYOUT,EQUIVALENCE_COLLAPSE,RESIDUAL_ARCHIVE,QD_PARENT_CHOICE,SCALE_BALANCED_SELECTION
    SEQUENCE_LAYOUT=maps.get(SEQUENCE_LAYOUT_KEY)
    # Settings that postdate a checkpoint resume with the behaviour it was searched with.
    RESIDUAL_ARCHIVE=bool(state.get("residual_archive",False)); QD_PARENT_CHOICE=state.get("qd_parent_choice","legacy")
    SCALE_BALANCED_SELECTION=bool(state.get("scale_balanced_selection",False))
    # Checkpoints from before equivalence keys searched without them; keep that.
    EQUIVALENCE_COLLAPSE=bool(state.get("equivalence_collapse",False)); EQUIVALENCE_STATS["children_redrawn"]=0
    ops,nodes,depth,affine_on,coev=(state[k] for k in ("operators","nodes","depth","affine_on","coev"))
    constraints=compile_constraints(state.get("profile","general"),state.get("constraint_metadata",{}))
    if getattr(bayes,"legacy_head_banks",False) and any(labels is not None and len(labels)>2 for labels in cats):
        raise ValueError("Checkpoint has legacy independent multiclass Bayesian heads; start a new calibrated run")
    checkpoint_path=Path(args.resume); rate=state.get("bayesian_proposal_rate",args.bayesian_proposal_rate); crossover_rate=state.get("crossover_rate",args.crossover_rate)
    loss_tolerance=state.get("selection_loss_tolerance",args.selection_loss_tolerance)
    nsga_normalization=state.get("nsga_normalization","legacy")
    parsimony_quality_tolerance=state.get("parsimony_quality_tolerance",0.)
    island_config=dict(state.get("island_config",{}))
    if state.get("island_states"):
        islands=[island_from_snapshot(item,len(Xt),parsimony_quality_tolerance) for item in state["island_states"]]
        island_config.setdefault("count",len(islands))
    else:
        if "quality_diversity" not in state: raise ValueError("Checkpoint lacks Quality-Diversity state and cannot resume")
        semantic_qd,structural_qd,qd_controller=qd_from_snapshot(state["quality_diversity"])
        if "mutation_portfolio" not in state: raise ValueError("Checkpoint lacks mutation-portfolio state and cannot resume exactly")
        portfolio=MutationPortfolio(); portfolio.restore(state["mutation_portfolio"])
        library=FragmentLibrary.from_snapshot(state.get("fragment_library",{}))
        if coev and "case_population" not in state: raise ValueError("Checkpoint lacks co-evolution case-population state and cannot resume exactly")
        cases=CasePopulation.from_snapshot(state["case_population"],len(Xt)) if coev else CasePopulation(len(Xt))
        best_models=BestModelArchive.from_snapshot(state.get("best_model",{}),parsimony_quality_tolerance)
        if best_models.model is None: best_models.update([*archive.items,*pop])
        pressure=DynamicPressureController.from_snapshot(state.get("dynamic_pressure",{"enabled":state.get("dynamic_pressure_enabled",False),"base_parsimony":parsimony_quality_tolerance,"base_uniform":qd_controller.uniform_rate}))
        islands=[IslandRuntime(pop,bayes,archive,portfolio,cases,semantic_qd,structural_qd,qd_controller,best_models,pressure,library,ADFRegistry.from_snapshot(state.get("adf_registry",{})),EvaluationBudget.from_snapshot(state.get("evaluation_budget",{"mode":args.evaluation_budget,"refresh":args.evaluation_refresh})))]
        island_config={"count":1,"migration_interval":0,"migrants_per_island":0,"topology":"ring","migration_events":0}
    island_config["stages"]=island_config_stages(island_config); stage_count=island_config["stages"]["count"]
    island_config["roles"]=role_config(**island_config.get("roles",{}))
    if island_config.get("count",0)*stage_count!=len(islands): raise ValueError("Checkpoint island configuration does not match its saved island states")
    for index,cell in enumerate(islands):
        if "island_states" in state and not any("stage" in item for item in state["island_states"]): cell.island_index,cell.stage=divmod(index,stage_count)
    qd_mode=state.get("qd_mode","adaptive_dual")
    workers=resolve_worker_count(args.workers if args.workers else state.get("evaluation_workers",0),sum(len(island.population) for island in islands))
    evaluator=ModelEvaluator(workers,{"train":(Xt,Yt)},affine_on,cats,constraints,out_names)
    topology="single population" if island_config["count"]==1 else f"{island_config['count']} islands; ring migration every {island_config.get('migration_interval',0)} generations"
    if stage_count>1: topology+=f"; {stage_count} {island_config['stages']['mode']} stages per island"
    print(f"Resumed generation {generation} from {checkpoint_path} (seed {state['run_seed']}); {topology}; model scoring uses {'serial evaluation' if workers==1 else f'{workers} worker processes'}.")
    started=time.time(); stop=GracefulStop().__enter__(); interrupted=False
    try:
        while (not args.max_generations or generation < args.max_generations) and not stop.requested:
            for island_index,island in enumerate(islands):
                def progress(gen,elite,sample,island=island,island_index=island_index):
                    if len(islands)>1 and gen%10==0: print(cell_label(island,island_config["count"],stage_count),flush=True)
                    evolution_progress(gen,elite,sample,started=started,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,coev=coev,cases=island.cases,archive=island.archive,semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,qd_controller=island.qd_controller,pressure=island.pressure,bayes=island.bayes,library=island.library,budget=island.budget,evaluator=evaluator,adf_registry=island.adf_registry,population=island.population,best_so_far=island.best_models.model,loss_tolerance=loss_tolerance)
                cell_crossover,cell_proposals,cell_nodes,cell_weights=cell_search_settings(island,crossover_rate,rate,nodes)
                island.population=evolve_generation(island.population,generation,X=X,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,cats=cats,constraints=constraints,out_names=out_names,
                                  ops=ops,nodes=cell_nodes,depth=depth,case_weights=cell_weights,affine_on=affine_on,coev=coev,bayes=island.bayes,archive=island.archive,
                                  semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,residual_qd=island.residual_qd,qd_controller=island.qd_controller,best_models=island.best_models,
                                  pressure=island.pressure,cases=island.cases,portfolio=island.portfolio,library=island.library,evaluator=evaluator,bayesian_proposal_rate=cell_proposals,
                                  crossover_rate=cell_crossover,qd_mode=qd_mode,lexicase_cases=state.get("lexicase_cases",args.lexicase_cases),nsga_normalization=nsga_normalization,bayesian_mode=state.get("bayesian_mode",args.bayesian_mode),adf_registry=island.adf_registry,budget=island.budget,progress=progress,population_size=island.population_size)
            generation+=1
            advance_topology(islands,island_config,generation,X=Xt,n_features=X.shape[1],ops=ops,nodes=nodes,depth=depth,
                             nsga_normalization=nsga_normalization,parsimony_quality_tolerance=parsimony_quality_tolerance,evaluator=evaluator,
                             Y=Yt,cats=cats,Xv=Xv,Yv=Yv,crossover_rate=crossover_rate,bayesian_proposal_rate=rate)
            if args.checkpoint_every and generation%args.checkpoint_every==0:
                snapshot_islands(state,islands,island_config); save_checkpoint(checkpoint_path,generation,islands[0].population,islands[0].bayes,islands[0].archive,state)
    except KeyboardInterrupt: interrupted=True
    finally: stop.__exit__(None,None,None)
    if stop.requested: print("\nResume interrupted"+(" mid-generation; that generation will not replay exactly from this checkpoint." if interrupted else " after a whole generation."))
    evaluator.reopen()
    snapshot_islands(state,islands,island_config); state["evaluation_workers"]=workers; save_checkpoint(checkpoint_path,generation,islands[0].population,islands[0].bayes,islands[0].archive,state)
    for island in islands:
        if island.adf_registry.enabled:
            particle_models=[particle for bank in island.bayes.banks for particle in [*bank.particles.catalog,*bank.particles.particles]]
            island.adf_registry.attach([*island.population,*island.archive.items,*qd_cell_models(island.semantic_qd,island.structural_qd,island.residual_qd),island.best_models.model,*particle_models])
        refresh_persistent_scores(island.archive,island.best_models,island.semantic_qd,island.structural_qd,evaluator,island.residual_qd)
        evaluator.assess(island.population,"train"); island.best_models.update(island.population); island.archive.update(island.population,Xt)
    f=[model for island in islands for model in [*island.archive.items,*island.population,island.best_models.model] if model is not None]
    chosen,selection=select_best_model(f,Xv,Yv,cats,loss_tolerance,constraints,out_names) if Xv is not None else select_best_model(f,loss_tolerance=loss_tolerance)
    state["selection"]={**selection,"selected_choice":"default","default_selected":True,
                        "selected_metrics":selection["metrics"],"selected_objectives":selection["objectives"]}
    snapshot_islands(state,islands,island_config)
    save_checkpoint(checkpoint_path,generation,islands[0].population,islands[0].bayes,islands[0].archive,state)
    print(f"Resume complete at generation {generation}. {selection['source'].title()} loss-tolerance shortest-MDL model selected: {equations(chosen,names,out_names,cats)}")
    print(f"{selection['source'].title()} selection scores: mean loss={selection['metrics']['loss']:.6g}, mean shape={selection['metrics']['shape']:.6g}, MDL bits={selection['metrics']['mdl_bits']:.6g}")
    export_model(chosen,names,out_names,cats,maps,state["source_columns"],state["types"],state.get("export_fixture"),state.get("input_ranges")); evaluator.close()

def build_arg_parser():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-generations",type=int,default=0); ap.add_argument("--population",type=int,default=160); ap.add_argument("--seed",type=int); ap.add_argument("--workers",type=int,default=0,help="Model-scoring processes; 0=auto, 1=serial (default: 0)"); ap.add_argument("--adf-mode",choices=("off","flat","nested"),default="nested",help="ADF experiment mode; nested is v2, flat is the v1-style ablation, off disables ADFs")
    ap.add_argument("--bayesian-proposal-rate",type=float,default=.25,help="Fraction of offspring drawn from the Bayesian equation generator (0..1)")
    ap.add_argument("--bayesian-mode",choices=("off","grammar","fixed","adaptive"),default="adaptive",help="Bayesian injection policy: off, grammar-only, fixed particle mix, or adaptive particle mix")
    ap.add_argument("--crossover-rate",type=float,default=.35,help="Fraction of non-Bayesian offspring made by subtree crossover (0..1)")
    ap.add_argument("--lexicase-cases",type=int,default=0,help="Informed max training cases for lexicase; 0 uses all")
    ap.add_argument("--checkpoint-every",type=int,default=100,help="Save full evolutionary state every N generations; 0 disables periodic saves")
    ap.add_argument("--resume",help="Resume from a safe AFPO checkpoint without interactive setup")
    ap.add_argument("--allow-unsafe-pickle",action="store_true",help="Allow a trusted legacy pickle checkpoint; pickle files can execute code when loaded")
    ap.add_argument("--migrate-checkpoint",nargs=2,metavar=("SOURCE","DESTINATION"),help="Convert a trusted legacy checkpoint to the safe JSON format (requires --allow-unsafe-pickle)")
    ap.add_argument("--test-csv",help="Final held-out CSV; reported only, never used for selection")
    ap.add_argument("--selection-loss-tolerance",type=float,default=.01,help="Relative loss tolerance for selecting the shortest-MDL validation-equivalent model (default: 0.01)")
    ap.add_argument("--nsga-normalization",choices=NSGA_NORMALIZATIONS,default="intercept",help="Fixed NSGA-III normalization; legacy is retained only for paired ablations")
    ap.add_argument("--parsimony-quality-tolerance",type=float,default=.01,help="Per-objective relative near-tie band for lower-MDL survivor preference; 0 disables it (default: 0.01)")
    ap.add_argument("--profile",choices=PROFILES,default="general",help="Explicit optional prior profile; AFPO never auto-detects one")
    ap.add_argument("--constraint-metadata",help="JSON file containing explicit profile constraints keyed by output name/index")
    ap.add_argument("--bayesian-particles",type=int,default=96,help="Adaptive catalogue capacity per output")
    ap.add_argument("--qd-parent-rate",type=float,default=.20,help="Initial adaptive QD parent fraction; controller stays within 0.10..0.30 (default: 0.20)")
    ap.add_argument("--qd-mode",choices=("adaptive_dual","adaptive_semantic","fixed_semantic"),default="adaptive_dual",help="Adaptive dual (default), adaptive semantic-only, or fixed semantic 20%% QD")
    ap.add_argument("--evaluation-budget",choices=("baseline","adaptive"),default="baseline",help="Large-data co-evolution budget; adaptive uses a smaller screen sample plus anchor/full refreshes")
    ap.add_argument("--evaluation-refresh",type=int,default=25,help="Adaptive evaluation full-refresh interval in generations")
    ap.add_argument("--sequence-group",action="append",default=[],metavar="NAME=COL1,COL2,...",help="Ordered numeric input columns forming one sequence (repeatable, equal lengths); enables seqsum/seqprod with per-position features NAME[i], prod(NAME[<i]), sum(NAME[<i]) and i")
    ap.add_argument("--stagnation-window",type=int,default=100,help="Generations without quality/QD activity before bounded pressure escalates")
    ap.add_argument("--equivalence-collapse",choices=("on","off"),default="on",help="Treat algebraically equal equations (x+y vs y+x, x+x vs 2*x, x*x vs square(x)) as one candidate in offspring, deduplication and archives (default: on)")
    ap.add_argument("--residual-archive",choices=("on","off"),default="on",help="Keep a third QD archive keyed by where each model errs (target-size bins and input regions), so complementary partial models survive (default: on)")
    ap.add_argument("--qd-parent-choice",choices=QD_PARENT_CHOICES,default="quality_coverage",help="How QD archives pick parent cells beyond the uniform share: success x bounded quality rank x coverage bonus, or legacy success-only (default: quality_coverage)")
    ap.add_argument("--scale-balanced-selection",choices=("on","off"),default="off",help="Selection-only: give every target-magnitude band equal weight and compare asinh-compressed errors in lexicase parent choice; reported loss is unchanged (default: off)")
    ap.add_argument("--gui",action="store_true",help="Start the browser GUI (training, live Pareto frontier, model explorer) instead of the terminal prompts")
    ap.add_argument("--port",type=int,default=8778,help="Browser GUI port (default: 8778)")
    return ap

def parse_cli(argv=None):
    """Parse and validate options; the GUI's training process goes through this same path."""
    ap=build_arg_parser(); args=ap.parse_args(argv)
    if not 0 <= args.bayesian_proposal_rate <= 1 or not 0 <= args.crossover_rate <= 1: ap.error("proposal and crossover rates must be between 0 and 1")
    if args.bayesian_particles < 1: ap.error("--bayesian-particles must be positive")
    if args.selection_loss_tolerance < 0: ap.error("--selection-loss-tolerance must be non-negative")
    if args.parsimony_quality_tolerance < 0: ap.error("--parsimony-quality-tolerance must be non-negative")
    if not .10 <= args.qd_parent_rate <= .30: ap.error("--qd-parent-rate must be between 0.10 and 0.30")
    if args.evaluation_refresh < 1 or args.stagnation_window < 1: ap.error("evaluation refresh and stagnation window must be positive")
    return ap,args

def main():
    ap,args=parse_cli()
    if args.gui:
        from afpo_gui import run_gui
        run_gui(port=args.port); return
    if args.migrate_checkpoint:
        if not args.allow_unsafe_pickle: ap.error("--migrate-checkpoint requires --allow-unsafe-pickle for the trusted source")
        source,destination=args.migrate_checkpoint
        generation,pop,bayes,archive,state=load_checkpoint(source,True)
        save_checkpoint(destination,generation,pop,bayes,archive,state)
        print(f"Migrated trusted checkpoint to {destination}")
        return
    if args.resume: resume_main(args); return
    mode=ask("Mode: 0=train, 1=continue, 2=cross validation, 3=browser GUI","0")
    if mode not in {"0","1","2","3"}: raise ValueError("Mode must be 0, 1, 2, or 3")
    if mode=="3":
        from afpo_gui import run_gui
        run_gui(port=args.port); return
    if mode=="1":
        checkpoint=Path(ask("Checkpoint path","checkpoint_latest.json"))
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        args.resume=str(checkpoint); resume_main(args); return
    if mode=="2":
        script=Path(__file__).with_name("cross_validate.py")
        if not script.is_file(): raise FileNotFoundError(f"Cross validation needs {script}, which is not installed next to afpo.py")
        dataset=ask("Cross-validation dataset path")
        delimiter=ask("Delimiter [0=comma, 1=semicolon, 2=space, 3=tab]","0")
        column_types=ask("Column types (comma-separated, e.g. 1,5)")
        folds=ask("Folds","5"); generations=ask("Generations","50"); population=ask("Population","48"); seed=ask("Seed","1")
        delimiter_value={"0":",","1":";","2":" ","3":"\\t"}.get(delimiter,delimiter)
        command=[sys.executable,str(script),dataset,"--folds",folds,"--column-types",column_types,"--generations",generations,"--population",population,"--seed",seed,"--delimiter",delimiter_value]
        completed=subprocess.run(command)
        if completed.returncode: raise SystemExit(completed.returncode)
        return
    train_from_setup(args,collect_training_setup(args))

def collect_training_setup(args):
    """Ask the interactive training questions.  The GUI builds the same dict from its form."""
    path=Path(ask("Dataset path"))
    if not path.is_file(): raise FileNotFoundError(path)
    df,types,delimiter=configure(path)
    ops=choose_operator_groups()
    print("Structural objective = MDL model-description bits (uniform enabled grammar; exact constants and affine coefficients included).")
    affine_on=yes(ask("Affine scaling? 1=yes, 0=no","1")); coev=yes(ask("Use co-evolution/minibatches? 1=yes, 0=no","0"))
    dynamic_pressure_on=yes(ask("Dynamic evolution pressure? 1=yes, 0=no","1"))
    adf_enabled=yes(ask("Enable automatically defined function (ADF) mining? 1=yes, 0=no","0")) and args.adf_mode!="off"
    nodes=max(3,int(ask("Maximum nodes per output","31"))); depth=max(1,int(ask("Maximum depth","6")))
    island_count=max(1,int(ask("Island count (1=single population)","1")))
    if island_count>args.population//8: raise ValueError("Island count needs at least eight models per island; raise --population or choose fewer islands")
    migration_interval=0; migrants_per_island=0
    if island_count>1:
        migration_interval=max(1,int(ask("Migration interval (generations)","25")))
        migrants_per_island=max(1,int(ask("Migrants per island","2")))
    roles=role_config()
    if island_count>1 and yes(ask("Self-organising island roles (island 1 generalist, the rest specialise)? 1=yes, 0=no","0")):
        roles=role_config(True,interval=int(ask("Role update interval (generations)","10")))
    stages=ask_stage_setup()
    if island_count*stages["count"]>args.population//8: raise ValueError("Islands x stages need at least eight models each; raise --population or choose fewer islands/stages")
    val_path=ask("Validation CSV (blank=random split; 0=disabled)","")
    if args.constraint_metadata:
        metadata=json.loads(Path(args.constraint_metadata).read_text())
    elif args.profile != "general":
        prompts={
            "physics":"Physics metadata JSON (SI dimensions and declared symmetries)",
            "chemistry":"Chemistry metadata JSON (dimensions, non-negativity, fraction bounds, conservation groups)",
            "biology":"Biology metadata JSON (non-negativity, output bounds, domains, monotonicity)",
            "mathematics":"Mathematics metadata JSON (domains, nonzero/positive/integer, symmetry, rational constants)",
        }
        metadata=json.loads(ask(prompts[args.profile]+"; blank={} ","{}"))
    else: metadata={}
    validation_percent=None if val_path else float(ask("Validation percentage (0 disables validation; minimum 5 rows when possible)","20"))
    return {"path":path,"df":df,"types":types,"delimiter":delimiter,"ops":ops,"affine_on":affine_on,"coev":coev,
            "dynamic_pressure_on":dynamic_pressure_on,"adf_enabled":adf_enabled,"nodes":nodes,"depth":depth,
            "island_count":island_count,"migration_interval":migration_interval,"migrants_per_island":migrants_per_island,"stages":stages,"roles":roles,
            "val_path":val_path,"validation_percent":validation_percent,"metadata":metadata}

def ask_stage_setup():
    """Interactive AFPO stage questions (see promote_stages)."""
    answer=ask("AFPO stages: 0=off, 1=fitness (HFC), 2=age (ALPS), 3=both","0")
    modes={"0":"off","1":"fitness","2":"age","3":"both"}
    if answer not in modes: raise ValueError("Stage mode must be 0, 1, 2, or 3")
    mode=modes[answer]
    if mode=="off": return stage_config()
    options={"mode":mode,"count":int(ask("Stages per island","3")),"interval":int(ask("Promotion interval (generations)","5"))}
    if mode in ("fitness","both"):
        options["threshold_quantile"]=float(ask("Admission bar: loss quantile of the stage above (0-1]","0.5"))
    if mode in ("age","both"):
        options["age_gap"]=int(ask("Age gap (generations)","10"))
        schedule=ask("Age limits: 0=linear, 1=polynomial, 2=exponential","1")
        if schedule not in ("0","1","2"): raise ValueError("Age schedule must be 0, 1, or 2")
        options["schedule"]=STAGE_AGE_SCHEDULES[int(schedule)]
    return stage_config(**options)

def choose_model_interactively(labels, choices, evaluation):
    while True:
        answer=ask("Model to save (1=Best Score)","1")
        try: return max(0,min(len(choices)-1,int(answer)-1))
        except ValueError: print(f"Enter a number from 1 to {len(choices)}.")

def train_from_setup(args, setup, choose_model=None):
    """Run a fresh search from collected setup answers; returns where its outputs went.

    ``choose_model(labels, choices, evaluation)`` returns the index of the model
    to save; the default asks at the terminal."""
    global EQUIVALENCE_COLLAPSE,RESIDUAL_ARCHIVE,QD_PARENT_CHOICE,SCALE_BALANCED_SELECTION
    EQUIVALENCE_COLLAPSE=getattr(args,"equivalence_collapse","on")=="on"; EQUIVALENCE_STATS["children_redrawn"]=0
    RESIDUAL_ARCHIVE=getattr(args,"residual_archive","on")=="on"; QD_PARENT_CHOICE=getattr(args,"qd_parent_choice","quality_coverage")
    SCALE_BALANCED_SELECTION=getattr(args,"scale_balanced_selection","off")=="on"
    run_seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng.seed(run_seed); np.random.seed(run_seed)
    print(f"Run seed: {run_seed}")
    path,df,types,delimiter,ops=(setup[key] for key in ("path","df","types","delimiter","ops"))
    affine_on,coev,dynamic_pressure_on,adf_enabled,nodes,depth=(setup[key] for key in ("affine_on","coev","dynamic_pressure_on","adf_enabled","nodes","depth"))
    island_count,migration_interval,migrants_per_island,val_path=(setup[key] for key in ("island_count","migration_interval","migrants_per_island","val_path"))
    adf_enabled=adf_enabled and args.adf_mode!="off"
    stages=stage_config(**(setup.get("stages") or {}))
    roles=role_config(**(setup.get("roles") or {}))
    if roles["enabled"] and island_count<2: raise ValueError("Island roles need at least two islands (one generalist plus specialists)")
    cell_count=island_count*stages["count"]
    if cell_count>args.population//8: raise ValueError("Islands x stages need at least eight models each; raise --population or choose fewer islands/stages")
    perceptron_enabled=any(operator.startswith("perceptron") for operator in ops)
    metadata=setup.get("metadata")
    if metadata is None: metadata=json.loads(Path(args.constraint_metadata).read_text()) if args.constraint_metadata else {}
    external_validation=None
    if val_path=="0":
        train_indices=np.arange(len(df)); validation_indices=np.array([],dtype=int); train_df=df; validation_df=None
    elif val_path:
        validation_df=pd.read_csv(val_path,sep=delimiter,engine="python"); train_df=df; train_indices=np.arange(len(df)); validation_indices=np.arange(len(validation_df))
        external_validation={"path":str(Path(val_path).resolve()),"sha256":dataset_sha256(val_path),"rows":len(validation_df)}
    else:
        pct=setup["validation_percent"]
        if pct is None or pct<=0:
            train_indices=np.arange(len(df)); validation_indices=np.array([],dtype=int); train_df=df; validation_df=None
        else:
            requested=max(1,math.ceil(len(df)*pct/100))
            val_rows=max(5,requested)
            if len(df)-val_rows < 4:
                print("Validation disabled: this dataset cannot retain both 5 holdout rows and 4 training rows.")
                train_indices=np.arange(len(df)); validation_indices=np.array([],dtype=int); train_df=df; validation_df=None
            else:
                if val_rows != requested:
                    print(f"Using {val_rows} validation rows (minimum for reliable affine-scaled scoring).")
                train_indices,validation_indices=holdout_split_indices(len(df),val_rows,run_seed)
                train_df=df.iloc[train_indices]; validation_df=df.iloc[validation_indices]
    global SEQUENCE_GROUP_REQUEST
    SEQUENCE_GROUP_REQUEST=tuple(parse_sequence_group(item) for item in args.sequence_group)
    # Fit every fill value and category vocabulary on training rows only.
    Xt,Yt,names,out_names,cats,maps=encode(train_df,types)
    if SEQUENCE_LAYOUT is not None:
        ops=list(dict.fromkeys([*ops,"seqsum","seqprod"]))
        print(f"Sequence groups: {', '.join(group['name'] for group in SEQUENCE_LAYOUT['groups'])} (length {SEQUENCE_LAYOUT['length']}); added seqsum/seqprod.")
    else: ops=[op for op in ops if op not in ("seqsum","seqprod")]
    X,Y=Xt,Yt
    constraints=compile_constraints(args.profile,metadata); constraints.validate(Xt.shape[1],cats,out_names)
    Xv=Yv=None
    if validation_df is not None:
        Xv,Yv,names2,out2,cats2,_=encode(validation_df,types,maps)
        if names2!=names or out2!=out_names or cats2!=cats: raise ValueError("Validation CSV columns/types do not match training data")
    Xtest=Ytest=None
    if args.test_csv:
        test_df=pd.read_csv(args.test_csv,sep=delimiter,engine="python")
        Xtest,Ytest,test_names,test_outputs,test_cats,_=encode(test_df,types,maps)
        if test_names!=names or test_outputs!=out_names or test_cats!=cats: raise ValueError("Test CSV columns/types do not match training data")
    if Xv is not None and len(Xv)<3:
        raise ValueError("Validation needs at least three rows when affine scaling is enabled")
    if len(Xt)<4 and Xv is not None: raise ValueError("Need at least four training rows after validation split")
    if coev and len(Xt)<=512: print(f"Co-evolution subsamples only above 512 training rows; with {len(Xt)} rows every generation scores all rows.")
    hypotheses=discover_hypotheses(Xt,Yt,names,out_names)
    interaction_discovery=discover_interaction_fragments(Xt,Yt,names,out_names,ops,Xv,Yv)
    if interaction_discovery["status"]=="ok":
        print(f"Interaction probe: screened {interaction_discovery['screened']} pairwise fragments; admitted {len(interaction_discovery['accepted'])} held-out-supported fragments.")
    else:
        print(f"Interaction probe: {interaction_discovery['status'].replace('_',' ')}; no pre-evolution fragments admitted.")
    manifest_path=write_run_manifest(path,run_seed,{
        "delimiter":delimiter,"column_types":types,"operators":ops,"affine_scaling":affine_on,
        "co_evolution":coev,"max_nodes":nodes,"max_depth":depth,"perceptrons":perceptron_enabled,"evaluation_workers":resolve_worker_count(args.workers,args.population),
        "bayesian_proposal_rate":args.bayesian_proposal_rate,"bayesian_mode":args.bayesian_mode,"crossover_rate":args.crossover_rate,"validation_request":val_path,
        "evaluation_budget":{"mode":args.evaluation_budget,"refresh":args.evaluation_refresh,"adaptive_screen":"max(64,n/16)","anchor_rows":min(128,len(Xt))},
        "quality_diversity":{"mode":args.qd_mode,"initial_parent_rate":args.qd_parent_rate,"parent_rate_bounds":[.10,.30],"uniform_cell_rate":.25,"cells":144,"probe_rows":min(256,len(Xt)),"semantic_descriptor":QualityDiversityArchive.policy,"structural_descriptor":StructuralQualityDiversityArchive.policy,"admission":"feasible_loss_at_or_below_generation_median","archive_split":"equal"},
        "selection_policy":"loss_tolerance_shortest_mdl","selection_loss_tolerance":args.selection_loss_tolerance,
        "nsga_normalization":args.nsga_normalization,"parsimony_quality_tolerance":args.parsimony_quality_tolerance,"dynamic_pressure":{"enabled":dynamic_pressure_on,"window":args.stagnation_window,"policy":"bounded_novelty"},"adf_mining":adf_enabled,"adf_mode":args.adf_mode,
        "profile":args.profile,"constraint_metadata":metadata,"constraints":constraints.describe(),
        "preprocessing":{"schema_version":1,"fit_rows":len(Xt),"fit_scope":"training_rows_only"},
        "interaction_discovery":interaction_discovery,
        "bayesian_particles_per_output":args.bayesian_particles,
        "islands":{"count":island_count,"population_total":args.population,"migration_interval":migration_interval,"migrants_per_island":migrants_per_island,"topology":"ring","state":"independent population, Bayesian banks, archive, QD, fragment library, pressure, ADF, and budget",
                   "stages":{key:stages[key] for key in ("mode","count","interval","age_gap","schedule","threshold_quantile")},
                   "roles":{key:roles[key] for key in ("enabled","interval","mix","retire_after")}},
        "equivalence_collapse":EQUIVALENCE_COLLAPSE,"residual_archive":RESIDUAL_ARCHIVE,"qd_parent_choice":QD_PARENT_CHOICE,"scale_balanced_selection":SCALE_BALANCED_SELECTION,
        "mdl_policy":MDL_POLICY,"objective_schema":"per_output_loss_shape[,per_output_constraint_violation],mdl_bits,age",
        "test_csv":str(Path(args.test_csv).resolve()) if args.test_csv else None,
    },df,train_indices,validation_indices,external_validation)
    print(f"Run manifest: {manifest_path}")
    checkpoint_path=manifest_path.parent / "checkpoint_latest.json"
    input_ranges=training_input_ranges(train_df,list(df.columns),types)
    checkpoint_state={"dataset_path":str(path.resolve()),"types":types,"operators":ops,"affine_on":affine_on,
        "coev":coev,"nodes":nodes,"depth":depth,"X":X,"Y":Y,"Xt":Xt,"Yt":Yt,"Xv":Xv,"Yv":Yv,
        "names":names,"out_names":out_names,"cats":cats,"maps":maps,"encoding_schema":{"version":1,"fit_scope":"training_rows_only","maps":maps},"source_columns":list(df.columns),
        "export_fixture":train_df.head(16).copy(),"input_ranges":input_ranges,
        "run_seed":run_seed,"manifest":str(manifest_path.resolve()),"bayesian_proposal_rate":args.bayesian_proposal_rate,"bayesian_mode":args.bayesian_mode,"crossover_rate":args.crossover_rate,"evaluation_workers":resolve_worker_count(args.workers,args.population),
        "qd_parent_rate":args.qd_parent_rate,"qd_mode":args.qd_mode,"lexicase_cases":args.lexicase_cases,
        "selection_policy":"loss_tolerance_shortest_mdl","selection_loss_tolerance":args.selection_loss_tolerance,
        "nsga_normalization":args.nsga_normalization,"parsimony_quality_tolerance":args.parsimony_quality_tolerance,"dynamic_pressure_enabled":dynamic_pressure_on,"adf_registry":ADFRegistry(adf_enabled,allow_nested=args.adf_mode=="nested").snapshot(),
        "profile":args.profile,"constraint_metadata":metadata,"constraints":constraints.describe(),"bayesian_particles":args.bayesian_particles,"interaction_discovery":interaction_discovery,
        "island_config":{"count":island_count,"migration_interval":migration_interval,"migrants_per_island":migrants_per_island,"topology":"ring","migration_events":0,"stages":stages,"roles":roles},
        "equivalence_collapse":EQUIVALENCE_COLLAPSE,"residual_archive":RESIDUAL_ARCHIVE,"qd_parent_choice":QD_PARENT_CHOICE,"scale_balanced_selection":SCALE_BALANCED_SELECTION,
        "mdl_policy":MDL_POLICY,"objective_schema":"per_output_loss_shape[,per_output_constraint_violation],mdl_bits,age"}
    head_count=sum(len(heads) for heads in classification_layout(cats)[0])
    population_sizes=cell_population_sizes(args.population,cell_count)
    islands=[new_island_runtime(size,X=X,Xt=Xt,cats=cats,ops=ops,nodes=nodes,depth=depth,head_count=head_count,
                                bayesian_particles=args.bayesian_particles,run_seed=run_seed,island_index=index,cell=divmod(index,stages["count"]),
                                qd_parent_rate=args.qd_parent_rate,nsga_normalization=args.nsga_normalization,
                                parsimony_quality_tolerance=args.parsimony_quality_tolerance,dynamic_pressure_on=dynamic_pressure_on,
                                stagnation_window=args.stagnation_window,adf_enabled=adf_enabled,adf_mode=args.adf_mode,
                                evaluation_budget=args.evaluation_budget,evaluation_refresh=args.evaluation_refresh,
                                interaction_discovery=interaction_discovery,Yt=Yt)
             for index,size in enumerate(population_sizes)]
    if roles["enabled"]: assign_role_parameters(islands,island_count,args.crossover_rate,args.bayesian_proposal_rate,nodes)
    snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
    workers=checkpoint_state["evaluation_workers"]
    evaluator=ModelEvaluator(workers,{"train":(Xt,Yt)},affine_on,cats,constraints,out_names)
    gen=0; start=time.time()
    topology=("single population" if island_count==1 else f"{island_count} islands, ring migration every {migration_interval} generations")
    if stages["count"]>1: topology+=f", {stages['count']} {stages['mode']} stages per island (promotion every {stages['interval']} generations)"
    if roles["enabled"]: topology+=f", self-organising roles (generalist + {island_count-1} specialist(s), update every {roles['interval']} generations)"
    if cell_count>1: topology+=f" ({', '.join(map(str,population_sizes))} models per cell)"
    print(f"Searching indefinitely with {args.bayesian_proposal_rate:.0%} Bayesian proposals, {topology}, and {'serial evaluation' if workers==1 else f'{workers} worker processes'}; press Ctrl-C to choose and save a model.")
    stop=GracefulStop().__enter__(); interrupted=False
    try:
        while (not args.max_generations or gen<args.max_generations) and not stop.requested:
            for island_index,island in enumerate(islands):
                def progress(generation,elite,sample,island=island,island_index=island_index):
                    if cell_count>1 and generation%10==0: print(cell_label(island,island_count,stages["count"]),flush=True)
                    evolution_progress(generation,elite,sample,started=start,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,coev=coev,cases=island.cases,archive=island.archive,semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,qd_controller=island.qd_controller,pressure=island.pressure,bayes=island.bayes,library=island.library,budget=island.budget,evaluator=evaluator,adf_registry=island.adf_registry,population=island.population,best_so_far=island.best_models.model,loss_tolerance=args.selection_loss_tolerance)
                cell_crossover,cell_proposals,cell_nodes,cell_weights=cell_search_settings(island,args.crossover_rate,args.bayesian_proposal_rate,nodes)
                island.population=evolve_generation(island.population,gen,X=X,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,cats=cats,constraints=constraints,out_names=out_names,
                                  ops=ops,nodes=cell_nodes,depth=depth,case_weights=cell_weights,affine_on=affine_on,coev=coev,bayes=island.bayes,archive=island.archive,
                                  semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,residual_qd=island.residual_qd,qd_controller=island.qd_controller,best_models=island.best_models,
                                  pressure=island.pressure,cases=island.cases,portfolio=island.portfolio,library=island.library,evaluator=evaluator,bayesian_proposal_rate=cell_proposals,
                                  crossover_rate=cell_crossover,qd_mode=args.qd_mode,lexicase_cases=args.lexicase_cases,nsga_normalization=args.nsga_normalization,bayesian_mode=args.bayesian_mode,adf_registry=island.adf_registry,budget=island.budget,progress=progress,population_size=island.population_size)
            gen+=1
            advance_topology(islands,checkpoint_state["island_config"],gen,X=Xt,n_features=X.shape[1],ops=ops,nodes=nodes,depth=depth,
                             nsga_normalization=args.nsga_normalization,parsimony_quality_tolerance=args.parsimony_quality_tolerance,evaluator=evaluator,
                             Y=Yt,cats=cats,Xv=Xv,Yv=Yv,crossover_rate=args.crossover_rate,bayesian_proposal_rate=args.bayesian_proposal_rate)
            if args.checkpoint_every and gen % args.checkpoint_every == 0:
                snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
                save_checkpoint(checkpoint_path,gen,islands[0].population,islands[0].bayes,islands[0].archive,checkpoint_state)
                print(f"Checkpoint: {checkpoint_path}")
    except KeyboardInterrupt: interrupted=True
    finally: stop.__exit__(None,None,None)
    if stop.requested:
        snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
        save_checkpoint(checkpoint_path,gen,islands[0].population,islands[0].bayes,islands[0].archive,checkpoint_state)
        note=" (interrupted mid-generation: that generation will not replay exactly on resume)" if interrupted else ""
        print(f"\nSearch stopped; checkpoint saved to {checkpoint_path}{note}; evaluating final candidates...")
    # An interrupt during parallel scoring shuts the pool down; reopen it so
    # the final rescoring is not serial.
    evaluator.reopen()
    for island in islands:
        if island.adf_registry.enabled:
            particle_models=[particle for bank in island.bayes.banks for particle in [*bank.particles.catalog,*bank.particles.particles]]
            island.adf_registry.attach([*island.population,*island.archive.items,*qd_cell_models(island.semantic_qd,island.structural_qd,island.residual_qd),island.best_models.model,*particle_models])
        refresh_persistent_scores(island.archive,island.best_models,island.semantic_qd,island.structural_qd,evaluator,island.residual_qd)
        evaluator.assess(island.population,"train")
        island.best_models.update(island.population); island.archive.update(island.population,Xt)
    f=[model for island in islands for model in [*island.archive.items,*island.population,island.best_models.model] if model is not None]
    evaluation=selection_evaluation(f,Xv,Yv,cats,constraints,out_names)
    labels,choices,selection=model_options(f,cats=cats,loss_tolerance=args.selection_loss_tolerance,evaluation=evaluation)
    print_frontier(f,names,out_names,cats,recommendations=(labels,choices),evaluation=evaluation)
    if len(islands)==1: print(islands[0].archive.stats())
    else: print(f"Island archives: {' | '.join(island.archive.stats() for island in islands)}")
    if EQUIVALENCE_COLLAPSE: print(f"Equivalence collapse: redrew {EQUIVALENCE_STATS['children_redrawn']} offspring equivalent to an existing or sibling equation.")
    selected_index=max(0,min(len(choices)-1,int((choose_model or choose_model_interactively)(labels,choices,evaluation))))
    chosen=choices[selected_index]
    print(f"Selected model: {equations(chosen,names,out_names,cats)}")
    print("Fitted constants:", [constant_vector(tree) for tree in chosen.trees])
    if Xv is not None:
        metrics=frozen_metrics(chosen,Xv,Yv,cats,constraints,out_names)
        print(f"Validation (used for selection): loss={metrics['loss']:.6g}, shape={metrics['shape']:.6g} | output losses={output_loss_summary(metrics['losses'],out_names)}")
    if Xtest is not None:
        metrics=frozen_metrics(chosen,Xtest,Ytest,cats,constraints,out_names)
        print(f"Final held-out test (not used for selection): loss={metrics['loss']:.6g}, shape={metrics['shape']:.6g} | output losses={output_loss_summary(metrics['losses'],out_names)}")
    export_model(chosen,names,out_names,cats,maps,list(df.columns),types,train_df,input_ranges)
    selected_entry=next(entry for entry in evaluation[1] if entry[0] is chosen)
    selection={**selection,"selected_choice":labels[selected_index],"default_selected":selected_index==0,
               "selected_metrics":selected_entry[2],"selected_objectives":tuple(selected_entry[1].objectives)}
    checkpoint_state["selection"]=selection
    snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
    save_checkpoint(checkpoint_path,gen,islands[0].population,islands[0].bayes,islands[0].archive,checkpoint_state)
    record_selection_manifest(manifest_path,selection)
    card=write_model_card(manifest_path,chosen,names,out_names,constraints,hypotheses,islands[0].bayes,{"train_rows":len(Xt),"validation_rows":0 if Xv is None else len(Xv),"preprocessing":"fit_on_training_rows_only","schema_version":1},cats=cats,selection=selection,quality_diversity={"islands":[{"semantic":island.semantic_qd.diagnostics(),"structural":island.structural_qd.diagnostics(),"controller":island.qd_controller.diagnostics(),
                                                                                         "residual":None if island.residual_qd is None else island.residual_qd.diagnostics()} for island in islands]},survival={"nsga_normalization":args.nsga_normalization,"parsimony_quality_tolerance":args.parsimony_quality_tolerance,"islands":checkpoint_state["island_config"]},adf_diagnostics=[island.adf_registry.diagnostics() for island in islands if island.adf_registry.enabled] or None,evaluation={"budgets":[island.budget.snapshot() for island in islands],"evaluator":evaluator.diagnostics()},interaction_discovery=interaction_discovery,island_diagnostics={"config":checkpoint_state["island_config"],"bayesian_posteriors":[[bank.particles.last_predictive for bank in island.bayes.banks] for island in islands]})
    evaluator.close()
    print(f"Model card: {card}")
    print("Saved best_model.py")
    return {"checkpoint":str(checkpoint_path.resolve()),"manifest":str(manifest_path.resolve()),"model_card":str(Path(card).resolve()),
            "generation":gen,"selected":labels[selected_index],"equation":equations(chosen,names,out_names,cats)}

if __name__=="__main__":
    try: main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc: print(f"Configuration error: {exc}",file=sys.stderr); sys.exit(2)
