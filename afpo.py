#!/usr/bin/env python3
"""Interactive AFPO/NSGA-II symbolic regression.

Only numpy and pandas are required; matplotlib is optional and only used by
the exported model.  Start it with ``python afpo.py``.
Use ``--max-generations N`` for a bounded unattended run (useful in CI).

Speed: with Cython and a C compiler installed, constant fitting (the
Levenberg-Marquardt loop and the jump-constant scan), the Huber affine
readout, the numeric guard check and the numeric inversion of semantic
backpropagation use the compiled kernels in afpo_lib/fitcore.pyx (built on
first use; every operator except seqsum/seqprod is compiled; they agree with
Python to round-off; ``--fit-backend python`` opts out).  Runs with several
island/stage cells evolve the cells in parallel processes (``--cell-workers``;
results are identical to serial).  Above 8192 rows trees are evaluated in
cache-sized row blocks (identical values), so many ``--workers`` no longer
starve each other of memory bandwidth.

Island roles: with roles on, each island after the first is "auto"
(self-organising) or a fixed role picked per island: generalist, simplifier,
explorer, refiner, or an operator family.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from html import escape as xml_escape
import hashlib
import inspect
import io
import json
import math
import multiprocessing
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
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

# Tree and ADF-definition bits depend only on the trees, the grammar, the
# feature count and the called ADFs; scoring asks for them for every model it
# scores and the Bayesian banks for every particle, each time building the full
# audit record.  Memoize that structural part (exact integers, so the total is
# unchanged); the readout coefficients are priced fresh.  Invalid models raise
# from model_description() and are never cached.
# Cache keys of trees: a 128-bit digest of their repr (and ADF signature)
# instead of the repr itself.  With large trees the repr strings, one per
# cached entry, outweighed the cached values: at 63-node trees the evaluation
# cache held ~150 MB of key text beside ~130 MB of arrays.  repr, not the
# tuple, still decides identity (0.0/-0.0 and 1/1.0 evaluate differently), and
# blake2b is the same in every process (Python's hash() is randomised per
# process), so keys merge across cell workers.  A tree's fingerprint is
# memoized per tree object: trees are immutable tuples shared by every clone,
# so a large tree is serialized and hashed once, not on every cache lookup.
# Each memo entry holds its tree, so the id cannot be reused while cached.
_TREE_FINGERPRINTS={}
TREE_FINGERPRINT_LIMIT=100_000
def _is_tree(value):
    return isinstance(value,tuple) and bool(value) and isinstance(value[0],str)
def tree_fingerprint(tree):
    """128-bit digest of repr(tree), memoized per tree object."""
    hit=_TREE_FINGERPRINTS.get(id(tree))
    if hit is not None and hit[0] is tree: return hit[1]
    digest=hashlib.blake2b(repr(tree).encode(),digest_size=16).digest()
    if len(_TREE_FINGERPRINTS)>=TREE_FINGERPRINT_LIMIT: _TREE_FINGERPRINTS.clear()
    _TREE_FINGERPRINTS[id(tree)]=(tree,digest)
    return digest
def tree_digest(*parts):
    """128-bit key of trees, lists of trees and any other repr-able parts (ADF signatures, scales)."""
    digest=hashlib.blake2b(digest_size=16)
    for part in parts:
        if _is_tree(part): digest.update(b"t"); digest.update(tree_fingerprint(part))
        elif isinstance(part,list) and part and all(_is_tree(item) for item in part):
            digest.update(b"l%d:"%len(part))
            for item in part: digest.update(tree_fingerprint(item))
        else: digest.update(b"r"); digest.update(repr(part).encode())
        digest.update(b"\0")
    return digest.digest()
# One shared tuple per distinct grammar, so cache keys reference it instead of
# each copying the run's operator list (~100 names).
_GRAMMARS={}
def interned_grammar(operators):
    operators=tuple(operators); hit=_GRAMMARS.get(operators)
    if hit is None:
        if len(_GRAMMARS)>=1024: _GRAMMARS.clear()
        hit=_GRAMMARS[operators]=operators
    return hit
_DESCRIPTION_BITS_CACHE={}
def model_description_bits(model, n_features=None, operators=None, adfs=None):
    n_features=model.mdl_feature_count if n_features is None else n_features
    operators=tuple(model.mdl_operators if operators is None else operators)
    adfs=getattr(model,"adfs",{}) if adfs is None else adfs
    key=(tree_digest(model.trees,adf_signature(model.trees,adfs)),n_features,interned_grammar(operators))
    structural=_DESCRIPTION_BITS_CACHE.get(key)
    if structural is None:
        description=model_description(model,n_features,operators,adfs)
        if len(_DESCRIPTION_BITS_CACHE)>=50_000: _DESCRIPTION_BITS_CACHE.clear()
        _DESCRIPTION_BITS_CACHE[key]=description["tree_bits"]+description["adf_definition_bits"]
        return float(description["total_bits"])
    return float(structural+sum(exact_float_code(scale)["bits"]+exact_float_code(intercept)["bits"] for scale,intercept in model.scales))

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
def _op_eval_unguarded_state(op, a):
    """fast_op_eval for callers already inside np.errstate(all="ignore");
    entering it per node was a measurable share of constant fitting."""
    fn=_OP_TABLE.get(op)
    if fn is None: return op_eval(op,a)
    z=fn(a[0],a)
    # In range and NaN-free (a NaN fails the comparison), clean() would return
    # these exact values; one reduction replaces its clamp and NaN scan.
    if type(z) is np.ndarray and z.dtype==np.float64 and z.size and np.abs(z).max()<=CLIP: return z
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
    # 128-bit digests (see tree_digest) instead of repr strings: the key and
    # value text of every cached entry grew with the tree.  Keys are only ever
    # compared for identity, never ordered or stored.
    raw=tree_fingerprint(tree)
    if not EQUIVALENCE_COLLAPSE: return raw
    hit=_EQUIVALENCE_CACHE.get(raw)
    if hit is None:
        if len(_EQUIVALENCE_CACHE)>=16384: _EQUIVALENCE_CACHE.clear()
        hit=_EQUIVALENCE_CACHE[raw]=hashlib.blake2b(repr(_equivalence_key(tree)).encode(),digest_size=16).digest()
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
# Trees are evaluated in blocks of rows on large data.  Every operator works
# row by row, so the values are bit-identical, but a block's intermediates
# (~5 arrays of 32 KB) stay in the CPU cache.  Whole-column intermediates
# (800 KB each at 100k rows) streamed through RAM instead, and parallel workers
# then starved each other of memory bandwidth (15 ran slower than 4).
EVALUATION_BLOCK_ROWS = 4096
def _row_blocks(n):
    """Row slices for blocked evaluation, or None when n is small enough to do whole."""
    if n<=2*EVALUATION_BLOCK_ROWS: return None
    return [slice(start,min(n,start+EVALUATION_BLOCK_ROWS)) for start in range(0,n,EVALUATION_BLOCK_ROWS)]
def evaluate(t, X, adfs=None, arguments=None, position=None):
    # One error-state context per tree instead of one per node; the values are
    # identical (np.errstate only changes how floating-point errors are reported).
    blocks=_row_blocks(len(X)) if isinstance(X,np.ndarray) and X.ndim==2 else None
    with np.errstate(all="ignore"):
        if blocks is None: return _evaluate(t,X,adfs,arguments,position)
        out=None
        for rows in blocks:
            part=_evaluate(t,X[rows],adfs,None if arguments is None else [argument[rows] for argument in arguments],position)
            if out is None: out=np.empty(len(X),dtype=np.asarray(part).dtype)
            out[rows]=part
        return out
def _evaluate(t, X, adfs=None, arguments=None, position=None):
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
        values=[_evaluate(q,X,adfs,arguments,position) for q in t[1:]]
        return _evaluate(item["tree"],X,adfs,values,position)
    if t[0] in ("seqsum","seqprod"):
        if SEQUENCE_LAYOUT is None: raise ValueError(f"{t[0]} needs --sequence-group")
        parts=[_evaluate(t[1],X,adfs,arguments,i) for i in range(SEQUENCE_LAYOUT["length"])]
        with np.errstate(all="ignore"): return clean(np.sum(parts,axis=0) if t[0]=="seqsum" else np.prod(parts,axis=0))
    return _op_eval_unguarded_state(t[0],[_evaluate(q,X,adfs,arguments,position) for q in t[1:]])

# Numeric guards exist to keep evaluation finite, not to model anything.  A
# model whose output depends on one is rejected.  Every operator result is
# clamped to +/-CLIP and several operators clip inputs (pow's exponent, exp,
# sinh, cosh, tan, ...), and the search learns to use them as hidden min/max:
# min(x, 17) as -1.34e-11*((x-3.59)*-7.45e10)+3.59 (the product saturates at
# CLIP exactly when x reaches 17), or a pow of an exp_decay plateau.  Such
# equations read as something else, break under exact re-evaluation, and
# depend on the guard constants.  The check evaluates each tree twice in one
# pass, guarded and with the exact unguarded mathematics, and flags the model
# when a value reaches the clamp or the two outputs differ beyond round-off
# relative to the output's spread.  Natural saturation (a steep sigmoid step,
# exp of a very negative number) changes outputs by ~1e-22 and stays legal.
# Protected-division/log EPS offsets are standard GP semantics and not checked.
GUARD_EXPLOIT_CHECK = True
GUARD_TOLERANCE = 1e-6
_UNGUARDED = {
    "pow":lambda x,a: np.sign(x)*np.abs(x)**a[1],
    "exp":lambda x,a: np.exp(x), "expm1":lambda x,a: np.expm1(x), "10^x":lambda x,a: 10.**x,
    "exp_decay":lambda x,a: np.exp(-x*a[1]), "gaussian":lambda x,a: np.exp(-x*x),
    "sigmoid":lambda x,a: 1/(1+np.exp(-x)), "perceptronSigma1":lambda x,a: 1/(1+np.exp(-x)),
    "perceptronSigma2":lambda x,a: 1/(1+np.exp(-(x+a[1]))),
    "perceptronCustom1":lambda x,a: x/(1+np.exp(-x)), "perceptronCustom2":lambda x,a: (x+a[1])/(1+np.exp(-(x+a[1]))),
    "sinh":lambda x,a: np.sinh(x), "cosh":lambda x,a: np.cosh(x), "tan":lambda x,a: np.tan(x),
}
_GUARD_CACHE = {}
class _GuardEngaged(Exception):
    pass
def _guard_walk(t, X, adfs, arguments=None, position=None):
    """(guarded value, exact unguarded value) of a node; raises _GuardEngaged at the value clamp."""
    if t[0]=="arg":
        if arguments is None or not isinstance(t[1],int) or not 0<=t[1]<len(arguments): raise ValueError("Invalid ADF argument")
        return arguments[t[1]]
    if t[0] in ("x","c"):
        value=evaluate(t,X,adfs,None,position); return value,value
    if t[0].startswith("adf_"):
        item=(adfs or {}).get(t[0])
        if item is None or len(t)-1!=int(item["arity"]): raise ValueError(f"Unknown or malformed ADF {t[0]!r}")
        pairs=[_guard_walk(q,X,adfs,arguments,position) for q in t[1:]]
        return _guard_walk(item["tree"],X,adfs,pairs,position)
    with np.errstate(all="ignore"):
        if t[0] in ("seqsum","seqprod"):
            pairs=[_guard_walk(t[1],X,adfs,arguments,i) for i in range(SEQUENCE_LAYOUT["length"])]
            reduce=np.sum if t[0]=="seqsum" else np.prod
            raw=reduce([v for v,_ in pairs],axis=0); exact=reduce([r for _,r in pairs],axis=0)
        else:
            pairs=[_guard_walk(q,X,adfs,arguments,position) for q in t[1:]]
            values=[v for v,_ in pairs]; references=[r for _,r in pairs]
            raw=np.asarray(_OP_TABLE[t[0]](values[0],values),float)
            # While every input is still its exact reference and the operator
            # has no unguarded form, the exact pass would repeat this very
            # computation; share it (the common case: no guard on the path).
            if t[0] not in _UNGUARDED and all(v is r for v,r in pairs): exact=None
            else: exact=np.asarray(_UNGUARDED.get(t[0],_OP_TABLE[t[0]])(references[0],references),float)
    value=clean(raw)
    if np.any(np.abs(value)>=CLIP): raise _GuardEngaged("value_clamp")
    if exact is None:
        # Below the clamp, clean() changes only NaNs (to 0): with none, the
        # guarded value is bitwise the exact one and can stand for it.
        if np.isnan(raw).any(): exact=raw
        else: return value,value
    return value,exact
_COMPILED_TREES={}
def _compiled_tree(tree, feature_count):
    """(core, program, constants) of a tree the compiled kernels can run, else None (memoized)."""
    if FIT_BACKEND!="auto": return None
    key=(tree_fingerprint(tree),feature_count); hit=_COMPILED_TREES.get(key)
    if hit is None:
        core=compiled_fitter(); flat=None if core is None else _FlatTree.build(tree)
        program=None if flat is None else _compiled_program(flat,core,feature_count)
        hit=(None,) if program is None else (core,program,np.array([flat.payload[index] for index in flat.constants],float))
        if len(_COMPILED_TREES)>=cache_limit(20000): _COMPILED_TREES.clear()
        _COMPILED_TREES[key]=hit
    return None if hit[0] is None else hit
def _guard_changes_output(value, exact):
    spread=float(np.std(value)) if np.size(value)>1 else 0.
    tolerance=GUARD_TOLERANCE*(spread+1e-9*(1.+np.abs(value)))
    return bool(np.any(~(np.abs(exact-value)<=tolerance)))
def guard_engagement(trees, X, adfs=None):
    """'' when no numeric guard shapes the trees' values on X, else the guard's name."""
    if not GUARD_EXPLOIT_CHECK: return ""
    X=np.asarray(X,float); interface=X.__array_interface__
    key=(tree_digest(*trees,adf_signature(trees,adfs)),interface["data"][0],X.shape,X.strides)
    hit=_GUARD_CACHE.get(key)
    if hit is not None: return hit[0]
    reason=""; blocks=_row_blocks(len(X)) if X.ndim==2 else None
    try:
        for tree in trees:
            compiled=_compiled_tree(tree,X.shape[1]) if X.ndim==2 else None
            if compiled is not None:
                # One compiled pass computes both the guarded and the exact
                # values (blocked like evaluate()), with the same verdicts.
                verdict=compiled[0].guard_verdict(*compiled[1],compiled[2],X,GUARD_TOLERANCE,EVALUATION_BLOCK_ROWS)
                if verdict==1: raise _GuardEngaged("value_clamp")
                if verdict==2: reason="input_clip"; break
                continue
            if blocks is None: value,exact=_guard_walk(tree,X,adfs)
            else:
                # Blocked like evaluate(); a clamp in any block is a clamp on X,
                # and the spread-based comparison sees the reassembled columns.
                parts=[_guard_walk(tree,X[rows],adfs) for rows in blocks]
                value=np.concatenate([np.broadcast_to(v,(rows.stop-rows.start,)) for (v,_),rows in zip(parts,blocks)])
                exact=np.concatenate([np.broadcast_to(r,(rows.stop-rows.start,)) for (_,r),rows in zip(parts,blocks)])
            if _guard_changes_output(value,exact): reason="input_clip"; break
    except _GuardEngaged as engaged: reason=str(engaged)
    if len(_GUARD_CACHE)>=20000: _GUARD_CACHE.clear()
    _GUARD_CACHE[key]=(reason,X)      # holding X keeps its buffer address from being reused
    return reason

# Deterministic top-level tree outputs, reused across the many places one
# generation re-predicts the same trees on the same rows (mutation baselines,
# behavioral dedup, lexicase, QD cells, serial scoring).  Entries hold the
# data array itself, so its buffer address cannot be recycled while cached;
# results are read-only so no caller can corrupt a shared entry.
_EVALUATION_CACHE={}; _EVALUATION_CACHE_SIZE=[0]
EVALUATION_CACHE_ELEMENTS=16_000_000
# --cache-memory MB sets this budget (8 bytes per element).  Every scoring
# worker (--workers) builds its own caches, so with the full budget each one
# grew as large as the main process (4 workers on 63-node trees: ~1 GB in
# all).  Workers see each model about once (the main process deduplicates
# before sending), so they get WORKER_CACHE_SHARE of the budget.
WORKER_CACHE_SHARE = 1/8
# The entry caps of the compiled-tree and score caches scale with the same
# setting (CACHE_SCALE = MB/128): their entries grow with the tree size.
CACHE_SCALE = 1.
def cache_limit(entries):
    return max(500,int(entries*CACHE_SCALE))
def configure_cache_memory(megabytes):
    global EVALUATION_CACHE_ELEMENTS,CACHE_SCALE
    megabytes=float(megabytes)
    if not megabytes>0: raise ValueError("--cache-memory must be positive")
    EVALUATION_CACHE_ELEMENTS=max(10_000,int(megabytes*1_000_000/8)); CACHE_SCALE=megabytes/128
# Charged per entry beside its array (key, tuples, array header, dict slot,
# ~400 bytes), so few-row data cannot fill the budget with entries whose
# overhead outweighs their values.
EVALUATION_CACHE_ENTRY_COST=48
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
def rows_digest(rows):
    """128-bit content key of a row-index array.  Raw index bytes as a cache key
    cost 8 bytes per sampled row in every entry: tens of thousands of score
    cache entries on a large co-evolution sample held gigabytes of keys."""
    return hashlib.blake2b(np.ascontiguousarray(rows,dtype=np.int64).tobytes(),digest_size=16).digest()
_ROW_SUBSETS={}
def row_subset(X, rows):
    """X[rows], returning the same array object for the same rows of the same
    array.  Fancy indexing copies, and the evaluation caches key on buffer
    addresses: a fresh copy per call never hit and pinned its own rows."""
    if isinstance(rows,slice): return X[rows]
    rows=np.asarray(rows)
    interface=X.__array_interface__
    key=(interface["data"][0],X.shape,X.strides,interface["typestr"],rows.dtype.str,rows_digest(rows) if rows.dtype.kind in "iu" else rows.tobytes())
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
    key=(tree_digest(t,adf_signature((t,),adfs)),interface["data"][0],X.shape,X.strides,interface["typestr"])
    hit=_EVALUATION_CACHE.get(key)
    if hit is not None: return hit[0]
    value=np.asarray(evaluate(t,X,adfs))
    if value.ndim!=1 or len(value)!=len(X): return value
    value.setflags(write=False)
    _EVALUATION_CACHE[key]=(value,X); _EVALUATION_CACHE_SIZE[0]+=value.size+EVALUATION_CACHE_ENTRY_COST
    while _EVALUATION_CACHE_SIZE[0]>EVALUATION_CACHE_ELEMENTS and _EVALUATION_CACHE:
        oldest=next(iter(_EVALUATION_CACHE)); _EVALUATION_CACHE_SIZE[0]-=_EVALUATION_CACHE.pop(oldest)[0].size+EVALUATION_CACHE_ENTRY_COST
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

def canonical_adf_template(tree, parametric=False):
    """Replace up to three distinct feature leaves with ordered ADF arguments.

    parametric=True also turns every constant into its own argument (up to
    ADF_MAX_ARITY in all): a constant inside an ADF body is never fitted
    (constant_paths stops at the call), while one at the call site is, and
    tanh(2*x+3*y) and tanh(7*x-.4*y) become one motif instead of two."""
    features=[]
    def convert(node):
        if node[0]=="x":
            if ("x",node[1]) not in features: features.append(("x",node[1]))
            return ("arg",features.index(("x",node[1])))
        if node[0]=="c" and parametric:
            features.append(("c",len(features))); return ("arg",len(features)-1)
        if node[0] in ("c","arg"): return node
        return tuple([node[0]]+[convert(child) for child in node[1:]])
    result=convert(simplify_tree(tree))
    inputs=sum(kind=="x" for kind,_ in features)
    if parametric: return (result,len(features)) if inputs>=1 and len(features)<=ADF_MAX_ARITY else (None,0)
    return (result,len(features)) if 1<=len(features)<=3 else (None,0)
def _operator_count(tree):
    return 0 if tree[0] in ("x","c","arg") else 1+sum(_operator_count(child) for child in tree[1:])
def adf_net_gain(item):
    """Nodes saved by calling a motif instead of inlining it at each supporting
    founder (inline size minus call size), less the definition's own size."""
    size=node_size(item["tree"])
    return len(item["founders"])*(size-1-item["arity"])-size

# ADF mining (schema 3).  Schema 2 kept the most widely supported template and
# broke ties toward the smallest, so its ADFs were single-operator aliases of
# the grammar (arg0*arg1, arg0/arg1: 3.3 nodes on average on bench_complex)
# with frozen inner constants.  Schema 3 parametrises constants, skips
# single-operator templates, ranks by adf_net_gain and promotes up to
# ADF_PROMOTIONS_PER_CHECK motifs per check.  Resumed schema-2 registries
# keep the old rule.
ADF_MAX_ARITY = 6
ADF_PROMOTIONS_PER_CHECK = 3

class ADFRegistry:
    """ADF catalog: founder-supported, bounded, and dependency-aware (schema 3 mining, see above)."""
    def __init__(self, enabled=False, capacity=8, allow_nested=True):
        self.enabled=bool(enabled); self.capacity=int(capacity); self.allow_nested=bool(allow_nested); self.definitions={}; self.active=[]; self.history={}; self.last_used={}; self.schema_version=3; self.invalid_calls=0
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
                    template,arity=canonical_adf_template(subtree,parametric=self.schema_version>=3)
                    if template is None: continue
                    key=repr(template); seen.setdefault(key,{"tree":template,"arity":arity,"founders":set()})["founders"].update(model.founder_ids)
        for key,item in seen.items():
            record=self.history.setdefault(key,{"tree":item["tree"],"arity":item["arity"],"founders":[],"last":generation})
            record["founders"]=sorted(set(record["founders"]).union(item["founders"]))[-64:]; record["last"]=generation
        for key in list(self.history):
            if generation-self.history[key]["last"]>50: del self.history[key]
        choices=[(key,item) for key,item in self.history.items() if len(item["founders"])>=3 and self._valid_definition(item["tree"],item["arity"]) and all(existing["tree"]!=item["tree"] for existing in self.definitions.values())]
        if self.schema_version>=3:
            choices=[pair for pair in choices if _operator_count(pair[1]["tree"])>=2 and adf_net_gain(pair[1])>0]
            ranked=sorted(choices,key=lambda pair:(-adf_net_gain(pair[1]),pair[0]))[:ADF_PROMOTIONS_PER_CHECK]
        else:
            ranked=[max(choices,key=lambda pair:(len(pair[1]["founders"]),-node_size(pair[1]["tree"]),pair[0]))] if choices else []
        promoted=False
        for key,item in ranked:
            name="adf_"+hashlib.sha256(key.encode()).hexdigest()[:10]
            if name in self.definitions or not self._acyclic(name,item["tree"]): continue
            self.definitions[name]={"tree":item["tree"],"arity":item["arity"],"dependencies":self._dependencies(item["tree"]),"supporting_founders":list(item["founders"]),"activated_generation":generation,"retired_generation":None,"elite_uses":0,"validation":[]}; self.active.append(name); self.last_used[name]=generation
            promoted=True
        self._retire(generation); return promoted
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
        with np.errstate(all="ignore"):
            for index in range(len(self.kind)-1,-1,-1):
                kind=self.kind[index]
                if kind=="x": values[index]=X[:,payload[index]]
                elif kind=="c": values[index]=np.full(len(X),payload[index])
                else: values[index]=_op_eval_unguarded_state(kind,[values[child] for child in self.children[index]])
        return values
    def nudged_root(self, values, constant, value, X):
        """Root output when only one constant changes, reusing every off-path value."""
        changed=self.constants[constant]; current=np.full(len(X),float(value)); index=self.parent[changed]
        with np.errstate(all="ignore"):
            while index>=0:
                current=_op_eval_unguarded_state(self.kind[index],[current if child==changed else values[child] for child in self.children[index]])
                changed,index=index,self.parent[index]
        return current

# Compiled constant fitter (afpo_lib/fitcore.pyx, built on first use with
# Cython's pyximport).  "auto" uses it when it builds and the tree's operators
# are all compiled, else the Python fitter below; "python" always uses Python
# (bit-for-bit reproduction of runs made before it existed).  The two agree to
# round-off, not bit for bit: compiled sums and libm differ in the last bit.
FIT_BACKEND = "auto"
_FITCORE = []
def compiled_fitter():
    """The compiled fitter module, or None when Cython or a compiler is unavailable."""
    if not _FITCORE:
        try:
            import pyximport
            importers=pyximport.install(language_level=3)
            try: from afpo_lib import fitcore
            finally: pyximport.uninstall(*importers)
            _FITCORE.append(fitcore)
        except Exception as error:
            print(f"Compiled constant fitter unavailable ({type(error).__name__}: {error}); using the Python fitter.",file=sys.stderr)
            _FITCORE.append(None)
    return _FITCORE[0]
def _compiled_program(flat, core, feature_count):
    """(code, arg, kid) arrays of a flat tree, or None when it needs the Python fitter."""
    codes=core.OPERATOR_CODES; nodes=len(flat.kind)
    code=np.empty(nodes,np.int32); arg=np.zeros(nodes,np.int32); kid=np.full((nodes,3),-1,np.int32)
    slot={index:position for position,index in enumerate(flat.constants)}
    for index,kind in enumerate(flat.kind):
        if kind=="x":
            feature=flat.payload[index]
            if not isinstance(feature,(int,np.integer)) or not 0<=feature<feature_count: return None
            code[index]=0; arg[index]=feature
        elif kind=="c": code[index]=1; arg[index]=slot[index]
        else:
            op=codes.get(kind); children=flat.children[index]
            if op is None or len(children)>3: return None
            code[index]=op; kid[index,:len(children)]=children
    return code,arg,kid

# A constant that only decides where a jump falls (a mod period, a comparison
# threshold, the scale inside floor) has a zero finite-difference gradient
# almost everywhere, so Levenberg-Marquardt never moves it and only random
# nudges did.  Before the gradient fit, a small derivative-free scan tries
# data-driven values for such constants (midpoints between the compared
# values, fractions of a mod/floordiv operand's range, a log grid around the
# current value, and simple numbers) and keeps the best strict improvement.
JUMP_CONSTANT_SCAN = True
JUMP_SCAN_MAX_CONSTANTS = 3
_JUMP_OPERATORS = frozenset(("mod","floordiv","quantize","gt","lt","gte","lte","eq","ne","round2","floor2","ceil2",
                             "round","floor","ceil","int","frac","sign","signbit","oom","bitwise_and","bitwise_or",
                             "bitwise_xor","bitwise_not","lshift","rshift","gcd","lcm","cat","x_at_pos_y"))
_JUMP_ARGUMENTS = {"if_else":(0,)}   # only the condition of if_else jumps
_THRESHOLD_OPERATORS = frozenset(("gt","lt","gte","lte","eq","ne"))
_PERIOD_OPERATORS = frozenset(("mod","floordiv","quantize"))
_SIMPLE_NUMBERS = (.5,1.,1.5,2.,2.5,3.,4.,5.,6.,8.,10.)
JUMP_SCAN_STATS = {"scans":0,"improved":0}
def jump_constant_slots(flat):
    """Positions (in constant order) of constants that sit under a jump."""
    slots=[]
    for slot,index in enumerate(flat.constants):
        child,parent=index,flat.parent[index]
        while parent>=0:
            kind=flat.kind[parent]
            if kind in _JUMP_OPERATORS or flat.children[parent].index(child) in _JUMP_ARGUMENTS.get(kind,()):
                slots.append(slot); break
            child,parent=parent,flat.parent[parent]
    return slots
def _jump_candidates(flat, nodes, slot, value):
    index=flat.constants[slot]; parent=flat.parent[index]
    kind=flat.kind[parent] if parent>=0 else None
    siblings=[child for child in flat.children[parent] if child!=index] if parent>=0 else []
    candidates=[value*factor for factor in (.25,.5,.7,.85,1.2,1.4,2.,4.)]
    candidates+=[sign*number for number in _SIMPLE_NUMBERS for sign in (1.,-1.)]
    if siblings:
        other=nodes[siblings[0]]
        if kind in _THRESHOLD_OPERATORS or kind=="if_else":
            levels=np.unique(other)
            if len(levels)>1:
                middles=(levels[1:]+levels[:-1])/2
                candidates+=middles[np.unique(np.linspace(0,len(middles)-1,min(24,len(middles))).astype(int))].tolist()
        elif kind in _PERIOD_OPERATORS and flat.children[parent].index(index)==1:
            span=float(np.ptp(other))
            if span>EPS: candidates+=[span/k for k in range(1,13)]
    return [c for c in dict.fromkeys(candidates) if math.isfinite(c) and abs(c)<=CONSTANT_LIMIT]
def _scan_costs(predictions, y, scale, fit_readout):
    """Huber cost after a least-squares readout, for each row of a (K, n) batch."""
    P=np.asarray(predictions,float)
    if fit_readout:
        mu=P.mean(axis=1,keepdims=True); du=P-mu; my=float(np.mean(y))
        suu=np.einsum("kn,kn->k",du,du)
        degenerate=~(suu>1e-12*np.maximum(np.einsum("kn,kn->k",P,P),EPS))
        slope=np.einsum("kn,n->k",du,y-my)/np.where(degenerate,1.,suu)
        slope=np.where(degenerate|~np.isfinite(slope),0.,np.clip(slope,-AFFINE_COEFFICIENT_BOUND,AFFINE_COEFFICIENT_BOUND))
        P=slope[:,None]*du+my
    r=np.abs((P-y)/scale); huber=loss_delta()
    costs=np.where(r<=huber,.5*r*r,huber*(r-.5*huber)).sum(axis=1)
    return np.where(np.isfinite(costs),costs,np.inf)
def _nudged_roots(flat, nodes, slot, candidates, X):
    """Root outputs (K, n) for K values of one constant, every other node reused."""
    changed=flat.constants[slot]; current=np.asarray(candidates,float)[:,None]; index=flat.parent[changed]
    with np.errstate(all="ignore"):
        while index>=0:
            current=_op_eval_unguarded_state(flat.kind[index],[current if child==changed else nodes[child] for child in flat.children[index]])
            changed,index=index,flat.parent[index]
    return np.broadcast_to(current,(len(candidates),len(X)))
def scan_jump_constants(flat, start, X, y, scale, fit_readout=True, program=None):
    """Constant vector with jump constants placed by a candidate scan, or None.

    Every candidate value of a constant is scored in one batched pass along
    its path to the root (a (K, rows) array), not one evaluation per value.
    ``program`` (from _compiled_program) runs the scan in the compiled kernel."""
    slots=jump_constant_slots(flat)
    if not slots: return None
    if len(slots)>JUMP_SCAN_MAX_CONSTANTS:
        # Drawn from the tree itself, not the run's random stream: scoring runs
        # in --workers processes whose stream copies diverge with scheduling,
        # so a stream draw here made parallel runs irreproducible.
        seed=hashlib.blake2b(repr((flat.kind,flat.payload,[float(value) for value in start])).encode(),digest_size=8).digest()
        slots=random.Random(int.from_bytes(seed,"little")).sample(slots,JUMP_SCAN_MAX_CONSTANTS)
    JUMP_SCAN_STATS["scans"]+=1
    current=np.array(start,float)
    core=compiled_fitter() if program is not None else None
    if core is not None:
        code,arg,kid=program; Xc=np.ascontiguousarray(X,dtype=float); yc=np.ascontiguousarray(y,dtype=float)
        bound,huber=AFFINE_COEFFICIENT_BOUND,loss_delta()
        def node_values(constants): return core.node_values(code,arg,kid,constants,Xc)
        def root_cost(nodes): return float(core.prediction_costs(nodes[0][None,:],yc,scale,fit_readout,huber,bound)[0])
        def candidate_costs(nodes, slot, candidates):
            changed=flat.constants[slot]; path=[]; index=flat.parent[changed]
            while index>=0: path.append(index); index=flat.parent[index]
            return core.scan_costs(code,kid,nodes,changed,np.asarray(path,np.int32),np.asarray(candidates,float),yc,scale,fit_readout,huber,bound)
    else:
        def node_values(constants): return flat.values(constants,X)
        def root_cost(nodes): return float(_scan_costs(nodes[0][None,:],y,scale,fit_readout)[0])
        def candidate_costs(nodes, slot, candidates): return _scan_costs(_nudged_roots(flat,nodes,slot,candidates,X),y,scale,fit_readout)
    try:
        nodes=node_values(current); best=initial=root_cost(nodes)
        for slot in slots:
            chosen=current[slot]
            candidates=_jump_candidates(flat,nodes,slot,chosen)
            # Then two shrinking local grids: the cost is piecewise constant.
            for round_ in range(3):
                if round_:
                    step=(.05 if round_==1 else .005)*max(abs(chosen),1e-3)
                    candidates=[chosen+step*k for k in (-8,-6,-4,-3,-2,-1,-.5,.5,1,2,3,4,6,8)]
                costs=candidate_costs(nodes,slot,candidates)
                pick=int(np.argmin(costs))
                if costs[pick]<best: best,chosen=float(costs[pick]),float(candidates[pick])
            if chosen!=current[slot]:
                current[slot]=chosen; nodes=node_values(current)
    except (ArithmeticError, IndexError, ValueError): return None
    if not best<initial: return None
    JUMP_SCAN_STATS["improved"]+=1
    return current

def fit_tree_constants(tree, X, y, adfs=None, fit_readout=True, iterations=None, robust=True):
    """Levenberg-Marquardt on a tree's inner constants (variable projection).

    The affine readout a*f+b is re-solved in closed form for every constant
    vector, so only the constants inside the tree are searched.  Returns the
    tree unchanged unless its residual cost strictly improves.  ``robust``
    fits what assess() scores: a Huber-reweighted readout and robust_loss's
    Huber cost.  Plain least squares let a few outliers drag correct
    constants away, and the result is written back into the model.
    """
    if iterations is None: iterations=CONSTANT_FIT_ITERATIONS  # read at call time: --fit-iterations sets it
    start=np.asarray(constant_vector(tree),float)
    if not len(start) or not len(y): return tree
    scale=loss_scale(y); flat=_FlatTree.build(tree); huber=loss_delta()
    base=loss_base_weights(y); ones=np.ones(len(y)) if base is None else base*base
    program=None
    if flat is not None and FIT_BACKEND=="auto" and LOSS_MODE=="huber" and isinstance(X,np.ndarray) and X.ndim==2:
        core=compiled_fitter(); program=None if core is None else _compiled_program(flat,core,X.shape[1])
    if JUMP_CONSTANT_SCAN and flat is not None and isinstance(X,np.ndarray) and X.ndim==2:
        scanned=scan_jump_constants(flat,start,X,y,scale,fit_readout,program)
        if scanned is not None:
            placed=with_constants(tree,scanned)
            # The gradient fit below returns its own input unless it improves
            # further, so the scanned tree becomes that input.
            if not guarded_constant_divisor(placed) and not guard_engagement([placed],X,adfs): tree,start=placed,scanned
    if program is not None:
        result=core.fit(*program,start,np.ascontiguousarray(X,dtype=float),np.ascontiguousarray(y,dtype=float),scale,
                        fit_readout,robust,iterations,AFFINE_COEFFICIENT_BOUND,huber,CONSTANT_LIMIT)
        if result is None: return tree
        current,cost,initial=result
        if cost>=initial: return tree
        tuned=with_constants(tree,current)
        return tree if guarded_constant_divisor(tuned) or guard_engagement([tuned],X,adfs) else tuned
    # readout() runs ~30 times per fit on <=256 rows, so numpy's per-call
    # overhead is the cost: scalars are hoisted and wrappers (np.clip/np.all
    # on scalars) avoided.  Every value is computed exactly as before.
    huber_scale=huber*scale; twice_huber=2*huber; huber_squared=huber*huber; bound=AFFINE_COEFFICIENT_BOUND
    def readout(pred):
        if fit_readout:
            # Mirror affine()'s coefficient bound, or the fitter would treat any
            # overall scale as free and never move a large constant into the tree.
            weights=ones; fitted=None
            for _ in range(3 if robust else 1):
                line=_weighted_line(pred,y,weights)
                if line is None: fitted=np.full(len(y),float(np.mean(y))); break
                slope,intercept=line
                if abs(slope)>bound: slope=float(np.sign(slope))*bound; intercept=float(np.average(y-slope*pred,weights=weights))
                fitted=slope*pred+min(max(float(intercept),-bound),bound)
                if robust:
                    huber_weights=np.minimum(1.,huber_scale/np.maximum(np.abs(fitted-y),EPS))
                    if (huber_weights>=1.).all(): break  # no point is in the Huber tail: least squares is exact
                    weights=ones*huber_weights
            pred=fitted
        r=(pred-y)/scale
        if robust:
            # sum(rho**2)/2 equals the Huber loss, so least squares on rho is Huber.
            magnitude=np.abs(r); inner=magnitude<=huber
            if not inner.all(): r=np.where(inner,r,np.sign(r)*np.sqrt(np.maximum(twice_huber*magnitude-huber_squared,0.)))
        return r if np.isfinite(r).all() else None
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
        if not J.any(): break
        g=J.T@r; H=J.T@J; improved=False
        regulariser=np.diag(np.diag(H))+1e-12*np.eye(len(current)); descent=-g
        for _ in range(8):
            try: delta=np.linalg.solve(H+damping*regulariser,descent)
            except np.linalg.LinAlgError: damping*=10; continue
            candidate=np.minimum(np.maximum(current+delta,-CONSTANT_LIMIT),CONSTANT_LIMIT); rc,candidate_nodes=residual(candidate)
            candidate_cost=float(rc@rc) if rc is not None else None
            if rc is not None and candidate_cost<cost:
                gain=cost-candidate_cost; current,r,nodes,cost=candidate,rc,candidate_nodes,candidate_cost; damping=max(damping/3,1e-9); improved=True; break
            damping*=4
        if not improved or gain<=1e-10*max(cost,1e-30): break
    if cost>=initial: return tree
    tuned=with_constants(tree,current)
    # A fit that only improves by driving a value into a numeric guard would be
    # rejected at scoring; keep the untuned constants instead.
    return tree if guarded_constant_divisor(tuned) or guard_engagement([tuned],X,adfs) else tuned

def _classifier_log_loss(raw, truth, class_count):
    scales=fit_classifier_affine(raw,truth,class_count)
    scores=np.column_stack([a*raw[:,k]+b for k,(a,b) in enumerate(scales)])
    probabilities=binary_probabilities(scores[:,0]) if raw.shape[1]==1 else stable_softmax(scores)
    truth=np.asarray(np.rint(truth),int); valid=(truth>=0)&(truth<probabilities.shape[1])
    if not np.any(valid): return 0.
    losses=-np.log(np.maximum(probabilities[np.arange(len(truth))[valid],truth[valid]],EPS))
    return float(np.average(losses,weights=class_balance_weights(truth,class_count)[valid]))

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
            if len(cats[j])>=2 and affine_on:
                # Input coverage alone can leave a rare class with no rows at all.
                class_rows=class_balanced_rows(X,Y[:,j],len(cats[j]),CONSTANT_FIT_ROWS) if CLASS_BALANCE else rows
                if _tune_classifier_heads(trees,heads,row_subset(X,class_rows),row_subset(Y,class_rows)[:,j],len(cats[j]),model.adfs): changed=True
            continue
        for head in heads:
            original=trees[head]
            multiterm=READOUT_MODE=="multiterm" and affine_on
            grammar=model.mdl_operators or None  # never write an operator the run did not select
            tree=multiterm_refit(original,Xs,Ys[:,j],model.adfs,grammar) if multiterm else original
            tree=fit_tree_constants(tree,Xs,Ys[:,j],model.adfs,fit_readout=affine_on)
            # Re-solve the linear coefficients after the joint fit: the readout
            # scale and the term coefficients share one degree of freedom, and
            # leaving it to Levenberg-Marquardt lets them drift apart.
            if multiterm and tree is not original: tree=multiterm_refit(tree,Xs,Ys[:,j],model.adfs,grammar)
            # Pruning a term can leave part of an input relation exposed; keep the rule-abiding original.
            if tree is not original and RELATION_OF_FEATURE and relation_violation(tree) and not relation_violation(original): continue
            if tree is not original: trees[head]=tree; changed=True
    if changed: model.trees=trees
    return changed

# --readout multiterm (multigene GP: GPTIPS, MRGP).  The top-level +/- terms
# of a regression tree each get their own least-squares coefficient,
# b + a1*T1 + a2*T2 + ..., solved in closed form (Huber IRLS on an SVD
# least-squares solve) during constant tuning and written back into the tree
# as constants, so the export format and the scored semantics are unchanged
# and MDL charges exactly the coefficients that survive.  Terms that are
# negligible or collinear with the others are pruned (nearly identical terms
# would otherwise get huge cancelling coefficients), and at most MAX_TERMS
# are kept.  On by default since bench_diag v1 (2026-10-02, alone and with
# the residual term): more exact solves and +0.46 held-out digits, beyond
# paired seed noise.  Checkpoints from before the flag resume with affine.
READOUT_MODE = "multiterm"
MAX_TERMS = 4
GENE_CROSSOVER_RATE = 0.    # share of crossovers that exchange whole terms in multiterm mode
MULTITERM_CONDITION_LIMIT = 1e8
MULTITERM_MIN_CONTRIBUTION = 1e-9  # of the target scale

def split_terms(tree, sign=1.):
    """Top-level additive terms as (coefficient, body); body None is a pure constant."""
    op=tree[0]
    if op=="+": return split_terms(tree[1],sign)+split_terms(tree[2],sign)
    if op=="-": return split_terms(tree[1],sign)+split_terms(tree[2],-sign)
    if op=="neg": return split_terms(tree[1],-sign)
    if op=="c": return [(sign*float(tree[1]),None)]
    if op=="*" and tree[1][0]=="c" and tree[2][0]!="c": return [(sign*float(tree[1][1]),tree[2])]
    if op=="*" and tree[2][0]=="c" and tree[1][0]!="c": return [(sign*float(tree[2][1]),tree[1])]
    return [(sign,tree)]

def join_terms(terms, ops=None):
    """Inverse of split_terms for (coefficient, body) pairs with a body, written
    only with operators in ops (None allows any).  A sign goes into "-" or "neg"
    when allowed, else into the coefficient; returns None when ops cannot
    express the sum (no "+", or a coefficient other than 1 without "*")."""
    allowed=lambda op: ops is None or op in ops
    def scaled(coefficient, body):
        if coefficient==1.: return body
        return ("*",("c",coefficient),body) if allowed("*") else None
    if not terms: return ("c",0.)
    tree=None
    for coefficient,body in terms:
        magnitude=abs(coefficient); negative=coefficient<0
        if tree is None:
            if negative and allowed("neg") and scaled(magnitude,body) is not None: tree=("neg",scaled(magnitude,body))
            else: tree=scaled(coefficient,body)
        elif negative and allowed("-") and scaled(magnitude,body) is not None: tree=("-",tree,scaled(magnitude,body))
        elif allowed("+") and scaled(coefficient,body) is not None: tree=("+",tree,scaled(coefficient,body))
        else: return None
        if tree is None: return None
    return tree

def _multiterm_solve(F, y):
    """Huber-reweighted least squares y ~ F @ a + b; returns (a, b, loss, condition)."""
    centre=F.mean(axis=0); spread=F.std(axis=0); spread=np.where(spread>EPS,spread,1.)
    A=np.column_stack(((F-centre)/spread,np.ones(len(y)))); cutoff=loss_delta()*loss_scale(y)
    base=loss_base_weights(y); base=np.ones(len(y)) if base is None else base; weights=base; coefficients=None
    for _ in range(6):
        solution,_,_,singular=np.linalg.lstsq(A*weights[:,None],y*weights,rcond=1e-12)
        if coefficients is not None and np.allclose(solution,coefficients,rtol=1e-10,atol=1e-14): coefficients=solution; break
        coefficients=solution
        weights=base*np.sqrt(np.minimum(1.,cutoff/np.maximum(np.abs(A@coefficients-y),EPS)))
    a=coefficients[:-1]/spread; b=float(coefficients[-1]-np.dot(a,centre))
    condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else float("inf")
    return a,b,robust_loss(F@a+b,y),condition

def multiterm_refit(tree, X, y, adfs=None, ops=None):
    """Re-solve the coefficients of a tree's top-level terms; returns the tree unchanged unless it improves or shrinks at equal loss."""
    terms=[(coefficient,body) for coefficient,body in split_terms(tree) if body is not None]
    if len(terms)<2: return tree
    try:
        F=np.column_stack([evaluate_cached(body,X,adfs) for _,body in terms]).astype(float)
        current_raw=evaluate_cached(tree,X,adfs)
    except (ArithmeticError, IndexError, RecursionError, ValueError): return tree
    if not np.isfinite(F).all(): return tree
    a,b=affine(current_raw,y); before=robust_loss(a*current_raw+b,y)
    scale=target_scale(y); active=[index for index in range(len(terms)) if F[:,index].std()>EPS]
    if len(active)<1: return tree
    def solve(columns): return _multiterm_solve(F[:,columns],y)
    coefficients,_,loss,condition=solve(active)
    while len(active)>1:
        contribution=np.abs(coefficients)*F[:,active].std(axis=0)
        if condition<=MULTITERM_CONDITION_LIMIT and len(active)<=MAX_TERMS and contribution.min()>MULTITERM_MIN_CONTRIBUTION*scale: break
        # Drop the term whose removal costs least; ties go to the larger body.
        options=[]
        for position in range(len(active)):
            rest=active[:position]+active[position+1:]
            options.append((solve(rest)[2],-node_size(terms[active[position]][1]),position))
        _,_,position=min(options); active.pop(position)
        coefficients,_,loss,condition=solve(active)
    if not np.isfinite(coefficients).all() or np.max(np.abs(coefficients))>CONSTANT_LIMIT: return tree
    pruned=len(active)<len(terms)
    if loss>before+max(LOSS_NOISE_FLOOR,1e-12*abs(before)) or (not pruned and loss>=before): return tree
    tolerance=np.sqrt(np.finfo(float).eps)
    rebuilt=join_terms([(1. if abs(c-1.)<=tolerance else -1. if abs(c+1.)<=tolerance else float(c),terms[index][1]) for c,index in zip(coefficients,active)],ops)
    if rebuilt is None or rebuilt==tree or guarded_constant_divisor(rebuilt) or guard_engagement([rebuilt],X,adfs): return tree
    return rebuilt

def gene_crossover(left, right, max_nodes, max_depth, ops=None):
    """Multigene crossover: add one of right's top-level terms to left, or swap it for one of left's."""
    left_terms=[(c,body) for c,body in split_terms(left) if body is not None]
    right_terms=[(c,body) for c,body in split_terms(right) if body is not None]
    if not right_terms: return left
    for _ in range(4):
        donor=rng.choice(right_terms); terms=list(left_terms)
        if terms and (len(terms)>=MAX_TERMS or rng.random()<.5): terms[rng.randrange(len(terms))]=donor
        else: terms.append(donor)
        joined=join_terms(terms,ops)
        if joined is None: continue
        child=simplify_tree(joined)
        if child!=left and node_size(child)<=max_nodes and node_depth(child)<=max_depth: return child
    return left
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
                "mdl_operators":model.mdl_operators,"mdl_feature_count":model.mdl_feature_count,"adfs":checkpoint_adfs(model),"founder_ids":model.founder_ids,"birth_generation":model.birth_generation,"history":list(model.history)}

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
            return float(np.average(np.where(valid,-np.log(np.where(truth==1,probability,1-probability)+EPS),-np.log(EPS)),weights=class_balance_weights(truth,2)))
        if self.likelihood_kind=="categorical":
            probability=stable_softmax(raw); truth=np.asarray(np.rint(Y[:,0]),int); valid=(truth>=0)&(truth<probability.shape[1]); row=np.arange(len(truth))
            return float(np.average(np.where(valid,-np.log(np.maximum(probability[row,np.clip(truth,0,probability.shape[1]-1)],EPS)),-np.log(EPS)),weights=class_balance_weights(truth,probability.shape[1])))
        return float("inf")

    def _cached_likelihood_energy(self, model, X, Y, cats):
        """_likelihood_energy, memoized: the catalog carries over between
        generations, so the same particles were re-predicted every update."""
        if not (isinstance(X,np.ndarray) and isinstance(Y,np.ndarray)): return self._likelihood_energy(model,X,Y,cats)
        key=(tree_digest(model.trees,model.scales,adf_signature(model.trees,model.adfs)),array_digest(X),array_digest(Y),
             repr(cats),self.likelihood_kind,self.noise_scale,self.student_t_df)
        hit=_LIKELIHOOD_ENERGY_CACHE.get(key)
        if hit is None:
            hit=self._likelihood_energy(model,X,Y,cats)
            if len(_LIKELIHOOD_ENERGY_CACHE)>=20000: _LIKELIHOOD_ENERGY_CACHE.clear()
            _LIKELIHOOD_ENERGY_CACHE[key]=hit
        return hit

    def _energies(self, models, X=None, Y=None, cats=None):
        if X is not None and self.likelihood_kind!="legacy":
            return np.asarray([self._cached_likelihood_energy(model,X,Y,cats)+self.complexity_prior*model_complexity(model) for model in models])
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
        # A cumulative sum that rounds to just below 1 would index one past the end.
        indices=np.minimum(np.searchsorted(np.cumsum(self.weights),positions,side="right"),len(self.particles)-1)
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
            correct=total=0; log_losses=[]; confidences=[]; briers=[]; recalls=[]
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
                recalls.extend(float(np.mean(chosen[valid&(truth==label)]==label)) for label in np.unique(truth[valid]))
                row=np.arange(len(truth))[valid]
                log_losses.extend(-np.log(np.maximum(probabilities[row,truth[valid]],EPS)))
                confidences.extend(np.max(probabilities[valid],axis=1))
                one_hot=np.eye(class_count)[truth[valid]]; briers.extend(np.sum((probabilities[valid]-one_hot)**2,axis=1))
            if total:
                result["classification"]={
                    "accuracy":correct/total,
                    "balanced_accuracy":float(np.mean(recalls)),
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
                parts.append(f"class accuracy={classification['accuracy']:.1%} (balanced {classification.get('balanced_accuracy',classification['accuracy']):.1%}), log loss={classification['log_loss']:.4g}, "
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
_LIKELIHOOD_ENERGY_CACHE={}
def assess_particle_cached(particle, X, Y, affine_on, cats):
    key=(tree_digest(particle.trees,adf_signature(particle.trees,particle.adfs)),interned_grammar(particle.mdl_operators),particle.mdl_feature_count,
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
        for key in ("base_exploration","decay","floor","temperature","exploration","pressure_exploration","op_alpha","feature_alpha","op_draw","feature_draw","entropy","depth_alpha","constant_abs_sum","constant_weight","constant_scale"):
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
                for head in heads: trees[head]=admissible_random_tree(n_features,ops,max_nodes,max_depth,proposal=bank,adfs=adfs)
        return (trees,scales,sources) if return_sources else (trees,scales)
    particle=bayes.sample_particle()
    if particle_mode!="grammar" and particle is not None and len(particle.trees)==n_outputs and rng.random()<.65:
        trees=[mutate(tree,n_features,ops,max_nodes,max_depth,proposal=bayes,adfs=adfs) for tree in particle.trees]
        return (trees,list(particle.scales),[particle]) if return_sources else (trees,list(particle.scales))
    trees=[admissible_random_tree(n_features,ops,max_nodes,max_depth,proposal=bayes,adfs=adfs) for _ in range(n_outputs)]
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
def prune_mutate(t):
    """Collapse one inner node, anywhere in the tree, into one of its children
    or a constant (refitted with the child), so the tree always gets smaller.

    shrink and hoist only cut at the root; the simplifier island uses this
    to remove the redundant middle of an expression."""
    paths=[path for path in subtree_paths(t) if subtree_at(t,path)[0] not in ("x","c","arg")]
    if not paths: return t
    path=rng.choice(paths); node=subtree_at(t,path)
    replacement=("c",1.) if rng.random()<.25 else rng.choice(node[1:])
    return simplify_tree(replace_subtree(t,path,replacement))
def jump_mutate(t, n_features, ops, max_nodes, max_depth):
    """Wrap a subtree in a whole jump block: mod(s,c), floordiv(s,c) or
    if_else(gt(x,c), s, s') with s' a mutated copy of s.

    A discontinuous target gives no partial credit for a comparison or a
    modulo built one node at a time; offering the block as one move lets the
    jump-constant scan place c right away when the child is tuned."""
    forms=[form for form in ("mod","floordiv") if form in ops]
    if "if_else" in ops and "gt" in ops: forms.append("branch")
    paths=[path for path in subtree_paths(t) if subtree_at(t,path)[0]!="c"]
    if not forms or not paths: return t
    path=rng.choice(paths); target=subtree_at(t,path); form=rng.choice(forms)
    if form=="branch":
        other=mutate(target,n_features,ops,max(1,node_size(target)+2),max(1,node_depth(target)+1))
        block=("if_else",("gt",("x",rng.randrange(n_features)),("c",rng.uniform(-1,1))),target,other)
    else: block=(form,target,("c",float(rng.choice(_SIMPLE_NUMBERS))))
    child=replace_subtree(t,path,block)
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t

# Each squashing operator written as a + b*tanh(k*z), matched in level,
# range and slope at z=0 (exact for sigmoid; erf to within 2% of its range).
_SQUASH_FORMS = {"sigmoid":(.5,.5,.5), "tanh":(0.,1.,1.), "erf":(0.,1.,2./math.sqrt(math.pi))}
def _scaled(tree, factor):
    return tree if abs(factor-1.)<1e-12 else ("*",("c",float(factor)),tree)
def squash_swap_mutate(t, ops, max_nodes, max_depth):
    """Swap one sigmoid/tanh/erf for another, rewritten to the same level,
    range and slope: sigmoid(z) -> 0.5+0.5*erf(0.443*z), tanh(z) -> erf(0.886*z).

    A plain point swap turns x*sigmoid(kx) into x*erf(kx), an even function
    (R^2 0.23 against GELU), because sigmoid sits on 0.5 and erf on 0; no
    single move then reaches x*(1+erf(kx)).  This move lands next to the
    parent, so the constant fitter only refines it."""
    paths=[path for path in subtree_paths(t) if subtree_at(t,path)[0] in _SQUASH_FORMS]
    if not paths or "*" not in ops: return t
    path=rng.choice(paths); node=subtree_at(t,path); source=node[0]
    targets=[op for op in _SQUASH_FORMS if op!=source and op in ops]
    if not targets: return t
    target=rng.choice(targets)
    (a_f,b_f,k_f),(a_g,b_g,k_g)=_SQUASH_FORMS[source],_SQUASH_FORMS[target]
    offset,scale=a_f-b_f*a_g/b_g,b_f/b_g
    block=_scaled((target,_scaled(node[1],k_f/k_g)),scale)
    if abs(offset)>1e-12:
        if "+" not in ops: return t
        block=("+",("c",float(offset)),block)
    child=simplify_tree(replace_subtree(t,path,block))
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t
def activation_moves_apply(ops, kind):
    """Whether squash_swap/smooth_swap/gate_mutate can ever change a tree built from ops."""
    ops=set(ops); squash=ops&set(_SQUASH_FORMS)
    if "*" not in ops: return False
    if kind=="squash": return len(squash)>=2
    if kind=="gate": return bool(squash)
    return bool(("relu" in ops and ops&{"softplus","sigmoid"}) or ("abs" in ops and ops&{"tanh","erf"})
                or ("sign" in ops and "tanh" in ops) or {"softplus","relu"}<=ops)
_SMOOTH_SHARPNESS = 4.
def smooth_swap_mutate(t, ops, max_nodes, max_depth):
    """Trade a hard piece for its smooth relative, or back, as one move:
    relu(z) <-> softplus(4z)/4 or z*sigmoid(4z); abs(z) -> z*tanh(4z);
    sign(z) -> tanh(4z).  The constant 4 is fitted afterwards.

    A smooth target (SiLU, GELU, softplus) is otherwise approximated by more
    and more relu kinks, each a local optimum the next kink only refines."""
    c=_SMOOTH_SHARPNESS*float(np.exp(rng.gauss(0,.3)))
    choices=[]
    for path in subtree_paths(t):
        node=subtree_at(t,path); op=node[0]
        if op=="relu":
            if "softplus" in ops and "*" in ops: choices.append((path,_scaled(("softplus",_scaled(node[1],c)),1./c)))
            if "sigmoid" in ops and "*" in ops: choices.append((path,("*",node[1],("sigmoid",_scaled(node[1],c)))))
        elif op=="abs":
            for squash in ("tanh","erf"):
                if squash in ops and "*" in ops: choices.append((path,("*",node[1],(squash,_scaled(node[1],c)))))
        elif op=="sign" and "tanh" in ops and "*" in ops: choices.append((path,("tanh",_scaled(node[1],c))))
        elif op=="softplus" and "relu" in ops: choices.append((path,("relu",node[1])))
    if not choices: return t
    path,block=rng.choice(choices); child=simplify_tree(replace_subtree(t,path,block))
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t
def gate_mutate(t, n_features, ops, max_nodes, max_depth):
    """Multiply a subtree by a fitted soft gate as one move: s*sigmoid(c*u),
    s*(1+erf(c*u)) or s*(1+tanh(c*u)), with u = s itself or an input.

    x*sigmoid(x) (SiLU), x*(1+erf(x/sqrt 2)) (GELU) and smooth switches
    between two regimes are gates; built a node at a time, the half-built
    gate scores worse than the ungated parent and is discarded."""
    gates=[op for op in _SQUASH_FORMS if op in ops]
    if "*" not in ops or not gates: return t
    paths=[path for path in subtree_paths(t) if subtree_at(t,path)[0]!="c"]
    if not paths: return t
    path=rng.choice(paths); target=subtree_at(t,path); gate=rng.choice(gates)
    driver=target if rng.random()<.5 else ("x",rng.randrange(n_features))
    c=float(rng.choice((-1.,1.))*np.exp(rng.gauss(0,.5)))
    factor=(gate,("*",("c",c),driver))
    if gate!="sigmoid":
        if "+" not in ops: return t
        factor=("+",("c",1.),factor)
    child=simplify_tree(replace_subtree(t,path,("*",target,factor)))
    return child if node_size(child)<=max_nodes and node_depth(child)<=max_depth else t

# Off by default: it did not solve the linear-system benchmark and was neutral
# to harmful elsewhere (4-way ablation).  Set > 0 to re-enable.
BILINEAR_MUTATION_WEIGHT = 0.
# Initial weight of jump_mutate; the portfolio adapts it like the others.
# It only acts when mod/floordiv or if_else+gt are in the grammar.
JUMP_MUTATION_WEIGHT = 1.
# Initial weights of squash_swap_mutate, smooth_swap_mutate and gate_mutate.
# Each acts only when its operators (sigmoid/tanh/erf, relu/softplus/abs/sign)
# are in the grammar.
SQUASH_SWAP_WEIGHT = 1.
SMOOTH_SWAP_WEIGHT = 1.
GATE_MUTATION_WEIGHT = 1.
# Semantic backpropagation mutation (RDO; Pawlak, Wieloch and Krawiec 2015).
# The desired raw output of the parent tree, T* = (y - b) / a from its affine
# readout, is inverted down a random path to the values the subtree there
# should output; the subtree is replaced by the library entry that best
# matches them after an affine fit.  The library is every input, every unary
# operator of an input and every binary operator of two inputs (bounded),
# plus the fragment library.  "exact" inverts only the single-valued
# operators (+ - * / neg exp log tanh sigmoid); "generic" also inverts every
# other operator numerically per row, taking the solution nearest the
# subtree's current value (the branch it is already on).
BACKPROP_MUTATION_WEIGHT = 0.
BACKPROP_INVERSE = "generic"
BACKPROP_LIBRARY_LIMIT = 2000
BACKPROP_MIN_VALID = .5            # share of rows whose desired value must be defined
_BACKPROP_CONTEXT = {"X":None,"desired":None,"fragments":()}
_BACKPROP_LIBRARY_CACHE = {}
_INVERSE_GRID = np.concatenate([-np.logspace(-6,6,121)[::-1],[0.],np.logspace(-6,6,121)])
_INVERSE_WIDE, _INVERSE_NARROW = np.linspace(-4,4,161), np.linspace(-.25,.25,51)  # offsets around the current value, in spans
_LIBRARY_EXCLUDED = {"python_rng","perlin_noise","seqsum","seqprod","cat","x_at_pos_y"}

def set_backprop_context(X=None, desired=None, fragments=()):
    _BACKPROP_CONTEXT.update(X=X,desired=desired,fragments=tuple(fragments))

def exact_inverse(op, k, desired, args):
    """Desired value of child k of op, or None when op is not single-valued there."""
    d=desired
    with np.errstate(all="ignore"):
        if op=="+": return d-args[1-k]
        if op=="-": return d+args[1] if k==0 else args[0]-d
        if op=="*":
            other=args[1-k]; return np.where(np.abs(other)>EPS,d/other,np.nan)
        if op=="/":
            if k==0: return d*np.where(np.abs(args[1])<EPS,EPS,args[1])
            return np.where(np.abs(d)>EPS,args[0]/d,np.nan)
        if op=="neg": return -d
        if op=="exp": return np.where(d>0,np.log(d),np.nan)
        if op=="log":  # log(|x|+eps): keep the sign the child already has
            magnitude=np.exp(np.clip(d,-700,700))-EPS
            return np.where(magnitude>0,np.where(args[0]<0,-magnitude,magnitude),np.nan)
        if op=="tanh": return np.where(np.abs(d)<1,np.arctanh(np.clip(d,-1+1e-15,1-1e-15)),np.nan)
        if op=="sigmoid": return np.where((d>0)&(d<1),np.log(d/(1-d)),np.nan)
    return None

def numeric_inverse(op, k, desired, args):
    """Per-row solution z of op(..., z at k, ...) = desired nearest the current child value.

    With --fit-backend auto the compiled kernel (afpo_lib/fitcore.pyx) runs the
    same grid, bisection and dip search row by row: about 5x faster, equal to
    round-off (bit for bit for the exactly compiled operators)."""
    core=compiled_fitter() if FIT_BACKEND=="auto" else None
    code=None if core is None else core.OPERATOR_CODES.get(op)
    if code is not None and len(args)<=3:
        return core.numeric_inverse(code,k,np.broadcast_to(np.asarray(desired,float),(len(args[k]),)),
                                    [np.asarray(value,float) for value in args],_INVERSE_WIDE,_INVERSE_NARROW,_INVERSE_GRID)
    current=np.asarray(args[k],float); n=len(current)
    span=np.maximum(np.abs(current),1.)
    grid=np.concatenate([current[:,None]+span[:,None]*_INVERSE_WIDE[None,:],current[:,None]+span[:,None]*_INVERSE_NARROW[None,:],np.broadcast_to(_INVERSE_GRID,(n,len(_INVERSE_GRID)))],axis=1)
    grid.sort(axis=1); G=grid.shape[1]
    def apply(z):
        values=[np.repeat(np.asarray(a,float),z.shape[1]) if i!=k else z.ravel() for i,a in enumerate(args)]
        return op_eval(op,values).reshape(z.shape)
    try: f=apply(grid)-desired[:,None]
    except (ArithmeticError, IndexError, ValueError): return None
    f=np.where(np.isfinite(f),f,np.nan)
    crossing=(np.sign(f[:,:-1])*np.sign(f[:,1:])<=0)&np.isfinite(f[:,:-1])&np.isfinite(f[:,1:])
    distance=np.where(crossing,np.abs(.5*(grid[:,:-1]+grid[:,1:])-current[:,None]),np.inf)
    best=np.argmin(distance,axis=1); found=np.isfinite(distance[np.arange(n),best])
    lo=grid[np.arange(n),best]; hi=grid[np.arange(n),best+1]
    flo=f[np.arange(n),best]; fhi=f[np.arange(n),best+1]
    with np.errstate(all="ignore"):
        for _ in range(30):  # bisection keeps a sign change bracketed, so jumps cannot fool it
            mid=.5*(lo+hi); fmid=apply(mid[:,None])[:,0]-desired
            left=np.sign(flo)*np.sign(fmid)<=0
            hi=np.where(left,mid,hi); fhi=np.where(left,fmid,fhi); lo=np.where(left,lo,mid); flo=np.where(left,flo,fmid)
        root=np.where(np.abs(flo)<=np.abs(fhi),lo,hi)
        tolerance=1e-6*np.maximum(np.abs(desired),1.)
        # Two roots inside one grid cell (or a touch at an extremum, sin
        # reaching 1) show no sign change: also minimise |f| around the
        # nearest dip of |f| and keep whichever valid root is closer.
        magnitude=np.where(np.isfinite(f),np.abs(f),np.inf)
        padded=np.pad(magnitude,((0,0),(1,1)),constant_values=np.inf)
        dip=(magnitude<=padded[:,:-2])&(magnitude<padded[:,2:])&np.isfinite(magnitude)  # strict on one side: duplicate grid points are not dips
        near=np.where(dip,np.abs(grid-current[:,None]),np.inf); index=np.argmin(near,axis=1)
        a_=grid[np.arange(n),np.maximum(index-1,0)]; b_=grid[np.arange(n),np.minimum(index+1,G-1)]
        for _ in range(60):
            m1=a_+(b_-a_)/3; m2=b_-(b_-a_)/3
            f1=np.abs(apply(m1[:,None])[:,0]-desired); f2=np.abs(apply(m2[:,None])[:,0]-desired)
            smaller=np.where(np.isfinite(f1),f1,np.inf)<=np.where(np.isfinite(f2),f2,np.inf)
            b_=np.where(smaller,m2,b_); a_=np.where(smaller,a_,m1)
        dip_root=.5*(a_+b_)
        crossing_ok=found&(np.abs(apply(root[:,None])[:,0]-desired)<=tolerance)
        dip_ok=np.abs(apply(dip_root[:,None])[:,0]-desired)<=tolerance
        closer=dip_ok&(~crossing_ok|(np.abs(dip_root-current)<np.abs(root-current)))
        root=np.where(closer,dip_root,root)
        residual=np.abs(apply(root[:,None])[:,0]-desired)
    return np.where(residual<=tolerance,root,np.nan)

def child_desired(op, k, desired, args):
    exact=exact_inverse(op,k,desired,args)
    if exact is not None or BACKPROP_INVERSE=="exact": return exact
    return numeric_inverse(op,k,desired,args)

def backprop_library(X, ops, adfs=None):
    """(trees, semantics matrix) of small candidate subtrees, cached per data and grammar."""
    key=(array_digest(X),tuple(ops))
    cached=_BACKPROP_LIBRARY_CACHE.get(key)
    if cached is not None: return cached
    n_features=X.shape[1]; trees=[("x",i) for i in range(n_features)]
    usable=[op for op in ops if op in OPS and op not in _LIBRARY_EXCLUDED]
    trees+=[(op,("x",i)) for op in usable if OPS[op][0]==1 for i in range(n_features)]
    pairs=[(op,("x",i),("x",j)) for op in usable if OPS[op][0]==2 for i in range(n_features) for j in range(n_features) if i!=j]
    local=random.Random(len(X)*1009+n_features)  # fixed subsample; the run's RNG streams are untouched
    room=max(0,BACKPROP_LIBRARY_LIMIT-len(trees))
    trees+=pairs if len(pairs)<=room else local.sample(pairs,room)
    kept=[]; columns=[]; seen=set()
    for tree in trees:
        try: values=np.asarray(evaluate_cached(tree,X,adfs),float)
        except (ArithmeticError, IndexError, RecursionError, ValueError): continue
        if not np.isfinite(values).all() or values.std()<=EPS: continue
        signature=np.round(values,10).tobytes()
        if signature in seen: continue
        seen.add(signature); kept.append(tree); columns.append(values)
    result=(kept,np.column_stack(columns) if columns else np.zeros((len(X),0)))
    if len(_BACKPROP_LIBRARY_CACHE)>8: _BACKPROP_LIBRARY_CACHE.clear()
    _BACKPROP_LIBRARY_CACHE[key]=result
    return result

def _best_affine_match(L, D):
    """Index, slope, intercept and SSE of the column of L best matching D affinely."""
    Lc=L-L.mean(axis=0); Dc=D-D.mean(); var=np.einsum("ij,ij->j",Lc,Lc)
    with np.errstate(all="ignore"):
        slope=np.where(var>EPS,(Lc.T@Dc)/var,0.)
        sse=np.where(var>EPS,float(Dc@Dc)-slope*slope*var,np.inf)
    index=int(np.argmin(sse)); intercept=float(D.mean()-slope[index]*L[:,index].mean())
    return index,float(slope[index]),intercept,float(sse[index])

# Residual-driven term (boosting-style): add to the parent the library entry
# that best matches what its readout still misses, T + c*L with L fitted to
# (y - b)/a - T.  Shares the backpropagation library and context.  On by
# default since bench_diag v1 (+0.48 held-out digits, beyond seed noise).
RESIDUAL_TERM_WEIGHT = 1.

def residual_term_mutate(t, ops, max_nodes, max_depth, adfs=None):
    X=_BACKPROP_CONTEXT["X"]; target=_BACKPROP_CONTEXT["desired"]
    if X is None or target is None or len(target)!=len(X): return t
    try: residual=np.asarray(target,float)-np.asarray(evaluate_cached(t,X,adfs),float)
    except (ArithmeticError, IndexError, RecursionError, ValueError): return t
    valid=np.isfinite(residual)
    if valid.sum()<5 or residual[valid].std()<=EPS: return t
    trees,L=backprop_library(X,ops,adfs)
    if not L.shape[1]: return t
    index,slope,_,sse=_best_affine_match(L[valid],residual[valid])
    centred=residual[valid]-residual[valid].mean()
    if not np.isfinite(sse) or sse>=float(centred@centred)*(1-1e-6) or abs(slope)<=EPS: return t
    if "+" not in ops: return t
    term=trees[index] if abs(slope-1.)<=1e-9 or "*" not in ops else ("*",("c",slope),trees[index])
    for addition in (term,trees[index]):
        child=simplify_tree(("+",t,addition))
        if child!=t and node_size(child)<=max_nodes and node_depth(child)<=max_depth: return child
    return t

# --forbid-nesting (PySR's nested_constraints, simplified): outer>inner pairs
# such as exp>exp or sin>sin.  Offspring that break a rule are redrawn, and
# any model that still contains one is infeasible.
NESTING_RULES = frozenset()

def nesting_violation(tree, enclosing=frozenset(), opaque=frozenset()):
    """The first forbidden outer>inner pair in the tree, or ''.  Subtrees in
    opaque are leaves (see Model.opaque): they are not looked into."""
    if not NESTING_RULES or tree[0] in ("x","c","arg") or (opaque and tree in opaque): return ""
    for outer in enclosing:
        if (outer,tree[0]) in NESTING_RULES: return f"{outer}>{tree[0]}"
    inner=enclosing|{tree[0]}
    for child in tree[1:]:
        found=nesting_violation(child,inner,opaque)
        if found: return found
    return ""

# --units (dimensional analysis, as in PySR): units per input column, e.g.
# "x=m,t=s,F=kg*m/s^2".  Trees whose subexpressions add, compare or feed
# a transcendental function with inconsistent units are infeasible, and such
# offspring are redrawn.  Constants are unit wildcards (they may carry any
# unit), as is the affine readout, so the output unit is not constrained.
# Columns without a unit are wildcards too.
UNIT_FEATURES = {}           # feature index -> unit vector (tuple over UNIT_BASES)
UNIT_SPEC = ""
UNIT_BASES = ()
WILDCARD = None
_COMPARE_OPS = {"gt","lt","gte","lte","eq","ne"}
_UNIT_PRESERVING = {"abs","neg","relu","leaky_relu","floor","ceil","round","int","perceptronReLU1"}
_POWERS = {"square":2,"cube":3,**{f"pow{k}":k for k in range(4,11)},"sqrt":.5,"inv":-1,**{f"root{k}":1/k for k in range(3,11)}}

def parse_unit(text):
    """'kg*m/s^2' -> {base: exponent}; '' or '1' is dimensionless."""
    text=str(text).strip().replace(" ","")
    result={}
    if text in ("","1","-"): return result
    sign=1; token=""
    def flush(token, sign):
        if not token or token=="1": return
        base,_,exponent=token.partition("^")
        if not base.replace("_","").isalpha(): raise ValueError(f"Bad unit {base!r}")
        value=float(eval_fraction(exponent)) if exponent else 1.
        result[base]=result.get(base,0.)+sign*value
    depth=0
    for char in text:
        depth+=(char=="(")-(char==")")
        if char in "*/" and depth==0:  # m^(1/2): a fraction exponent stays in its token
            flush(token,sign); token=""; sign=1 if char=="*" else -1
        else: token+=char
    flush(token,sign)
    return {base:value for base,value in result.items() if value}

def eval_fraction(text):
    numerator,_,denominator=text.strip("()").partition("/")
    return float(numerator)/(float(denominator) if denominator else 1.)

def configure_units(spec, feature_names):
    """Set UNIT_FEATURES from 'name=unit,...'; returns {name: unit dict}."""
    global UNIT_FEATURES,UNIT_BASES,UNIT_SPEC
    UNIT_SPEC=str(spec or ""); parsed={}
    for item in str(spec or "").split(","):
        if not item.strip(): continue
        name,separator,unit=item.partition("=")
        if not separator: raise ValueError(f"Bad unit entry {item.strip()!r}: use column=unit")
        if name.strip() not in feature_names: raise ValueError(f"Unknown column {name.strip()!r} in --units")
        parsed[name.strip()]=parse_unit(unit)
    UNIT_BASES=tuple(sorted({base for unit in parsed.values() for base in unit}))
    UNIT_FEATURES={feature_names.index(name):tuple(unit.get(base,0.) for base in UNIT_BASES) for name,unit in parsed.items()}
    return parsed

def tree_units(tree, opaque=frozenset()):
    """(unit or WILDCARD, violation text or '')."""
    if tree[0]=="x": return UNIT_FEATURES.get(tree[1],WILDCARD),""
    if tree[0] in ("c","arg") or (opaque and tree in opaque): return WILDCARD,""
    children=[]
    for child in tree[1:]:
        unit,violation=tree_units(child,opaque)
        if violation: return None,violation
        children.append(unit)
    op=tree[0]; zero=tuple(0. for _ in UNIT_BASES)
    def same(units):
        concrete=[u for u in units if u is not WILDCARD]
        if any(not np.allclose(u,concrete[0]) for u in concrete[1:]): return None
        return concrete[0] if concrete else WILDCARD
    def dimensionless(unit): return unit is WILDCARD or np.allclose(unit,zero)
    if op in ("+","-","max","min","delta","hypot","distance_2","harmonic"):
        unit=same(children); return (unit,"") if unit is not None else (None,f"{op} of unlike units")
    if op in ("mod","quantize","copysign","geometric"):
        if op=="geometric":
            return (WILDCARD if WILDCARD in children else tuple(.5*(a+b) for a,b in zip(*children))),""
        if op=="copysign": return children[0],""
        unit=same(children); return (unit,"") if unit is not None else (None,f"{op} of unlike units")
    if op in _COMPARE_OPS:
        return (WILDCARD,"") if same(children) is not None else (None,f"{op} of unlike units")
    if op=="if_else":
        unit=same(children[1:]); return (unit,"") if unit is not None else (None,"if_else branches of unlike units")
    if op=="lerp":
        unit=same(children[:2]); ok=dimensionless(children[2]) and unit is not None
        return (unit,"") if ok else (None,"lerp of unlike units")
    if op in ("if_in_range","if_out_of_range"):
        unit=same(children); return (unit,"") if unit is not None else (None,f"{op} of unlike units")
    if op=="*": return (WILDCARD if WILDCARD in children else tuple(a+b for a,b in zip(*children))),""
    if op=="/": return (WILDCARD if WILDCARD in children else tuple(a-b for a,b in zip(*children))),""
    if op=="floordiv": return (WILDCARD if WILDCARD in children else tuple(a-b for a,b in zip(*children))),""
    if op in _UNIT_PRESERVING: return children[0],""
    if op in _POWERS: return (WILDCARD if children[0] is WILDCARD else tuple(_POWERS[op]*a for a in children[0])),""
    if op=="pow":
        if not dimensionless(children[1]): return None,"pow exponent has units"
        exponent=tree[2]
        if children[0] is WILDCARD or dimensionless(children[0]): return children[0],""
        return (tuple(float(exponent[1])*a for a in children[0]),"") if exponent[0]=="c" else (None,"pow of a dimensional base by a non-constant exponent")
    if op=="atan2":
        return (zero,"") if same(children) is not None else (None,"atan2 of unlike units")
    if op=="sign": return zero,""
    if op=="exp_decay":
        # exp(-x*y): only the product must be dimensionless, so exp_decay(t, k)
        # with t in s and k in 1/s is fine (as exp(neg(t*k)) already is).
        if WILDCARD in children or dimensionless(tuple(a+b for a,b in zip(*children))): return zero,""
        return None,"exp_decay of a dimensional product"
    if op in ("round2","floor2","ceil2","lshift","rshift"):
        # x rounded to y places / shifted by y bits keeps x's unit; y is a count.
        return (children[0],"") if dimensionless(children[1]) else (None,f"{op} count has units")
    if op=="perceptronReLU2":
        unit=same(children); return (unit,"") if unit is not None else (None,"perceptronReLU2 of unlike units")
    # Everything else (exp, log, sin, tanh, sigmoid, erf, ...) needs dimensionless arguments.
    if all(dimensionless(unit) for unit in children): return zero,""
    return None,f"{op} of a dimensional argument"

def unit_violation(tree, opaque=frozenset()):
    if not UNIT_FEATURES or tree[0] in ("x","c","arg"): return ""
    return tree_units(tree,opaque)[1]

# --input-relations "HH1,MM1;HH2,MM2": input columns that belong together.
# Inside a relation its columns combine freely; across relations (and with
# inputs in no relation, each its own one-column relation) only *complete*
# subexpressions combine, ones that read every column of their relation.  So
# (60*HH2+MM2)-(60*HH1+MM1) is allowed but HH2-HH1 or HH1*MM2 is not: the
# relation is a module whose result may meet other modules' results, while
# its variables never meet outside variables directly.  A categorical input
# is one column (any of its one-hot features reads it).  Trees that break the
# rule are infeasible, and offspring that break it are redrawn.
INPUT_RELATIONS = ()           # tuple of column-name tuples, as given
RELATION_SPEC = ""
RELATION_OF_FEATURE = {}       # feature index -> (relation index, column name)
RELATION_COLUMNS = {}          # relation index -> frozenset of its column names

def parse_relations(spec):
    """'a,b;c,d' (or a list of such strings) -> (('a','b'),('c','d'))."""
    items=[spec] if isinstance(spec,str) else list(spec or ())
    relations=[]
    for item in items:
        for part in str(item).split(";"):
            columns=tuple(dict.fromkeys(column.strip() for column in part.split(",") if column.strip()))
            if not columns: continue
            if len(columns)<2: raise ValueError(f"A relation needs at least two columns: {part.strip()!r}")
            relations.append(columns)
    seen={}
    for index,columns in enumerate(relations):
        for column in columns:
            if column in seen: raise ValueError(f"Column {column!r} is in two relations")
            seen[column]=index
    return tuple(relations)

def format_relations(relations): return ";".join(",".join(columns) for columns in relations)

def configure_input_relations(spec, feature_names, source_columns=(), types=()):
    """Map every feature of a related input column to its relation."""
    global INPUT_RELATIONS,RELATION_SPEC,RELATION_OF_FEATURE,RELATION_COLUMNS
    relations=parse_relations(spec)
    INPUT_RELATIONS=relations; RELATION_SPEC=format_relations(relations); RELATION_OF_FEATURE={}; RELATION_COLUMNS={}
    if not relations: return
    kinds=dict(zip(source_columns,types))
    for index,columns in enumerate(relations):
        RELATION_COLUMNS[index]=frozenset(columns)
        for column in columns:
            kind=kinds.get(column)
            if kind is None and source_columns: raise ValueError(f"Input relation names unknown column {column!r}")
            if kind is not None and kind not in (1,2): raise ValueError(f"Input relation column {column!r} is not an input")
            features=[position for position,name in enumerate(feature_names) if name==column or (kind!=1 and name.startswith(f"{column}="))]
            if not features: raise ValueError(f"Input relation column {column!r} has no encoded feature")
            for position in features: RELATION_OF_FEATURE[position]=(index,column)

def _relation_summary(tree, opaque=frozenset()):
    """({relation: columns read}, violation) of a subtree; unrelated inputs are complete one-column relations."""
    if tree[0]=="x":
        found=RELATION_OF_FEATURE.get(tree[1])
        return ({("feature",tree[1]):frozenset()} if found is None else {found[0]:frozenset((found[1],))}),""
    if tree[0] in ("c","arg"): return {},""
    # An opaque subtree stands for the staged feature it replaced: a complete one-column relation.
    if opaque and tree in opaque: return {("opaque",tree):frozenset()},""
    children=[]
    for child in tree[1:]:
        summary,violation=_relation_summary(child,opaque)
        if violation: return None,violation
        children.append(summary)
    merged={}
    for summary in children:
        for relation,columns in summary.items(): merged[relation]=merged.get(relation,frozenset())|columns
    if len(merged)>1:
        for summary in children:
            if len(summary)!=1: continue
            (relation,columns),=summary.items()
            if relation in RELATION_COLUMNS and not columns>=RELATION_COLUMNS[relation]:
                return None,f"{tree[0]} mixes part of relation {','.join(INPUT_RELATIONS[relation])} with other inputs"
    return merged,""

def relation_violation(tree, opaque=frozenset()):
    if not RELATION_OF_FEATURE or tree[0] in ("x","c","arg"): return ""
    return _relation_summary(tree,opaque)[1]

def structural_violation(tree, opaque=frozenset()):
    """The first --forbid-nesting, --units or --input-relations violation of a tree, or ''."""
    return nesting_violation(tree,opaque=opaque) or unit_violation(tree,opaque) or relation_violation(tree,opaque)

def admissible_random_tree(n_features, ops, max_nodes, max_depth, attempts=25, **kwargs):
    """A random tree that passes --forbid-nesting, --units and --input-relations, when one turns up within a few draws."""
    tree=random_tree(n_features,ops,max_nodes,max_depth,**kwargs)
    for _ in range(attempts):
        if not structural_violation(tree): break
        tree=random_tree(n_features,ops,max_nodes,max_depth,**kwargs)
    return tree

def regression_head_outputs(cats):
    """Tree index -> output column for regression heads (classifier heads have no desired value)."""
    targets,_=classification_layout(cats)
    return {heads[0]:j for j,heads in enumerate(targets) if cats[j] is None}

def backprop_desired(model, head, Y, head_outputs):
    """Desired raw tree output (y - b) / a from the model's affine readout, or None when a is ~0."""
    j=head_outputs.get(head)
    if j is None or head>=len(model.scales): return None
    a,b=model.scales[head]; y=np.asarray(Y[:,j],float)
    if not np.isfinite(a) or abs(a)<=1e-9*max(target_scale(y),EPS): return None
    return (y-b)/a

def backprop_mutate(t, ops, max_nodes, max_depth, adfs=None):
    """Replace a random subtree by the library entry closest to its back-propagated desired output."""
    X=_BACKPROP_CONTEXT["X"]; target=_BACKPROP_CONTEXT["desired"]
    if X is None or target is None or len(target)!=len(X): return t
    path=rng.choice(subtree_paths(t)); desired=np.asarray(target,float); node=t
    try:
        for k in path:
            args=[np.asarray(evaluate_cached(child,X,adfs),float) for child in node[1:]]
            desired=child_desired(node[0],k,desired,args)
            if desired is None: return t
            node=node[k+1]
        current=np.asarray(evaluate_cached(node,X,adfs),float)
    except (ArithmeticError, IndexError, RecursionError, ValueError): return t
    valid=np.isfinite(desired)
    if valid.sum()<max(5,BACKPROP_MIN_VALID*len(desired)): return t
    centre=np.median(desired[valid]); spread=np.median(np.abs(desired[valid]-centre))+EPS
    valid&=np.abs(np.where(valid,desired,centre)-centre)<=50*spread  # inversions near a pole explode
    if valid.sum()<max(5,BACKPROP_MIN_VALID*len(desired)): return t
    trees,L=backprop_library(X,ops,adfs)
    fragments=[tree for tree in _BACKPROP_CONTEXT["fragments"] if tree[0]!="c"]
    if fragments:
        extra=[];kept=[]
        for tree in fragments:
            try: values=np.asarray(evaluate_cached(tree,X,adfs),float)
            except (ArithmeticError, IndexError, RecursionError, ValueError): continue
            if np.isfinite(values).all(): extra.append(values); kept.append(tree)
        if extra: trees=[*trees,*kept]; L=np.column_stack([L,*extra])
    if not L.shape[1]: return t
    D=desired[valid]; index,slope,intercept,sse=_best_affine_match(L[valid],D)
    _,_,_,own=_best_affine_match(current[valid][:,None],D)
    if not np.isfinite(sse) or sse>=own*(1-1e-6): return t
    candidate=trees[index]
    tolerance=1e-9*max(abs(slope),1.)
    # Only wrap with operators the run selected, or the child leaves its grammar.
    if abs(slope-1.)>tolerance and "*" in ops: candidate=("*",("c",slope),candidate)
    if abs(intercept)>1e-9*(abs(slope)+1.)*max(spread,1.) and "+" in ops: candidate=("+",candidate,("c",intercept))
    for replacement in (candidate,trees[index]):
        child=simplify_tree(replace_subtree(t,path,replacement))
        if child!=t and node_size(child)<=max_nodes and node_depth(child)<=max_depth: return child
    return t

class MutationPortfolio:
    def __init__(self):
        self.weights={"subtree":1.,"point":1.,"constant":1.,"hoist":.7,"shrink":.7,"parametrize":1.,"bilinear":BILINEAR_MUTATION_WEIGHT,"jump":JUMP_MUTATION_WEIGHT,
                      "squash":SQUASH_SWAP_WEIGHT,"smooth":SMOOTH_SWAP_WEIGHT,"gate":GATE_MUTATION_WEIGHT,"backprop":BACKPROP_MUTATION_WEIGHT,"residual_term":RESIDUAL_TERM_WEIGHT}
        self.tries={k:0 for k in self.weights}; self.wins={k:0 for k in self.weights}
    # Island-role multipliers on the adaptive weights, set before each
    # generation by the island's role (never checkpointed: the role is).
    # A kind only a role uses (prune) starts at weight 1 times its bias.
    bias=None
    def choose(self):
        if not self.bias: return rng.choices(list(self.weights),weights=list(self.weights.values()))[0]
        kinds=list(dict.fromkeys([*self.weights,*self.bias]))
        return rng.choices(kinds,weights=[self.weights.get(k,1.)*self.bias.get(k,1.) for k in kinds])[0]
    def record(self, kind, improved):
        self.tries[kind]=self.tries.get(kind,0)+1; self.wins[kind]=self.wins.get(kind,0)+int(improved)
        if self.tries[kind]%8==0: self.weights[kind]=max(.15,min(4.,.5+3*self.wins[kind]/self.tries[kind]))
    def apply(self,t,n_features,ops,max_nodes,max_depth,proposal=None,adfs=None):
        kind=self.choose()
        if kind=="point": return point_mutate(t,ops,adfs),kind
        if kind=="hoist": return hoist_mutate(t),kind
        if kind=="shrink": return shrink_mutate(t),kind
        if kind=="prune": return prune_mutate(t),kind
        if kind=="parametrize": return parametrize_mutate(t,ops,max_nodes,max_depth),kind
        if kind=="bilinear": return bilinear_mutate(t,n_features,ops,max_nodes,max_depth),kind
        if kind=="jump": return jump_mutate(t,n_features,ops,max_nodes,max_depth),kind
        if kind in ("backprop","residual_term"):
            child=(backprop_mutate if kind=="backprop" else residual_term_mutate)(t,ops,max_nodes,max_depth,adfs)
            if child!=t: return child,kind
            kind="subtree"  # no defined desired output or no better match: make an ordinary move
        if kind in ("squash","smooth","gate"):
            child=(squash_swap_mutate(t,ops,max_nodes,max_depth) if kind=="squash" else smooth_swap_mutate(t,ops,max_nodes,max_depth) if kind=="smooth"
                   else gate_mutate(t,n_features,ops,max_nodes,max_depth))
            # These moves need particular operators or nodes; where they do not
            # apply (no sigmoid/tanh/erf in the grammar, say), make an ordinary
            # move instead of spending the mutation slot on a no-op.
            if child!=t: return child,kind
            kind="subtree"
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
# --semantic-max-delta.  The 2026-10-02 audit found 33-38% of subtree, point
# and hoist tries rejected as too big; inf trended up on cond2d (26 vs 23/30,
# not significant), so the default stays 5 until a harder benchmark decides.
SEMANTIC_MAX_DELTA = 5.0
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
def constant_baselines(residual, halves):
    """Held-out loss of the best constant, per cross-fitting direction: the
    part of cross_fitted_reduction that depends on the residual alone, so one
    model's fragments can share it."""
    if halves is None: return None
    baselines=[]
    for fit,score in (halves,halves[::-1]):
        _,level=affine(np.zeros(len(fit)),residual[fit])
        baselines.append(robust_loss(np.full(len(score),level),residual[score]))
    return baselines
def cross_fitted_reduction(values, residual, halves, baselines=None):
    """Held-out residual reduction of an affinely fitted fragment beyond the best constant.

    Fit on one half, score on the other, both ways, and average.  A fragment
    that is constant on the rows earns nothing (it can only move the mean,
    which the model's own affine readout already does).  ``baselines``: a
    list the caller shares across one residual's fragments; it is filled with
    constant_baselines(residual, halves) on first need."""
    values=np.asarray(values,float)
    if not np.all(np.isfinite(values)) or float(np.std(values))<=1e-12*(1.+float(np.mean(np.abs(values)))): return 0.
    if halves is None: return 0.
    if baselines is None: baselines=[]
    if not baselines: baselines.extend(constant_baselines(residual,halves))
    gains=[]
    for (fit,score),baseline in zip((halves,halves[::-1]),baselines):
        scale,offset=affine(values[fit],residual[fit])
        gains.append(baseline-robust_loss(clean(scale*values[score]+offset),residual[score]))
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
                tree=model.trees[heads[0]]; baselines=[]
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
                        try: reduction=cross_fitted_reduction(evaluate_cached(fragment,X,model.adfs),residual,halves,baselines)
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

def semantic_mutate(tree, X, portfolio, n_features, ops, max_nodes, max_depth, proposal=None, adfs=None, library=None, min_delta=1e-8, max_delta=None, neutral_shrink=False):
    """neutral_shrink (simplifier islands) also accepts a smaller child with
    unchanged output: exactly the rewrite a simplifier is looking for."""
    if max_delta is None: max_delta=SEMANTIC_MAX_DELTA
    # A returned kind of None means "no move was applied": the unchanged
    # parent must not be credited (or blamed) to any mutation kind.
    try: baseline=evaluate_cached(tree,X,adfs)
    except ValueError: return tree,None,False
    for _ in range(6):
        child,kind=portfolio.apply(tree,n_features,ops,max_nodes,max_depth,proposal,adfs)
        if RELATION_OF_FEATURE and relation_violation(child): continue
        try: delta=semantic_distance(baseline,evaluate_cached(child,X,adfs))
        except ValueError: continue
        if min_delta < delta <= max_delta: return child,kind,False
        if neutral_shrink and delta<=min_delta and node_size(child)<node_size(tree): return child,kind,False
    # A rare macro lane keeps the normal semantic guard as the default while
    # allowing a finite, genuinely different step across distant basins.
    if library is not None and rng.random()<library.macro_rate:
        for _ in range(3):
            child,kind=portfolio.apply(tree,n_features,ops,max_nodes,max_depth,proposal,adfs)
            if RELATION_OF_FEATURE and relation_violation(child): continue
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
        if RELATION_OF_FEATURE and relation_violation(child): continue
        if node_size(child)<=max_nodes and node_depth(child)<=max_depth and child!=left: return child
    return left

def semantic_crossover(left, right, X, max_nodes, max_depth, adfs=None, min_delta=1e-8, max_delta=None):
    """Prefer bounded semantic moves over syntactically random exchanges."""
    if max_delta is None: max_delta=SEMANTIC_MAX_DELTA
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

# --class-balance (default on): every class of a categorical output carries the
# same total weight in its log loss, its error rate (the shape objective
# becomes the balanced error rate), the classifier readout fit, classifier
# constant tuning, the Bayesian likelihood and lexicase case order.
# Unweighted, a 95/5 target is 95% "accurate" with a constant head, and every
# one of those signals preferred dropping the rare class to fitting it.  The
# weights come from the rows being scored, so validation is balanced too.
CLASS_BALANCE = True
def class_balance_weights(truth, class_count):
    """Row weights giving every present class the same total weight, with mean 1
    over the valid rows; rows with an unknown label (-1) keep weight 1."""
    truth=np.asarray(np.rint(truth),int); weights=np.ones(len(truth))
    if not CLASS_BALANCE: return weights
    valid=(truth>=0)&(truth<class_count)
    if not np.any(valid): return weights
    counts=np.bincount(truth[valid],minlength=class_count).astype(float)
    weights[valid]=np.count_nonzero(valid)/(np.count_nonzero(counts)*counts[truth[valid]])
    return weights
_BALANCED_ROWS={}
def class_balanced_rows(X, truth, class_count, maximum):
    """Deterministic training rows with every present class equally represented:
    a large class keeps its maximin input-coverage probe, a small one repeats its
    rows (duplication is how an unweighted least-squares fit sees a class weight)."""
    truth=np.asarray(np.rint(truth),int)
    key=(int(maximum),int(class_count),array_digest(np.asarray(X)),rows_digest(truth))
    cached=_BALANCED_ROWS.get(key)
    if cached is not None: return cached
    groups=[np.flatnonzero(truth==label) for label in range(class_count)]
    groups=[group for group in groups if len(group)]
    if not groups: return stratified_probe_indices(X,maximum) if len(X)>maximum else slice(None)
    quota=max(1,min(max(len(group) for group in groups),int(maximum)//len(groups)))
    parts=[group[stratified_probe_indices(X[group],quota)] if len(group)>quota else np.resize(group,quota) for group in groups]
    rows=np.sort(np.concatenate(parts))
    if len(_BALANCED_ROWS)>=64: _BALANCED_ROWS.clear()
    _BALANCED_ROWS[key]=rows
    return rows

CLASSIFIER_RIDGE=1e-3
def fit_classifier_affine(raw, truth, class_count):
    """Fit per-head (scale, offset) minimizing ridge-penalized log loss by damped Newton.

    raw has one column per head; binary targets use one head whose log-odds are
    score-0.5, so the returned offset already includes that +0.5 shift.  Rows
    carry class_balance_weights, so the offsets do not encode the class prior."""
    raw=np.asarray(raw,float); n,heads=raw.shape
    truth=np.asarray(np.rint(truth),int); valid=(truth>=0)&(truth<class_count)
    if not np.any(valid): return [(1.,0.)]*heads
    raw=raw[valid]; truth=truth[valid]; n=len(truth); row_weights=class_balance_weights(truth,class_count)
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
        return float(-np.mean(row_weights*log_probabilities[np.arange(n),truth])+.5*CLASSIFIER_RIDGE*t@t),np.exp(log_probabilities)
    value,probabilities=objective(theta)
    for _ in range(30):
        weighted=((probabilities-one_hot)*row_weights[:,None])[:,:,None]*features
        gradient=weighted.reshape(n,-1).mean(axis=0)+CLASSIFIER_RIDGE*theta
        # Softmax Hessian block (k,l) = sum_n w_n (p_k[k=l] - p_k p_l) f_k f_l^T.
        # g.T@g on one buffer (numpy's symmetric product), so unit weights reproduce the unweighted fit bit for bit.
        weighted_features=(probabilities[:,:,None]*features).reshape(n,-1)*np.sqrt(row_weights)[:,None]
        hessian=(same_class*((flat*np.repeat(probabilities*row_weights[:,None],2,axis=1)).T@flat)-weighted_features.T@weighted_features)/n+ridge
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
# --loss: huber (default) is the MAD-scaled Huber loss with --huber-delta;
# squared is the same with an infinite delta; relative applies the Huber
# loss to (pred - y) / max(|y|, 1e-3 * median |y|), for targets spanning
# orders of magnitude.  The scorer, the constant fitter, the jump scan and
# both readouts all go through loss_scale() and loss_delta().
LOSS_MODE = "huber"
RELATIVE_LOSS_FLOOR = 1e-3
def loss_delta(): return float("inf") if LOSS_MODE=="squared" else ROBUST_LOSS_DELTA
def loss_scale(y):
    """Residual divisor: the target's robust scale, or per row in relative mode."""
    if LOSS_MODE!="relative": return target_scale(y)
    magnitude=np.abs(np.asarray(y,float))
    return np.maximum(magnitude,max(RELATIVE_LOSS_FLOOR*float(np.median(magnitude)),EPS))
def loss_base_weights(y):
    """Least-squares row weights that make a readout fit minimise the relative residual."""
    return None if LOSS_MODE!="relative" else 1./loss_scale(y)
def robust_loss(pred, y, delta=None):
    """MAD-scaled Huber loss; a few extreme target values cannot dominate."""
    delta=loss_delta() if delta is None else delta
    scale=loss_scale(y)
    r=np.abs((pred-y)/scale)
    return float(np.mean(np.where(r<=delta,.5*r*r,delta*(r-.5*delta))))
AFFINE_COEFFICIENT_BOUND = 1e9
def _weighted_line(u, y, w):
    """Closed-form weighted least-squares line y ~ c0*u+c1; None when degenerate."""
    s=float(w.sum())
    if not s>0: return None
    mu=float(np.dot(w,u))/s; my=float(np.dot(w,y))/s; du=u-mu
    suu=float(np.dot(w,du*du))
    if not math.isfinite(suu) or suu<=1e-12*max(float(np.dot(w,u*u)),EPS): return None
    c0=float(np.dot(w,du*(y-my)))/suu
    # Plain floats: the same IEEE doubles, without an array per call.
    return (c0,my-c0*mu) if math.isfinite(c0) else None
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
        # The robust location depends on the loss, so the loss settings are part of the key.
        key=(LOSS_MODE,ROBUST_LOSS_DELTA,y.tobytes() if len(y)<=4096 else array_digest(y))
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
    varying=spread>=EPS; spread=spread if varying else 1.
    bound=AFFINE_COEFFICIENT_BOUND
    # (A constant prediction is a degenerate line the compiled path would only hand back.)
    if varying and FIT_BACKEND=="auto" and LOSS_MODE=="huber":
        # The compiled IRLS streams the rows twice per iteration instead of ~25
        # times (memory bandwidth on large data) and has no per-iteration numpy
        # overhead (which dominated on small data).  It agrees to the solver's
        # sqrt(eps) tolerance; rare degenerate/bounded fits fall through.
        core=compiled_fitter()
        if core is not None:
            fitted=core.robust_affine(np.ascontiguousarray(pred,dtype=float),np.ascontiguousarray(y,dtype=float),centre,spread,loss_delta()*target_scale(y),bound)
            if fitted is not None: return fitted
    u=(pred-centre)/spread; A=None
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
        base=loss_base_weights(y); base=np.ones(len(pred)) if base is None else base
        coefficients=weighted_fit(base)
        cutoff=loss_delta()*loss_scale(y)
        # Solver termination limits are numerical safeguards, not search settings.
        for _ in range(200):
            residual=coefficients[0]*pred+coefficients[1]-y
            weights=base*np.sqrt(np.minimum(1.,cutoff/np.maximum(np.abs(residual),EPS)))
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
    # Subtrees the structural rules treat as leaves: in a merged separate-output
    # model, each inlined earlier-stage equation, which was validated in its own
    # search and stands for the staged feature its reader was validated with.
    opaque:tuple=field(default_factory=tuple,compare=False)
    # Main-line history (see HISTORY_LIMIT): observational only, never read
    # by the search.  Records are never mutated in place, so clones share them.
    history:tuple=field(default_factory=tuple,compare=False)
    def __post_init__(self):
        if not self.founder_ids: self.founder_ids=(self.lineage_id,)
        else: self.founder_ids=tuple(sorted(set(self.founder_ids)))
        if not isinstance(self.history,tuple): self.history=tuple(self.history or ())
    def clone(self): return Model(trees=list(self.trees),scales=list(self.scales),age=self.age,objectives=tuple(self.objectives),lineage_id=self.lineage_id,origin=self.origin,parent_ids=tuple(self.parent_ids),feasible=self.feasible,invalid_reason=self.invalid_reason,constraint_count=self.constraint_count,mdl_operators=tuple(self.mdl_operators),mdl_feature_count=self.mdl_feature_count,adfs=dict(self.adfs),founder_ids=tuple(self.founder_ids),birth_generation=self.birth_generation,history=self.history,opaque=self.opaque)

# Main-line model history.  Every model carries a short timeline of its main
# line of descent: where it was born (generation, island, stage, role, how),
# runs of variation on one island (consecutive mutations collapsed into one
# record with the generation span, the count and kinds of moves, and MDL bits
# and loss at the start and end of the run), crossovers (partner lineage id
# and the island and role the partner was last made on), migrations (ring or
# gathered by a simplifier), stage promotions and the final constant
# snapping.  A child inherits its main parent's timeline (the first parent of
# a crossover).  Runs never merge across an island, stage, role, crossover or
# migration.  Past HISTORY_LIMIT records the oldest block of variation and
# crossover records is merged into one summary record, so births,
# migrations, promotions and snapping are kept preferentially.  Purely
# observational: no equivalence key, MDL, selection, deduplication or random
# draw reads it.
HISTORY_LIMIT = 64
HISTORY_LANDMARKS = frozenset(("born","migrated","promoted","snapped"))
def history_place(cell=None, role=None):
    """{"island", "stage", "role"} of a cell (island/stage numbers are 0-based)."""
    if cell is None: return {"island":0,"stage":0,"role":role}
    return {"island":int(cell.island_index),"stage":int(cell.stage),"role":role if role is not None else (role_kind(cell) if cell.role else None)}
def _history_number(value):
    """4 significant digits keep timelines readable and checkpoints small."""
    return float(f"{value:.4g}") if value is not None and np.isfinite(value) else None
def _history_score(model):
    bits=model_complexity(model)
    return (int(round(bits)) if np.isfinite(bits) else None),_history_number(aggregate_loss(model))
def _history_start(parent, place):
    """A parent's timeline, or a birth record built from the parent itself
    (Bayesian particles and pre-history checkpoints carry none)."""
    if parent.history: return parent.history
    bits,loss=_history_score(parent)
    return ({"event":"born","generation":parent.birth_generation,**place,"how":parent.origin,"bits":bits,"loss":loss},)
def _history_same_place(record, place):
    return all(record.get(key)==place.get(key) for key in ("island","stage","role"))
def _history_compact(history):
    """Merge the oldest run of 2+ adjacent variation/crossover records into one summary record."""
    history=list(history)
    while len(history)>HISTORY_LIMIT:
        start=None
        for index in range(len(history)-1):
            if history[index]["event"] not in HISTORY_LANDMARKS and history[index+1]["event"] not in HISTORY_LANDMARKS:
                start=index; break
        if start is None:
            # No two adjacent ordinary records: drop the oldest one (or, failing
            # that, the oldest landmark after the birth).
            index=next((i for i,r in enumerate(history) if r["event"] not in HISTORY_LANDMARKS),1)
            del history[index]; continue
        end=start
        while end+1<len(history) and history[end+1]["event"] not in HISTORY_LANDMARKS: end+=1
        block=history[start:end+1]; kinds={}; count=0
        for record in block:
            if record["event"]=="crossover": kinds["crossover"]=kinds.get("crossover",0)+1; count+=1
            else:
                for kind,n in record.get("kinds",{}).items(): kinds[kind]=kinds.get(kind,0)+n
                count+=record.get("count",1)
        first,last=block[0],block[-1]
        places={(r.get("island"),r.get("stage"),r.get("role")) for r in block}
        island,stage,role=next(iter(places)) if len(places)==1 else (None,None,None)
        history[start:end+1]=[{"event":"variation","start":first.get("start",first.get("generation")),"end":last.get("end",last.get("generation")),
                               "island":island,"stage":stage,"role":role,"count":count,"kinds":kinds,"merged":True,
                               "bits":[first["bits"][0],last["bits"][1]],"loss":[first["loss"][0],last["loss"][1]]}]
    return tuple(history)
def history_append(history, record):
    return _history_compact((*history,record)) if len(history)>=HISTORY_LIMIT else (*history,record)
def history_born(model, generation, place):
    """Start a timeline for a model that has none (seeds, fresh and injected models)."""
    if model.history: return
    bits,loss=_history_score(model)
    model.history=({"event":"born","generation":int(generation),**place,"how":model.origin,"bits":bits,"loss":loss},)
def history_varied(child, parent, generation, place, kinds):
    """A mutation-like move on the parent's island: extend the open run or start one."""
    history=_history_start(parent,place)
    before,after=_history_score(parent),_history_score(child)
    kinds=[kind for kind in kinds if kind]
    last=history[-1]
    if last["event"]=="variation" and not last.get("merged") and _history_same_place(last,place):
        merged=dict(last.get("kinds",{}))
        for kind in kinds: merged[kind]=merged.get(kind,0)+1
        child.history=(*history[:-1],{**last,"end":int(generation),"count":last["count"]+1,"kinds":merged,
                                       "bits":[last["bits"][0],after[0]],"loss":[last["loss"][0],after[1]]})
    else:
        counted={}
        for kind in kinds: counted[kind]=counted.get(kind,0)+1
        child.history=history_append(history,{"event":"variation","start":int(generation),"end":int(generation),**place,"count":1,
                                              "kinds":counted,"bits":[before[0],after[0]],"loss":[before[1],after[1]]})
def history_last_made(model):
    """{"island", "role"} where a model was last made (its latest non-migration record), or None."""
    for record in reversed(model.history):
        if record["event"] in ("born","variation","crossover"): return {"island":record.get("island"),"role":record.get("role")}
    return None
def history_crossed(child, first, second, generation, place):
    history=_history_start(first,place)
    before,after=_history_score(first),_history_score(child)
    child.history=history_append(history,{"event":"crossover","generation":int(generation),**place,"partner":second.lineage_id,
                                          "partner_from":history_last_made(second),"bits":[before[0],after[0]],"loss":[before[1],after[1]]})
def _history_where(place):
    if not place or place.get("island") is None: return "several islands"
    text=f"island {place['island']+1}"
    if place.get("stage"): text+=f" stage {place['stage']+1}"
    return text+(f" ({place['role']})" if place.get("role") else "")
def _history_span(record):
    start,end=record.get("start",record.get("generation")),record.get("end",record.get("generation"))
    if start is None: return "gen ?"
    return f"gen {start}" if start==end or end is None else f"gen {start}-{end}"
def _history_change(record):
    """' (74->96 bits, loss 0.076->0.0043)' from a record's bits/loss pairs (or single values)."""
    def pair(value): return list(value) if isinstance(value,(list,tuple)) else [None,value]
    pieces=[]
    for (before,after),unit,form in ((pair(record.get("bits")),"bits","{:g}"),(pair(record.get("loss")),"loss","{:.4g}")):
        if after is None: continue
        text=form.format(after) if before is None or before==after else f"{form.format(before)}->{form.format(after)}"
        pieces.append(f"{text} bits" if unit=="bits" else f"loss {text}")
    return f" ({', '.join(pieces)})" if pieces else ""
def describe_history(history):
    """One readable line per history record (CLI output and the model card)."""
    lines=[]
    for record in history:
        event=record.get("event"); span=_history_span(record); change=_history_change(record)
        if event=="born":
            lines.append(f"{span}: born on {_history_where(record)} as {record.get('how')}{change}")
        elif event=="variation":
            kinds=", ".join(f"{kind} {count}" for kind,count in sorted(record.get("kinds",{}).items(),key=lambda item:-item[1]))
            count=record.get("count",1); noun="variation" if count==1 else "variations"
            detail=f" [{kinds}]" if kinds else ""
            lines.append(f"{span}: {count} {noun} on {_history_where(record)}{detail}{change}")
        elif event=="crossover":
            partner=record.get("partner_from")
            detail=f" (last made on {_history_where(partner)})" if partner else ""
            lines.append(f"{span}: crossover on {_history_where(record)} with #{record.get('partner')}{detail}{change}")
        elif event in ("migrated","promoted"):
            how=record.get("how"); detail=f" [{how}]" if how else ""
            lines.append(f"{span}: {event} {_history_where(record.get('from'))} -> {_history_where(record.get('to'))}{detail}")
        elif event=="snapped":
            values=", ".join(f"{a:g}->{b:g}" for a,b in record.get("values",[])[:6])
            detail=f" ({values})" if values else ""
            simplified=" and simplified" if record.get("changed") else ""
            lines.append(f"final: snapped {record.get('constants')} constant(s){detail}{simplified}{change}")
        else: lines.append(str(record))
    return lines
def history_path(history):
    """Places a model's main line passed through, in order ("island 2 (explorer)", ...), repeats collapsed."""
    path=[]
    for record in history:
        place=record.get("to") if record.get("event") in ("migrated","promoted") else record if record.get("event") in ("born","variation","crossover") else None
        if place is None or place.get("island") is None: continue
        where=_history_where(place)
        if not path or path[-1]!=where: path.append(where)
    return path
def history_moved(model, event, generation, source, destination, how=None):
    """A migration (between islands) or stage promotion."""
    record={"event":event,"generation":None if generation is None else int(generation),"from":source,"to":destination}
    if how: record["how"]=how
    model.history=history_append(model.history,record)

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
def _mean(values):
    """float(np.mean(values)); a single value (one output) skips numpy, with the same result."""
    return (0.+float(values[0]))/1. if len(values)==1 else float(np.mean(values))
def aggregate_loss(model_or_objectives):
    values=model_losses(model_or_objectives) if isinstance(model_or_objectives,Model) else tuple(model_or_objectives[:-2:2])
    return _mean(values) if values else float("inf")
def secondary_key(model):
    """Deterministic non-fitness key for duplicate handling and diagnostics."""
    # Losses that differ only by round-off are ties, so the shorter model wins.
    return (_noise_rounded(aggregate_loss(model)),_noise_rounded(_mean(model_shapes(model))),model_complexity(model),repr(model.trees))
# Loss/shape differences below this (afpo's scaled units) are rounding noise,
# not fit: a CSV holding ~7 significant digits leaves an exact model at a loss
# of ~1e-11, and a bigger model with spare constants can "beat" it by 1e-12.
# Near-tie comparisons use max(relative tolerance, this floor).
LOSS_NOISE_FLOOR = 1e-9
# The floor is set per run from the targets' own precision (below): a
# float64 target lets an exact model reach 1e-20..1e-30, and a fixed 1e-9
# then ranked a 1e-11 approximation as equal to it, so the shorter
# approximation won the final choice and the best-so-far archive kept it.
# The 1e-18 minimum (relative error ~1e-9) is where the constant fitter's own
# convergence leaves exact models, so those still tie and the shortest wins.
LOSS_NOISE_FLOOR_MAX, LOSS_NOISE_FLOOR_MIN, LOSS_NOISE_MULTIPLE = 1e-9, 1e-18, 1e3
def estimate_loss_noise_floor(Y, cats=None, Yv=None, sample=4000):
    """LOSS_NOISE_MULTIPLE x the loss that rounding the numeric targets to the
    digits they are written with would cost, in [LOSS_NOISE_FLOOR_MIN, LOSS_NOISE_FLOOR_MAX].

    Each value's rounding step is its last significant digit (shortest decimal
    that reads back as the same double); the median over a column guards
    against short values such as 0.5 in an otherwise 7-digit column."""
    Y=np.asarray(Y,float)
    if Y.ndim==1: Y=Y[:,None]
    if Yv is not None and len(Yv): Y=np.vstack([Y,np.asarray(Yv,float).reshape(len(Yv),-1)])
    floors=[]
    for j in range(Y.shape[1]):
        if cats is not None and cats[j] is not None: continue
        column=Y[:,j]; column=column[np.isfinite(column)]
        if len(column)<2: continue
        scale=float(target_scale(column))
        values=column[column!=0]
        if len(values)>sample: values=values[np.random.default_rng(len(values)).choice(len(values),sample,replace=False)]
        if not len(values) or not scale>0: continue
        steps=[]
        for value in values.tolist():
            digits=next(d for d in range(1,18) if float(f"{value:.{d}g}")==value)
            steps.append(.5*10.**(math.floor(math.log10(abs(value)))-digits+1))
        step=float(np.median(steps))/scale
        floors.append(LOSS_NOISE_MULTIPLE*.5*step*step/3.)
    if not floors: return LOSS_NOISE_FLOOR_MAX
    return float(min(LOSS_NOISE_FLOOR_MAX,max(LOSS_NOISE_FLOOR_MIN,max(floors))))
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
        if left>right+max(tolerance*abs(right),LOSS_NOISE_FLOOR): return False
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
    comparisons=[(left,right,max(tolerance*abs(right),LOSS_NOISE_FLOOR)) for left,right in zip(_quality_objectives(candidate),_quality_objectives(incumbent))]
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
        # A bigger model only noise-better than a simpler one is not a trade-off.
        self.items=parsimony_candidates(self.items,self.parsimony_quality_tolerance)
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

    # Descriptors are pure functions of a model's trees, readout and called
    # ADFs (and the archive's fixed probe), but update() needs each one twice
    # (initialize, then cell) and every cell model is re-binned each
    # generation, so most were recomputed.  The memo is per archive, bounded,
    # and never pickled or snapshotted.
    DESCRIPTOR_CACHE_LIMIT=8192
    def descriptor_key(self, model): return (repr(model.trees),repr(model.scales),adf_signature(model.trees,model.adfs))
    def cached_descriptor(self, model):
        cache=self.__dict__.setdefault("_descriptor_cache",{})
        key=self.descriptor_key(model); hit=cache.get(key)
        if hit is None:
            if len(cache)>=self.DESCRIPTOR_CACHE_LIMIT: cache.clear()
            hit=np.asarray(self.descriptor(model)); hit.setflags(write=False); cache[key]=hit
        return hit
    def __getstate__(self):
        state=dict(self.__dict__); state.pop("_descriptor_cache",None); return state

    def initialize(self, candidates):
        descriptors=[self.cached_descriptor(model) for model in candidates if model.feasible]
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
        descriptor=self.cached_descriptor(model)
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
    def descriptor_key(self, model): return repr(model.trees)

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
def residual_regions(probe_X, probe_Y, cats, class_groups=True):
    """Fixed row groups for residual signatures: target-magnitude bins per
    numeric output, one group per class of a categorical output (class_groups;
    a model that only misses class A and one that only misses class B are
    complementary), and input-region bins, each a list of row-index arrays."""
    X=np.asarray(probe_X,float); groups=[]
    for j,labels in enumerate(cats):
        if labels is not None:
            if class_groups and len(labels)>=2:
                truth=np.rint(np.asarray(probe_Y[:,j],float))
                groups.append([np.flatnonzero(truth==label) for label in range(len(labels))])
            continue
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
    def __init__(self, probe, probe_targets, cats, seed, capacity=64, landmarks=None, class_groups=True):
        super().__init__(probe,cats,seed,capacity,landmarks)
        self.probe_targets=np.asarray(probe_targets,float); self.class_groups=bool(class_groups)
        self.groups=residual_regions(self.probe,self.probe_targets,self.cats,self.class_groups)

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
        data=super().snapshot(); data["probe_targets"]=self.probe_targets; data["class_groups"]=self.class_groups; return data

    @classmethod
    def from_snapshot(cls, data):
        # Archives saved before class groups keep their descriptor length (and landmarks).
        result=cls(data["probe"],data["probe_targets"],data["cats"],data["seed"],data["capacity"],data.get("landmarks"),data.get("class_groups",False))
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

SCALE_BALANCED_SELECTION = True    # --scale-balanced-selection (default on since the 2026-09-29 3-seed benchmark)
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
    per_case=None if case_weights is None else np.tile(np.asarray(case_weights,float),Y.shape[1])
    if CLASS_BALANCE and any(labels is not None for labels in cats):
        # A rare class's rows are few cases, so uniform order almost never
        # examines them first; equal class weight puts them first as often.
        balance=np.concatenate([np.ones(len(Y)) if labels is None else class_balance_weights(Y[:,j],len(labels)) for j,labels in enumerate(cats)])
        if not np.all(balance==1.): per_case=balance if per_case is None else per_case*balance
    weights=None if per_case is None else np.maximum(per_case[cases],EPS)
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

CSV_CHUNK_ROWS=50_000
def csv_read_options(delimiter):
    """pandas' C parser for one-character delimiters: several times faster and
    far leaner than the Python engine, which only multi-character (regex)
    separators need."""
    sep=delimiter or ","
    return {"sep":sep,"engine":"c" if len(sep)==1 else "python"}

class _MixedChunkTypes(Exception): pass

def _sample_csv_rows(path, options, max_rows, seed, as_text):
    """(kept chunk pieces, kept file row numbers, file row count) of a streamed uniform sample.

    Each row gets a random key and the max_rows smallest keys survive, so only
    about max_rows rows are ever held.  Typed chunks raise _MixedChunkTypes when
    a column parses as text in one chunk and as numbers in another."""
    generator=np.random.default_rng(seed)
    kept=[]; keys=np.empty(0); rows=np.empty(0,dtype=np.int64); total=0; threshold=np.inf; kinds=None
    extra={"dtype":str,"keep_default_na":False} if as_text else {}
    for chunk in pd.read_csv(path,chunksize=CSV_CHUNK_ROWS,**extra,**options):
        if not as_text:
            chunk_kinds=["number" if dtype.kind in "iuf" else str(dtype) for dtype in chunk.dtypes]
            if kinds is None: kinds=chunk_kinds
            elif chunk_kinds!=kinds: raise _MixedChunkTypes
        chunk_keys=generator.random(len(chunk)); chunk_rows=np.arange(total,total+len(chunk)); total+=len(chunk)
        chosen=np.flatnonzero(chunk_keys<threshold)
        if not len(chosen): continue
        kept.append(chunk.iloc[chosen]); keys=np.concatenate([keys,chunk_keys[chosen]]); rows=np.concatenate([rows,chunk_rows[chosen]])
        if len(keys)>max_rows:
            frame=pd.concat(kept,ignore_index=True)
            order=np.sort(np.argpartition(keys,max_rows-1)[:max_rows])
            kept=[frame.iloc[order]]; keys=keys[order]; rows=rows[order]; threshold=float(keys.max())
    return kept,rows,total

def read_dataset(path, delimiter=",", max_rows=0, seed=0):
    """Read a CSV; above ``max_rows`` rows keep a seeded uniform sample, in file order.

    The sample is drawn while streaming the file in chunks, so a file far larger
    than memory is never held whole.  Columns keep the types a full read gives
    (int and float chunks unify to float as they would); if a column reads as
    text in some chunks and numbers in others, the file is streamed again as
    text and the kept rows are parsed together.  ``frame.attrs
    ["afpo_row_sample"]`` records the file's row count and the kept row numbers."""
    options=csv_read_options(delimiter)
    if not max_rows or max_rows<=0: return pd.read_csv(path,**options)
    try:
        kept,rows,total=_sample_csv_rows(path,options,max_rows,seed,as_text=False)
        frame=pd.concat(kept,ignore_index=True) if kept else None
    except _MixedChunkTypes:
        kept,rows,total=_sample_csv_rows(path,options,max_rows,seed,as_text=True)
        text=io.StringIO(); pd.concat(kept,ignore_index=True).to_csv(text,index=False); del kept
        text.seek(0); frame=pd.read_csv(text)
    if total<=max_rows: return pd.read_csv(path,**options)
    frame.attrs["afpo_row_sample"]={"source_rows":int(total),"max_rows":int(max_rows),"seed":int(seed),"rows":rows.tolist()}
    return frame

def csv_shape(path, delimiter=","):
    """(row count, column names) without holding the file: the header, then one column streamed."""
    options=csv_read_options(delimiter)
    columns=list(pd.read_csv(path,nrows=0,**options).columns)
    if not columns: return 0,columns
    rows=sum(len(chunk) for chunk in pd.read_csv(path,usecols=[0],dtype=str,keep_default_na=False,chunksize=CSV_CHUNK_ROWS*4,**options))
    return rows,columns

def describe_row_sample(frame):
    sample=frame.attrs.get("afpo_row_sample")
    return f"; sampled {len(frame):,} of {sample['source_rows']:,} rows (--max-rows, seed {sample['seed']})" if sample else ""

def holdout_split_indices(n_rows, validation_rows, seed, strata=None):
    """Deterministic disjoint train/validation split used by every run.

    strata (one label per row, e.g. the class of a categorical output) gives
    each stratum its proportional share of the validation rows, and always
    leaves it at least one training row: a plain random split can put every
    row of a rare class in validation, where its label was never seen."""
    if not 0 < validation_rows < n_rows: raise ValueError("Validation rows must be between 1 and n_rows - 1")
    indices=np.random.default_rng(seed).permutation(n_rows)
    if strata is None: return indices[:-validation_rows],indices[-validation_rows:]
    _,codes=np.unique(np.asarray(strata,dtype=str),return_inverse=True)
    counts=np.bincount(codes); share=validation_rows*counts/n_rows; room=counts-1
    quota=np.minimum(np.floor(share).astype(int),room)
    # Largest remainder first, then any stratum with room left.
    for stratum in [*np.argsort(-(share-quota),kind="stable"),*np.argsort(-room,kind="stable")]:
        if quota.sum()>=validation_rows: break
        if quota[stratum]<room[stratum]: quota[stratum]+=1
    if quota.sum()<validation_rows: return indices[:-validation_rows],indices[-validation_rows:]
    # The last members of each stratum in permutation order become validation.
    ordered=codes[indices]; rank=np.empty(n_rows,dtype=int)
    for stratum in range(len(counts)):
        members=np.flatnonzero(ordered==stratum); rank[members]=np.arange(len(members))[::-1]
    holdout=rank<quota[ordered]
    return indices[~holdout],indices[holdout]

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
    """Persist enough provenance to reproduce a run's data and split.

    ``df`` is the loaded frame or just its ``(row_count, columns)``.  Under
    --max-rows the split indices are rows of the sample; configuration
    ["row_sample"]["rows"] maps them to file rows."""
    rows,columns=df if isinstance(df,tuple) else (len(df),list(df.columns))
    stamp=time.strftime("%Y%m%d-%H%M%S")
    run_dir=Path("afpo_runs") / f"{stamp}-seed{seed}"
    suffix=1
    while run_dir.exists():
        suffix+=1; run_dir=Path("afpo_runs") / f"{stamp}-seed{seed}-{suffix}"
    run_dir.mkdir(parents=True)
    manifest={
        "format_version":2, "seed":seed, "dataset":{"path":str(Path(path).resolve()), "sha256":dataset_sha256(path),
        "rows":rows, "columns":columns}, "configuration":config,
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
          "history":{"records":list(model.history),"summary":describe_history(model.history)},
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

# 16 stores numeric arrays as compressed base64 bytes; 15 (JSON number lists) still loads.
SAFE_CHECKPOINT_FORMAT=16
READABLE_CHECKPOINT_FORMATS=(15,16)

def _checkpoint_state_for_save(state):
    """The run state minus X/Y when they are the training arrays themselves.

    A fresh run sets X,Y=Xt,Yt, so saving both wrote the whole dataset twice;
    loading restores them from Xt/Yt (see checkpoint_arrays)."""
    if state.get("X") is state.get("Xt") and state.get("Y") is state.get("Yt") and "Xt" in state:
        return {key:value for key,value in state.items() if key not in ("X","Y")}
    return state

def checkpoint_arrays(state):
    """(X, Y, Xt, Yt, Xv, Yv) of a loaded state; X/Y default to the training arrays."""
    Xt,Yt=state["Xt"],state["Yt"]
    return state.get("X",Xt),state.get("Y",Yt),Xt,Yt,state["Xv"],state["Yv"]

def _is_model_data(value):
    return isinstance(value,dict) and "history" in value and "trees" in value and "lineage_id" in value
def _walk_model_data(value, visit):
    """Call visit(data) once on every serialized model (PosteriorParticlePopulation._model_data)
    in a payload; containers shared between places (island 0's runtime is also
    mirrored into the run state) are visited once."""
    stack=[value]; seen=set()
    while stack:
        item=stack.pop()
        if isinstance(item,(dict,list,tuple)):
            if id(item) in seen: continue
            seen.add(id(item))
        if isinstance(item,dict):
            if _is_model_data(item): visit(item)
            stack.extend(item.values())
        elif isinstance(item,(list,tuple)): stack.extend(item)
def _intern_histories(payload):
    """Store each distinct history record once (siblings share most of their
    timeline); models keep indices into payload["history_records"].  Some
    model dicts are live run state (island snapshots), so the returned undo
    list puts their record lists back once the payload is written."""
    table=[]; by_id={}; by_text={}; undo=[]
    def visit(data):
        undo.append((data,data["history"]))
        indices=[]
        for record in data["history"]:
            index=by_id.get(id(record))
            if index is None:
                text=json.dumps(record,sort_keys=True,default=str)
                index=by_text.get(text)
                if index is None: index=by_text[text]=len(table); table.append(record)
                by_id[id(record)]=index
            indices.append(index)
        data["history"]=indices
    _walk_model_data(payload,visit)
    payload["history_records"]=table
    return undo
def _restore_histories(payload):
    table=payload.pop("history_records",None)
    if table is None: return
    def visit(data): data["history"]=[table[index] for index in data["history"]]
    _walk_model_data(payload,visit)
def save_checkpoint(path, generation, population, bayes, archive, state):
    """Atomically persist all stochastic/evolutionary state as safe JSON."""
    state=_checkpoint_state_for_save(state)
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
    undo=_intern_histories(payload)
    try:
        encoded=_json_checkpoint_value(payload)
        body=json.dumps(encoded,sort_keys=True,separators=(",",":"),allow_nan=False); del encoded
    finally:
        for data,history in undo: data["history"]=history
    checksum=hashlib.sha256(body.encode()).hexdigest()
    temporary=Path(str(path)+".tmp")
    # Exactly json.dumps(wrapper,sort_keys=True,separators=(",",":")), written
    # around the body instead of serialising the whole payload a second time.
    with temporary.open("w") as handle:
        handle.write(f'{{"checksum":{json.dumps(checksum)},"format_version":{SAFE_CHECKPOINT_FORMAT},"payload":')
        handle.write(body); handle.write("}\n")
    temporary.replace(path)

def load_checkpoint(path, allow_unsafe_pickle=False):
    global _NEXT_LINEAGE_ID
    source=Path(path)
    try:
        wrapper=json.loads(source.read_text())
    except (UnicodeDecodeError,json.JSONDecodeError):
        if not allow_unsafe_pickle: raise ValueError("Refusing legacy pickle checkpoint; rerun with --allow-unsafe-pickle only for a trusted local file")
        with source.open("rb") as handle: payload=pickle.load(handle)
    else:
        if wrapper.get("format_version") not in READABLE_CHECKPOINT_FORMATS: raise ValueError("Unknown safe checkpoint format")
        body=json.dumps(wrapper["payload"],sort_keys=True,separators=(",",":"),allow_nan=False)
        if hashlib.sha256(body.encode()).hexdigest()!=wrapper.get("checksum"): raise ValueError("Checkpoint checksum does not match")
        payload=_from_json_checkpoint_value(wrapper["payload"])
        _restore_histories(payload)
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
LARGE_DATASET_ROWS=200_000
def row_sample_seed(args):
    """--max-rows samples with --seed when given, else a fixed seed, so a rerun keeps the same rows."""
    return 0 if getattr(args,"seed",None) is None else int(args.seed)

def report_loaded(df, max_rows=0):
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns{describe_row_sample(df)}: {list(df.columns)}")
    if len(df)>LARGE_DATASET_ROWS and not max_rows:
        print(f"Large dataset: every generation scores every training row. --max-rows N (e.g. 100000) trains on a "
              f"uniform sample of N rows and uses far less memory; co-evolution scores a subsample each generation.")

def configure(path, max_rows=0, seed=0):
    delim=parse_delimiter(ask("Delimiter: 0=comma, 1=semicolon, 2=space, 3=tab, 4=custom", "0"))
    df=read_dataset(path,delim,max_rows,seed)
    report_loaded(df,max_rows)
    types=column_types(df)
    if not any(t in (5,6) for t in types) or not any(t in (1,2) for t in types): raise ValueError("Select at least one input and one output")
    return df,types,delim
def parse_sequence_group(text):
    name,sep,columns=str(text).partition("=")
    if not sep or not name.strip(): raise ValueError(f"--sequence-group expects NAME=COL1,COL2,...; got {text!r}")
    return name.strip(),[column.strip() for column in columns.split(",") if column.strip()]
ONE_HOT_WARNING_BYTES=1<<30
def encode(df, types, fitted_maps=None):
    # Columns are planned first and written straight into one preallocated
    # matrix: stacking a list of per-feature arrays held every feature twice.
    plan=[]; names=[]; outputs=[]; output_names=[]; categorical=[]; maps={}
    numeric_fills=dict((fitted_maps or {}).get("__afpo_numeric_fills__",{}))
    for col,t in zip(df.columns,types):
        s=df[col]
        if t==1:
            z=pd.to_numeric(s,errors="coerce").to_numpy(float)
            fill=float(numeric_fills.get(col,np.nanmedian(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.))
            numeric_fills[col]=fill; plan.append(("numeric",col,z,fill)); names.append(col)
        elif t==2:
            vals=s.fillna("__MISSING__").astype(str)
            classes=(fitted_maps or {}).get(col, sorted(vals.unique())); maps[col]=classes
            # Class positions (-1 if unseen): one pass over the rows instead of one per class.
            plan.append(("one_hot",col,pd.Index(classes).get_indexer(vals),len(classes)))
            names.extend(f"{col}={cl}" for cl in classes)
            if len(classes)*len(df)*8>=ONE_HOT_WARNING_BYTES:
                print(f"Warning: categorical input {col!r} has {len(classes):,} categories; its one-hot columns take "
                      f"{len(classes)*len(df)*8/2**30:.1f} GB. Ignore it (type 0) if it is an identifier.")
        elif t in (5,6):
            if t==5:
                z=pd.to_numeric(s,errors="coerce").to_numpy(float); mask=np.isfinite(z); fill=np.nanmedian(z[mask]) if mask.any() else 0.; outputs.append(np.where(mask,z,fill)); categorical.append(None)
            else:
                vals=s.fillna("__MISSING__").astype(str)
                classes=(fitted_maps or {}).get(col, sorted(vals.unique())); maps[col]=classes
                # Unseen validation labels are deliberately treated as a loss
                # (code -1), not silently added as a new output dimension.
                outputs.append(pd.Index(classes).get_indexer(vals).astype(float)); categorical.append(classes)
            output_names.append(col)
    maps["__afpo_numeric_fills__"]=numeric_fills
    if not names: raise ValueError("need at least one array to concatenate")
    X=np.empty((len(df),len(names))); offset=0
    for item in plan:
        if item[0]=="numeric":
            _,_,z,fill=item; X[:,offset]=np.where(np.isfinite(z),z,fill); offset+=1
        else:
            _,_,codes,width=item; block=X[:,offset:offset+width]; block.fill(0.)
            present=codes>=0; block[np.flatnonzero(present),codes[present]]=1.; offset+=width
    del plan
    layout=(fitted_maps or {}).get(SEQUENCE_LAYOUT_KEY) or build_sequence_layout(names,SEQUENCE_GROUP_REQUEST)
    # Assigned even when None: a layout left over from an earlier run in this
    # process would otherwise add seqsum/seqprod to a run without sequences.
    global SEQUENCE_LAYOUT
    SEQUENCE_LAYOUT=layout
    if layout is not None:
        maps[SEQUENCE_LAYOUT_KEY]=layout
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

# Memorisation check.  A model can fit the training rows and still be wrong
# everywhere between them: a short-period mod() sawtooth whose period nearly
# divides the row spacing looks like a smooth ramp on the training grid (and
# even at exact midpoints) but is noise anywhere else.  Pair each sampled row
# with its nearest neighbour (standardised inputs) and probe at a random but
# fixed fraction of the way between them; random fractions matter, because
# probes at a fixed fraction of the spacing are aliased along with the grid.
# Each prediction is measured against the band spanned by the two neighbours'
# targets (widened by half its width plus INTERPOLATION_BAND_SLACK target
# scales), and only the *excess* over the model's own miss at those two rows
# is charged, as a median over probes: a memoriser hits the rows and misses
# between most of them, an honest model that is wrong somewhere is about as
# wrong at the rows, a steep but imperfect jump disturbs only the probes that
# straddle it, and an exactly correct model pays nothing.
INTERPOLATION_CHECK = True
INTERPOLATION_CHECK_WEIGHT = 1.
INTERPOLATION_BAND_SLACK = .05
_NEIGHBOUR_PROBE_CACHE = {}
NEAREST_ROW_BLOCK_BYTES = 1<<25
def nearest_other_rows(standard, rows):
    """(nearest row, has one) for each of standard[rows], ignoring rows at squared distance <= 1e-18.

    The same per-pair arithmetic as one (len(rows), n, features) difference
    array, done a block of rows at a time with a running minimum (ties keep
    the first row, as argmin does), so results are identical while memory
    stays at NEAREST_ROW_BLOCK_BYTES: the all-at-once array needed 26 GB for
    64 probes on a million 52-feature rows."""
    query=standard[rows]; best=np.full(len(rows),np.inf); partner=np.zeros(len(rows),dtype=np.intp)
    step=max(1,NEAREST_ROW_BLOCK_BYTES//max(1,8*len(rows)*standard.shape[1])); positions=np.arange(len(rows))
    for start in range(0,len(standard),step):
        distance=np.sum((query[:,None,:]-standard[None,start:start+step,:])**2,axis=2); distance[distance<=1e-18]=np.inf
        local=np.argmin(distance,axis=1); value=distance[positions,local]
        better=value<best; best[better]=value[better]; partner[better]=local[better]+start
    return partner,np.isfinite(best)
def neighbour_probes(X):
    """(probe inputs, row indices, neighbour indices) for nearest-neighbour pairs, or None."""
    if not isinstance(X,np.ndarray) or X.ndim!=2 or len(X)<4: return None
    key=(X.__array_interface__["data"][0],X.shape,X.strides)
    cached=_NEIGHBOUR_PROBE_CACHE.get(key)
    if cached is not None and cached[3] is X: return cached[:3]
    values=np.asarray(X,float); scale=values.std(axis=0)
    standard=np.divide(values-values.mean(axis=0),scale,out=np.zeros_like(values),where=scale>EPS)
    generator=np.random.default_rng(len(X)+7919)
    rows=generator.choice(len(X),min(INTERPOLATION_PROBES,len(X)),replace=False)
    partner,usable=nearest_other_rows(standard,rows)
    rows,partner=rows[usable],partner[usable]
    if not len(rows): return None
    fraction=generator.uniform(.15,.85,len(rows))[:,None]
    # Few-valued columns (one-hot flags, small codes) are copied, not interpolated.
    discrete=np.array([len(np.unique(values[:,j]))<=10 for j in range(values.shape[1])])
    probes=np.where(discrete,values[rows],values[rows]+(values[partner]-values[rows])*fraction)
    if len(_NEIGHBOUR_PROBE_CACHE)>32: _NEIGHBOUR_PROBE_CACHE.clear()
    _NEIGHBOUR_PROBE_CACHE[key]=(probes,rows,partner,X)
    return probes,rows,partner
def interpolation_band_excess(between, at_left, at_right, left, right, y):
    """Median excess (in target scales) of the between-neighbour miss over the at-row miss.

    The median charges pervasive misbehaviour (a memoriser misses between
    most row pairs) but not a steep, slightly misplaced jump, which only
    disturbs the one or two probes that straddle it."""
    low=np.minimum(left,right); high=np.maximum(left,right); scale=max(target_scale(y),EPS)
    slack=.5*(high-low)+INTERPOLATION_BAND_SLACK*scale
    def miss(prediction): return np.minimum(np.maximum(0.,np.maximum(low-slack-prediction,prediction-high-slack))/scale,1e6)
    return float(np.median(np.maximum(0.,miss(between)-.5*(miss(at_left)+miss(at_right)))))

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
    reason=""; opaque=frozenset(m.opaque)
    for tree in m.trees:
        violation=structural_violation(tree,opaque) if NESTING_RULES or UNIT_FEATURES or RELATION_OF_FEATURE else ""
        kind="nesting" if NESTING_RULES and nesting_violation(tree,opaque=opaque) else "units" if UNIT_FEATURES and unit_violation(tree,opaque) else "relations"
        reason=valid(tree) or (f"{kind}:{violation}" if violation else "")
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
    guard=guard_engagement(m.trees,X,m.adfs)
    if guard:
        m.feasible=False; m.invalid_reason=f"numeric_guard:{guard}"; INVALID_DIAGNOSTICS[m.invalid_reason]=INVALID_DIAGNOSTICS.get(m.invalid_reason,0)+1
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
            local=neighbour_probes(X) if INTERPOLATION_CHECK else None
            if local is not None:
                inputs,rows,partner=local
                try: fitted=clean(scales[heads[0]][0]*evaluate_cached(m.trees[heads[0]],inputs,m.adfs)+scales[heads[0]][1])
                except (ArithmeticError, IndexError, RecursionError, ValueError): fitted=np.full(len(inputs),CLIP)
                loss+=INTERPOLATION_CHECK_WEIGHT*interpolation_band_excess(fitted,p[rows],p[partner],Y[rows,j],Y[partner,j],Y[:,j])
            losses.append(loss); shapes.append(shape_error(p,Y[:,j]))
            decoded.append(p)
        elif len(cats[j])>2:
            scores=np.column_stack([clean(scales[head][0]*raw[:,head]+scales[head][1]) for head in heads]); probabilities=stable_softmax(scores)
            truth=np.asarray(np.rint(Y[:,j]),dtype=int); valid=(truth>=0)&(truth<len(cats[j])); row=np.arange(len(truth))
            cross_entropy=np.where(valid,-np.log(np.maximum(probabilities[row,np.clip(truth,0,len(cats[j])-1)],EPS)),-np.log(EPS))
            balance=class_balance_weights(truth,len(cats[j]))
            labels=np.argmax(probabilities,axis=1); losses.append(float(np.average(cross_entropy,weights=balance))); shapes.append(float(np.average(labels!=truth,weights=balance))); decoded.append(labels)
        else:
            balance=class_balance_weights(Y[:,j],len(cats[j]))
            labels=binary_labels(p,len(cats[j])); error_rate=float(np.average(labels!=Y[:,j],weights=balance))
            if len(cats[j])==2:
                # Log loss rewards moving the decision score toward the right side of 0.5, which a 0/1 error cannot.
                truth=np.asarray(np.rint(Y[:,j]),dtype=int); positive=binary_probabilities(p)[:,1]
                chosen=np.where(truth==1,positive,np.where(truth==0,1.-positive,0.))
                losses.append(float(np.average(-np.log(np.maximum(chosen,EPS)),weights=balance)))
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
    # A smaller evaluation cache per scoring worker (see WORKER_CACHE_SHARE);
    # inherited entries go too, so the budget holds from the first batch.
    global EVALUATION_CACHE_ELEMENTS,CACHE_SCALE
    EVALUATION_CACHE_ELEMENTS=max(10_000,int(EVALUATION_CACHE_ELEMENTS*WORKER_CACHE_SHARE)); CACHE_SCALE*=WORKER_CACHE_SHARE
    _EVALUATION_CACHE.clear(); _EVALUATION_CACHE_SIZE[0]=0; _COMPILED_TREES.clear()

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
        self.journal=None  # set in forked cell processes: new cache entries to merge back
        # Load (and on first use, build) the compiled fitter before forking, so
        # workers inherit it instead of each racing to compile it.
        if FIT_BACKEND=="auto": compiled_fitter()
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
        if len(self._score_cache)>cache_limit(self.SCORE_CACHE_LIMIT): self._score_cache.clear()
    def _cache_key(self, model, dataset, indices, fit_affine, tune=False):
        sample=None if indices is None else rows_digest(indices)
        scales=() if fit_affine else tuple((float(a),float(b)) for a,b in model.scales)
        return (dataset,sample,fit_affine,tune,tree_digest(model.trees,adf_signature(model.trees,model.adfs)),scales,interned_grammar(model.mdl_operators),model.mdl_feature_count)
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
            if self.journal is not None: self.journal.append((key,data))
            # The tuned tree is itself a finished model: its later untuned
            # rescoring (stable copies, archives) is the same computation.
            if tune:
                tuned_key=self._cache_key(model,dataset,indices,fit_affine); self._score_cache[tuned_key]=data
                if self.journal is not None: self.journal.append((tuned_key,data))
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

def classification_summary(m, X, Y, cats, output_names):
    """'name: accuracy=..., balanced accuracy=...' for each categorical output, or ''."""
    if cats is None or all(labels is None for labels in cats): return ""
    try: prediction=predict_targets(m,X,cats)
    except (ArithmeticError, IndexError, RecursionError, ValueError): return ""
    parts=[]
    for j,labels in enumerate(cats):
        if labels is None: continue
        truth=np.asarray(np.rint(Y[:,j]),int); hits=np.asarray(np.rint(prediction[:,j]),int)==truth
        recalls=[float(np.mean(hits[truth==label])) for label in np.unique(truth)]
        parts.append(f"{output_names[j]}: accuracy={np.mean(hits):.1%}, balanced accuracy={np.mean(recalls):.1%}")
    return "; ".join(parts)

def _selection_metrics(model, objectives, cats):
    """Extract final-selection metrics without treating age as model quality."""
    target_count=len(cats) if cats is not None else len(_quality_objectives(model))//2
    quality=objectives[:2*target_count]
    return {"loss":float(np.mean(quality[::2])),"shape":float(np.mean(quality[1::2])),
            "mdl_bits":float(objectives[-2])}

def selection_identity(model):
    """Different fitted coefficients or ADF definitions are different predictors."""
    return repr(model.trees),tuple(model.scales),adf_signature(model.trees,model.adfs),tuple(model.mdl_operators),model.mdl_feature_count

# The validation rows of a gridded or integer dataset sit on the same lattice
# as the training rows, so an equation that only memorises lattice points
# (mod/round tricks) can tie the exact law on validation and win on MDL.  The
# final choice therefore also probes between nearest-neighbour rows of all
# known data (training + validation): a candidate whose in-between predictions
# leave the neighbours' target band far more often than the best candidate's
# is not recommended.  A real jump only disturbs the few probes straddling it.
SELECTION_PROBE_FILTER = True
SELECTION_PROBE_ROWS, SELECTION_PROBE_SLACK, SELECTION_PROBE_MARGIN = 200, .05, .10
_SELECTION_PROBES = None
def set_selection_probe_data(X, Y, Xv=None, Yv=None, cats=None):
    """Build the between-row probes used by the final model choice (None disables)."""
    global _SELECTION_PROBES
    _SELECTION_PROBES=None
    if not SELECTION_PROBE_FILTER or X is None: return
    if Xv is not None and len(Xv): X,Y=np.vstack([X,Xv]),np.vstack([Y,Yv])
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim!=2 or len(X)<4: return
    numeric=[j for j in range(Y.shape[1]) if cats is None or cats[j] is None]
    if not numeric: return
    scale=X.std(axis=0); standard=np.divide(X-X.mean(axis=0),scale,out=np.zeros_like(X),where=scale>EPS)
    generator=np.random.default_rng(len(X)+104729)
    rows=generator.choice(len(X),min(SELECTION_PROBE_ROWS,len(X)),replace=False)
    partner,usable=nearest_other_rows(standard,rows)
    rows,partner=rows[usable],partner[usable]
    if not len(rows): return
    fraction=generator.uniform(.15,.85,len(rows))[:,None]
    # Few-valued and integer-valued columns have nothing between their rows
    # (a parity or Collatz rule is undefined at n=3.4): copy, not interpolate.
    discrete=np.array([len(np.unique(X[:,j]))<=10 or bool(np.all(X[:,j]==np.round(X[:,j]))) for j in range(X.shape[1])])
    if discrete.all(): return
    probes=np.where(discrete,X[rows],X[rows]+(X[partner]-X[rows])*fraction)
    bands=[]
    for j in numeric:
        slack=SELECTION_PROBE_SLACK*max(float(np.ptp(Y[:,j])),EPS)
        bands.append((j,np.minimum(Y[rows,j],Y[partner,j])-slack,np.maximum(Y[rows,j],Y[partner,j])+slack))
    _SELECTION_PROBES=(probes,bands)
def between_row_violation(model, cats=None):
    """Share of between-row probes whose prediction leaves the neighbours' band, or None."""
    if _SELECTION_PROBES is None: return None
    probes,bands=_SELECTION_PROBES
    try: prediction=predict_targets(model,probes,cats if cats is not None else [None]*len(model.trees))
    except (ArithmeticError, IndexError, RecursionError, ValueError): return None
    return float(np.mean([np.mean((prediction[:,j]<low)|(prediction[:,j]>high)) for j,low,high in bands]))
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
        violation=between_row_violation(model,cats)
        if violation is not None: metrics["between_row_violation"]=violation
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
    probed=[e[2]["between_row_violation"] for e in entries if "between_row_violation" in e[2]]
    excluded=0
    if probed:
        limit=min(probed)+SELECTION_PROBE_MARGIN
        kept=[e for e in entries if e[2].get("between_row_violation",0.)<=limit]
        excluded=len(entries)-len(kept); entries=kept
    candidates=[e[0] for e in entries]; vectors=[tuple(e[1].objectives) for e in entries]; metrics=[e[2] for e in entries]
    best_loss=min(item["loss"] for item in metrics)
    allowed_loss=best_loss+max(loss_tolerance*abs(best_loss),LOSS_NOISE_FLOOR)
    eligible=[i for i,item in enumerate(metrics) if item["loss"]<=allowed_loss]
    index=min(eligible,key=lambda i:(metrics[i]["mdl_bits"],metrics[i]["shape"],metrics[i]["loss"],repr(candidates[i].trees)))
    return candidates[index],{"source":source,"objectives":vectors[index],"metrics":metrics[index],
        "policy":"loss_tolerance_shortest_mdl","loss_tolerance":loss_tolerance,"best_loss":best_loss,
        "allowed_loss":allowed_loss,"eligible_candidates":len(eligible),"between_row_excluded":excluded}

def constant_selection_warning(model, source):
    """Explain a recommended model that reads no input column, else None."""
    if used_feature_indices(model): return None
    if source=="validation":
        return ("No candidate that uses the inputs beat a constant on the validation rows, so the recommended model is a constant: "
                "the search did not find structure that carries over to held-out data (common with tiny validation sets or rules it never found). "
                "The 'Lowest Training Loss' choice shows the best training fit.")
    return "The recommended model is a constant: no candidate that uses the inputs fit the training data better."
def simplifier_identities(cells, models_of=lambda cell:[*cell.archive.items,*cell.population,cell.best_models.model]):
    """selection_identity keys of every model held by a simplifier island (any stage)."""
    return {selection_identity(model) for cell in cells if role_kind(cell)=="simplifier" for model in models_of(cell) if model is not None}
def simplifier_choice(entries, simplifier_keys, band=None):
    """The shortest simplifier-island model whose selection loss is within band of the best
    candidate's (the anchored lane's band, noise-floor aware), or None.

    The Best Score choice uses the tighter --selection-loss-tolerance across all
    islands, and an accurate anchor can push the simplifier's short models just
    outside it; this entry keeps them selectable."""
    if not simplifier_keys or not entries: return None
    band=SIMPLIFIER_BAND if band is None else band
    best=min(e[2]["loss"] for e in entries); limit=best+max(band*abs(best),LOSS_NOISE_FLOOR)
    inside=[e for e in entries if e[2]["loss"]<=limit and selection_identity(e[0]) in simplifier_keys]
    return min(inside,key=lambda e:(e[2]["mdl_bits"],e[2]["loss"],e[2]["shape"]))[0] if inside else None
def model_options(models, X=None, Y=None, cats=None, loss_tolerance=.01, constraints=None, output_names=(), best_so_far=None, evaluation=None, simplifier_keys=None):
    """Return the deduplicated candidates shown when the user saves a model.

    simplifier_keys (simplifier_identities) adds the simplifier island's
    shortest model within SIMPLIFIER_BAND of the best loss as its own choice."""
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
        *(((f"Shortest within {SIMPLIFIER_BAND:.0%} of the best {selection['source']} loss (simplifier island)",shortest),)
          if (shortest:=simplifier_choice(entries,simplifier_keys)) is not None else ()),
        ("Shortest MDL Model",min(entries,key=lambda e:(e[2]["mdl_bits"],e[2]["loss"],e[2]["shape"]))[0]),
        ("Most Correct Shape",min(entries,key=lambda e:(e[2]["shape"],e[2]["loss"],e[2]["mdl_bits"]))[0]),
        ("Youngest Model",min(entries,key=lambda e:(model_age(e[0]),e[2]["loss"],e[2]["mdl_bits"]))[0]),
    ]
    selection["warning"]=constant_selection_warning(best,selection["source"])
    if selection["source"]=="validation":
        # Validation can reject everything the search found; keep the best
        # training fit visible (clearly labelled, and second when the default
        # is a constant) instead of hiding it behind the constant.
        fit=("Lowest Training Loss (not supported by validation)",min(entries,key=lambda e:(aggregate_loss(e[0]),e[2]["mdl_bits"]))[0])
        if selection["warning"]: candidates.insert(1,fit)
        else: candidates.append(fit)
    labels=[]; choices=[]; seen=set()
    for label,model in candidates:
        key=selection_identity(model)
        if key not in seen:
            labels.append(label); choices.append(model); seen.add(key)
    return labels,choices,selection

# --constant-snapping final: before the final choice, fitted constants of the
# strongest candidates are rounded to simpler values (integers, p/q, multiples
# of pi/e/sqrt2/ln2, powers of ten, short decimals) when neither training nor
# validation loss rises by more than max(noise floor, SNAP_TOLERANCE * loss)
# and the numeric guard and constraints still pass.  Validation is required
# too: a snapped jump threshold can match every sampled row and still move
# the jump between them.  Selection only sees the result; search is untouched.
CONSTANT_SNAPPING = "final"
SNAP_TOLERANCE = 1e-6
SNAP_WINDOW = .02          # only values within 2% (absolute .02 near zero) are tried
SNAP_CANDIDATE_LIMIT = 64  # models snapped: the loss/MDL front plus the lowest-loss rest
_SNAP_BASES = ((math.pi,"pi"),(math.e,"e"),(math.sqrt(2.),"sqrt2"),(math.log(2.),"ln2"))
_SNAP_MULTIPLIERS = (1.,2.,.5,3.,1/3,4.,.25)

def snap_values(value):
    """Simpler candidate values near ``value``, simplest first."""
    value=float(value); window=SNAP_WINDOW*max(abs(value),1.)
    options=[0.,float(round(value))]
    options+=[round(value*q)/q for q in range(2,13)]
    sign=1. if value>=0 else -1.
    options+=[sign*base*multiplier for base,_ in _SNAP_BASES for multiplier in _SNAP_MULTIPLIERS]
    if value: options.append(sign*10.**round(math.log10(abs(value))))
    options+=[float(f"{value:.{digits}g}") for digits in range(1,5)]
    seen=set(); result=[]
    for option in options:
        option=float(option)
        if option==value or option in seen or not math.isfinite(option) or abs(option-value)>window: continue
        seen.add(option); result.append(option)
    return result

def _snap_scores(model, Xt, Yt, Xv, Yv, affine_on, cats, constraints, output_names):
    """Re-fit the readout on training data and return (model, train loss, validation loss, violations)."""
    scored=model.clone()
    assess(scored,Xt,Yt,affine_on,cats,fit_affine=True,constraints=constraints,output_names=output_names)
    if not scored.feasible: return scored,float("inf"),float("inf"),()
    violations=tuple(scored.objectives[2*len(cats):2*len(cats)+scored.constraint_count])
    validation=frozen_metrics(scored,Xv,Yv,cats,constraints,output_names)["loss"] if Xv is not None else 0.
    return scored,aggregate_loss(scored),validation,violations

def snap_model_constants(model, Xt, Yt, Xv, Yv, affine_on, cats, constraints=None, output_names=()):
    """Greedy constant snapping; returns (model, constants snapped).

    Every snap is compared against the unsnapped model, so tolerances do not
    accumulate across constants."""
    if not any(constant_paths(tree) for tree in model.trees): return model,0
    base,train,validation,violations=_snap_scores(model,Xt,Yt,Xv,Yv,affine_on,cats,constraints,output_names)
    if not base.feasible or not np.isfinite(train): return model,0
    train_bound=train+max(LOSS_NOISE_FLOOR,SNAP_TOLERANCE*abs(train))
    validation_bound=validation+max(LOSS_NOISE_FLOOR,SNAP_TOLERANCE*abs(validation))
    current=base; snapped=0
    for head in range(len(model.trees)):
        for path in constant_paths(current.trees[head]):
            value=float(subtree_at(current.trees[head],path)[1])
            for option in snap_values(value):
                trial=current.clone(); trial.trees[head]=replace_subtree(trial.trees[head],path,("c",option))
                scored,trial_train,trial_validation,trial_violations=_snap_scores(trial,Xt,Yt,Xv,Yv,affine_on,cats,constraints,output_names)
                if (scored.feasible and trial_train<=train_bound and trial_validation<=validation_bound
                        and all(a<=b+1e-12 for a,b in zip(trial_violations,violations))):
                    current=scored; snapped+=1; break
    if not snapped: return model,0
    # Snapping to 0 or 1 can make subtrees removable; keep the shorter tree only if it scores the same.
    simplified=current.clone(); simplified.trees=[simplify_tree(tree) for tree in current.trees]
    if simplified.trees!=current.trees:
        scored,trial_train,trial_validation,trial_violations=_snap_scores(simplified,Xt,Yt,Xv,Yv,affine_on,cats,constraints,output_names)
        if scored.feasible and trial_train<=train_bound and trial_validation<=validation_bound: current=scored
    current.origin=getattr(model,"origin","")
    before,after=[constant_vector(tree) for tree in model.trees],[constant_vector(tree) for tree in current.trees]
    current.history=history_append(model.history,{"event":"snapped","constants":snapped,"changed":current.trees!=model.trees,
        "values":[[float(a),float(b)] for old,new in zip(before,after) if len(old)==len(new) for a,b in zip(old,new) if a!=b],
        "bits":[_history_score(model)[0],_history_score(current)[0]],"loss":[_history_number(train),_history_number(aggregate_loss(current))]})
    return current,snapped

def report_snapping(summary):
    if summary["mode"]=="final" and summary["models_tried"]:
        print(f"Constant snapping: {summary['constants_snapped']} constant(s) simplified in {summary['models_snapped']} of {summary['models_tried']} final candidates.")

def snap_final_candidates(models, Xt, Yt, Xv, Yv, affine_on, cats, constraints=None, output_names=(), limit=None):
    """Return ``models`` with the strongest candidates replaced by snapped copies, plus a summary."""
    limit=SNAP_CANDIDATE_LIMIT if limit is None else limit
    summary={"mode":CONSTANT_SNAPPING,"tolerance":SNAP_TOLERANCE,"models_tried":0,"models_snapped":0,"constants_snapped":0}
    if CONSTANT_SNAPPING!="final": return list(models),summary
    unique={}
    for model in models:
        if model is not None and model.feasible and np.all(np.isfinite(model.scales)): unique.setdefault(selection_identity(model),model)
    pool=sorted(unique.values(),key=lambda m:(aggregate_loss(m),model_complexity(m)))
    front=[]; best_bits=float("inf")
    for model in pool:  # loss/MDL front: each member is shorter than every lower-loss model
        bits=model_complexity(model)
        if bits<best_bits: front.append(model); best_bits=bits
    chosen=list({id(model):model for model in [*front,*pool]}.values())[:limit]
    replacements={}
    for model in chosen:
        summary["models_tried"]+=1
        snapped,count=snap_model_constants(model,Xt,Yt,Xv,Yv,affine_on,cats,constraints,output_names)
        if count:
            replacements[selection_identity(model)]=snapped; summary["models_snapped"]+=1; summary["constants_snapped"]+=count
    # Replace every copy, or an unsnapped twin would still compete for "Lowest Loss".
    return [replacements.get(selection_identity(model),model) if model is not None else None for model in models],summary

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

# --symbolic-export: the chosen model written twice to best_model_symbolic.txt,
# each with its LaTeX:
# - exact: afpo's protected operators as evaluated (log(|x|+eps), guarded
#   division, signed roots, clipped exp...) with full-precision constants, so
#   it reproduces the model's predictions.  Only the final +/-CLIP clamp and
#   NaN -> 0 that every node applies are left out.
# - raw: the same model with the protections removed (log(x), x/y, x**y,
#   x**(1/3)...) and constants written as fractions or multiples of pi when
#   within 1e-8 relative of one, else to six significant digits.  It is the
#   readable form, valid only where its operators are defined, so the export
#   reports how closely it matches the model on the training rows.
# Inputs that are positive on the training data are declared positive, so
# Abs() and sign() of them simplify away.  Operators without a closed form
# stay as named functions.
SYMBOLIC_EXPORT = "on"
SYMBOLIC_SIMPLIFY_NODES = 40
READABLE_CONSTANT_TOLERANCE = 1e-8
SYMBOLIC_TIME_LIMIT = 20.  # seconds of SymPy rearranging per form

def readable_constant(value, exact=False):
    """A SymPy number for value: a fraction or multiple of pi when it is one
    (exactly for exact=True, within READABLE_CONSTANT_TOLERANCE otherwise),
    else the shortest decimal (exact) or six significant digits (raw)."""
    import sympy as sp
    from fractions import Fraction
    value=float(value)
    if not math.isfinite(value): return sp.Float(value)
    if value==int(value) and abs(value)<1e15: return sp.Integer(int(value))
    tolerance=0. if exact else READABLE_CONSTANT_TOLERANCE*max(1.,abs(value))
    for base,symbol,limit in ((1.,sp.Integer(1),100),(math.pi,sp.pi,12)):
        ratio=Fraction(value/base).limit_denominator(limit)
        if ratio and abs(float(ratio)*base-value)<=tolerance:
            return sp.Rational(ratio.numerator,ratio.denominator)*symbol
    text=repr(value) if exact else f"{value:.6g}"
    digits=len(text.lower().split("e")[0].replace("-","").replace(".","").lstrip("0")) or 1
    return sp.Float(text,max(digits,1))

def sympy_expression(tree, symbols, adfs=None, mode="exact", constant=None):
    """tree as a SymPy expression; mode "exact" keeps afpo's guards, "raw" drops them."""
    import sympy as sp
    constant=constant or (lambda v: readable_constant(v,mode=="exact"))
    eps=sp.Float(EPS,1)
    def clip(v, lo, hi): return sp.Min(sp.Max(v,lo),hi)
    def nonzero(v, fill): return sp.Piecewise((fill,sp.Abs(v)<eps),(v,True))
    def build(t, args=None):
        if t[0]=="x": return symbols[t[1]]
        if t[0]=="c": return constant(t[1])
        if t[0]=="arg": return args[t[1]]
        values=[build(child,args) for child in t[1:]]
        op=t[0]
        if op.startswith("adf_") and adfs and op in adfs: return build(adfs[op]["tree"],values)
        x=values[0]; y=values[1] if len(values)>1 else None
        powers={"square":2,"cube":3,**{f"pow{k}":k for k in range(4,11)}}
        if op in powers: return x**powers[op]
        if op.startswith("root") and op[4:].isdigit():
            n=sp.Rational(1,int(op[4:]))
            return sp.sign(x)*sp.Abs(x)**n if mode=="exact" else x**n
        sigmoid=lambda v: 1/(1+sp.exp(-v))
        table={
            "+":lambda:x+y,"-":lambda:x-y,"*":lambda:x*y,"/":lambda:x/y,"neg":lambda:-x,"delta":lambda:sp.Abs(x-y),
            "max":lambda:sp.Max(x,y),"min":lambda:sp.Min(x,y),"pow":lambda:x**y,
            "hypot":lambda:sp.sqrt(x**2+y**2),"distance_2":lambda:sp.sqrt(x**2+y**2),
            "log_base":lambda:sp.log(x)/sp.log(y),"exp_decay":lambda:sp.exp(-x*y),"rbf":lambda:sp.exp(-(x-y)**2),
            "atan2":lambda:sp.atan2(x,y),"geometric":lambda:sp.sqrt(x*y),"harmonic":lambda:2*x*y/(x+y),
            "mod":lambda:sp.Mod(x,y),"floordiv":lambda:sp.floor(x/y),"copysign":lambda:sp.Abs(x)*sp.sign(y),
            "gt":lambda:sp.Piecewise((1,x>y),(0,True)),"lt":lambda:sp.Piecewise((1,x<y),(0,True)),
            "gte":lambda:sp.Piecewise((1,x>=y),(0,True)),"lte":lambda:sp.Piecewise((1,x<=y),(0,True)),
            "eq":lambda:sp.Piecewise((1,sp.Eq(x,y)),(0,True)),"ne":lambda:sp.Piecewise((0,sp.Eq(x,y)),(1,True)),
            "sin":lambda:sp.sin(x),"cos":lambda:sp.cos(x),"tan":lambda:sp.tan(x),"exp":lambda:sp.exp(x),"expm1":lambda:sp.exp(x)-1,
            "10^x":lambda:10**x,"log":lambda:sp.log(x),"log1p":lambda:sp.log(1+x),"log10":lambda:sp.log(x,10),
            "sqrt":lambda:sp.sqrt(x),"abs":lambda:sp.Abs(x),"inv":lambda:1/x,"oom":lambda:sp.floor(sp.log(x,10)),
            "frac":lambda:x-sp.floor(x),"round":lambda:sp.Function("round")(x),"floor":lambda:sp.floor(x),"ceil":lambda:sp.ceiling(x),
            "int":lambda:sp.Function("trunc")(x),"sigmoid":lambda:sigmoid(x),"tanh":lambda:sp.tanh(x),"sinh":lambda:sp.sinh(x),
            "cosh":lambda:sp.cosh(x),"relu":lambda:sp.Max(x,0),"atan":lambda:sp.atan(x),"gaussian":lambda:sp.exp(-x**2),
            "softplus":lambda:sp.log(1+sp.exp(x)),"sign":lambda:sp.sign(x),"sinc":lambda:sp.sinc(x),"xlogx":lambda:x*sp.log(x),
            "erf":lambda:sp.erf(x),"deg2rad":lambda:x*sp.pi/180,"rad2deg":lambda:x*180/sp.pi,
            "perceptronSigma1":lambda:sigmoid(x),"perceptronReLU1":lambda:sp.Max(x,0),"perceptronCustom1":lambda:x*sigmoid(x),
            "perceptronSigma2":lambda:sigmoid(x+y),"perceptronReLU2":lambda:sp.Max(x+y,0),"perceptronCustom2":lambda:(x+y)*sigmoid(x+y),
            "if_else":lambda:sp.Piecewise((values[1],x>sp.Rational(1,2)),(values[2],True)),
            "if_in_range":lambda:clip(x,sp.Min(values[1],values[2]),sp.Max(values[1],values[2])),
            "lerp":lambda:x+(y-x)*values[2],"leaky_relu":lambda:sp.Piecewise((x,x>=0),(x/100,True)),
            "distance_3":lambda:sp.sqrt(x**2+y**2+values[2]**2),
        }
        if mode=="exact":
            # The guards of op_eval, operator by operator.
            log_abs=lambda v: sp.log(sp.Abs(v)+eps)
            table.update({
                "/":lambda:x/nonzero(y,eps),"pow":lambda:sp.sign(x)*sp.Abs(x)**clip(y,-12,12),
                "log_base":lambda:log_abs(x)/sp.log(sp.Abs(y)+sp.Float("1.000001",7)),
                "exp_decay":lambda:sp.exp(-clip(x*y,-50,50)),"rbf":lambda:sp.exp(-sp.Min((x-y)**2,50)),
                "geometric":lambda:sp.sqrt(sp.Abs(x*y)),"harmonic":lambda:2*x*y/(sp.Abs(x+y)+eps),
                "mod":lambda:sp.Mod(x,nonzero(y,1)),"floordiv":lambda:sp.floor(x/nonzero(y,1)),
                "tan":lambda:sp.tan(clip(x,sp.Float("-1.55",3),sp.Float("1.55",3))),
                "exp":lambda:sp.exp(clip(x,-50,50)),"expm1":lambda:sp.exp(clip(x,-50,50))-1,"10^x":lambda:10**clip(x,-12,12),
                "log":lambda:log_abs(x),"log1p":lambda:sp.log(1+sp.Abs(x)),"log10":lambda:sp.log(sp.Abs(x)+eps,10),
                "sqrt":lambda:sp.sign(x)*sp.sqrt(sp.Abs(x)),"inv":lambda:1/(sp.Abs(x)+eps),
                "oom":lambda:sp.floor(sp.log(sp.Abs(x)+eps,10)),"xlogx":lambda:x*log_abs(x),
                "sigmoid":lambda:sigmoid(clip(x,-50,50)),"perceptronSigma1":lambda:sigmoid(clip(x,-50,50)),
                "perceptronCustom1":lambda:x*sigmoid(clip(x,-50,50)),"perceptronSigma2":lambda:sigmoid(clip(x+y,-50,50)),
                "perceptronCustom2":lambda:(x+y)*sigmoid(clip(x+y,-50,50)),
                "sinh":lambda:sp.sinh(clip(x,-20,20)),"cosh":lambda:sp.cosh(clip(x,-20,20)),
                "gaussian":lambda:sp.exp(-sp.Min(x**2,50)),
            })
        return table[op]() if op in table else sp.Function(op)(*values)
    return build(tree)

GREEK_LETTERS = {"alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda","mu","nu","xi",
                 "pi","rho","sigma","tau","upsilon","phi","chi","psi","omega","Gamma","Delta","Theta","Lambda","Xi",
                 "Pi","Sigma","Upsilon","Phi","Psi","Omega"}

def latex_symbol_name(name):
    """x3 -> x_{3}, alpha -> \\alpha; any other multi-letter name is one upright
    word (\\mathrm{speed\\_ms}), not a product of italic letters."""
    match=re.fullmatch(r"([A-Za-z]|"+"|".join(sorted(GREEK_LETTERS,key=len,reverse=True))+r")_?(\d+)",name)
    letter=lambda head: "\\"+head if head in GREEK_LETTERS else head
    if match: return f"{letter(match.group(1))}_{{{match.group(2)}}}"
    if name in GREEK_LETTERS or len(name)==1: return letter(name)
    return r"\mathrm{"+re.sub(r"([_&%$#{}])",r"\\\1",name)+"}"

def mathml_expression(expression, symbols):
    """Presentation MathML, with each multi-letter input name as one upright
    word (SymPy would split speed_ms into speed with subscript ms)."""
    import sympy as sp
    from html import escape
    text=sp.mathml(expression,printer="presentation")
    for symbol in sorted(symbols,key=lambda q: -len(q.name)):
        if latex_symbol_name(symbol.name).startswith(r"\mathrm"):
            text=text.replace(sp.mathml(symbol,printer="presentation"),f'<mi mathvariant="normal">{escape(symbol.name)}</mi>')
    return text

def _symbolic_form(model, head, symbols, mode):
    """Simplified exact or raw form.  The raw form is rearranged with full
    constants and only then rounded, so expanding a product does not compound
    rounded factors; of the candidate forms the one with the shortest LaTeX wins."""
    import sympy as sp
    a,b=model.scales[head]
    constant=lambda v: readable_constant(v,True)
    expression=constant(a)*sympy_expression(model.trees[head],symbols,model.adfs,mode,constant)+constant(b)
    finish=(lambda e: e) if mode=="exact" else (lambda e: e.xreplace({f:readable_constant(float(f)) for f in e.atoms(sp.Float)}))
    candidates=[expression]
    if node_size(model.trees[head])<=SYMBOLIC_SIMPLIFY_NODES:
        def rearrange():
            candidates.append(sp.simplify(expression))
            if mode=="raw": candidates.extend([sp.expand(candidates[-1]),sp.factor_terms(sp.expand(candidates[-1]))])
        try: _within_time_limit(rearrange,SYMBOLIC_TIME_LIMIT)
        except Exception: pass  # simplification is cosmetic; keep what finished
    shown=[]
    for candidate in candidates:
        try: candidate=finish(candidate); shown.append((len(sp.latex(candidate)),candidate))
        except Exception: pass  # SymPy can fail on an odd form (e.g. Max of zoo); skip that candidate
    if not shown: raise ValueError("no printable form")
    return min(shown,key=lambda item: item[0])[1]

class _SymbolicTimeout(Exception): pass

def _within_time_limit(function, seconds):
    """Run function, raising _SymbolicTimeout after seconds where SIGALRM is
    usable (Unix main thread); elsewhere it runs unguarded."""
    if not hasattr(signal,"SIGALRM") or threading.current_thread() is not threading.main_thread():
        return function()
    def expire(*_): raise _SymbolicTimeout()
    previous=signal.signal(signal.SIGALRM,expire); signal.setitimer(signal.ITIMER_REAL,seconds)
    try: return function()
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)

def _raw_agreement(expression, symbols, X, reference):
    """(share of rows where the raw form is finite, max |raw - model| / output spread) or None."""
    import sympy as sp
    try:
        function=sp.lambdify(symbols,expression,"numpy")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values=np.broadcast_to(np.asarray(function(*[X[:,i] for i in range(X.shape[1])]),dtype=complex),reference.shape)
    except Exception:
        return None
    defined=np.isfinite(values)&(np.abs(values.imag)<=1e-12*(1+np.abs(values.real)))
    spread=max(float(np.ptp(reference)),float(np.max(np.abs(reference))),1e-300)
    gap=float(np.max(np.abs(values.real[defined]-reference[defined])))/spread if defined.any() else float("nan")
    return float(defined.mean()),gap

def latex_text(value):
    """\text{...} of a class label, LaTeX specials escaped."""
    escaped=re.sub(r"([&%$#_{}])",r"\\\1",str(value)).replace("~",r"\textasciitilde{}").replace("^",r"\textasciicircum{}")
    return r"\text{"+escaped+"}"
def classification_decision(name, labels, score_names):
    """How a classifier's score equations become a class: LaTeX lines and plain text.

    Mirrors predict_targets: one head for two classes (the second label when
    the score rounds to 1, probability sigmoid(s - 1/2)); one score per class
    for three or more (arg max, softmax probabilities)."""
    target=latex_symbol_name(name.replace(" ","_"))
    hat=(lambda t: rf"\hat{{{t}}}") if len(name)==1 else (lambda t: rf"\widehat{{{t}}}")
    if len(labels)==1:
        return {"latex":[rf"{hat(target)} = {latex_text(labels[0])}"],"text":[f"{name} is always {labels[0]!r}"]}
    if len(labels)==2:
        s=score_names[0]
        return {"latex":[rf"{hat(target)} = \begin{{cases}} {latex_text(labels[1])} & {s} \ge \tfrac{{1}}{{2}} \\ {latex_text(labels[0])} & \text{{otherwise}} \end{{cases}}",
                         rf"P\left({target} = {latex_text(labels[1])}\right) = \frac{{1}}{{1 + e^{{-\left({s} - \frac{{1}}{{2}}\right)}}}}"],
                "text":[f"{name} = {labels[1]!r} if score >= 0.5, else {labels[0]!r}",f"P({name} = {labels[1]!r}) = 1 / (1 + exp(-(score - 0.5)))"]}
    return {"latex":[rf"{hat(target)} = \operatorname*{{arg\,max}}_{{k}}\; s_{{k}}",
                     rf"P\left({target} = k\right) = \frac{{e^{{s_{{k}}}}}}{{\sum_{{j}} e^{{s_{{j}}}}}}"],
            "text":[f"{name} = the class with the highest score",f"P({name} = k) = softmax of the class scores"]}
def symbolic_model(model, feature_names, output_names, cats, positive=(), X=None):
    """{name: {"exact": (expr, latex), "raw": (expr, latex), "agreement": ...}}
    per equation head; None without SymPy.  agreement compares the raw form
    with the model on X (None without X).  Regression outputs key by output
    name; classifier score heads key as "name score" (two classes) or
    "name[label] score" and also carry "output", "name_latex" (s_y or s_label)
    and, on the target's last head, "decision" (classification_decision)."""
    try: import sympy as sp
    except ImportError: return None
    symbols=[sp.Symbol(name.replace(" ","_"),real=True,positive=True if index in positive else None) for index,name in enumerate(feature_names)]
    names={symbol:latex_symbol_name(symbol.name) for symbol in symbols}
    predictions=None if X is None else predict_model(model,X)
    targets,_=classification_layout(cats); result={}
    for j,heads in enumerate(targets):
        labels=cats[j]; score_names=[]
        for k,head in enumerate(heads):
            entry={}
            for mode in ("exact","raw"):
                # An export must never stop a run: a form SymPy cannot handle is reported, not raised.
                try:
                    expression=_symbolic_form(model,head,symbols,mode)
                    entry[mode]=(expression,sp.latex(expression,symbol_names=names,ln_notation=True,mul_symbol=r"\,"))
                except Exception as error:
                    entry[mode]=(None,None); entry.setdefault("errors",{})[mode]=f"{type(error).__name__}: {error}"
            entry["agreement"]=None if predictions is None or entry["raw"][0] is None else _raw_agreement(entry["raw"][0],symbols,X,predictions[:,head])
            if labels is None: result[output_names[j]]=entry; continue
            many=len(labels)>2
            entry["output"]=output_names[j]
            entry["name_latex"]=f"s_{{{latex_text(labels[k])}}}" if many else f"s_{{{latex_symbol_name(output_names[j].replace(' ','_'))}}}"
            score_names.append(entry["name_latex"])
            if k==len(heads)-1: entry["decision"]=classification_decision(output_names[j],labels,score_names)
            result[f"{output_names[j]}[{labels[k]}] score" if many else f"{output_names[j]} score"]=entry
    return result

def write_symbolic_export(model, feature_names, output_names, cats, X=None, path="best_model_symbolic.txt"):
    if SYMBOLIC_EXPORT!="on": return None
    positive=() if X is None else tuple(index for index in range(X.shape[1]) if np.all(X[:,index]>0))
    try: result=symbolic_model(model,feature_names,output_names,cats,positive,X)
    except Exception as error:  # the export is optional; it must never stop a run
        print(f"Symbolic export skipped: {type(error).__name__}: {error}"); return None
    if result is None:
        print("Symbolic export skipped: install sympy for a simplified equation and LaTeX."); return None
    lines=["# afpo symbolic export",
           "# exact: afpo's protected operators as evaluated, full-precision constants (the per-node +/-1e12 clamp and NaN->0 are omitted)",
           "# raw: protections removed, constants as fractions or 6 significant digits; valid only where its operators are defined",""]
    for name,entry in result.items():
        for mode in ("exact","raw"):
            expression,latex=entry[mode]
            if expression is None: lines.append(f"{name} ({mode}): not converted ({entry['errors'][mode]})"); continue
            lines+=[f"{name} ({mode}) = {expression}",f"LaTeX ({mode}): {name} = {latex}"]
        agreement=entry["agreement"]
        if agreement is not None:
            share,gap=agreement
            lines.append(f"raw vs model on training rows: defined on {100*share:.1f}% of rows, max |difference| = {gap:.2g} of the output range")
        decision=entry.get("decision")
        if decision:
            lines+=[f"decision: {text}" for text in decision["text"]]+[f"LaTeX (decision): {latex}" for latex in decision["latex"]]
        lines.append("")
        if entry["raw"][0] is not None: print(f"Symbolic: {name} = {entry['raw'][0]}")
    Path(path).write_text("\n".join(lines))
    return result

# --constant-intervals: approximate 95% intervals for the chosen model's
# fitted constants and readout from the linearised least-squares covariance
# s^2 (J^T J)^-1 on the training rows.  A constant whose interval spans zero
# or is enormous is not pinned down by the data.  Local and approximate:
# jump constants (zero gradient) get no interval.
CONSTANT_INTERVALS = "on"

def constant_intervals(model, X, Y, cats):
    """[{output, kind, path, value, se, low, high}] for regression heads."""
    targets,_=classification_layout(cats); report=[]
    for j,heads in enumerate(targets):
        if cats[j] is not None: continue
        head=heads[0]; tree=model.trees[head]; a,b=model.scales[head]; y=np.asarray(Y[:,j],float)
        values=np.asarray(constant_vector(tree),float); paths=constant_paths(tree)
        def predict(theta):
            return theta[-2]*np.asarray(evaluate_cached(with_constants(tree,theta[:-2]),X,model.adfs),float)+theta[-1]
        theta=np.concatenate([values,[a,b]])
        try:
            base=predict(theta); J=np.empty((len(y),len(theta)))
            for k in range(len(theta)):
                step=1e-6*max(abs(theta[k]),1.); probe=theta.copy(); probe[k]+=step
                J[:,k]=(predict(probe)-base)/step
        except (ArithmeticError, IndexError, RecursionError, ValueError): continue
        if not np.isfinite(J).all() or not np.isfinite(base).all(): continue
        dof=len(y)-len(theta)
        if dof<=0: continue
        sigma2=float(np.sum((base-y)**2))/dof
        identified=np.linalg.norm(J,axis=0)>1e-12*max(np.linalg.norm(base),1.)
        covariance=np.full((len(theta),len(theta)),np.nan)
        if identified.any():
            covariance[np.ix_(identified,identified)]=sigma2*np.linalg.pinv(J[:,identified].T@J[:,identified])
        for k,value in enumerate(theta):
            se=float(np.sqrt(max(covariance[k,k],0.))) if np.isfinite(covariance[k,k]) else None
            report.append({"output":j,"kind":"constant" if k<len(values) else ("scale" if k==len(values) else "intercept"),
                           "path":list(paths[k]) if k<len(values) else None,"value":float(value),"se":se,
                           "low":None if se is None else float(value-1.96*se),"high":None if se is None else float(value+1.96*se)})
    return report

def print_constant_intervals(report):
    if not report: return
    shown=", ".join(f"{item['value']:.6g}"+(f" ± {1.96*item['se']:.2g}" if item["se"] is not None else " (not identified)") for item in report)
    print(f"Approximate 95% intervals (constants, then scale and intercept): {shown}")

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
    X=transform(df); raw=np.column_stack([clean(a*ev(t,X)+b) for t,(a,b) in zip(MODEL['trees'],MODEL['scales'])]); out=pd.DataFrame(index=df.index); head=0
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
            labels=actual.astype(str).to_numpy(); matches=predicted[output].astype(str).to_numpy()==labels
            accuracy=100.*float(matches.mean())
            # Mean per-class recall: a model that never predicts a rare class cannot hide behind the majority.
            balanced=100.*float(np.mean([matches[labels==label].mean() for label in np.unique(labels)]))
            report[output]={{'type':'categorical','rows':len(frame),'accuracy':accuracy,'balanced_accuracy':balanced}}
            print(f'{{output}}: categorical accuracy={{accuracy:.6g}}%, balanced accuracy={{balanced:.6g}}% ({{int(matches.sum())}}/{{len(frame)}} exact matches)')
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
    # Per-cell random streams and lineage-id counter (see cell_streams); None
    # means the cell draws from the shared module streams (one-cell runs and
    # checkpoints made before cells could run in parallel).
    streams:Any=None
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
        **({"streams":island.streams} if island.streams is not None else {}),
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
        data.get("streams"),
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

def migrate_islands(islands, migrant_count, *, X, nsga_normalization, parsimony_quality_tolerance, evaluator=None, generation=None):
    """Send local Pareto elites around a ring, keeping each island's size fixed."""
    if len(islands)<2 or migrant_count<1: return 0
    outgoing=[]
    for island in islands:
        if evaluator is not None:
            refresh_persistent_scores(island.archive,island.best_models,island.semantic_qd,island.structural_qd,evaluator,island.residual_qd)
            evaluator.assess(island.population,"train")
        pool=[model for model in island.population if model.feasible]
        if not pool: pool=island.population
        leaving=(anchored_emigrants(pool,min(migrant_count,len(pool)),nsga_normalization,parsimony_quality_tolerance)
                 if (island.role.get("params") or {}).get("anchored") else
                 select_nsga(pool,min(migrant_count,len(pool)),nsga_normalization,parsimony_quality_tolerance))
        outgoing.append([model.clone() for model in leaving])
    # A simplifier island takes every other island's elites, not just its
    # ring neighbour's: shortening them is its whole job.  Every island's
    # arrivals are drawn before any is stamped, so a gathered copy never
    # inherits the ring copy's migration record.
    gathers=[bool((island.role.get("params") or {}).get("gather_migrants")) for island in islands]
    arrivals=[[(source,model.clone()) for source,models in enumerate(outgoing) if source!=index for model in models] if gathers[index] else
              [((index-1)%len(islands),model) for model in outgoing[(index-1)%len(islands)]] for index in range(len(islands))]
    for index,island in enumerate(islands):
        destination=history_place(island)
        for source,migrant in arrivals[index]:
            anchored=(islands[source].role.get("params") or {}).get("anchored")
            history_moved(migrant,"migrated",generation,history_place(islands[source]),destination,
                          "gathered" if gathers[index] else "anchored emigrant" if anchored else "ring")
        incoming=[migrant for _,migrant in arrivals[index]]
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
        trees=[admissible_random_tree(n_features,active_ops,nodes,depth,adfs=definitions) for _ in range(head_count)]
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
            for m in incoming:
                m.origin="stage_promotion"; history_moved(m,"promoted",generation,history_place(source),history_place(receiver))
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
            migrated+=migrate_islands(level,island_config["migrants_per_island"],X=X,nsga_normalization=nsga_normalization,generation=generation,
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
        contributions=", ".join(f"{cell_label(cell,island_count,stage_count)} ({role_kind(cell)})={cell.role.get('contribution',0.):.3g}" for cell in cells if cell.island_index>0)
        extra="".join([f"; retired {retired}" if retired else "",f"; split {collapsed} collapsed pair(s)" if collapsed else ""])
        print(f"Island roles {roles['updates']}: held-out contribution {contributions}{extra}.",flush=True)
# Self-organising island roles ("auto").  No niche is named in advance: each
# auto island's parent selection weights the training rows it already handles
# better than the other islands (soft responsibilities, as in a mixture of
# experts), so the positive feedback splits the data into niches for any
# island count.  Island 0 of each stage level stays a generalist.  Auto islands
# also get a deterministic spread of search settings (tree size, crossover,
# Bayesian proposals), a held-out complementarity check retires specialists
# that stop contributing, and fragment libraries migrate with the ring.  All
# of it is selection-only: reported losses, archives, and the final pick use
# the true unweighted objective.
ROLE_FRAGMENT_MIGRANTS = 4
ROLE_COLLAPSE_CORRELATION = .95
ROLE_MIN_CONTRIBUTION = 1e-3
# Island roles.  Every island after the first gets one: "auto" is the
# self-organising specialist above; the others are fixed presets the user
# picks per island.  A preset only changes how that island searches (tree
# size, variation mix, step size, parsimony, which operators build new
# structure, where its migrants come from); every model is still scored,
# priced (MDL over the run's whole grammar) and selected on the same
# objectives, so roles cannot bias the final choice.
ISLAND_ROLES={
    "auto":"Auto: learns its niche from the data (self-organising specialist)",
    "generalist":"Generalist: default settings, no specialisation",
    "simplifier":"Simplifier: gathers every island's elites and searches for shorter equivalents",
    "explorer":"Explorer: fresh random structures and large jumps to escape plateaus",
    "refiner":"Refiner: small steps that polish the structures it receives",
}
FAMILY_ROLE_PREFIX="family:"
def family_role_choices(ops):
    """{role: label} for the operator-family roles available with these operators.

    A family island builds new structure from arithmetic (group 1) plus one
    other operator group, restricted to the run's operators."""
    enabled=set(ops); arithmetic=set(OPERATOR_GROUPS["1"][1])&enabled; choices={}
    for key,(name,members) in OPERATOR_GROUPS.items():
        if not set(members)&enabled: continue
        label=f"Operators: {name.lower()}" if key=="1" or not arithmetic else f"Operators: arithmetic + {name.lower()}"
        choices[FAMILY_ROLE_PREFIX+key]=label
    return choices
def island_role_choices(ops):
    """Every role selectable for a run with these operators, in menu order."""
    return {**ISLAND_ROLES,**family_role_choices(ops)}
def family_role_operators(role, ops):
    """Operators a family-role island builds new structure from."""
    group=role[len(FAMILY_ROLE_PREFIX):]
    if group not in OPERATOR_GROUPS: raise ValueError(f"Unknown operator group in island role {role!r}")
    members=set(OPERATOR_GROUPS[group][1])|set(OPERATOR_GROUPS["1"][1])
    chosen=[op for op in ops if op in members]
    if not chosen: raise ValueError(f"Island role {role!r} has no operators among the selected ones")
    return chosen
def validate_island_roles(assignments, island_count, ops):
    """Per-island roles for islands 2..N; an empty list means all auto."""
    assignments=[str(role).strip() for role in (assignments or [])]
    if not assignments: assignments=["auto"]*max(0,island_count-1)
    if len(assignments)!=island_count-1:
        raise ValueError(f"Island roles need one entry per island after the first ({island_count-1}), got {len(assignments)}")
    choices=island_role_choices(ops)
    for role in assignments:
        if role not in choices: raise ValueError(f"Unknown island role {role!r}; choose from {', '.join(choices)}")
    return assignments
def role_config(enabled=False, interval=10, mix=.5, retire_after=5, assignments=(), **state):
    if int(interval)<1 or int(retire_after)<1: raise ValueError("Role interval and retirement window must be positive")
    if not 0<=float(mix)<=.9: raise ValueError("Role mix must be in [0, 0.9] so every island still sees all rows")
    return {"enabled":bool(enabled),"interval":int(interval),"mix":float(mix),"retire_after":int(retire_after),
            "assignments":[str(role) for role in (assignments or [])],
            "updates":int(state.get("updates",0)),"retirements":int(state.get("retirements",0)),
            "collapses":int(state.get("collapses",0)),"fragment_migrants":int(state.get("fragment_migrants",0))}
def role_parameters(t, crossover_rate, bayesian_proposal_rate, nodes):
    """Search settings along one axis t in [0,1]: small, proposal-driven trees
    at 0; large, crossover-driven trees at 1."""
    t=float(np.clip(t,0.,1.))
    return {"t":t,"crossover_rate":float(np.clip(crossover_rate*(.5+t),0.,.9)),
            "bayesian_proposal_rate":float(np.clip(bayesian_proposal_rate*(1.5-t),0.,1.)),
            "nodes":max(3,int(round(nodes*(.5+.5*t))))}
# Mutation-portfolio multipliers per preset (kinds not listed keep x1).
ROLE_MUTATION_BIAS={
    "simplifier":{"prune":4.,"hoist":3.,"shrink":3.,"constant":1.5,"subtree":.5,"bilinear":0.,"residual_term":0.,"backprop":.5,
                  "jump":.3,"gate":.3,"squash":.5,"smooth":.5},
    "explorer":{"subtree":2.,"point":1.5,"constant":.5},
    "refiner":{"constant":3.,"parametrize":2.,"point":1.5,"subtree":.5,"jump":.5,"hoist":.3,"shrink":.3,"bilinear":.5},
}
# Anchored simplifier.  Plain Pareto survival with a parsimony near-tie band
# left the simplifier island holding migrant copies: different structures
# rarely land within a few percent of each other's loss, so a bigger, better
# model always kept its slot.  The simplifier therefore solves
#     min MDL  subject to  loss <= anchor + max(band*|anchor|, noise floor)
# in one lane of its population, where the anchor is the lowest loss the
# island holds (it gathers every island's elites, so roughly the run's best):
# SIMPLIFIER_LANE_SHARE of survivors are the shortest in-band models no
# larger than the anchor, SIMPLIFIER_PARENT_SHARE of parents come from that
# lane (and their children may not outgrow the anchor), and the rest of the
# island is ordinary Pareto survival with the usual size allowance, a supply
# of strong material.  Emigrants are the shortest models within the final
# choice's band (SIMPLIFIER_FINAL_BAND, the default --selection-loss-tolerance),
# else within SIMPLIFIER_BAND.  Selection-only: scores and the final pick are
# unchanged.
SIMPLIFIER_BAND = .05
SIMPLIFIER_FINAL_BAND = .01
SIMPLIFIER_LANE_SHARE = .5
SIMPLIFIER_PARENT_SHARE = 2/3
EXPLORER_NOVELTY = .25          # share of an explorer's offspring drawn as fresh random trees
REFINER_MAX_DELTA = 1.          # refiner's semantic step cap (target standard deviations)
def preset_role_parameters(role, crossover_rate, bayesian_proposal_rate, nodes, ops):
    """Search settings of a fixed (non-auto) island role."""
    if role=="generalist": return {}
    if role=="simplifier":
        return {"crossover_rate":.5*crossover_rate,"bayesian_proposal_rate":.5*bayesian_proposal_rate,
                "anchored":True,"neutral_shrink":True,"gather_migrants":True,"mutation_bias":ROLE_MUTATION_BIAS["simplifier"]}
    if role=="explorer":
        return {"crossover_rate":float(min(.9,1.5*crossover_rate)),"bayesian_proposal_rate":float(min(1.,1.5*bayesian_proposal_rate)),
                "novelty":EXPLORER_NOVELTY,"semantic_max_delta":float("inf"),"mutation_bias":ROLE_MUTATION_BIAS["explorer"]}
    if role=="refiner":
        return {"crossover_rate":.5*crossover_rate,"bayesian_proposal_rate":.3*bayesian_proposal_rate,
                "semantic_max_delta":REFINER_MAX_DELTA,"mutation_bias":ROLE_MUTATION_BIAS["refiner"]}
    if role.startswith(FAMILY_ROLE_PREFIX): return {"ops":family_role_operators(role,ops)}
    raise ValueError(f"Unknown island role {role!r}")
def assign_role_parameters(cells, island_count, crossover_rate, bayesian_proposal_rate, nodes, assignments=(), ops=()):
    """Give every island after the first its role (all stages of an island share it).

    Auto islands spread their settings evenly over [0,1] in island order;
    island 0 keeps the defaults."""
    assignments=list(assignments) or ["auto"]*max(0,island_count-1)
    auto=[index for index in range(1,island_count) if assignments[index-1]=="auto"]
    for cell in cells:
        if cell.island_index==0: cell.role={"kind":"generalist"}; continue   # named for histories; no settings
        role=assignments[cell.island_index-1]
        if role=="auto":
            t=auto.index(cell.island_index)/max(1,len(auto)-1)
            cell.role={"params":role_parameters(t,crossover_rate,bayesian_proposal_rate,nodes),"stale":0}
        else:
            cell.role={"kind":role,"params":preset_role_parameters(role,crossover_rate,bayesian_proposal_rate,nodes,ops)}
def anchored_band(models, band):
    """(anchor, loss limit, in-band models) of the anchored simplifier.

    The anchor is the lowest-loss feasible model (ties: the shorter); the
    limit adds max(band*|anchor loss|, LOSS_NOISE_FLOOR), so an exact anchor
    (loss ~0) still leaves room for models that differ only by round-off."""
    feasible=[m for m in models if m.feasible and np.isfinite(aggregate_loss(m))]
    if not feasible: return None,None,[]
    anchor=min(feasible,key=secondary_key); best=aggregate_loss(anchor)
    limit=best+max(band*abs(best),LOSS_NOISE_FLOOR)
    return anchor,limit,[m for m in feasible if aggregate_loss(m)<=limit]
def tree_size_cap(model):
    """Largest single tree of a model: the per-tree node cap of its simplification lane."""
    return max(3,max(node_size(tree) for tree in model.trees))
def anchored_lane(models, band=SIMPLIFIER_BAND):
    """(lane models shortest first, per-tree node cap): in-band models no bigger than the anchor."""
    anchor,_,inside=anchored_band(models,band)
    if anchor is None: return [],None
    cap=tree_size_cap(anchor)
    lane=[m for m in inside if tree_size_cap(m)<=cap]
    return sorted(lane,key=lambda m:(model_complexity(m),aggregate_loss(m),repr(m.trees))),cap
def anchored_survivors(pool, count, normalization, tolerance):
    """SIMPLIFIER_LANE_SHARE of the slots to the shortest lane models, the rest by NSGA."""
    lane,_=anchored_lane(pool)
    kept=lane[:int(count*SIMPLIFIER_LANE_SHARE)]
    kept_ids={id(m) for m in kept}
    rest=[m for m in pool if id(m) not in kept_ids]
    return kept+select_nsga(rest,min(count-len(kept),len(rest)),normalization,tolerance)
def anchored_emigrants(pool, count, normalization, tolerance):
    """Shortest models within the final choice's band, else within the wider lane band; NSGA fills the rest."""
    _,_,final=anchored_band(pool,SIMPLIFIER_FINAL_BAND)
    chosen=final or anchored_band(pool,SIMPLIFIER_BAND)[2]
    chosen=sorted(chosen,key=lambda m:(model_complexity(m),aggregate_loss(m),repr(m.trees)))[:count]
    chosen_ids={id(m) for m in chosen}
    rest=[m for m in pool if id(m) not in chosen_ids]
    return chosen+select_nsga(rest,min(count-len(chosen),len(rest)),normalization,tolerance)
def describe_island_roles(assignments):
    """'island 2 auto, island 3 simplifier' for the run banner."""
    return ", ".join(f"island {index+2} {role}" for index,role in enumerate(assignments))
def role_kind(cell):
    """'generalist' for island 0, 'auto' for self-organising islands, else the preset's name."""
    if cell.island_index==0 and not cell.role.get("kind"): return "generalist"
    return cell.role.get("kind","auto")
def cell_search_settings(cell, crossover_rate, bayesian_proposal_rate, nodes):
    """(crossover_rate, bayesian_proposal_rate, nodes, case_weights) for one cell's generation."""
    params=cell.role.get("params") or {}
    weights=cell.role.get("case_weights")
    return (params.get("crossover_rate",crossover_rate),params.get("bayesian_proposal_rate",bayesian_proposal_rate),
            params.get("nodes",nodes),None if weights is None else np.asarray(weights,float))
def cell_role_settings(cell):
    """The preset-only settings evolve_generation takes as role_settings (None for defaults)."""
    params=cell.role.get("params") or {}
    settings={key:params[key] for key in ("ops","parsimony","semantic_max_delta","novelty","neutral_shrink","mutation_bias","anchored") if key in params}
    return settings or None
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
            others=np.delete(held,index,axis=0).min(axis=0)
            if role_kind(cell)!="auto":
                # A fixed role keeps its preset: reported, never reweighted or retired.
                cell.role["contribution"]=float(np.mean(others-best))/scale; continue
            target=responsibility[index]*island_count
            previous=cell.role.get("specialization")
            if previous is None or len(previous)!=len(target):
                specialization=_normalized(.5*_normalized(target)+.5*_normalized(_random_affinity(len(target))))
            else: specialization=_normalized(.5*np.asarray(previous,float)+.5*_normalized(target))
            contribution=float(np.mean(others-best))/scale
            cell.role["contribution"]=contribution
            cell.role["stale"]=0 if contribution>ROLE_MIN_CONTRIBUTION else int(cell.role.get("stale",0))+1
            if cell.role["stale"]>=roles["retire_after"]:
                # A niche that no longer helps anywhere is abandoned for a fresh random one.
                specialization=_normalized(_random_affinity(len(target)))
                cell.role["params"]=role_parameters(rng.random(),crossover_rate,bayesian_proposal_rate,nodes)
                cell.role["stale"]=0; retired+=1
            cell.role["specialization"]=specialization
        specialists=[cell for cell in level[1:] if role_kind(cell)=="auto"]
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

# --sparse-seeding (FFX, McConaghy 2011; SINDy, Brunton 2016): before the
# search, a modest basis (inputs, unary operators of inputs, pairwise
# products and ratios, hinges at quartiles when max is in the grammar) is
# searched by orthogonal matching pursuit, branching on the best first
# terms, and the sparse fits with the best BIC become starting models.  It is
# an initialiser, not a search mechanism: the basis is never composed
# recursively.  SPARSE_SEED_STATS records the best seed's training R^2 so a
# benchmark can tell a good start from better evolution.
SPARSE_SEEDING = "off"
SPARSE_BASIS_SIZE = 300
SPARSE_SEED_SHARE = .1      # of each stage-0 cell's population
SPARSE_SEED_STATS = {"seeds":0,"best_r2":None,"basis":0}

def sparse_basis(X, ops, limit=None):
    """(trees, columns) of the seeding basis, in priority order, without duplicates."""
    limit=SPARSE_BASIS_SIZE if limit is None else limit; n_features=X.shape[1]
    trees=[("x",i) for i in range(n_features)]
    if "*" in ops: trees+=[("*",("x",i),("x",j)) for i in range(n_features) for j in range(i,n_features)]
    if "/" in ops: trees+=[("/",("x",i),("x",j)) for i in range(n_features) for j in range(n_features) if i!=j]
    unary=[op for op in ops if op in OPS and OPS[op][0]==1 and op not in _LIBRARY_EXCLUDED]
    trees+=[(op,("x",i)) for op in unary for i in range(n_features)]
    if "max" in ops:
        for i in range(n_features):
            trees+=[("max",("x",i),("c",float(q))) for q in np.unique(np.round(np.quantile(X[:,i],(.25,.5,.75)),6))]
    kept=[]; columns=[]; seen=set()
    for tree in trees:
        if len(kept)>=limit: break
        try: values=np.asarray(evaluate_cached(tree,X,None),float)
        except (ArithmeticError, IndexError, RecursionError, ValueError): continue
        if not np.isfinite(values).all() or values.std()<=EPS: continue
        signature=np.round((values-values.mean())/values.std(),9).tobytes()
        if signature in seen: continue
        seen.add(signature); kept.append(tree); columns.append(values)
    return kept,(np.column_stack(columns) if columns else np.zeros((len(X),0)))

def sparse_fits(B, y, max_terms=4, branches=6):
    """Orthogonal matching pursuit from the `branches` best first terms; returns [(bic, support, coefficients, intercept, r2)]."""
    n,p=B.shape
    if not p or n<4: return []
    Z=(B-B.mean(axis=0))/B.std(axis=0); yc=y-y.mean(); total=float(yc@yc)
    if total<=0: return []
    first=np.argsort(-np.abs(Z.T@yc))[:branches]; results={}
    for start in first:
        support=[int(start)]
        while True:
            A=np.column_stack([B[:,support],np.ones(n)])
            solution=np.linalg.lstsq(A,y,rcond=None)[0]; residual=y-A@solution
            # Floor at round-off, or an exact fit's extra terms 'win' BIC on 1e-28 vs 1e-27.
            rss=max(float(residual@residual),1e-20*total)
            key=tuple(sorted(support))
            if key not in results:
                results[key]=(n*math.log(rss/n)+(len(support)+1)*math.log(n),key,
                              {index:float(c) for index,c in zip(support,solution[:-1])},float(solution[-1]),1-rss/total)
            if len(support)>=max_terms: break
            Q,_=np.linalg.qr(Z[:,support]); perpendicular=Z-Q@(Q.T@Z)
            norms=np.einsum("ij,ij->j",perpendicular,perpendicular); norms[support]=np.inf
            scores=np.where(norms>1e-10*n,(perpendicular.T@residual)**2/np.maximum(norms,1e-300),-1.)
            scores[support]=-1.; best=int(np.argmax(scores))
            if scores[best]<=1e-12*total: break
            support.append(best)
    return sorted(results.values(),key=lambda item:item[0])

def sparse_seed_models(X, Y, cats, ops, max_nodes, max_depth, count, head_count):
    """Up to `count` seed models whose heads are sparse fits of their outputs; a
    classifier head fits its class indicator (one-vs-rest) on class-balanced
    rows, and the classifier readout calibrates it."""
    targets,_=classification_layout(cats)
    rows=stratified_probe_indices(X,1000) if len(X)>1000 else slice(None)
    Xs=X[rows]; trees_by_head={}; best_r2=None
    basis,B=sparse_basis(Xs,ops)
    def options_for(basis, B, y, track=True):
        nonlocal best_r2
        options=[]
        for _,support,coefficients,_,r2 in sparse_fits(B,y,max_terms=min(MAX_TERMS,4)):
            joined=join_terms([(coefficients[index],basis[index]) for index in support],ops)
            if joined is None: continue
            tree=simplify_tree(joined)
            if node_size(tree)<=max_nodes and node_depth(tree)<=max_depth and tree not in options and not structural_violation(tree):
                options.append(tree)
                if track: best_r2=r2 if best_r2 is None else max(best_r2,r2)  # regression fit quality only
        return options
    for j,heads in enumerate(targets):
        if cats[j] is not None:
            if len(cats[j])<2: continue
            class_rows=class_balanced_rows(X,Y[:,j],len(cats[j]),1000) if CLASS_BALANCE else rows
            class_basis,class_B=sparse_basis(X[class_rows],ops); truth=np.rint(np.asarray(Y[class_rows,j],float))
            for position,head in enumerate(heads):
                options=options_for(class_basis,class_B,(truth==(1 if len(heads)==1 else position)).astype(float),track=False)
                if options: trees_by_head[head]=options
            continue
        options=options_for(basis,B,np.asarray(Y[rows,j],float))
        if options: trees_by_head[heads[0]]=options
    SPARSE_SEED_STATS.update(basis=len(basis))
    if not trees_by_head: return []
    models=[]
    for index in range(min(count,max(len(options) for options in trees_by_head.values()))):
        trees=[trees_by_head[head][index%len(trees_by_head[head])] if head in trees_by_head else ("x",0) for head in range(head_count)]
        models.append(Model(trees,[(1.,0.)]*head_count,origin="sparse_seed",mdl_operators=tuple(ops),mdl_feature_count=X.shape[1]))
    SPARSE_SEED_STATS["seeds"]+=len(models)
    if best_r2 is not None: SPARSE_SEED_STATS["best_r2"]=max(best_r2,SPARSE_SEED_STATS["best_r2"] or -np.inf)
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
    if SPARSE_SEEDING=="on" and Yt is not None and (cell is None or cell[1]==0):
        seeds+=sparse_seed_models(Xt,Yt,cats,ops,nodes,depth,max(1,int(SPARSE_SEED_SHARE*population_size)),head_count)
    for seed in seeds: seed.adfs=dict(adf_registry.definitions)
    population=seeds+[Model([admissible_random_tree(X.shape[1],ops,nodes,depth) for _ in range(head_count)],[(1.,0.)]*head_count,
                      mdl_operators=tuple(ops),mdl_feature_count=X.shape[1],adfs=dict(adf_registry.definitions))
                for _ in range(population_size-len(seeds))]
    probe_indices=stratified_probe_indices(Xt,256)
    # Per-class residual groups need rows of every class; input coverage alone can miss a rare one.
    classified=[j for j,labels in enumerate(cats) if labels is not None and len(labels)>=2]
    residual_probe=(class_balanced_rows(Xt,np.asarray(Yt)[:,classified[0]],len(cats[classified[0]]),256)
                    if CLASS_BALANCE and classified and Yt is not None else probe_indices)
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
        residual_qd=(ResidualQualityDiversityArchive(Xt[residual_probe],np.asarray(Yt)[residual_probe],cats,run_seed ^ 0x5245 ^ island_index,class_groups=CLASS_BALANCE)
                     if RESIDUAL_ARCHIVE and Yt is not None else None),
    )

# Called once per island per generation with that island's live state (the
# browser GUI streams it); None keeps the terminal-only behaviour.
PROGRESS_HOOK = None
def evolution_progress(generation, elite, sample, *, started, Xt, Yt, Xv, Yv, names, out_names, cats, constraints, coev, cases, archive, semantic_qd, structural_qd, qd_controller, pressure, bayes, library=None, budget=None, evaluator=None, adf_registry=None, population=(), best_so_far=None, loss_tolerance=.01, cell=None):
    """Emit the same bounded search telemetry for fresh and resumed runs.

    cell=(island, stage) identifies the cell to PROGRESS_HOOK consumers."""
    if PROGRESS_HOOK is not None:
        PROGRESS_HOOK(generation=generation,elite=elite,population=population,archive=archive,best_so_far=best_so_far,started=started,
                      Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,pressure=pressure,
                      semantic_qd=semantic_qd,structural_qd=structural_qd,qd_controller=qd_controller,evaluator=evaluator,library=library,bayes=bayes,cell=cell)
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

def reset_run_caches():
    """Forget module-level results that depend on run settings (loss, numeric
    guard, interpolation check, units, nesting rules) rather than on the trees
    and data alone.  A fresh process starts empty; benchmarks and tests run
    several configurations in one process, which then reused the previous
    configuration's particle scores, fragment fits and constant readouts."""
    for cache in (_PARTICLE_SCORE_CACHE,_LIKELIHOOD_ENERGY_CACHE,_FRAGMENT_FIT_CACHE,_CONSTANT_AFFINE_CACHE): cache.clear()
    INVALID_DIAGNOSTICS.clear()

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
                      bayesian_proposal_rate, crossover_rate, qd_mode, lexicase_cases, nsga_normalization, progress=None, bayesian_mode="adaptive", adf_registry=None, budget=None, population_size=None, case_weights=None, residual_qd=None, role_settings=None, place=None):
    """Advance one generation; fresh and resumed runs share this exact path.

    case_weights: optional selection-only per-training-row weights (island roles).
    role_settings: a fixed island role's preset (cell_role_settings): "ops"
    (operators that build new structure; the MDL grammar stays the run's),
    "parsimony" (near-tie band floor), "semantic_max_delta", "novelty" (fresh
    random share of offspring), "neutral_shrink" and "mutation_bias"."""
    role_settings=role_settings or {}; place=place or history_place()
    portfolio.bias=role_settings.get("mutation_bias")
    role_delta=role_settings.get("semantic_max_delta"); neutral_shrink=bool(role_settings.get("neutral_shrink"))
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
    pressure.apply(bayes,qd_controller); effective_tolerance=max(pressure.effective_parsimony(),float(role_settings.get("parsimony",0.)))
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
    for model in pop: history_born(model,model.birth_generation if model.birth_generation is not None else generation,place)
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
    # The best-so-far tracker also sees offspring directly; offer it too, so the
    # archive (and the plotted frontier) never lacks the best model found.
    archive.update([*stable_elite,*([best_models.model] if best_models.model is not None else [])],Xt)
    if adf_registry is not None and adf_registry.enabled:
        particle_models=[particle for bank in (bayes.banks if isinstance(bayes,PerOutputBayesianBanks) else [bayes]) for particle in [*bank.particles.catalog,*bank.particles.particles]]
        adf_registry.mark_usage([*pop,*archive.items,*qd_cell_models(semantic_qd,structural_qd,residual_qd),*particle_models],generation,elite)
        active_ops=adf_registry.operators(ops); bayes.sync_operators(active_ops)
        for particle in particle_models: particle.adfs.update(adf_registry.definitions)
    else: active_ops=list(ops)
    # Operators that build new structure: a family role narrows them (ADF
    # calls stay available); models are still priced over active_ops, and the
    # Bayesian bank's particle catalogue keeps the run's grammar (its draws
    # are mutated with these operators before they become offspring).
    family=role_settings.get("ops")
    variation_ops=active_ops if not family else [op for op in active_ops if op in family or op.startswith("adf_")]
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
    if BACKPROP_MUTATION_WEIGHT>0 or RESIDUAL_TERM_WEIGHT>0:
        _,Ysb=behaviour_rows(Xs,Ys); head_outputs=regression_head_outputs(cats)
        fragment_trees=[item["tree"] for item in library.items.values()]
    parent_pool=novelty_pool(pop,Xs)
    parent_count=max(1,population_size//2); qd_count=dual_qd_parent_count(parent_count,qd_controller,semantic_qd,structural_qd,residual_qd)
    row_weights=None if case_weights is None else (np.asarray(case_weights) if isinstance(sample,slice) else np.asarray(case_weights)[sample])
    ordinary=lexicase_parents(parent_pool,parent_count-qd_count,Xs,Ys,cats,lexicase_cases,row_weights,SCALE_BALANCED_SELECTION)
    parents=(blend_dual_qd_parents(ordinary,semantic_qd,structural_qd,parent_count,qd_count,qd_controller.uniform_rate,residual_qd)
             if qd_mode=="adaptive_dual" else blend_fixed_semantic_parents(ordinary,semantic_qd,parent_count,qd_count)); children=[]; feedback=[]; credits=[]; injections=[]; discovery_children=[]
    # Anchored simplifier: most parents come from the in-band lane (a
    # shortest-of-two draw), and their children may not outgrow the anchor.
    lane_ids=set(); lane_cap=nodes
    if role_settings.get("anchored"):
        lane,cap=anchored_lane(pop)
        if lane:
            lane_count=min(len(parents),int(round(len(parents)*SIMPLIFIER_PARENT_SHARE)))
            lane_parents=[ParentChoice(min(rng.choice(lane),rng.choice(lane),key=lambda m:(model_complexity(m),aggregate_loss(m)))) for _ in range(lane_count)]
            parents=parents[:len(parents)-lane_count]+lane_parents
            lane_ids={id(choice) for choice in lane_parents}; lane_cap=min(nodes,cap)
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
        if (NESTING_RULES or UNIT_FEATURES or RELATION_OF_FEATURE) and unchanged<unchanged_limit and any(structural_violation(tree) for tree in trees):
            unchanged+=1; return True
        key=model_equivalence_key(trees)
        if EQUIVALENCE_COLLAPSE and key in seen and unchanged<unchanged_limit:
            unchanged+=1; EQUIVALENCE_STATS["children_redrawn"]+=1; return True
        seen.add(key); return False
    novelty_rate=max(pressure.novelty_rate(),float(role_settings.get("novelty",0.)))
    made=[]   # (child, main parent or None, kinds, crossover partner or None) for the history
    while len(children)<population_size:
        sources=[]
        if novelty_rate and rng.random()<novelty_rate:
            trees=[admissible_random_tree(X.shape[1],variation_ops,nodes,depth,adfs=adf_registry.definitions if adf_registry else None) for _ in range(len(pop[0].trees))]; scales=[(1.,0.)]*len(trees); child_age=0; child_origin="novelty_injection"
        elif bayesian_mode!="off" and rng.random() < bayesian_proposal_rate:
            trees,scales,sources=bayesian_injection_trees(bayes,len(pop[0].trees),X.shape[1],variation_ops,nodes,depth,"grammar" if bayesian_mode=="grammar" else bayesian_mode,adf_registry.definitions if adf_registry else None,return_sources=True)
            child_age=max((model.age+1 for model in sources),default=0); child_origin="bayesian_injection"
        elif library.items and rng.random()<library.fragment_rate:
            p=rng.choice(parents); trees=[]; composed=False
            for tree in p.model.trees:
                candidate=library.compose(tree,variation_ops,lane_cap if id(p) in lane_ids else nodes,depth)
                trees.append(candidate if candidate is not None else tree); composed|=candidate is not None
            if not composed and unchanged<unchanged_limit: unchanged+=1; continue
            scales=list(p.model.scales); child_age=p.model.age+1; child_origin="fragment"
            sources=[p.model]
        elif rng.random() < crossover_rate and len(parents)>=2:
            p,q=rng.sample(parents,2); child_nodes=lane_cap if id(p) in lane_ids else nodes
            trees=[gene_crossover(a,b,child_nodes,depth,variation_ops) if READOUT_MODE=="multiterm" and affine_on and rng.random()<GENE_CROSSOVER_RATE else semantic_crossover(a,b,Xsb,child_nodes,depth,adf_registry.definitions if adf_registry else None,max_delta=role_delta) for a,b in zip(p.model.trees,q.model.trees)]; scales=list(p.model.scales); child_age=max(p.model.age,q.model.age)+1
            if trees==list(p.model.trees) and unchanged<unchanged_limit: unchanged+=1; continue
            if duplicate(trees): continue
            child=Model(trees,scales,child_age,origin="crossover",parent_ids=(p.model.lineage_id,q.model.lineage_id),mdl_operators=grammar_for_trees(active_ops,trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),founder_ids=tuple(sorted(set(p.model.founder_ids).union(q.model.founder_ids))),birth_generation=generation+1-child_age)
            children.append(child); credits.append((child,(p,q))); made.append((child,p.model,None,q.model)); continue
        else:
            p=rng.choice(parents)
            if bayesian_mode!="off": bayes.begin_equation()
            trees=[]; kinds=[]; macro_used=False
            for index,tree in enumerate(p.model.trees):
                proposal=None if bayesian_mode=="off" else (bayes[index] if isinstance(bayes,PerOutputBayesianBanks) else bayes)
                if BACKPROP_MUTATION_WEIGHT>0 or RESIDUAL_TERM_WEIGHT>0: set_backprop_context(Xsb,backprop_desired(p.model,index,Ysb,head_outputs),fragment_trees)
                child_tree,kind,macro=semantic_mutate(tree,Xsb,portfolio,X.shape[1],variation_ops,lane_cap if id(p) in lane_ids else nodes,depth,proposal,adf_registry.definitions if adf_registry else None,library,max_delta=role_delta,neutral_shrink=neutral_shrink)
                trees.append(child_tree); kinds.append(kind); macro_used|=macro
            if trees==list(p.model.trees) and unchanged<unchanged_limit: unchanged+=1; continue
            scales=list(p.model.scales); child_age=p.model.age+1; child_origin="macro_mutation" if macro_used else "mutation"
            sources=[p.model]
        if duplicate(trees): continue
        child=Model(trees,scales,child_age,origin=child_origin,parent_ids=tuple(model.lineage_id for model in sources),mdl_operators=grammar_for_trees(active_ops,trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),founder_ids=tuple(sorted({founder for model in sources for founder in model.founder_ids})),birth_generation=generation+1-child_age)
        children.append(child)
        move=kinds+(["macro"] if macro_used else []) if child_origin in ("mutation","macro_mutation") else [{"fragment":"fragment","bayesian_injection":"bayesian"}.get(child_origin,child_origin)]
        made.append((child,sources[0] if sources else None,move,None))
        if child.origin=="bayesian_injection": injections.append(child)
        if child.origin in {"fragment","macro_mutation"}: discovery_children.append(child)
        if child.origin=="mutation": feedback.append((child,p.model,kinds)); credits.append((child,(p,)))
    # A correct structure with untuned constants otherwise scores like a wrong
    # one and is lost; fit every offspring's inner constants before scoring.
    evaluator.assess(children,"train",None if isinstance(sample,slice) else sample,tune=True)
    for child,parent,kinds,partner in made:
        if partner is not None: history_crossed(child,parent,partner,generation,place)
        elif parent is not None: history_varied(child,parent,generation,place,kinds)
        else: history_born(child,generation,place)
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
            fresh_trees=[admissible_random_tree(X.shape[1],variation_ops,nodes,depth,proposal=None if bayesian_mode=="off" else (bayes[index] if isinstance(bayes,PerOutputBayesianBanks) else bayes),adfs=adf_registry.definitions if adf_registry else None) for index in range(len(pop[0].trees))]
            batch.append(Model(fresh_trees,[(1.,0.)]*len(pop[0].trees),0,origin="novelty_injection",mdl_operators=grammar_for_trees(active_ops,fresh_trees,adf_registry.definitions if adf_registry else None),mdl_feature_count=X.shape[1],adfs={} if adf_registry is None else dict(adf_registry.definitions),birth_generation=generation+1))
        evaluator.assess(batch,"train",None if isinstance(sample,slice) else sample,tune=True)
        for model in batch: history_born(model,generation,place)
        survivor_pool=novelty_pool([*survivor_pool,*batch],Xs); attempts+=len(batch)
    survivors=(anchored_survivors(survivor_pool,min(population_size,len(survivor_pool)),nsga_normalization,effective_tolerance)
               if role_settings.get("anchored") else
               select_nsga(survivor_pool,min(population_size,len(survivor_pool)),nsga_normalization,effective_tolerance))
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

# Parallel island/stage cells.  In a multi-cell run every cell owns its random
# streams (Python and numpy) and a disjoint lineage-id range, swapped into the
# module globals while that cell evolves.  A cell's generation then depends only
# on its own state, so cells may evolve in any order or in parallel processes
# with identical results; migration, promotion and roles run serially between
# generations.  One-cell runs keep the shared module streams (unchanged results).
LINEAGE_RANGE_BITS=40
def seed_cell_streams(cells, run_seed):
    """Give every cell its own seeded streams (fresh multi-cell runs)."""
    for index,cell in enumerate(cells):
        numpy_seed=int(np.random.SeedSequence([int(run_seed)&0xFFFFFFFF,index]).generate_state(1)[0])
        cell.streams={"python":random.Random(f"afpo-cell:{run_seed}:{index}").getstate(),
                      "numpy":np.random.RandomState(numpy_seed).get_state(),"lineage_next":(index+1)<<LINEAGE_RANGE_BITS}

@contextlib.contextmanager
def cell_streams(cell):
    """Draw from the cell's own streams inside the block (no-op without them)."""
    global _NEXT_LINEAGE_ID
    if cell.streams is None:
        yield; return
    saved=(rng.getstate(),np.random.get_state(),_NEXT_LINEAGE_ID)
    rng.setstate(cell.streams["python"]); np.random.set_state(cell.streams["numpy"]); _NEXT_LINEAGE_ID=int(cell.streams["lineage_next"])
    try: yield
    finally:
        cell.streams={"python":rng.getstate(),"numpy":np.random.get_state(),"lineage_next":_NEXT_LINEAGE_ID}
        rng.setstate(saved[0]); np.random.set_state(saved[1]); _NEXT_LINEAGE_ID=saved[2]

def resolve_cell_workers(requested, cells):
    """Processes for evolving cells in parallel; 1 means serial."""
    if requested<0: raise ValueError("--cell-workers must be non-negative")
    if cells<2 or requested==1 or not sys.platform.startswith("linux"): return 1
    return max(1,min(cells,requested or max(1,(os.cpu_count() or 1)-1)))

def _buffer_identity(array):
    interface=array.__array_interface__
    return (interface["data"][0],array.shape,array.strides,interface["typestr"])
class _SharedPickler(pickle.Pickler):
    """Pickles the objects in `shared` (data arrays, the evaluator) by name.  An
    array with a shared array's exact buffer (a full view, Xt[slice(None)])
    maps to that array too: same address, same values."""
    def __init__(self, file, shared):
        super().__init__(file,protocol=pickle.HIGHEST_PROTOCOL); self._names={id(value):name for name,value in shared.items()}
        self._buffers={_buffer_identity(value):name for name,value in shared.items() if isinstance(value,np.ndarray)}
    def persistent_id(self, obj):
        name=self._names.get(id(obj))
        if name is None and type(obj) is np.ndarray: name=self._buffers.get(_buffer_identity(obj))
        return name
class _SharedUnpickler(pickle.Unpickler):
    def __init__(self, file, shared): super().__init__(file); self._shared=shared
    def persistent_load(self, name): return self._shared[name]
def _dumps_shared(value, shared):
    buffer=io.BytesIO(); _SharedPickler(buffer,shared).dump(value); return buffer.getvalue()
def _loads_shared(data, shared): return _SharedUnpickler(io.BytesIO(data),shared).load()

# Caches a child's new entries are merged back from, so the next generation's
# children inherit them as a serial run would (losing them each generation cost
# ~20%).  Content-keyed caches merge as they are; address-keyed ones (value
# (result, X)) only when X is an array the parent holds at that same address.
_CONTENT_KEYED_CACHES=("_FRAGMENT_FIT_CACHE","_PARTICLE_SCORE_CACHE")
_ADDRESS_KEYED_CACHES=("_EVALUATION_CACHE","_GUARD_CACHE")
def _cell_shared_objects(shared):
    """shared plus every array the parent's address-keyed caches anchor to."""
    shared=dict(shared); shared["_FRAGMENT_FIT_FAILED"]=_FRAGMENT_FIT_FAILED; seen={id(v) for v in shared.values()}
    for name in _ADDRESS_KEYED_CACHES:
        for _,anchor in list(globals()[name].values()):
            if isinstance(anchor,np.ndarray) and id(anchor) not in seen:
                seen.add(id(anchor)); shared[f"anchor:{len(shared)}"]=anchor
    return shared
def _merge_cache_entries(entries):
    """Add a child's new cache entries, applying each cache's size bound."""
    for key,value in entries.get("_FRAGMENT_FIT_CACHE",()):
        if len(_FRAGMENT_FIT_CACHE)>=100_000: _FRAGMENT_FIT_CACHE.clear()
        _FRAGMENT_FIT_CACHE[key]=value
    for key,value in entries.get("_PARTICLE_SCORE_CACHE",()):
        if len(_PARTICLE_SCORE_CACHE)>=20000: _PARTICLE_SCORE_CACHE.clear()
        _PARTICLE_SCORE_CACHE[key]=value
    for key,value in entries.get("_GUARD_CACHE",()):
        if len(_GUARD_CACHE)>=20000: _GUARD_CACHE.clear()
        _GUARD_CACHE[key]=value
    for key,value in entries.get("_EVALUATION_CACHE",()):
        if key in _EVALUATION_CACHE: continue
        _EVALUATION_CACHE[key]=value; _EVALUATION_CACHE_SIZE[0]+=value[0].size+EVALUATION_CACHE_ENTRY_COST
        while _EVALUATION_CACHE_SIZE[0]>EVALUATION_CACHE_ELEMENTS and _EVALUATION_CACHE:
            oldest=next(iter(_EVALUATION_CACHE)); _EVALUATION_CACHE_SIZE[0]-=_EVALUATION_CACHE.pop(oldest)[0].size+EVALUATION_CACHE_ENTRY_COST

_CELL_JOB=None
def _evolve_cell_in_child(index):
    """Forked child: evolve one cell serially; return its state and side effects."""
    global PROGRESS_HOOK
    step,cells,evaluator,shared=_CELL_JOB
    cell=cells[index]
    if evaluator is not None:
        _CHILD_KEEPALIVE.append(evaluator.executor)  # the parent's pool is not usable here; never collect it
        evaluator.executor=None; evaluator.journal=[]
        counts=(evaluator.cache_hits,evaluator.cache_misses,evaluator.row_model_evaluations)
    invalid=dict(INVALID_DIAGNOSTICS); redrawn=EQUIVALENCE_STATS["children_redrawn"]; frames=[]
    if PROGRESS_HOOK is not None:
        # Snapshot the hook's arguments when it fires, so the parent replays
        # exactly what a serial run would have reported at that moment.
        PROGRESS_HOOK=lambda **kw: frames.append(_dumps_shared(kw,shared))
    caches=(*_CONTENT_KEYED_CACHES,*_ADDRESS_KEYED_CACHES); before={name:set(globals()[name]) for name in caches}
    output=io.StringIO()
    with contextlib.redirect_stdout(output), cell_streams(cell): step(cell)
    anchors={_buffer_identity(value) for value in shared.values() if isinstance(value,np.ndarray)}
    new_entries={name:[(key,value) for key,value in globals()[name].items() if key not in before[name]
                       and (name in _CONTENT_KEYED_CACHES or (isinstance(value[1],np.ndarray) and _buffer_identity(value[1]) in anchors))]
                 for name in caches}
    side={"stdout":output.getvalue(),"frames":frames,"caches":new_entries,
          "invalid":{key:value-invalid.get(key,0) for key,value in INVALID_DIAGNOSTICS.items() if value!=invalid.get(key,0)},
          "redrawn":EQUIVALENCE_STATS["children_redrawn"]-redrawn}
    if evaluator is not None:
        side["journal"]=evaluator.journal
        side["counts"]=tuple(now-before for now,before in zip((evaluator.cache_hits,evaluator.cache_misses,evaluator.row_model_evaluations),counts))
    return _dumps_shared((cell,side),shared)
_CHILD_KEEPALIVE=[]

def evolve_cells(cells, step, workers, evaluator=None, shared=None):
    """Advance every cell one generation with step(cell), in cell order or in
    `workers` forked processes (identical results when every cell has streams)."""
    global _CELL_JOB
    if workers<=1 or len(cells)<2 or any(cell.streams is None for cell in cells):
        for cell in cells:
            with cell_streams(cell): step(cell)
        return
    shared=_cell_shared_objects(shared or {})
    if evaluator is not None: shared["evaluator"]=evaluator
    _CELL_JOB=(step,cells,evaluator,shared)
    try:
        with multiprocessing.get_context("fork").Pool(min(workers,len(cells)),initializer=_worker_init) as pool:
            results=pool.map(_evolve_cell_in_child,range(len(cells)),chunksize=1)
    finally: _CELL_JOB=None
    hook=PROGRESS_HOOK
    for index,data in enumerate(results):
        cell,side=_loads_shared(data,shared)
        cells[index]=cell
        if side["stdout"]: sys.stdout.write(side["stdout"]); sys.stdout.flush()
        for key,value in side["invalid"].items(): INVALID_DIAGNOSTICS[key]=INVALID_DIAGNOSTICS.get(key,0)+value
        EQUIVALENCE_STATS["children_redrawn"]+=side["redrawn"]
        _merge_cache_entries(side["caches"])
        if evaluator is not None:
            for key,value in side["journal"]: evaluator._score_cache[key]=value
            hits,misses,rows=side["counts"]; evaluator.cache_hits+=hits; evaluator.cache_misses+=misses; evaluator.row_model_evaluations+=rows
        if hook is not None:
            for frame in side["frames"]: hook(**_loads_shared(frame,shared))
    if evaluator is not None: evaluator.begin_generation()

def resume_main(args):
    reset_run_caches()
    generation,pop,bayes,archive,state=load_checkpoint(args.resume,args.allow_unsafe_pickle)
    # Island snapshots, top-up proposals and final ADF refresh all need
    # per-output banks; a single shared generator failed at the first save.
    if not isinstance(bayes,PerOutputBayesianBanks): raise ValueError("Checkpoint predates per-output Bayesian banks and cannot resume; start a new run")
    X,Y,Xt,Yt,Xv,Yv=checkpoint_arrays(state)
    names,out_names,cats,maps=(state[k] for k in ("names","out_names","cats","maps"))
    part=state.get("separate_output")
    if part:
        print(f"This checkpoint is output {part['output']!r} ({part['index']+1} of {part['count']}) of a separate-output run: resuming continues "
              "only this output's search and does not rebuild the merged model; rerun the full setup for that.")
    global SEQUENCE_LAYOUT,EQUIVALENCE_COLLAPSE,RESIDUAL_ARCHIVE,QD_PARENT_CHOICE,SCALE_BALANCED_SELECTION,CLASS_BALANCE,GUARD_EXPLOIT_CHECK
    SEQUENCE_LAYOUT=maps.get(SEQUENCE_LAYOUT_KEY)
    GUARD_EXPLOIT_CHECK=bool(state.get("numeric_guard_check",False))
    global INTERPOLATION_CHECK,FIT_BACKEND,JUMP_CONSTANT_SCAN
    configure_cache_memory(getattr(args,"cache_memory",128))
    INTERPOLATION_CHECK=bool(state.get("interpolation_check",False))
    JUMP_CONSTANT_SCAN=bool(state.get("jump_constant_scan",False))
    global SELECTION_PROBE_FILTER
    SELECTION_PROBE_FILTER=bool(state.get("selection_probe_filter",False))
    global JUMP_MUTATION_WEIGHT
    JUMP_MUTATION_WEIGHT=float(state.get("jump_mutation_weight",0.))
    global CONSTANT_FIT_ITERATIONS,SEMANTIC_MAX_DELTA,CONSTANT_SNAPPING,SNAP_TOLERANCE
    CONSTANT_FIT_ITERATIONS=int(state.get("fit_iterations",12)); SEMANTIC_MAX_DELTA=float(state.get("semantic_max_delta",5.))
    CONSTANT_SNAPPING=state.get("constant_snapping","off"); SNAP_TOLERANCE=float(state.get("snap_tolerance",1e-6))
    global READOUT_MODE,MAX_TERMS,GENE_CROSSOVER_RATE
    global BACKPROP_MUTATION_WEIGHT,BACKPROP_INVERSE
    global RESIDUAL_TERM_WEIGHT,NESTING_RULES,SYMBOLIC_EXPORT,LOSS_MODE,ROBUST_LOSS_DELTA
    SYMBOLIC_EXPORT=getattr(args,"symbolic_export","on")
    LOSS_MODE=state.get("loss","huber"); ROBUST_LOSS_DELTA=float(state.get("huber_delta",1.5))
    configure_units(state.get("units",""),list(names))
    configure_input_relations(state.get("input_relations",""),list(names),state.get("source_columns",()),state.get("types",()))
    RESIDUAL_TERM_WEIGHT=float(state.get("residual_term_weight",0.)); NESTING_RULES=parse_nesting_rules(state.get("forbid_nesting",""))
    BACKPROP_MUTATION_WEIGHT=float(state.get("backprop_mutation_weight",0.)); BACKPROP_INVERSE=state.get("backprop_inverse","generic")
    READOUT_MODE=state.get("readout","affine"); MAX_TERMS=int(state.get("max_terms",4)); GENE_CROSSOVER_RATE=float(state.get("gene_crossover_rate",0.))
    global LOSS_NOISE_FLOOR
    LOSS_NOISE_FLOOR=float(state.get("loss_noise_floor",LOSS_NOISE_FLOOR_MAX))
    global SQUASH_SWAP_WEIGHT,SMOOTH_SWAP_WEIGHT,GATE_MUTATION_WEIGHT
    SQUASH_SWAP_WEIGHT=float(state.get("squash_swap_weight",0.)); SMOOTH_SWAP_WEIGHT=float(state.get("smooth_swap_weight",0.)); GATE_MUTATION_WEIGHT=float(state.get("gate_mutation_weight",0.))
    set_selection_probe_data(Xt,Yt,Xv,Yv,cats)
    FIT_BACKEND=state.get("fit_backend","python")
    # Settings that postdate a checkpoint resume with the behaviour it was searched with.
    RESIDUAL_ARCHIVE=bool(state.get("residual_archive",False)); QD_PARENT_CHOICE=state.get("qd_parent_choice","legacy")
    SCALE_BALANCED_SELECTION=bool(state.get("scale_balanced_selection",False))
    CLASS_BALANCE=bool(state.get("class_balance",False))
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
    cell_workers=resolve_cell_workers(getattr(args,"cell_workers",0),len(islands))
    if cell_workers>1 and any(island.streams is None for island in islands):
        print("This checkpoint predates per-cell random streams; its cells evolve serially so the run continues exactly.")
    topology="single population" if island_config["count"]==1 else f"{island_config['count']} islands; ring migration every {island_config.get('migration_interval',0)} generations"
    if stage_count>1: topology+=f"; {stage_count} {island_config['stages']['mode']} stages per island"
    print(f"Resumed generation {generation} from {checkpoint_path} (seed {state['run_seed']}); {topology}; model scoring uses {'serial evaluation' if workers==1 else f'{workers} worker processes'}.")
    started=time.time(); stop=GracefulStop().__enter__(); interrupted=False
    try:
        while (not args.max_generations or generation < args.max_generations) and not stop.requested and not stop_rule_reached(args,started,islands):
            def step(island):
                def progress(gen,elite,sample):
                    if len(islands)>1 and gen%10==0: print(cell_label(island,island_config["count"],stage_count),flush=True)
                    evolution_progress(gen,elite,sample,started=started,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,coev=coev,cases=island.cases,archive=island.archive,semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,qd_controller=island.qd_controller,pressure=island.pressure,bayes=island.bayes,library=island.library,budget=island.budget,evaluator=evaluator,adf_registry=island.adf_registry,population=island.population,best_so_far=island.best_models.model,loss_tolerance=loss_tolerance,cell=(island.island_index,island.stage))
                cell_crossover,cell_proposals,cell_nodes,cell_weights=cell_search_settings(island,crossover_rate,rate,nodes)
                island.population=evolve_generation(island.population,generation,X=X,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,cats=cats,constraints=constraints,out_names=out_names,
                                  ops=ops,nodes=cell_nodes,depth=depth,case_weights=cell_weights,affine_on=affine_on,coev=coev,bayes=island.bayes,archive=island.archive,
                                  semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,residual_qd=island.residual_qd,qd_controller=island.qd_controller,best_models=island.best_models,
                                  pressure=island.pressure,cases=island.cases,portfolio=island.portfolio,library=island.library,evaluator=evaluator,bayesian_proposal_rate=cell_proposals,
                                  crossover_rate=cell_crossover,qd_mode=qd_mode,lexicase_cases=state.get("lexicase_cases",args.lexicase_cases),nsga_normalization=nsga_normalization,bayesian_mode=state.get("bayesian_mode",args.bayesian_mode),adf_registry=island.adf_registry,budget=island.budget,progress=progress,population_size=island.population_size,
                                  role_settings=cell_role_settings(island),place=history_place(island))
            evolve_cells(islands,step,cell_workers,evaluator,shared={"X":X,"Xt":Xt,"Yt":Yt,"Xv":Xv,"Yv":Yv,"constraints":constraints})
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
    f,snapping=snap_final_candidates(f,Xt,Yt,Xv,Yv,affine_on,cats,constraints,out_names); report_snapping(snapping)
    chosen,selection=select_best_model(f,Xv,Yv,cats,loss_tolerance,constraints,out_names) if Xv is not None else select_best_model(f,loss_tolerance=loss_tolerance)
    state["selection"]={**selection,"selected_choice":"default","default_selected":True,
                        "selected_metrics":selection["metrics"],"selected_objectives":selection["objectives"],"constant_snapping":snapping}
    snapshot_islands(state,islands,island_config)
    save_checkpoint(checkpoint_path,generation,islands[0].population,islands[0].bayes,islands[0].archive,state)
    print(f"Resume complete at generation {generation}. {selection['source'].title()} loss-tolerance shortest-MDL model selected: {equations(chosen,names,out_names,cats)}")
    warning=constant_selection_warning(chosen,selection["source"])
    if warning: print(f"WARNING: {warning}"); state["selection"]["warning"]=warning
    print(f"{selection['source'].title()} selection scores: mean loss={selection['metrics']['loss']:.6g}, mean shape={selection['metrics']['shape']:.6g}, MDL bits={selection['metrics']['mdl_bits']:.6g}")
    if chosen.history: print("History:\n  "+"\n  ".join(describe_history(chosen.history)))
    if part and part.get("reads"):
        # Its inputs include other outputs' predicted values, which no CSV holds.
        print(f"Not exporting best_model.py: {part['output']!r} reads predicted {', '.join(part['reads'])}; only the full separate-output run exports it.")
        evaluator.close(); return
    if part: print(f"Exporting best_model.py for {part['output']!r} alone.")
    export_model(chosen,names,out_names,cats,maps,state["source_columns"],state["types"],state.get("export_fixture"),state.get("input_ranges")); evaluator.close()
    write_symbolic_export(chosen,names,out_names,cats,Xt)

def parse_nesting_rules(text):
    """'exp>exp,sin>cos' -> {("exp","exp"),("sin","cos")}; outer>inner forbids inner anywhere below outer."""
    rules=set()
    for item in str(text or "").split(","):
        if not item.strip(): continue
        outer,separator,inner=item.strip().partition(">")
        if not separator or outer.strip() not in OPS or inner.strip() not in OPS: raise ValueError(f"Bad nesting rule {item.strip()!r}: use outer>inner with operator names")
        rules.add((outer.strip(),inner.strip()))
    return frozenset(rules)

def stop_rule_reached(args, started, islands):
    """--max-time (seconds of search) and --stop-at-loss (best training loss) end the search like Ctrl-C."""
    limit=getattr(args,"max_time",0.) or 0.
    if limit and time.time()-started>=limit:
        print(f"Stopping: --max-time {limit:g}s reached."); return True
    target=getattr(args,"stop_at_loss",None)
    if target is not None:
        best=min((aggregate_loss(island.best_models.model) for island in islands if island.best_models.model is not None),default=float("inf"))
        if best<=target:
            print(f"Stopping: best training loss {best:.6g} reached --stop-at-loss {target:g}."); return True
    return False

def build_arg_parser():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-generations",type=int,default=0); ap.add_argument("--population",type=int,default=160); ap.add_argument("--seed",type=int); ap.add_argument("--workers",type=int,default=0,help="Model-scoring processes; 0=auto, 1=serial.  Results are identical for any count (default: 0)"); ap.add_argument("--adf-mode",choices=("off","flat","nested"),default="nested",help="ADF experiment mode; nested is v2, flat is the v1-style ablation, off disables ADFs")
    ap.add_argument("--bayesian-proposal-rate",type=float,default=.25,help="Fraction of offspring drawn from the Bayesian equation generator (0..1)")
    ap.add_argument("--bayesian-mode",choices=("off","grammar","fixed","adaptive"),default="adaptive",help="Bayesian injection policy: off, grammar-only, fixed particle mix, or adaptive particle mix")
    ap.add_argument("--crossover-rate",type=float,default=.35,help="Fraction of non-Bayesian offspring made by subtree crossover (0..1)")
    ap.add_argument("--lexicase-cases",type=int,default=0,help="Informed max training cases for lexicase; 0 uses all")
    ap.add_argument("--checkpoint-every",type=int,default=100,help="Save full evolutionary state every N generations; 0 disables periodic saves")
    ap.add_argument("--resume",help="Resume from a safe AFPO checkpoint without interactive setup")
    ap.add_argument("--allow-unsafe-pickle",action="store_true",help="Allow a trusted legacy pickle checkpoint; pickle files can execute code when loaded")
    ap.add_argument("--migrate-checkpoint",nargs=2,metavar=("SOURCE","DESTINATION"),help="Convert a trusted legacy checkpoint to the safe JSON format (requires --allow-unsafe-pickle)")
    ap.add_argument("--test-csv",help="Final held-out CSV; reported only, never used for selection")
    ap.add_argument("--max-rows",type=int,default=0,help="Read at most this many rows from each CSV (training, validation and test): a uniform random sample (seeded by --seed, else fixed) kept in file order and drawn while streaming the file, so a large CSV never sits in memory whole and every generation scores fewer rows; the kept file rows are recorded in the run manifest; 0 reads every row (default: 0)")
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
    ap.add_argument("--class-balance",choices=("on","off"),default="on",help="Give every class of a categorical output equal weight: class-weighted log loss, balanced error rate as its shape objective, class-weighted readout fit and constant tuning, class-balanced lexicase case order, per-class residual-archive groups, and a train/validation split stratified by class (default: on)")
    ap.add_argument("--scale-balanced-selection",choices=("on","off"),default="on",help="Selection-only: give every target-magnitude band equal weight and compare asinh-compressed errors in lexicase parent choice; reported loss is unchanged (default: on)")
    ap.add_argument("--cell-workers",type=int,default=0,help="Processes that evolve island/stage cells in parallel; 0=auto (one per cell, up to CPUs-1), 1=serial.  Results are identical either way (default: 0)")
    ap.add_argument("--cache-memory",type=float,default=128,metavar="MB",help="Budget of the tree-output cache in the main process; each --workers scoring process gets an eighth of it. Lower it if large trees or many workers run out of memory: results are identical, only speed changes (default: 128)")
    ap.add_argument("--fit-backend",choices=("auto","python"),default="auto",help="Compiled kernels: auto uses the Cython constant fitter, jump-constant scan, affine readout and numeric guard check when they build: several times faster, agreeing with Python to round-off.  python always uses the pure-Python code (default: auto)")
    ap.add_argument("--numeric-guard-check",choices=("on","off"),default="on",help="Reject models whose values depend on afpo's numeric safety guards (the +/-1e12 value clamp, sinh/cosh/tan input clips) instead of letting them use a guard as a hidden min/max (default: on)")
    ap.add_argument("--interpolation-check",choices=("on","off"),default="on",help="Add a loss term scoring predictions between nearest-neighbour rows against interpolated targets, so equations that only memorise the training rows (e.g. short-period mod sawtooths) lose (default: on)")
    ap.add_argument("--jump-constant-scan",choices=("on","off"),default="on",help="Before the gradient constant fit, scan data-driven values for constants that only move a jump (mod periods, comparison thresholds, floor scales), which the gradient fit cannot move (default: on)")
    ap.add_argument("--fit-iterations",type=int,default=12,help="Levenberg-Marquardt iterations per constant fit; the 2026-10-02 audit found 15%% of fits stop at 12 while still improving (default: 12)")
    ap.add_argument("--semantic-max-delta",type=float,default=5.,help="Largest output change (in target standard deviations) a mutation or crossover may make before the constant fit; inf disables the cap (default: 5)")
    ap.add_argument("--constant-snapping",choices=("off","final"),default="final",help="final: before the final choice, round fitted constants of the strongest candidates to simpler values (integers, p/q, pi, e, sqrt2, ln2, powers of ten, short decimals) when training and validation loss and the numeric guard are preserved (default: final)")
    ap.add_argument("--snap-tolerance",type=float,default=1e-6,help="Relative loss increase a snapped constant may cause on training and on validation data, never below the loss noise floor (default: 1e-6)")
    ap.add_argument("--readout",choices=("affine","multiterm"),default="multiterm",help="Output readout: affine fits a*tree+b; multiterm (multigene GP) gives each top-level +/- term of a regression tree its own least-squares coefficient, pruning negligible and collinear terms (default: multiterm)")
    ap.add_argument("--max-terms",type=int,default=4,help="Most top-level terms a multiterm tree keeps (default: 4)")
    ap.add_argument("--gene-crossover-rate",type=float,default=0.,help="Share of crossovers that add or swap one whole top-level term in multiterm mode; at 0.5 it cut bench_complex solves (2/15 vs 5/15 at 0 on products6, sum8, reuse_poly3) (default: 0)")
    ap.add_argument("--backprop-mutation-weight",type=float,default=0.,help="Initial portfolio weight of semantic backpropagation: invert the tree from its readout down to a random node and replace that subtree with the small expression or fragment that best matches the desired values; 0 disables it (default: 0)")
    ap.add_argument("--backprop-inverse",choices=("exact","generic"),default="generic",help="exact inverts only + - * / neg exp log tanh sigmoid; generic also solves every other operator numerically per row, on the branch the subtree is already on (default: generic)")
    ap.add_argument("--residual-term-weight",type=float,default=1.,help="Initial portfolio weight of the residual-term mutation, which adds the small expression that best matches what the parent still misses (boosting-style); 0 disables it (default: 1)")
    ap.add_argument("--forbid-nesting",default="",metavar="OUTER>INNER,...",help="Operator pairs that may not nest, e.g. exp>exp,log>exp,sin>sin: INNER may not appear anywhere below OUTER (default: none)")
    ap.add_argument("--symbolic-export",choices=("on","off"),default="on",help="Also write the chosen model to best_model_symbolic.txt when sympy is installed: an exact form with afpo's protected operators and a readable raw form without them, each with LaTeX (default: on)")
    ap.add_argument("--loss",choices=("huber","squared","relative"),default="huber",help="Regression loss: huber (MAD-scaled, robust to outliers), squared (plain least squares) or relative (Huber on the error relative to |y|, for targets spanning orders of magnitude) (default: huber)")
    ap.add_argument("--huber-delta",type=float,default=1.5,help="Huber threshold in robust target-scale units for --loss huber and relative (default: 1.5)")
    ap.add_argument("--constant-intervals",choices=("on","off"),default="on",help="Print and record approximate 95%% intervals for the chosen model's constants from the linearised covariance (default: on)")
    ap.add_argument("--output-mode",choices=OUTPUT_MODES,default="separate",help="With several output columns: separate gives each column (a categorical column with all its classes) its own search, Pareto front, MDL and node limit, run one after another and merged into one exported model; joint searches one model holding every output (default: separate)")
    ap.add_argument("--output-relations",action="append",default=[],metavar="'A -> B[; C -> D]'",help="Staged prediction (separate mode) as a dependency graph, e.g. 'HH -> MM' or 'lat -> lon; year -> month -> day' ('a, b -> c' feeds both into c; repeatable). Each output may read the predicted (never the true) values of all its ancestors as extra inputs, ancestors are searched first, and the export inlines them (default: none)")
    ap.add_argument("--input-relations",action="append",default=[],metavar="A,B[;C,D]",help="Input columns that belong together, e.g. HH1,MM1;HH2,MM2 (repeatable). Inside a relation columns combine freely; with other inputs only as complete subexpressions that read every column of the relation, so HH2-HH1 is rejected but (60*HH2+MM2)-(60*HH1+MM1) is not (default: none)")
    ap.add_argument("--units",default="",metavar="COLUMN=UNIT,...",help="Units of input columns for dimensional analysis, e.g. x=m,t=s,F=kg*m/s^2 (exponents with ^, fractions in parentheses like m^(1/2)); trees that add, compare or exponentiate unlike units are rejected; constants and unlisted columns are unit-free wildcards (default: none)")
    ap.add_argument("--max-time",type=float,default=0.,help="Stop the search after this many seconds and go to the final choice; 0 = no limit (default: 0)")
    ap.add_argument("--stop-at-loss",type=float,default=None,help="Stop the search once the best training loss (the loss printed during the run) is at or below this value (default: off)")
    ap.add_argument("--sparse-seeding",choices=("on","off"),default="off",help="Seed the initial population with sparse linear fits over a modest basis (inputs, unary operators of inputs, pairwise products and ratios, hinges), found by orthogonal matching pursuit (default: off)")
    ap.add_argument("--sparse-basis-size",type=int,default=300,help="Most basis terms the sparse seeding searches (default: 300)")
    ap.add_argument("--jump-mutation-weight",type=float,default=1.,help="Initial portfolio weight of the jump mutation, which wraps a subtree in mod(s,c), floordiv(s,c) or if_else(gt(x,c),s,s') as one move; adapted like the other mutation kinds; 0 disables it (default: 1)")
    ap.add_argument("--loss-noise-floor",default="auto",help="Loss differences below this count as ties (the shorter model wins). 'auto' derives it from the targets' written precision: about 3e-12 for 7-digit CSV values, down to 1e-18 for full doubles, never above the old fixed 1e-9 (default: auto)")
    ap.add_argument("--squash-swap-weight",type=float,default=1.,help="Initial portfolio weight of the squash swap, which replaces one sigmoid/tanh/erf with another rewritten to the same level, range and slope (sigmoid(z) -> 0.5+0.5*erf(0.443z)); 0 disables it (default: 1)")
    ap.add_argument("--smooth-swap-weight",type=float,default=1.,help="Initial portfolio weight of the smooth swap: relu(z) <-> softplus(4z)/4 or z*sigmoid(4z), abs(z) -> z*tanh(4z), sign(z) -> tanh(4z); 0 disables it (default: 1)")
    ap.add_argument("--gate-mutation-weight",type=float,default=1.,help="Initial portfolio weight of the gate mutation, which multiplies a subtree s by sigmoid(c*u), 1+erf(c*u) or 1+tanh(c*u) with u = s or an input; 0 disables it (default: 1)")
    ap.add_argument("--selection-probe-filter",choices=("on","off"),default="on",help="Final model choice: drop candidates whose predictions between neighbouring data rows leave the neighbours' target band much more often than the best candidate's (memorised lattice tricks that tie on validation) (default: on)")
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
    if args.fit_iterations < 1: ap.error("--fit-iterations must be positive")
    if args.max_rows < 0: ap.error("--max-rows must be non-negative (0 reads every row)")
    if 0 < args.max_rows < 10: ap.error("--max-rows needs at least 10 rows to split training and validation data")
    try:
        for item in args.units.split(","):
            if item.strip():
                if "=" not in item: raise ValueError(f"Bad unit entry {item.strip()!r}: use column=unit")
                parse_unit(item.partition("=")[2])
    except ValueError as error: ap.error(str(error))
    try: parse_nesting_rules(args.forbid_nesting)
    except ValueError as error: ap.error(str(error))
    if not args.huber_delta > 0: ap.error("--huber-delta must be positive")
    if args.max_time < 0: ap.error("--max-time must be non-negative")
    if args.residual_term_weight < 0: ap.error("--residual-term-weight must be non-negative")
    if args.backprop_mutation_weight < 0: ap.error("--backprop-mutation-weight must be non-negative")
    if args.sparse_basis_size < 1: ap.error("--sparse-basis-size must be positive")
    if args.max_terms < 2: ap.error("--max-terms must be at least 2")
    if not 0 <= args.gene_crossover_rate <= 1: ap.error("--gene-crossover-rate must be between 0 and 1")
    if not args.semantic_max_delta > 0: ap.error("--semantic-max-delta must be positive (inf disables the cap)")
    if not 0 <= args.snap_tolerance < 1: ap.error("--snap-tolerance must be in [0, 1)")
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
    df,types,delimiter=configure(path,getattr(args,"max_rows",0),row_sample_seed(args))
    inputs=[column for column,kind in zip(df.columns,types) if kind in (1,2)]; outputs=[column for column,kind in zip(df.columns,types) if kind in (5,6)]
    if len(inputs)>=2 and not getattr(args,"input_relations",None):
        answer=ask(f"Input relations among {', '.join(map(str,inputs))} (e.g. A,B;C,D; blank=none)","")
        if answer: parse_relations(answer); args.input_relations=[answer]
    if len(outputs)>=2 and getattr(args,"output_mode","separate")=="separate" and not getattr(args,"output_relations",None):
        answer=ask(f"Output relations among {', '.join(map(str,outputs))} (e.g. A -> B; blank=none: each output is searched on its own)","")
        if answer: separate_output_plan(list(df.columns),types,parse_output_relations(answer)); args.output_relations=[answer]
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
    if island_count>1 and yes(ask("Island roles (island 1 generalist, the rest specialise)? 1=yes, 0=no","0")):
        choices=island_role_choices(ops)
        print("Island roles: "+"; ".join(f"{name} = {label}" for name,label in choices.items()))
        answer=ask(f"Roles for islands 2-{island_count}, comma separated (blank = all auto)","")
        assignments=validate_island_roles([item for item in answer.split(",") if item.strip()],island_count,ops)
        roles=role_config(True,interval=int(ask("Role update interval (generations)","10")),assignments=assignments)
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

# --output-mode separate (default): with several output columns, each column
# gets its own search -- its own population, Pareto front, MDL count, node
# limit and final choice -- one after another, and the chosen equations are
# merged into one exported model.  In one joint search every model carries a
# tree per output, every mutation and crossover changes all of them at once,
# MDL is one sum over the trees and selection compares mean loss, so outputs
# fought for fitness and --nodes had to be large enough for all of them at
# once.  A categorical column keeps its class heads together in one search.
# `joint` restores the single search.  --max-generations and --max-time
# apply to each output's search.
#
# --output-relations "HH,MM": staged prediction.  Within a relation the
# columns are searched in the order given, and each later column gets the
# earlier columns' *predicted* values (never their true values, which are
# unknown at prediction time) as extra input features: a numeric output as
# one column, a categorical one as one-hot columns of its predicted class.
# In the merged model those features are replaced by the earlier equations
# themselves (a class indicator by comparisons of the class scores), so the
# exported model reads only the original inputs.
OUTPUT_MODES=("separate","joint")

def parse_output_relations(spec):
    """Dependency edges [(source, target)] of --output-relations, in the order written.

    'year -> month -> day; lat -> lon' is a graph: each '->' feeds every
    column of the stage before it into every column of the stage after it
    ('a, b -> c' feeds both a and b into c).  A part without '->' is a chain
    in the order listed ('HH,MM' is 'HH -> MM')."""
    items=[spec] if isinstance(spec,str) else list(spec or ())
    edges=[]
    for item in items:
        for part in str(item).split(";"):
            if not part.strip(): continue
            if "->" in part:
                stages=[[column.strip() for column in stage.split(",") if column.strip()] for stage in part.split("->")]
                if len(stages)<2 or not all(stages): raise ValueError(f"Malformed output relation {part.strip()!r}")
            else:
                stages=[[column.strip()] for column in part.split(",") if column.strip()]
                if len(stages)<2: raise ValueError(f"An output relation needs at least two columns: {part.strip()!r}")
            for before,after in zip(stages,stages[1:]):
                edges+=[(source,target) for target in after for source in before if (source,target) not in edges]
    for source,target in edges:
        if source==target: raise ValueError(f"Output {source!r} cannot read its own prediction")
    return edges

def format_output_relations(edges): return "; ".join(f"{source} -> {target}" for source,target in edges)

def separate_output_plan(columns, types, edges):
    """[(output column, ancestors it may read)] in search order.

    Every output reads the predictions of all its ancestors in the relation
    graph (day reads month and year in 'year -> month -> day'), so ancestors
    are searched first: related outputs in the order they first appear, then
    the unrelated ones in column order.  A cycle is an error."""
    outputs=[column for column,kind in zip(columns,types) if kind in (5,6)]
    for column in dict.fromkeys(column for edge in edges for column in edge):
        if column not in outputs: raise ValueError(f"Output relation names {column!r}, which is not an output column")
    parents={}
    for source,target in edges: parents.setdefault(target,[]).append(source)
    related=list(dict.fromkeys(column for edge in edges for column in edge))
    order=[]; state={}
    def visit(column, path=()):
        if state.get(column)=="done": return
        if column in path: raise ValueError("Output relations form a cycle: "+" -> ".join([*path[path.index(column):],column]))
        for parent in parents.get(column,()): visit(parent,(*path,column))
        state[column]="done"; order.append(column)
    for column in related: visit(column)
    def ancestors(column, seen=None):
        seen=set() if seen is None else seen
        for parent in parents.get(column,()):
            if parent not in seen: seen.add(parent); ancestors(parent,seen)
        return seen
    plan=[(column,tuple(other for other in order if other in ancestors(column))) for column in order]
    return plan+[(column,()) for column in outputs if column not in order]

def _staged_name(column, taken):
    name=f"{column}_pred"
    while name in taken: name+="_"
    return name

def _predicted_values(result, frame):
    """A sub-run's chosen model applied to a frame: floats, or predicted class labels."""
    X=encode(frame,result["types"],result["maps"])[0]
    values=predict_targets(result["model"],X,result["cats"])[:,0]
    labels=result["cats"][0]
    return values.astype(float) if labels is None else np.array([str(labels[int(value)]) for value in values],dtype=object)

def _affine_tree(tree, a, b):
    if a!=1.: tree=("*",("c",float(a)),tree)
    return ("+",tree,("c",float(b))) if b!=0. else tree

def _class_indicator(scores, k):
    """1 where class k wins the argmax (ties go to the lower index, as np.argmax does), else 0."""
    factors=[("gt" if j<k else "gte",scores[k],scores[j]) for j in range(len(scores)) if j!=k]
    if not factors: return ("c",1.)
    tree=factors[0]
    for factor in factors[1:]: tree=("*",tree,factor)
    return tree

def _rename_adfs(tree, renames):
    if tree[0] in ("x","c","arg"): return tree
    return tuple([renames.get(tree[0],tree[0])]+[_rename_adfs(child,renames) for child in tree[1:]])

def merge_separate_models(plan, results, staged, feature_names, out_names, cats):
    """One Model over the original features from the per-output choices.

    staged maps a source output to the name of its predicted-value feature;
    every read of such a feature becomes the source's own equation."""
    index={name:position for position,name in enumerate(feature_names)}
    targets,_=classification_layout(cats)
    merged_trees={}; merged_scales={}; adfs={}; operators=[]; opaque=[]
    for column,_ in plan:
        result=results[column]; model=result["model"]; tag=out_names.index(column)
        # Each search numbers its own ADFs from zero; keep the merged names apart.
        renames={name:f"adf_o{tag}_{name[4:]}" for name in model.adfs}
        for name,item in model.adfs.items():
            renamed=dict(item); renamed["tree"]=_rename_adfs(item["tree"],renames)
            if "dependencies" in item: renamed["dependencies"]=[renames.get(dependency,dependency) for dependency in item["dependencies"]]
            adfs[renames[name]]=renamed
        def staged_tree(name):
            for source,staged_name in staged.items():
                if source not in merged_trees: continue
                source_cats=results[source]["cats"][0]; trees,scales=merged_trees[source],merged_scales[source]
                if source_cats is None:
                    if name==staged_name: return _affine_tree(trees[0],*scales[0])
                    continue
                if not name.startswith(f"{staged_name}="): continue
                k=[str(label) for label in source_cats].index(name[len(staged_name)+1:])
                if len(trees)>1: return _class_indicator([_affine_tree(tree,*scale) for tree,scale in zip(trees,scales)],k)
                if len(source_cats)<2: return ("c",1.)
                score=_affine_tree(trees[0],*scales[0])
                # binary_labels: class 1 exactly when the score is above 0.5.
                return ("gt",score,("c",.5)) if k==1 else ("lte",score,("c",.5))
            raise ValueError(f"Output {column!r} reads unknown feature {name!r}")
        def substitute(tree):
            if tree[0]=="x":
                name=result["names"][tree[1]]
                if name in index: return ("x",index[name])
                inlined=staged_tree(name); opaque.append(inlined); return inlined
            if tree[0] in ("c","arg"): return tree
            return tuple([renames.get(tree[0],tree[0])]+[substitute(child) for child in tree[1:]])
        merged_trees[column]=[substitute(tree) for tree in model.trees]
        merged_scales[column]=list(model.scales)
        operators+=[op for op in model.mdl_operators if op not in operators]
    trees=[None]*sum(len(heads) for heads in targets); scales=[None]*len(trees)
    for j,column in enumerate(out_names):
        for head,tree,scale in zip(targets[j],merged_trees[column],merged_scales[column]): trees[head]=tree; scales[head]=scale
    # Inlined class indicators may use comparisons the searches' grammars lacked.
    extra=sorted({node[0] for tree in trees for node in walk_tree(tree)
                  if node[0] not in ("x","c","arg") and not node[0].startswith("adf_") and node[0] not in operators})
    return Model(trees,scales,origin="separate_outputs",mdl_operators=tuple(operators+extra),mdl_feature_count=len(feature_names),adfs=adfs,
                 opaque=tuple(dict.fromkeys(opaque)))

def _metadata_for_output(metadata, out_names, column):
    """The constraint metadata of one output, keyed by its name."""
    outputs=(metadata or {}).get("outputs",metadata or {})
    if not isinstance(outputs,dict): return {}
    kept={column:spec for key,spec in outputs.items()
          if key==column or (isinstance(key,str) and key.isdigit() and int(key)<len(out_names) and out_names[int(key)]==column)}
    return {"outputs":kept} if kept else {}

def train_separate_outputs(args, setup, df, frames, run_seed, metadata, choose_model=None):
    """One search per output column (see OUTPUT_MODES), then one merged, exported model."""
    types=list(setup["types"]); columns=list(df.columns)
    train_indices,validation_indices,train_df,validation_df,external_validation=frames
    edges=parse_output_relations(getattr(args,"output_relations",None) or ())
    plan=separate_output_plan(columns,types,edges)
    out_names=[column for column,kind in zip(columns,types) if kind in (5,6)]
    test_df=None
    if args.test_csv:
        test_df=read_dataset(args.test_csv,setup["delimiter"],getattr(args,"max_rows",0),row_sample_seed(args))
        if test_df.attrs.get("afpo_row_sample"): print(f"Test CSV{describe_row_sample(test_df)}.")
    print(f"Separate output searches ({len(plan)}): "+"; ".join(f"{column}"+(f" (reads predicted {', '.join(sources)})" if sources else "") for column,sources in plan))
    # Predicted columns of finished outputs on every frame: the full table (the
    # internal split takes its rows from it), an external validation file, the test file.
    predicted={}; staged={}; results={}
    internal_split=train_df is not df
    for number,(column,sources) in enumerate(plan,start=1):
        kind=types[columns.index(column)]
        print(f"\n=== Output {number}/{len(plan)}: {column} ({'categorical' if kind==6 else 'numeric'}"+(f"; reads predicted {', '.join(sources)}" if sources else "")+") ===")
        types_o=[kind_ if kind_ not in (5,6) or name==column else 0 for name,kind_ in zip(columns,types)]
        def augment(frame, part):
            if frame is None: return None
            frame=frame.copy()
            for source in sources: frame[staged[source]]=predicted[source][part]
            return frame
        for source in sources: types_o.append(1 if results[source]["cats"][0] is None else 2)
        df_o=augment(df,"df")
        train_o=df_o.iloc[train_indices] if internal_split else df_o
        if external_validation: validation_o=augment(validation_df,"validation")
        elif validation_df is None: validation_o=None
        else: validation_o=df_o.iloc[validation_indices]
        args_o=argparse.Namespace(**vars(args)); args_o.test_csv=None; args_o.seed=run_seed
        setup_o={**setup,"df":df_o,"types":types_o,"metadata":_metadata_for_output(metadata,out_names,column),
                 "_frames":(train_indices,validation_indices,train_o,validation_o,external_validation),
                 "_subrun":{"output":column,"index":number-1,"count":len(plan),"reads":list(sources)}}
        result=train_from_setup(args_o,setup_o,choose_model)
        result["frames"]=(train_o,validation_o); results[column]=result
        if any(column in reads for _,reads in plan):
            taken=set(columns)|set(staged.values())
            staged[column]=_staged_name(column,taken)
            predicted[column]={"df":_predicted_values(result,df_o)}
            if external_validation: predicted[column]["validation"]=_predicted_values(result,validation_o)
            if test_df is not None: predicted[column]["test"]=_predicted_values(result,augment(test_df,"test"))
    print(f"\n=== Merging {len(plan)} output searches ===")
    # The merged model lives in the original feature space of the whole table.
    Xt,Yt,names,out_names_all,cats,maps=encode(train_df,types)
    Xv=Yv=None
    if validation_df is not None: Xv,Yv=encode(validation_df,types,maps)[:2]
    Xtest=Ytest=None
    if test_df is not None:
        Xtest,Ytest,test_names,test_outputs,test_cats,_=encode(test_df,types,maps)
        if test_names!=names or test_outputs!=out_names_all or test_cats!=cats: raise ValueError("Test CSV columns/types do not match training data")
    merged=merge_separate_models(plan,results,staged,names,out_names_all,cats)
    constraints=compile_constraints(args.profile,metadata)
    # Every rule still applies to the merged trees, in the whole table's feature
    # space; only the inlined earlier equations are opaque (Model.opaque).
    configure_units(getattr(args,"units",""),list(names))
    configure_input_relations(getattr(args,"input_relations",None) or (),list(names),columns,types)
    assess(merged,Xt,Yt,setup["affine_on"],cats,fit_affine=False,constraints=constraints,output_names=out_names_all)
    if not merged.feasible: print(f"WARNING: the merged model is infeasible ({merged.invalid_reason})")
    # The inlined model must predict exactly what the searches chose.
    check_X=Xv if Xv is not None else Xt
    check=predict_targets(merged,check_X,cats); mismatches=[]
    for j,column in enumerate(out_names_all):
        sub=results[column]; frame=sub["frames"][1] if Xv is not None else sub["frames"][0]
        expected=predict_targets(sub["model"],encode(frame,sub["types"],sub["maps"])[0],sub["cats"])[:,0]
        same=np.isclose(check[:,j],expected,rtol=1e-9,atol=1e-9) if cats[j] is None else check[:,j]==expected
        if not np.all(same): mismatches.append(f"{column} ({int(np.sum(~same))} rows)")
    if mismatches: print(f"WARNING: the merged model differs from the per-output choices on {', '.join(mismatches)}")
    report={"train":frozen_metrics(merged,Xt,Yt,cats,constraints,out_names_all)}
    if Xv is not None: report["validation"]=frozen_metrics(merged,Xv,Yv,cats,constraints,out_names_all)
    if Xtest is not None: report["test"]=frozen_metrics(merged,Xtest,Ytest,cats,constraints,out_names_all)
    # Local complexity (what selection used: a staged feature is one leaf)
    # beside the expanded complexity of the inlined export, for information.
    targets,_=classification_layout(cats); complexity={}
    for j,column in enumerate(out_names_all):
        local=results[column]["model"]; heads=targets[j]
        expanded=Model([merged.trees[head] for head in heads],[merged.scales[head] for head in heads],mdl_operators=merged.mdl_operators,mdl_feature_count=len(names),adfs=merged.adfs)
        complexity[column]={"local_mdl_bits":model_description_bits(local),"local_nodes":int(sum(node_size(tree) for tree in local.trees)),
                            "expanded_mdl_bits":model_description_bits(expanded),"expanded_nodes":int(sum(node_size(tree) for tree in expanded.trees))}
    for column,sources in plan:
        item=complexity[column]
        print(f"{column}: {results[column]['equation']}"+(f"   [{', '.join(staged[source] for source in sources)} = predicted {', '.join(sources)}]" if sources else ""))
        print(f"  complexity: {item['local_mdl_bits']:.0f} bits, {item['local_nodes']} nodes"
              +(f" (expanded with inlined predictions: {item['expanded_mdl_bits']:.0f} bits, {item['expanded_nodes']} nodes)" if sources else ""))
    print(f"Merged model: {equations(merged,names,out_names_all,cats)}")
    if Xv is not None:
        print(f"Validation: loss={report['validation']['loss']:.6g}, shape={report['validation']['shape']:.6g} | output losses={output_loss_summary(report['validation']['losses'],out_names_all)}")
        summary=classification_summary(merged,Xv,Yv,cats,out_names_all)
        if summary: print(f"Validation classes: {summary}")
    if Xtest is not None:
        print(f"Final held-out test (not used for selection): loss={report['test']['loss']:.6g}, shape={report['test']['shape']:.6g} | output losses={output_loss_summary(report['test']['losses'],out_names_all)}")
        summary=classification_summary(merged,Xtest,Ytest,cats,out_names_all)
        if summary: print(f"Test classes: {summary}")
    export_model(merged,names,out_names_all,cats,maps,columns,types,train_df.head(16).copy(),training_input_ranges(train_df,columns,types))
    write_symbolic_export(merged,names,out_names_all,cats,Xt)
    run_dir=Path(results[plan[0][0]]["manifest"]).parent
    summary_path=run_dir.with_name(run_dir.name+"-outputs") / "separate_outputs.json"
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(_json_checkpoint_value({
        "schema_version":1,"output_mode":"separate","seed":run_seed,"output_relations":format_output_relations(edges),
        "input_relations":format_relations(parse_relations(getattr(args,"input_relations",None) or ())),"merged_equation":equations(merged,names,out_names_all,cats),
        "merge_check":"exact" if not mismatches else mismatches,
        "outputs":[{"output":column,"reads_predicted":list(sources),"staged_feature":staged.get(column),"equation":results[column]["equation"],
                    "complexity":complexity[column],"selected":results[column]["selected"],"generations":results[column]["generation"],"selected_metrics":results[column]["selected_metrics"],
                    "manifest":results[column]["manifest"],"checkpoint":results[column]["checkpoint"],"model_card":results[column]["model_card"]} for column,sources in plan],
        "metrics":report}),indent=2,default=str))
    print(f"Separate-output summary: {summary_path}")
    print("Saved best_model.py")
    return {"checkpoint":results[plan[0][0]]["checkpoint"],"checkpoints":[results[column]["checkpoint"] for column,_ in plan],
            "manifest":str(summary_path.resolve()),"model_card":str(summary_path.resolve()),
            "generation":max(results[column]["generation"] for column,_ in plan),
            "selected":"; ".join(f"{column}: {results[column]['selected']}" for column,_ in plan),
            "equation":equations(merged,names,out_names_all,cats),"outputs":{column:results[column]["equation"] for column,_ in plan}}

def split_frames(df, types, setup, run_seed, delimiter, max_rows=0, sample_seed=None):
    """(train_indices, validation_indices, train_df, validation_df, external_validation) for a run's setup."""
    val_path=setup["val_path"]
    external_validation=None
    if val_path=="0":
        train_indices=np.arange(len(df)); validation_indices=np.array([],dtype=int); train_df=df; validation_df=None
    elif val_path:
        validation_df=read_dataset(val_path,delimiter,max_rows,sample_seed); train_df=df; train_indices=np.arange(len(df)); validation_indices=np.arange(len(validation_df))
        external_validation={"path":str(Path(val_path).resolve()),"sha256":dataset_sha256(val_path),"rows":len(validation_df),
                             "row_sample":validation_df.attrs.get("afpo_row_sample")}
        if validation_df.attrs.get("afpo_row_sample"): print(f"Validation CSV{describe_row_sample(validation_df)}.")
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
                strata=None
                for col in [col for col,t in zip(df.columns,types) if t==6] if CLASS_BALANCE else ():
                    labels=df[col].fillna("__MISSING__").astype(str)
                    strata=labels if strata is None else strata+"\x1f"+labels
                strata=None if strata is None else strata.to_numpy()
                train_indices,validation_indices=holdout_split_indices(len(df),val_rows,run_seed,strata)
                train_df=df.iloc[train_indices]; validation_df=df.iloc[validation_indices]
    return train_indices,validation_indices,train_df,validation_df,external_validation

def train_from_setup(args, setup, choose_model=None):
    """Run a fresh search from collected setup answers; returns where its outputs went.

    ``choose_model(labels, choices, evaluation)`` returns the index of the model
    to save; the default asks at the terminal."""
    reset_run_caches()
    global EQUIVALENCE_COLLAPSE,RESIDUAL_ARCHIVE,QD_PARENT_CHOICE,SCALE_BALANCED_SELECTION,CLASS_BALANCE,GUARD_EXPLOIT_CHECK
    EQUIVALENCE_COLLAPSE=getattr(args,"equivalence_collapse","on")=="on"; EQUIVALENCE_STATS["children_redrawn"]=0
    GUARD_EXPLOIT_CHECK=getattr(args,"numeric_guard_check","on")=="on"
    global INTERPOLATION_CHECK,FIT_BACKEND,JUMP_CONSTANT_SCAN
    INTERPOLATION_CHECK=getattr(args,"interpolation_check","on")=="on"
    JUMP_CONSTANT_SCAN=getattr(args,"jump_constant_scan","on")=="on"
    global SELECTION_PROBE_FILTER
    SELECTION_PROBE_FILTER=getattr(args,"selection_probe_filter","on")=="on"
    global JUMP_MUTATION_WEIGHT
    JUMP_MUTATION_WEIGHT=float(getattr(args,"jump_mutation_weight",1.))
    global CONSTANT_FIT_ITERATIONS,SEMANTIC_MAX_DELTA,CONSTANT_SNAPPING,SNAP_TOLERANCE
    CONSTANT_FIT_ITERATIONS=int(getattr(args,"fit_iterations",12)); SEMANTIC_MAX_DELTA=float(getattr(args,"semantic_max_delta",5.))
    CONSTANT_SNAPPING=getattr(args,"constant_snapping","final"); SNAP_TOLERANCE=float(getattr(args,"snap_tolerance",1e-6))
    global READOUT_MODE,MAX_TERMS,GENE_CROSSOVER_RATE
    global SPARSE_SEEDING,SPARSE_BASIS_SIZE
    SPARSE_SEEDING=getattr(args,"sparse_seeding","off"); SPARSE_BASIS_SIZE=int(getattr(args,"sparse_basis_size",300)); SPARSE_SEED_STATS.update(seeds=0,best_r2=None,basis=0)
    global BACKPROP_MUTATION_WEIGHT,BACKPROP_INVERSE
    global RESIDUAL_TERM_WEIGHT,NESTING_RULES
    global SYMBOLIC_EXPORT,LOSS_MODE,ROBUST_LOSS_DELTA,CONSTANT_INTERVALS
    SYMBOLIC_EXPORT=getattr(args,"symbolic_export","on"); CONSTANT_INTERVALS=getattr(args,"constant_intervals","on")
    LOSS_MODE=getattr(args,"loss","huber"); ROBUST_LOSS_DELTA=float(getattr(args,"huber_delta",1.5))
    RESIDUAL_TERM_WEIGHT=float(getattr(args,"residual_term_weight",1.)); NESTING_RULES=parse_nesting_rules(getattr(args,"forbid_nesting",""))
    BACKPROP_MUTATION_WEIGHT=float(getattr(args,"backprop_mutation_weight",0.)); BACKPROP_INVERSE=getattr(args,"backprop_inverse","generic")
    READOUT_MODE=getattr(args,"readout","multiterm"); MAX_TERMS=int(getattr(args,"max_terms",4)); GENE_CROSSOVER_RATE=float(getattr(args,"gene_crossover_rate",0.))
    global SQUASH_SWAP_WEIGHT,SMOOTH_SWAP_WEIGHT,GATE_MUTATION_WEIGHT
    SQUASH_SWAP_WEIGHT=float(getattr(args,"squash_swap_weight",1.)); SMOOTH_SWAP_WEIGHT=float(getattr(args,"smooth_swap_weight",1.)); GATE_MUTATION_WEIGHT=float(getattr(args,"gate_mutation_weight",1.))
    FIT_BACKEND=getattr(args,"fit_backend","auto")
    configure_cache_memory(getattr(args,"cache_memory",128))
    RESIDUAL_ARCHIVE=getattr(args,"residual_archive","on")=="on"; QD_PARENT_CHOICE=getattr(args,"qd_parent_choice","quality_coverage")
    SCALE_BALANCED_SELECTION=getattr(args,"scale_balanced_selection","on")=="on"
    CLASS_BALANCE=getattr(args,"class_balance","on")=="on"
    run_seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng.seed(run_seed); np.random.seed(run_seed)
    print(f"Run seed: {run_seed}")
    # Dropped from the setup so the frame can be freed once encoded (see below).
    df=setup.pop("df")
    path,types,delimiter,ops=(setup[key] for key in ("path","types","delimiter","ops"))
    max_rows=getattr(args,"max_rows",0); sample_seed=row_sample_seed(args)
    affine_on,coev,dynamic_pressure_on,adf_enabled,nodes,depth=(setup[key] for key in ("affine_on","coev","dynamic_pressure_on","adf_enabled","nodes","depth"))
    island_count,migration_interval,migrants_per_island,val_path=(setup[key] for key in ("island_count","migration_interval","migrants_per_island","val_path"))
    adf_enabled=adf_enabled and args.adf_mode!="off"
    stages=stage_config(**(setup.get("stages") or {}))
    roles=role_config(**(setup.get("roles") or {}))
    if roles["enabled"] and island_count<2: raise ValueError("Island roles need at least two islands (one generalist plus specialists)")
    if roles["enabled"]: roles["assignments"]=validate_island_roles(roles["assignments"],island_count,ops)
    cell_count=island_count*stages["count"]
    if cell_count>args.population//8: raise ValueError("Islands x stages need at least eight models each; raise --population or choose fewer islands/stages")
    perceptron_enabled=any(operator.startswith("perceptron") for operator in ops)
    metadata=setup.get("metadata")
    if metadata is None: metadata=json.loads(Path(args.constraint_metadata).read_text()) if args.constraint_metadata else {}
    frames=setup.get("_frames") or split_frames(df,types,setup,run_seed,delimiter,max_rows,sample_seed)
    train_indices,validation_indices,train_df,validation_df,external_validation=frames
    output_columns=[col for col,t in zip(df.columns,types) if t in (5,6)]
    if not setup.get("_subrun") and len(output_columns)>1 and getattr(args,"output_mode","separate")=="separate":
        return train_separate_outputs(args,setup,df,frames,run_seed,metadata,choose_model)
    if not setup.get("_subrun") and parse_output_relations(getattr(args,"output_relations",None) or ()):
        print("Output relations need --output-mode separate and at least two outputs; ignoring them.")
    global SEQUENCE_GROUP_REQUEST
    SEQUENCE_GROUP_REQUEST=tuple(parse_sequence_group(item) for item in args.sequence_group)
    # Fit every fill value and category vocabulary on training rows only.
    Xt,Yt,names,out_names,cats,maps=encode(train_df,types)
    if SEQUENCE_LAYOUT is not None:
        ops=list(dict.fromkeys([*ops,"seqsum","seqprod"]))
        print(f"Sequence groups: {', '.join(group['name'] for group in SEQUENCE_LAYOUT['groups'])} (length {SEQUENCE_LAYOUT['length']}); added seqsum/seqprod.")
    else: ops=[op for op in ops if op not in ("seqsum","seqprod")]
    # A move that can never apply would only reshuffle the portfolio's draws.
    if not activation_moves_apply(ops,"squash"): SQUASH_SWAP_WEIGHT=0.
    if not activation_moves_apply(ops,"smooth"): SMOOTH_SWAP_WEIGHT=0.
    if not activation_moves_apply(ops,"gate"): GATE_MUTATION_WEIGHT=0.
    X,Y=Xt,Yt
    constraints=compile_constraints(args.profile,metadata); constraints.validate(Xt.shape[1],cats,out_names)
    Xv=Yv=None
    if validation_df is not None:
        Xv,Yv,names2,out2,cats2,_=encode(validation_df,types,maps)
        if names2!=names or out2!=out_names or cats2!=cats: raise ValueError("Validation CSV columns/types do not match training data")
    # Everything later needs only these from the frames; dropping them frees
    # the parsed text and the train/validation row copies (often several
    # times the encoded matrices) for the whole search.
    source_columns=list(df.columns); source_rows=len(df); row_sample=df.attrs.get("afpo_row_sample")
    input_ranges=training_input_ranges(train_df,source_columns,types); export_fixture=train_df.head(16).copy()
    del df,train_df,validation_df
    set_selection_probe_data(Xt,Yt,Xv,Yv,cats)
    global LOSS_NOISE_FLOOR
    floor_setting=getattr(args,"loss_noise_floor","auto")
    LOSS_NOISE_FLOOR=estimate_loss_noise_floor(Yt,cats,Yv) if floor_setting=="auto" else float(floor_setting)
    print(f"Loss noise floor: {LOSS_NOISE_FLOOR:.3g} ({'from the precision of the targets' if floor_setting=='auto' else 'set'})")
    Xtest=Ytest=None
    if args.test_csv:
        test_df=read_dataset(args.test_csv,delimiter,max_rows,sample_seed)
        if test_df.attrs.get("afpo_row_sample"): print(f"Test CSV{describe_row_sample(test_df)}.")
        Xtest,Ytest,test_names,test_outputs,test_cats,_=encode(test_df,types,maps); del test_df
        if test_names!=names or test_outputs!=out_names or test_cats!=cats: raise ValueError("Test CSV columns/types do not match training data")
    if Xv is not None and len(Xv)<3:
        raise ValueError("Validation needs at least three rows when affine scaling is enabled")
    if len(Xt)<4 and Xv is not None: raise ValueError("Need at least four training rows after validation split")
    if coev and len(Xt)<=512: print(f"Co-evolution subsamples only above 512 training rows; with {len(Xt)} rows every generation scores all rows.")
    configure_units(getattr(args,"units",""),list(names))
    if UNIT_FEATURES: print(f"Dimensional analysis on {len(UNIT_FEATURES)} column(s) over base units {', '.join(UNIT_BASES)}.")
    configure_input_relations(getattr(args,"input_relations",None) or (),list(names),source_columns,types)
    if INPUT_RELATIONS: print(f"Input relations: {'; '.join('('+', '.join(columns)+')' for columns in INPUT_RELATIONS)}; their columns combine with outside inputs only as complete subexpressions.")
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
                   "roles":{key:roles[key] for key in ("enabled","interval","mix","retire_after","assignments")}},
        "equivalence_collapse":EQUIVALENCE_COLLAPSE,"residual_archive":RESIDUAL_ARCHIVE,"qd_parent_choice":QD_PARENT_CHOICE,"scale_balanced_selection":SCALE_BALANCED_SELECTION,"class_balance":CLASS_BALANCE,
        "numeric_guard_check":GUARD_EXPLOIT_CHECK,"interpolation_check":INTERPOLATION_CHECK,"jump_constant_scan":JUMP_CONSTANT_SCAN,"selection_probe_filter":SELECTION_PROBE_FILTER,"jump_mutation_weight":JUMP_MUTATION_WEIGHT,"fit_iterations":CONSTANT_FIT_ITERATIONS,"semantic_max_delta":SEMANTIC_MAX_DELTA,"constant_snapping":CONSTANT_SNAPPING,"snap_tolerance":SNAP_TOLERANCE,"readout":READOUT_MODE,"max_terms":MAX_TERMS,"gene_crossover_rate":GENE_CROSSOVER_RATE,"backprop_mutation_weight":BACKPROP_MUTATION_WEIGHT,"backprop_inverse":BACKPROP_INVERSE,"residual_term_weight":RESIDUAL_TERM_WEIGHT,"loss":LOSS_MODE,"huber_delta":ROBUST_LOSS_DELTA,"forbid_nesting":",".join(sorted(f"{o}>{i}" for o,i in NESTING_RULES)),"units":UNIT_SPEC,"input_relations":RELATION_SPEC,"sparse_seeding":SPARSE_SEEDING,"sparse_basis_size":SPARSE_BASIS_SIZE,"squash_swap_weight":SQUASH_SWAP_WEIGHT,"smooth_swap_weight":SMOOTH_SWAP_WEIGHT,"gate_mutation_weight":GATE_MUTATION_WEIGHT,"loss_noise_floor":LOSS_NOISE_FLOOR,"fit_backend":FIT_BACKEND,"mdl_policy":MDL_POLICY,"objective_schema":"per_output_loss_shape[,per_output_constraint_violation],mdl_bits,age",
        "test_csv":str(Path(args.test_csv).resolve()) if args.test_csv else None,
        "row_sample":row_sample,
    },(source_rows,source_columns),train_indices,validation_indices,external_validation)
    print(f"Run manifest: {manifest_path}")
    checkpoint_path=manifest_path.parent / "checkpoint_latest.json"
    checkpoint_state={"dataset_path":str(path.resolve()),"types":types,"operators":ops,"affine_on":affine_on,
        "coev":coev,"nodes":nodes,"depth":depth,"X":X,"Y":Y,"Xt":Xt,"Yt":Yt,"Xv":Xv,"Yv":Yv,
        "names":names,"out_names":out_names,"cats":cats,"maps":maps,"encoding_schema":{"version":1,"fit_scope":"training_rows_only","maps":maps},"source_columns":source_columns,
        "export_fixture":export_fixture,"input_ranges":input_ranges,
        "run_seed":run_seed,"manifest":str(manifest_path.resolve()),"bayesian_proposal_rate":args.bayesian_proposal_rate,"bayesian_mode":args.bayesian_mode,"crossover_rate":args.crossover_rate,"evaluation_workers":resolve_worker_count(args.workers,args.population),
        "qd_parent_rate":args.qd_parent_rate,"qd_mode":args.qd_mode,"lexicase_cases":args.lexicase_cases,
        "selection_policy":"loss_tolerance_shortest_mdl","selection_loss_tolerance":args.selection_loss_tolerance,
        "nsga_normalization":args.nsga_normalization,"parsimony_quality_tolerance":args.parsimony_quality_tolerance,"dynamic_pressure_enabled":dynamic_pressure_on,"adf_registry":ADFRegistry(adf_enabled,allow_nested=args.adf_mode=="nested").snapshot(),
        "profile":args.profile,"constraint_metadata":metadata,"constraints":constraints.describe(),"bayesian_particles":args.bayesian_particles,"interaction_discovery":interaction_discovery,
        "island_config":{"count":island_count,"migration_interval":migration_interval,"migrants_per_island":migrants_per_island,"topology":"ring","migration_events":0,"stages":stages,"roles":roles},
        "equivalence_collapse":EQUIVALENCE_COLLAPSE,"residual_archive":RESIDUAL_ARCHIVE,"qd_parent_choice":QD_PARENT_CHOICE,"scale_balanced_selection":SCALE_BALANCED_SELECTION,"class_balance":CLASS_BALANCE,
        "numeric_guard_check":GUARD_EXPLOIT_CHECK,"interpolation_check":INTERPOLATION_CHECK,"jump_constant_scan":JUMP_CONSTANT_SCAN,"selection_probe_filter":SELECTION_PROBE_FILTER,"jump_mutation_weight":JUMP_MUTATION_WEIGHT,"fit_iterations":CONSTANT_FIT_ITERATIONS,"semantic_max_delta":SEMANTIC_MAX_DELTA,"constant_snapping":CONSTANT_SNAPPING,"snap_tolerance":SNAP_TOLERANCE,"readout":READOUT_MODE,"max_terms":MAX_TERMS,"gene_crossover_rate":GENE_CROSSOVER_RATE,"backprop_mutation_weight":BACKPROP_MUTATION_WEIGHT,"backprop_inverse":BACKPROP_INVERSE,"residual_term_weight":RESIDUAL_TERM_WEIGHT,"loss":LOSS_MODE,"huber_delta":ROBUST_LOSS_DELTA,"forbid_nesting":",".join(sorted(f"{o}>{i}" for o,i in NESTING_RULES)),"units":UNIT_SPEC,"input_relations":RELATION_SPEC,"sparse_seeding":SPARSE_SEEDING,"sparse_basis_size":SPARSE_BASIS_SIZE,"squash_swap_weight":SQUASH_SWAP_WEIGHT,"smooth_swap_weight":SMOOTH_SWAP_WEIGHT,"gate_mutation_weight":GATE_MUTATION_WEIGHT,"loss_noise_floor":LOSS_NOISE_FLOOR,"fit_backend":FIT_BACKEND,"mdl_policy":MDL_POLICY,"objective_schema":"per_output_loss_shape[,per_output_constraint_violation],mdl_bits,age"}
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
    if SPARSE_SEEDING=="on":
        best=SPARSE_SEED_STATS["best_r2"]
        print(f"Sparse seeding: {SPARSE_SEED_STATS['seeds']} seed model(s) from a {SPARSE_SEED_STATS['basis']}-term basis"+(f"; best seed training R2={best:.6g}" if best is not None else ""))
        checkpoint_state["sparse_seed_stats"]=dict(SPARSE_SEED_STATS)
    if roles["enabled"]: assign_role_parameters(islands,island_count,args.crossover_rate,args.bayesian_proposal_rate,nodes,roles["assignments"],ops)
    if cell_count>1: seed_cell_streams(islands,run_seed)
    cell_workers=resolve_cell_workers(getattr(args,"cell_workers",0),cell_count)
    snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
    workers=checkpoint_state["evaluation_workers"]
    evaluator=ModelEvaluator(workers,{"train":(Xt,Yt)},affine_on,cats,constraints,out_names)
    gen=0; start=time.time()
    topology=("single population" if island_count==1 else f"{island_count} islands, ring migration every {migration_interval} generations")
    if stages["count"]>1: topology+=f", {stages['count']} {stages['mode']} stages per island (promotion every {stages['interval']} generations)"
    if roles["enabled"]: topology+=f", island roles (island 1 generalist; {describe_island_roles(roles['assignments'])}; update every {roles['interval']} generations)"
    if cell_count>1: topology+=f" ({', '.join(map(str,population_sizes))} models per cell; {'cells evolve serially' if cell_workers==1 else f'cells evolve in {cell_workers} parallel processes'})"
    print(f"Searching indefinitely with {args.bayesian_proposal_rate:.0%} Bayesian proposals, {topology}, and {'serial evaluation' if workers==1 else f'{workers} worker processes'}; press Ctrl-C to choose and save a model.")
    stop=GracefulStop().__enter__(); interrupted=False
    try:
        while (not args.max_generations or gen<args.max_generations) and not stop.requested and not stop_rule_reached(args,start,islands):
            def step(island):
                def progress(generation,elite,sample):
                    if cell_count>1 and generation%10==0: print(cell_label(island,island_count,stages["count"]),flush=True)
                    evolution_progress(generation,elite,sample,started=start,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,names=names,out_names=out_names,cats=cats,constraints=constraints,coev=coev,cases=island.cases,archive=island.archive,semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,qd_controller=island.qd_controller,pressure=island.pressure,bayes=island.bayes,library=island.library,budget=island.budget,evaluator=evaluator,adf_registry=island.adf_registry,population=island.population,best_so_far=island.best_models.model,loss_tolerance=args.selection_loss_tolerance,cell=(island.island_index,island.stage))
                cell_crossover,cell_proposals,cell_nodes,cell_weights=cell_search_settings(island,args.crossover_rate,args.bayesian_proposal_rate,nodes)
                island.population=evolve_generation(island.population,gen,X=X,Xt=Xt,Yt=Yt,Xv=Xv,Yv=Yv,cats=cats,constraints=constraints,out_names=out_names,
                                  ops=ops,nodes=cell_nodes,depth=depth,case_weights=cell_weights,affine_on=affine_on,coev=coev,bayes=island.bayes,archive=island.archive,
                                  semantic_qd=island.semantic_qd,structural_qd=island.structural_qd,residual_qd=island.residual_qd,qd_controller=island.qd_controller,best_models=island.best_models,
                                  pressure=island.pressure,cases=island.cases,portfolio=island.portfolio,library=island.library,evaluator=evaluator,bayesian_proposal_rate=cell_proposals,
                                  crossover_rate=cell_crossover,qd_mode=args.qd_mode,lexicase_cases=args.lexicase_cases,nsga_normalization=args.nsga_normalization,bayesian_mode=args.bayesian_mode,adf_registry=island.adf_registry,budget=island.budget,progress=progress,population_size=island.population_size,
                                  role_settings=cell_role_settings(island),place=history_place(island))
            evolve_cells(islands,step,cell_workers,evaluator,shared={"X":X,"Xt":Xt,"Yt":Yt,"Xv":Xv,"Yv":Yv,"constraints":constraints})
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
    from_simplifier=[role_kind(island)=="simplifier" for island in islands for model in [*island.archive.items,*island.population,island.best_models.model] if model is not None]
    f,snapping=snap_final_candidates(f,Xt,Yt,Xv,Yv,affine_on,cats,constraints,out_names); report_snapping(snapping)
    # Snapping swaps in snapped copies position by position, so identify the
    # simplifier's candidates after it.
    simplifier_keys={selection_identity(model) for model,flag in zip(f,from_simplifier) if flag}
    evaluation=selection_evaluation(f,Xv,Yv,cats,constraints,out_names)
    labels,choices,selection=model_options(f,cats=cats,loss_tolerance=args.selection_loss_tolerance,evaluation=evaluation,simplifier_keys=simplifier_keys)
    print_frontier(f,names,out_names,cats,recommendations=(labels,choices),evaluation=evaluation)
    if selection.get("warning"): print(f"WARNING: {selection['warning']}")
    if len(islands)==1: print(islands[0].archive.stats())
    else: print(f"Island archives: {' | '.join(island.archive.stats() for island in islands)}")
    if EQUIVALENCE_COLLAPSE: print(f"Equivalence collapse: redrew {EQUIVALENCE_STATS['children_redrawn']} offspring equivalent to an existing or sibling equation.")
    selected_index=max(0,min(len(choices)-1,int((choose_model or choose_model_interactively)(labels,choices,evaluation))))
    chosen=choices[selected_index]
    print(f"Selected model: {equations(chosen,names,out_names,cats)}")
    print("Fitted constants:", [constant_vector(tree) for tree in chosen.trees])
    if chosen.history: print("History:\n  "+"\n  ".join(describe_history(chosen.history)))
    intervals=constant_intervals(chosen,Xt,Yt,cats) if CONSTANT_INTERVALS=="on" else None
    print_constant_intervals(intervals)
    if Xv is not None:
        metrics=frozen_metrics(chosen,Xv,Yv,cats,constraints,out_names)
        print(f"Validation (used for selection): loss={metrics['loss']:.6g}, shape={metrics['shape']:.6g} | output losses={output_loss_summary(metrics['losses'],out_names)}")
        summary=classification_summary(chosen,Xv,Yv,cats,out_names)
        if summary: print(f"Validation classes: {summary}")
    if Xtest is not None:
        metrics=frozen_metrics(chosen,Xtest,Ytest,cats,constraints,out_names)
        print(f"Final held-out test (not used for selection): loss={metrics['loss']:.6g}, shape={metrics['shape']:.6g} | output losses={output_loss_summary(metrics['losses'],out_names)}")
        summary=classification_summary(chosen,Xtest,Ytest,cats,out_names)
        if summary: print(f"Test classes: {summary}")
    subrun=setup.get("_subrun")
    if not subrun:
        export_model(chosen,names,out_names,cats,maps,source_columns,types,export_fixture,input_ranges)
        write_symbolic_export(chosen,names,out_names,cats,Xt)
    selected_entry=next(entry for entry in evaluation[1] if entry[0] is chosen)
    selection={**selection,"constant_snapping":snapping,"selected_choice":labels[selected_index],"default_selected":selected_index==0,
               "selected_metrics":selected_entry[2],"selected_objectives":tuple(selected_entry[1].objectives)}
    checkpoint_state["selection"]=selection
    if subrun: checkpoint_state["separate_output"]=dict(subrun)
    snapshot_islands(checkpoint_state,islands,checkpoint_state["island_config"])
    save_checkpoint(checkpoint_path,gen,islands[0].population,islands[0].bayes,islands[0].archive,checkpoint_state)
    record_selection_manifest(manifest_path,selection)
    card=write_model_card(manifest_path,chosen,names,out_names,constraints,hypotheses,islands[0].bayes,{"train_rows":len(Xt),"validation_rows":0 if Xv is None else len(Xv),"preprocessing":"fit_on_training_rows_only","schema_version":1},cats=cats,bootstrap=None if intervals is None else {"method":"linearised least-squares covariance on training rows","level":.95,"parameters":intervals},selection=selection,quality_diversity={"islands":[{"semantic":island.semantic_qd.diagnostics(),"structural":island.structural_qd.diagnostics(),"controller":island.qd_controller.diagnostics(),
                                                                                         "residual":None if island.residual_qd is None else island.residual_qd.diagnostics()} for island in islands]},survival={"nsga_normalization":args.nsga_normalization,"parsimony_quality_tolerance":args.parsimony_quality_tolerance,"islands":checkpoint_state["island_config"]},adf_diagnostics=[island.adf_registry.diagnostics() for island in islands if island.adf_registry.enabled] or None,evaluation={"budgets":[island.budget.snapshot() for island in islands],"evaluator":evaluator.diagnostics()},interaction_discovery=interaction_discovery,island_diagnostics={"config":checkpoint_state["island_config"],"bayesian_posteriors":[[bank.particles.last_predictive for bank in island.bayes.banks] for island in islands]})
    evaluator.close()
    print(f"Model card: {card}")
    result={"checkpoint":str(checkpoint_path.resolve()),"manifest":str(manifest_path.resolve()),"model_card":str(Path(card).resolve()),
            "generation":gen,"selected":labels[selected_index],"equation":equations(chosen,names,out_names,cats)}
    if subrun:
        result.update(model=chosen,names=list(names),maps=maps,cats=cats,out_names=list(out_names),types=list(types),selected_metrics=selected_entry[2])
        return result
    print("Saved best_model.py")
    return result

if __name__=="__main__":
    try: main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc: print(f"Configuration error: {exc}",file=sys.stderr); sys.exit(2)
