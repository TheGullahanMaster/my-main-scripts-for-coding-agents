#!/usr/bin/env python3
"""Pure AFPO symbolic regression baseline.

This executable deliberately shares the data handling, objectives, and optional
search infrastructure from :mod:`afpo`, while omitting every Bayesian proposal,
posterior update, SMC particle, and posterior-predictive operation.  It is meant
for seeded, apples-to-apples comparisons with ``afpo.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import afpo as core


CHECKPOINT_VERSION = 1


def model_data(model):
    """Checkpoint model state without creating a posterior container."""
    return {"trees": model.trees, "scales": model.scales, "age": model.age,
            "objectives": model.objectives, "lineage_id": model.lineage_id,
            "origin": model.origin, "parent_ids": model.parent_ids,
            "feasible": model.feasible, "invalid_reason": model.invalid_reason,
            "constraint_count": model.constraint_count,
            "mdl_operators": model.mdl_operators,
            "mdl_feature_count": model.mdl_feature_count, "adfs": model.adfs,
            "founder_ids": model.founder_ids}


def save_checkpoint(path, generation, population, archive, state):
    """Atomically save pure-AFPO state; Bayesian checkpoints are incompatible."""
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "algorithm": "pure_afpo",
        "generation": int(generation),
        "population": [model_data(model) for model in population],
        "archive": {"capacity": archive.capacity,
                    "items": [model_data(model) for model in archive.items]},
        "next_lineage_id": core._NEXT_LINEAGE_ID,
        "python_rng_state": core.rng.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "state": state,
    }
    encoded = core._json_checkpoint_value(payload)
    body = json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
    wrapper = {"format_version": core.SAFE_CHECKPOINT_FORMAT,
               "checksum": hashlib.sha256(body.encode()).hexdigest(), "payload": encoded}
    target = Path(path)
    temporary = Path(f"{target}.tmp")
    temporary.write_text(json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(target)


def load_checkpoint(path):
    """Load only pure-AFPO checkpoints and restore both random generators."""
    source = Path(path)
    try:
        wrapper = json.loads(source.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Pure AFPO resumes require a safe JSON pure-AFPO checkpoint") from error
    if wrapper.get("format_version") != core.SAFE_CHECKPOINT_FORMAT:
        raise ValueError("Unknown safe checkpoint format")
    body = json.dumps(wrapper.get("payload"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(body.encode()).hexdigest() != wrapper.get("checksum"):
        raise ValueError("Checkpoint checksum does not match")
    payload = core._from_json_checkpoint_value(wrapper["payload"])
    if payload.get("algorithm") != "pure_afpo" or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Checkpoint is not a pure-AFPO checkpoint; start a new pureAfpo.py run")
    population = [core.Model(**item) for item in payload["population"]]
    state = payload["state"]
    archive = core.ParetoArchive(payload["archive"]["capacity"],
                                 state.get("nsga_normalization", "intercept"),
                                 state.get("parsimony_quality_tolerance", 0.))
    archive.items = [core.Model(**item) for item in payload["archive"]["items"]]
    core._NEXT_LINEAGE_ID = int(payload.get("next_lineage_id", 0))
    core.rng.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    return int(payload["generation"]), population, archive, state


def snapshot_state(state, portfolio, cases, semantic_qd, structural_qd, qd_controller,
                   best_models, pressure, library, adf_registry, budget):
    state["mutation_portfolio"] = portfolio.snapshot()
    state["case_population"] = cases.snapshot()
    state["quality_diversity"] = core.qd_snapshot(semantic_qd, structural_qd, qd_controller)
    state["best_model"] = best_models.snapshot()
    state["dynamic_pressure"] = pressure.snapshot()
    state["fragment_library"] = library.snapshot()
    state["adf_registry"] = adf_registry.snapshot()
    state["evaluation_budget"] = budget.snapshot()


def apply_pressure(pressure, qd_controller):
    """Pure-AFPO pressure preserves QD/novelty changes, not posterior pressure."""
    qd_controller.uniform_rate = pressure.uniform_rate()


def pressure_stats(pressure):
    return (f"Dynamic pressure={'on' if pressure.enabled else 'off'}; level={pressure.level}; "
            f"parsimony={pressure.effective_parsimony():.0%}; "
            f"QD uniform={pressure.uniform_rate():.0%}; novelty={pressure.novelty_rate():.0%}")


def attach_adfs(registry, population, archive, semantic_qd, structural_qd, best_model):
    if registry.enabled:
        registry.attach([*population, *archive.items, *semantic_qd.cells.values(),
                         *structural_qd.cells.values(), best_model])


def progress(generation, elite, sample, *, started, Xt, Xv, Yv, names, out_names, cats,
             constraints, coev, cases, archive, semantic_qd, structural_qd, qd_controller,
             pressure, library, budget, evaluator, adf_registry):
    if generation % 10 == 0:
        best = min(elite, key=core.secondary_key)
        print(f"generation {generation:6d} | best mean loss {core.aggregate_loss(best):.6g} | "
              f"losses {core.output_loss_summary(core.model_losses(best), out_names)} | "
              f"mean shape {np.mean(core.model_shapes(best)):.6g} | "
              f"MDL bits {core.model_complexity(best):.6g} | {time.time()-started:.0f}s", flush=True)
        if Xv is not None:
            validation = core.frozen_metrics(best, Xv, Yv, cats, constraints, out_names)
            print(f"validation (read-only) | mean loss {validation['loss']:.6g} | "
                  f"losses {core.output_loss_summary(validation['losses'], out_names)} | "
                  f"shape {validation['shape']:.6g}", flush=True)
        if coev:
            print("case diagnostics | " + cases.diagnostics(Xt), flush=True)
        print("Pure AFPO: uniform random injection; no Bayesian posterior or SMC particles.", flush=True)
        print(semantic_qd.stats(), flush=True)
        print(structural_qd.stats(), flush=True)
        print(qd_controller.stats(), flush=True)
        print(pressure_stats(pressure), flush=True)
        print(library.stats(), flush=True)
        print(budget.stats(), flush=True)
        print("Evaluator:", evaluator.diagnostics(), flush=True)
        if adf_registry.enabled:
            info = adf_registry.diagnostics()
            print(f"ADF v2 | promotions={info['promotions']}; definitions={info['definitions']}; "
                  f"active={','.join(info['active']) or 'none'}; retired={info['retired']}; "
                  f"elite_uses={info['elite_uses']}; invalid={info['invalid_calls']}", flush=True)
    if generation and generation % 100 == 0:
        core.print_frontier(archive.items, names, out_names, cats)
        print(archive.stats())


def evolve_generation(population, generation, *, X, Xt, Yt, Xv, Yv, cats, constraints,
                      out_names, operators, nodes, depth, affine_on, coev, archive,
                      semantic_qd, structural_qd, qd_controller, best_models, pressure,
                      cases, portfolio, library, evaluator, random_injection_rate,
                      crossover_rate, qd_mode, lexicase_cases, nsga_normalization,
                      adf_registry, budget, progress_callback=None):
    """Advance one AFPO generation using uniform symbolic variation only."""
    attach_adfs(adf_registry, population, archive, semantic_qd, structural_qd, best_models.model)
    evaluator.begin_generation()
    apply_pressure(pressure, qd_controller)
    tolerance = pressure.effective_parsimony()
    qd_controller.begin_generation((semantic_qd, structural_qd))
    sample = cases.sample(budget.screen_count(len(Xt))) if coev and len(Xt) > 512 else slice(None)
    budget.record("screen", len(Xt) if isinstance(sample, slice) else len(sample))
    evaluator.assess(population, "train", None if isinstance(sample, slice) else sample)
    elite = core.select_nsga(population, max(8, len(population)//8), nsga_normalization, tolerance)
    if generation % 20 == 0:
        for candidate in elite[:2]:
            tuned = core.optimize_constants(candidate, Xt[sample], Yt[sample], affine_on, cats,
                                            constraints=constraints, output_names=out_names)
            if core.aggregate_loss(tuned) < core.aggregate_loss(candidate):
                population[next(i for i, model in enumerate(population) if model is candidate)] = tuned
        elite = core.select_nsga(population, max(8, len(population)//8), nsga_normalization, tolerance)
    if coev:
        cases.update(elite, Xt, Yt, cats)
    if adf_registry.enabled:
        adf_registry.observe(core.novelty_pool(elite, Xt[sample]), generation)
        attach_adfs(adf_registry, population, archive, semantic_qd, structural_qd, best_models.model)
        active_operators = adf_registry.operators(operators)
    else:
        active_operators = list(operators)
    quality_improved = best_models.update(population)
    archive.update(elite, Xt[sample])
    if adf_registry.enabled:
        adf_registry.mark_usage([*population, *archive.items, *semantic_qd.cells.values(),
                                 *structural_qd.cells.values()], generation, elite)
    pressure.observe(generation, quality_improved, (semantic_qd, structural_qd))
    apply_pressure(pressure, qd_controller)
    if generation % 10 == 0 and Xv is not None and adf_registry.enabled and elite:
        diagnostic = min(elite, key=core.secondary_key)
        adf_registry.record_validation(diagnostic,
            core.frozen_metrics(diagnostic, Xv, Yv, cats, constraints, out_names)["loss"], generation)
    if progress_callback is not None:
        progress_callback(generation, elite, sample)
    diverse = [*semantic_qd.cells.values(), *structural_qd.cells.values()]
    library.observe(core.unique_models([*elite, *archive.items, *diverse]), Xt[sample], Yt[sample], cats)
    parent_pool = core.novelty_pool(population, Xt[sample])
    parent_count = len(population)//2
    qd_count = core.dual_qd_parent_count(parent_count, qd_controller, semantic_qd, structural_qd)
    ordinary = core.lexicase_parents(parent_pool, parent_count-qd_count, Xt[sample], Yt[sample], cats, lexicase_cases)
    parents = (core.blend_dual_qd_parents(ordinary, semantic_qd, structural_qd, parent_count,
                                          qd_count, qd_controller.uniform_rate)
               if qd_mode == "adaptive_dual"
               else core.blend_fixed_semantic_parents(ordinary, semantic_qd, parent_count, qd_count))
    children, feedback, credits, discovery_children = [], [], [], []
    definitions = adf_registry.definitions if adf_registry.enabled else None
    while len(children) < len(population):
        if pressure.novelty_rate() and core.rng.random() < pressure.novelty_rate():
            trees = [core.random_tree(X.shape[1], active_operators, nodes, depth, adfs=definitions)
                     for _ in range(len(population[0].trees))]
            scales, child_age, origin, parent_ids, founders = [(1., 0.)]*len(trees), 0, "novelty_injection", (), ()
        elif core.rng.random() < random_injection_rate:
            trees = [core.random_tree(X.shape[1], active_operators, nodes, depth, adfs=definitions)
                     for _ in range(len(population[0].trees))]
            scales, child_age, origin, parent_ids, founders = [(1., 0.)]*len(trees), 0, "random_injection", (), ()
        elif library.items and core.rng.random() < library.fragment_rate:
            parent = core.rng.choice(parents)
            trees, composed = [], False
            for tree in parent.model.trees:
                candidate = library.compose(tree, active_operators, nodes, depth)
                trees.append(candidate if candidate is not None else tree)
                composed |= candidate is not None
            scales, child_age, origin = list(parent.model.scales), parent.model.age+1, "fragment"
            parent_ids, founders = (parent.model.lineage_id,), parent.model.founder_ids
        elif core.rng.random() < crossover_rate and len(parents) >= 2:
            left, right = core.rng.sample(parents, 2)
            trees = [core.semantic_crossover(a, b, Xt[sample], nodes, depth, definitions)
                     for a, b in zip(left.model.trees, right.model.trees)]
            child = core.Model(trees, list(left.model.scales), max(left.model.age, right.model.age)+1,
                               origin="crossover", parent_ids=(left.model.lineage_id, right.model.lineage_id),
                               mdl_operators=core.grammar_for_trees(active_operators, trees, definitions),
                               mdl_feature_count=X.shape[1], adfs={} if definitions is None else dict(definitions),
                               founder_ids=tuple(sorted(set(left.model.founder_ids).union(right.model.founder_ids))))
            children.append(child)
            credits.append((child, (left, right)))
            continue
        else:
            parent = core.rng.choice(parents)
            trees, kinds, macro_used = [], [], False
            for tree in parent.model.trees:
                child_tree, kind, macro = core.semantic_mutate(tree, Xt[sample], portfolio, X.shape[1],
                    active_operators, nodes, depth, None, definitions, library)
                trees.append(child_tree)
                kinds.append(kind)
                macro_used |= macro
            scales, child_age = list(parent.model.scales), parent.model.age+1
            origin = "macro_mutation" if macro_used else "mutation"
            parent_ids, founders = (parent.model.lineage_id,), parent.model.founder_ids
        child = core.Model(trees, scales, child_age, origin=origin, parent_ids=parent_ids,
                           mdl_operators=core.grammar_for_trees(active_operators, trees, definitions),
                           mdl_feature_count=X.shape[1], adfs={} if definitions is None else dict(definitions),
                           founder_ids=founders)
        children.append(child)
        if origin in {"fragment", "macro_mutation"}:
            discovery_children.append(child)
        if origin == "mutation":
            feedback.append((child, parent.model, kinds))
            credits.append((child, (parent,)))
    evaluator.assess(children, "train", None if isinstance(sample, slice) else sample)
    for child, parent_choices in credits:
        qd_controller.record(child, parent_choices, (semantic_qd, structural_qd))
    for child, parent, kinds in feedback:
        for kind in kinds:
            portfolio.record(kind, core.variation_improved(child, parent))
    if coev and len(Xt) > 512:
        promoted = [model.clone() for model in core.select_nsga(children, max(8, len(population)//4),
                                                                 nsga_normalization, tolerance)]
        anchor = budget.anchor_indices(Xt)
        evaluator.assess(promoted, "train", anchor)
        budget.record("anchor", len(anchor))
        best_models.update(promoted)
        archive.update(promoted, Xt[anchor])
    qd_candidates, qd_threshold = core.qd_eligible_candidates([*population, *children])
    semantic_qd.update(qd_candidates, qd_threshold)
    if qd_mode == "adaptive_dual":
        structural_qd.update(qd_candidates, qd_threshold)
    if qd_mode != "fixed_semantic":
        qd_controller.update_rate()
    for model in population:
        model.age += 1
        model.objectives = (*model.objectives[:-1], model.age)
    survivor_pool = core.novelty_pool(population + children, Xt[sample])
    attempts = 0
    while len(survivor_pool) < len(population) and attempts < len(population)*8:
        trees = [core.random_tree(X.shape[1], active_operators, nodes, depth, adfs=definitions)
                 for _ in range(len(population[0].trees))]
        fresh = core.Model(trees, [(1., 0.)]*len(trees), 0, origin="novelty_injection",
                           mdl_operators=core.grammar_for_trees(active_operators, trees, definitions),
                           mdl_feature_count=X.shape[1], adfs={} if definitions is None else dict(definitions))
        core.assess(fresh, Xt[sample], Yt[sample], affine_on, cats, constraints=constraints, output_names=out_names)
        survivor_pool = core.novelty_pool([*survivor_pool, fresh], Xt[sample])
        attempts += 1
    survivors = core.select_nsga(survivor_pool, min(len(population), len(survivor_pool)),
                                 nsga_normalization, tolerance)
    if budget.refresh_due(generation):
        refresh = [model.clone() for model in survivors]
        evaluator.assess(refresh, "train")
        budget.record("full", len(Xt))
        budget.last_full_refresh = generation
        best_models.update(refresh)
        archive.update(refresh, Xt)
    survivor_ids = {id(model) for model in survivors}
    for child in discovery_children:
        library.record(child.origin, id(child) in survivor_ids)
    return survivors


def pure_model_card(path, model, feature_names, output_names, constraints, hypotheses, split,
                    *, cats, selection, quality_diversity, survival, adf_diagnostics, evaluation,
                    random_injection_rate):
    card = {
        "format_version": 1, "algorithm": "pure_afpo",
        "formulae": core.equations(model, feature_names, output_names, cats),
        "adf_definitions": core.adf_display_definitions(model, feature_names),
        "profile": constraints.profile, "constraints": constraints.describe(),
        "mdl": core.model_description(model, len(feature_names)),
        "per_output_constraint_violations": list(core.model_violations(model)),
        "hypotheses": hypotheses, "posterior_diagnostics": None,
        "random_injection_rate": random_injection_rate, "split_provenance": split,
        "coefficient_intervals": "not estimated; bounded point fitting only",
        "feature_operator_stability": "not estimated unless bootstrap audits are enabled",
        "selection": selection, "quality_diversity": quality_diversity, "survival": survival,
        "evaluation": evaluation, "adf_registry": adf_diagnostics,
        "caveats": ["Pure AFPO uses uniform random injection and no Bayesian posterior or SMC particles.",
                    "Constraints are advisory soft objectives.",
                    "Hypotheses are training-only and are not inferred domain labels."],
    }
    target = Path(path).with_name("model_card.json")
    target.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    return target


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--max-generations", type=int, default=0)
    result.add_argument("--population", type=int, default=160)
    result.add_argument("--seed", type=int)
    result.add_argument("--workers", type=int, default=0,
                        help="Model-scoring processes; 0=auto, 1=serial")
    result.add_argument("--adf-mode", choices=("off", "flat", "nested"), default="nested")
    result.add_argument("--random-injection-rate", type=float, default=.25,
                        help="Fraction of offspring generated from the uniform grammar (default: 0.25)")
    result.add_argument("--crossover-rate", type=float, default=.35)
    result.add_argument("--lexicase-cases", type=int, default=0)
    result.add_argument("--checkpoint-every", type=int, default=100)
    result.add_argument("--resume", help="Resume a safe pure-AFPO checkpoint")
    result.add_argument("--test-csv", help="Final held-out CSV; never used for selection")
    result.add_argument("--selection-loss-tolerance", type=float, default=.01)
    result.add_argument("--nsga-normalization", choices=core.NSGA_NORMALIZATIONS, default="intercept")
    result.add_argument("--parsimony-quality-tolerance", type=float, default=.01)
    result.add_argument("--profile", choices=core.PROFILES, default="general")
    result.add_argument("--constraint-metadata")
    result.add_argument("--qd-parent-rate", type=float, default=.20)
    result.add_argument("--qd-mode", choices=("adaptive_dual", "adaptive_semantic", "fixed_semantic"), default="adaptive_dual")
    result.add_argument("--evaluation-budget", choices=("baseline", "adaptive"), default="baseline")
    result.add_argument("--evaluation-refresh", type=int, default=25)
    result.add_argument("--stagnation-window", type=int, default=100)
    return result


def validate_args(args):
    if not 0 <= args.random_injection_rate <= 1 or not 0 <= args.crossover_rate <= 1:
        raise ValueError("random injection and crossover rates must be between 0 and 1")
    if args.population < 2:
        raise ValueError("--population must be at least 2")
    if args.selection_loss_tolerance < 0 or args.parsimony_quality_tolerance < 0:
        raise ValueError("selection tolerances must be non-negative")
    if not .10 <= args.qd_parent_rate <= .30:
        raise ValueError("--qd-parent-rate must be between 0.10 and 0.30")
    if args.evaluation_refresh < 1 or args.stagnation_window < 1:
        raise ValueError("evaluation refresh and stagnation window must be positive")


def restore_runtime(args):
    generation, population, archive, state = load_checkpoint(args.resume)
    X, Xt, Yt, Xv, Yv = (state[key] for key in ("X", "Xt", "Yt", "Xv", "Yv"))
    names, out_names, cats, maps = (state[key] for key in ("names", "out_names", "cats", "maps"))
    operators, nodes, depth, affine_on, coev = (state[key] for key in ("operators", "nodes", "depth", "affine_on", "coev"))
    constraints = core.compile_constraints(state.get("profile", "general"), state.get("constraint_metadata", {}))
    workers = core.resolve_worker_count(args.workers if args.workers else state.get("evaluation_workers", 0), len(population))
    evaluator = core.ModelEvaluator(workers, {"train": (Xt, Yt)}, affine_on, cats, constraints, out_names)
    semantic_qd, structural_qd, qd_controller = core.qd_from_snapshot(state["quality_diversity"])
    portfolio = core.MutationPortfolio(); portfolio.restore(state["mutation_portfolio"])
    library = core.FragmentLibrary.from_snapshot(state.get("fragment_library", {}))
    cases = core.CasePopulation.from_snapshot(state["case_population"], len(Xt)) if coev else core.CasePopulation(len(Xt))
    tolerance = state.get("parsimony_quality_tolerance", args.parsimony_quality_tolerance)
    best_models = core.BestModelArchive.from_snapshot(state.get("best_model", {}), tolerance)
    if best_models.model is None:
        best_models.update([*archive.items, *population])
    pressure = core.DynamicPressureController.from_snapshot(state.get("dynamic_pressure", {}))
    budget = core.EvaluationBudget.from_snapshot(state.get("evaluation_budget", {}))
    adf_registry = core.ADFRegistry.from_snapshot(state.get("adf_registry", {}))
    return locals()


def train_new(args):
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    core.rng.seed(seed); np.random.seed(seed)
    print(f"Run seed: {seed}")
    path = Path(core.ask("Dataset path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    frame, types, delimiter = core.configure(path)
    operators = core.choose_operator_groups()
    affine_on = core.yes(core.ask("Affine scaling? 1=yes, 0=no", "1"))
    coev = core.yes(core.ask("Use co-evolution/minibatches? 1=yes, 0=no", "0"))
    dynamic_pressure_on = core.yes(core.ask("Dynamic evolution pressure? 1=yes, 0=no", "1"))
    adf_enabled = core.yes(core.ask("Enable automatically defined function (ADF) mining? 1=yes, 0=no", "0")) and args.adf_mode != "off"
    nodes = max(3, int(core.ask("Maximum nodes per output", "31")))
    depth = max(1, int(core.ask("Maximum depth", "6")))
    validation_path = core.ask("Validation CSV (blank=random split; 0=disabled)", "")
    if args.constraint_metadata:
        metadata = json.loads(Path(args.constraint_metadata).read_text())
    elif args.profile != "general":
        metadata = json.loads(core.ask("Constraint metadata JSON (blank={})", "{}"))
    else:
        metadata = {}
    if validation_path == "0":
        train_indices = np.arange(len(frame)); validation_indices = np.array([], dtype=int)
        train_frame, validation_frame = frame, None
    elif validation_path:
        validation_frame = pd.read_csv(validation_path, sep=delimiter, engine="python")
        train_frame, train_indices, validation_indices = frame, np.arange(len(frame)), np.arange(len(validation_frame))
    else:
        percentage = float(core.ask("Validation percentage (0 disables validation; minimum 5 rows when possible)", "20"))
        requested = max(1, math.ceil(len(frame)*percentage/100))
        validation_rows = max(5, requested)
        if percentage <= 0 or len(frame)-validation_rows < 4:
            train_indices = np.arange(len(frame)); validation_indices = np.array([], dtype=int)
            train_frame, validation_frame = frame, None
        else:
            train_indices, validation_indices = core.holdout_split_indices(len(frame), validation_rows, seed)
            train_frame, validation_frame = frame.iloc[train_indices], frame.iloc[validation_indices]
    Xt, Yt, names, out_names, cats, maps = core.encode(train_frame, types)
    constraints = core.compile_constraints(args.profile, metadata)
    constraints.validate(Xt.shape[1], cats, out_names)
    Xv = Yv = None
    if validation_frame is not None:
        Xv, Yv, names2, out2, cats2, _ = core.encode(validation_frame, types, maps)
        if names2 != names or out2 != out_names or cats2 != cats:
            raise ValueError("Validation CSV columns/types do not match training data")
    Xtest = Ytest = None
    if args.test_csv:
        test_frame = pd.read_csv(args.test_csv, sep=delimiter, engine="python")
        Xtest, Ytest, names2, out2, cats2, _ = core.encode(test_frame, types, maps)
        if names2 != names or out2 != out_names or cats2 != cats:
            raise ValueError("Test CSV columns/types do not match training data")
    hypotheses = core.discover_hypotheses(Xt, Yt, names, out_names)
    workers = core.resolve_worker_count(args.workers, args.population)
    manifest = core.write_run_manifest(path, seed, {
        "algorithm": "pure_afpo", "delimiter": delimiter, "column_types": types, "operators": operators,
        "affine_scaling": affine_on, "co_evolution": coev, "max_nodes": nodes, "max_depth": depth,
        "evaluation_workers": workers, "random_injection_rate": args.random_injection_rate,
        "crossover_rate": args.crossover_rate, "qd_mode": args.qd_mode,
        "evaluation_budget": {"mode": args.evaluation_budget, "refresh": args.evaluation_refresh},
        "profile": args.profile, "constraint_metadata": metadata,
        "constraints": constraints.describe(), "adf_mining": adf_enabled, "adf_mode": args.adf_mode,
        "nsga_normalization": args.nsga_normalization,
        "parsimony_quality_tolerance": args.parsimony_quality_tolerance,
        "selection_loss_tolerance": args.selection_loss_tolerance,
        "test_csv": str(Path(args.test_csv).resolve()) if args.test_csv else None,
    }, frame, train_indices, validation_indices)
    registry = core.ADFRegistry(adf_enabled, allow_nested=args.adf_mode == "nested")
    head_count = sum(len(heads) for heads in core.classification_layout(cats)[0])
    population = [core.Model([core.random_tree(Xt.shape[1], operators, nodes, depth) for _ in range(head_count)],
                             [(1., 0.)]*head_count, mdl_operators=tuple(operators),
                             mdl_feature_count=Xt.shape[1], adfs=dict(registry.definitions))
                  for _ in range(args.population)]
    cases = core.CasePopulation(len(Xt)); portfolio = core.MutationPortfolio(); library = core.FragmentLibrary()
    archive = core.ParetoArchive(normalization=args.nsga_normalization,
                                 parsimony_quality_tolerance=args.parsimony_quality_tolerance)
    probe = core.stratified_probe_indices(Xt, 256)
    semantic_qd = core.QualityDiversityArchive(Xt[probe], cats, seed ^ 0x5144)
    structural_qd = core.StructuralQualityDiversityArchive(Xt.shape[1], seed ^ 0x5354)
    qd_controller = core.QDOutcomeController(rate=float(np.clip(args.qd_parent_rate, .10, .30)))
    best_models = core.BestModelArchive(args.parsimony_quality_tolerance)
    pressure = core.DynamicPressureController(dynamic_pressure_on, args.parsimony_quality_tolerance,
                                              qd_controller.uniform_rate, args.stagnation_window)
    budget = core.EvaluationBudget(args.evaluation_budget, args.evaluation_refresh)
    evaluator = core.ModelEvaluator(workers, {"train": (Xt, Yt)}, affine_on, cats, constraints, out_names)
    state = {"algorithm": "pure_afpo", "dataset_path": str(path.resolve()), "types": types,
             "operators": operators, "affine_on": affine_on, "coev": coev, "nodes": nodes, "depth": depth,
             "X": Xt, "Xt": Xt, "Yt": Yt, "Xv": Xv, "Yv": Yv, "names": names, "out_names": out_names,
             "cats": cats, "maps": maps, "source_columns": list(frame.columns), "run_seed": seed,
             "manifest": str(manifest.resolve()), "evaluation_workers": workers,
             "random_injection_rate": args.random_injection_rate, "crossover_rate": args.crossover_rate,
             "qd_mode": args.qd_mode, "lexicase_cases": args.lexicase_cases,
             "selection_loss_tolerance": args.selection_loss_tolerance,
             "nsga_normalization": args.nsga_normalization,
             "parsimony_quality_tolerance": args.parsimony_quality_tolerance,
             "profile": args.profile, "constraint_metadata": metadata,
             "export_fixture": train_frame.head(16).copy(),
             "input_ranges": core.training_input_ranges(train_frame, list(frame.columns), types)}
    return locals()


def finish(runtime, args, generation, checkpoint_path):
    population = runtime["population"]; evaluator = runtime["evaluator"]
    evaluator.assess(population, "train")
    runtime["best_models"].update(population)
    runtime["archive"].update(population, runtime["Xt"])
    candidates = core.unique_models([*runtime["archive"].items, runtime["best_models"].model])
    chosen, selection = (core.select_best_model(candidates, runtime["Xv"], runtime["Yv"], runtime["cats"],
                              args.selection_loss_tolerance, runtime["constraints"], runtime["out_names"])
                         if runtime["Xv"] is not None else
                         core.select_best_model(candidates, loss_tolerance=args.selection_loss_tolerance))
    if runtime.get("Xtest") is not None:
        metrics = core.frozen_metrics(chosen, runtime["Xtest"], runtime["Ytest"], runtime["cats"],
                                      runtime["constraints"], runtime["out_names"])
        print(f"Final held-out test (not used for selection): loss={metrics['loss']:.6g}, "
              f"shape={metrics['shape']:.6g} | "
              f"output losses={core.output_loss_summary(metrics['losses'], runtime['out_names'])}")
    runtime["state"]["selection"] = {**selection, "selected_choice": "default", "default_selected": True}
    snapshot_state(runtime["state"], runtime["portfolio"], runtime["cases"], runtime["semantic_qd"],
                   runtime["structural_qd"], runtime["qd_controller"], runtime["best_models"], runtime["pressure"],
                   runtime["library"], runtime["registry"], runtime["budget"])
    save_checkpoint(checkpoint_path, generation, population, runtime["archive"], runtime["state"])
    core.export_model(chosen, runtime["names"], runtime["out_names"], runtime["cats"], runtime["maps"],
                      runtime["state"]["source_columns"], runtime["state"]["types"],
                      runtime["state"].get("export_fixture"), runtime["state"].get("input_ranges"))
    core.record_selection_manifest(runtime["manifest"], runtime["state"]["selection"])
    card = pure_model_card(runtime["manifest"], chosen, runtime["names"], runtime["out_names"],
        runtime["constraints"], runtime.get("hypotheses", []),
        {"train_rows": len(runtime["Xt"]), "validation_rows": 0 if runtime["Xv"] is None else len(runtime["Xv"]),
         "preprocessing": "fit_on_training_rows_only", "schema_version": 1}, cats=runtime["cats"],
        selection=runtime["state"]["selection"],
        quality_diversity={"semantic": runtime["semantic_qd"].diagnostics(),
                           "structural": runtime["structural_qd"].diagnostics(),
                           "controller": runtime["qd_controller"].diagnostics()},
        survival={"nsga_normalization": runtime["state"]["nsga_normalization"],
                  "parsimony_quality_tolerance": runtime["state"]["parsimony_quality_tolerance"]},
        adf_diagnostics=runtime["registry"].diagnostics() if runtime["registry"].enabled else None,
        evaluation={"budget": runtime["budget"].snapshot(), "evaluator": evaluator.diagnostics()},
        random_injection_rate=runtime["state"]["random_injection_rate"])
    evaluator.close()
    print(f"Pure AFPO complete at generation {generation}. Model card: {card}")


def main():
    args = parser().parse_args()
    validate_args(args)
    if args.resume:
        runtime = restore_runtime(args)
        generation = runtime["generation"]
        runtime["manifest"] = Path(runtime["state"]["manifest"])
        runtime["registry"] = runtime.pop("adf_registry")
        runtime["hypotheses"] = []
        print(f"Resumed pure AFPO generation {generation} from {args.resume} (seed {runtime['state']['run_seed']}).")
    else:
        runtime = train_new(args)
        generation = 0
        scoring = "serial evaluation" if runtime["workers"] == 1 else f"{runtime['workers']} worker processes"
        print(f"Pure AFPO search: {args.random_injection_rate:.0%} uniform random injections; "
              f"{scoring}.")
    checkpoint_path = Path(args.resume) if args.resume else runtime["manifest"].parent / "checkpoint_latest.json"
    started = time.time()
    try:
        while not args.max_generations or generation < args.max_generations:
            runtime["population"] = evolve_generation(
                runtime["population"], generation, X=runtime["Xt"], Xt=runtime["Xt"], Yt=runtime["Yt"],
                Xv=runtime["Xv"], Yv=runtime["Yv"], cats=runtime["cats"], constraints=runtime["constraints"],
                out_names=runtime["out_names"], operators=runtime["operators"], nodes=runtime["nodes"],
                depth=runtime["depth"], affine_on=runtime["affine_on"], coev=runtime["coev"], archive=runtime["archive"],
                semantic_qd=runtime["semantic_qd"], structural_qd=runtime["structural_qd"],
                qd_controller=runtime["qd_controller"], best_models=runtime["best_models"], pressure=runtime["pressure"],
                cases=runtime["cases"], portfolio=runtime["portfolio"], library=runtime["library"],
                evaluator=runtime["evaluator"], random_injection_rate=runtime["state"]["random_injection_rate"],
                crossover_rate=runtime["state"]["crossover_rate"], qd_mode=runtime["state"]["qd_mode"],
                lexicase_cases=runtime["state"]["lexicase_cases"], nsga_normalization=runtime["state"]["nsga_normalization"],
                adf_registry=runtime["registry"], budget=runtime["budget"],
                progress_callback=lambda gen, elite, sample: progress(
                    gen, elite, sample, started=started, Xt=runtime["Xt"], Xv=runtime["Xv"], Yv=runtime["Yv"],
                    names=runtime["names"], out_names=runtime["out_names"], cats=runtime["cats"],
                    constraints=runtime["constraints"], coev=runtime["coev"], cases=runtime["cases"],
                    archive=runtime["archive"], semantic_qd=runtime["semantic_qd"], structural_qd=runtime["structural_qd"],
                    qd_controller=runtime["qd_controller"], pressure=runtime["pressure"], library=runtime["library"],
                    budget=runtime["budget"], evaluator=runtime["evaluator"], adf_registry=runtime["registry"]))
            generation += 1
            if args.checkpoint_every and generation % args.checkpoint_every == 0:
                snapshot_state(runtime["state"], runtime["portfolio"], runtime["cases"], runtime["semantic_qd"],
                    runtime["structural_qd"], runtime["qd_controller"], runtime["best_models"], runtime["pressure"],
                    runtime["library"], runtime["registry"], runtime["budget"])
                save_checkpoint(checkpoint_path, generation, runtime["population"], runtime["archive"], runtime["state"])
                print(f"Checkpoint: {checkpoint_path}")
    except KeyboardInterrupt:
        print("\nPure AFPO interrupted; saving current checkpoint.")
    finish(runtime, args, generation, checkpoint_path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(2)
