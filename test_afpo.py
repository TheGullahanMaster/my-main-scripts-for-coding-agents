"""Regression checks for AFPO scoring and evolutionary-state invariants.

Run with: python -B -m unittest -v test_afpo
"""
import tempfile
import unittest
import io
import contextlib
from unittest.mock import patch
from pathlib import Path

import numpy as np

import afpo as a


def model(tree, age=0, ops=("+", "-", "*", "square"), features=1):
    return a.Model([tree], [(1., 0.)], age=age,
                   mdl_operators=ops, mdl_feature_count=features)


def island(X, cats, ops, *, size=12, coev=False, adfs=False, index=0):
    return a.new_island_runtime(
        size, X=X, Xt=X, cats=cats, ops=ops, nodes=15, depth=4,
        head_count=sum(map(len, a.classification_layout(cats)[0])),
        bayesian_particles=12, run_seed=12, island_index=index,
        qd_parent_rate=.20, nsga_normalization="intercept",
        parsimony_quality_tolerance=.01, dynamic_pressure_on=True,
        stagnation_window=100, adf_enabled=adfs, adf_mode="nested",
        evaluation_budget="adaptive" if coev else "baseline",
        evaluation_refresh=2, interaction_discovery={})


def advance(state, generation, X, Y, cats, ops, evaluator, *, coev=False, **kwargs):
    state.population = a.evolve_generation(
        state.population, generation, X=X, Xt=X, Yt=Y, Xv=None, Yv=None,
        cats=cats, constraints=evaluator.constraints, out_names=evaluator.output_names,
        ops=ops, nodes=15, depth=4, affine_on=evaluator.affine_on, coev=coev,
        bayes=state.bayes, archive=state.archive, semantic_qd=state.semantic_qd,
        structural_qd=state.structural_qd, qd_controller=state.qd_controller,
        best_models=state.best_models, pressure=state.pressure, cases=state.cases,
        portfolio=state.portfolio, library=state.library, evaluator=evaluator,
        bayesian_proposal_rate=.25, crossover_rate=.35, qd_mode="adaptive_dual",
        lexicase_cases=0, nsga_normalization="intercept",
        adf_registry=state.adf_registry, budget=state.budget,
        population_size=state.population_size, **kwargs)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(1., 6.)[:, None]
        self.training = 2*self.X
        self.validation = self.X
        self.overfit = model(("*", ("c", 2.), ("x", 0)), age=7)
        self.general = model(("x", 0), age=9)
        self.models = [self.overfit, self.general]
        for m in self.models:
            a.assess(m, self.X, self.training, False, [None])

    def test_validation_drives_every_quality_recommendation_and_display(self):
        original = [(m.objectives, list(m.scales), m.age) for m in self.models]
        evaluation = a.selection_evaluation(self.models, self.X, self.validation, [None])
        labels, choices, info = a.model_options(self.models, evaluation=evaluation)
        self.assertIs(choices[0], self.general)
        self.assertEqual(info['source'], 'validation')
        self.assertEqual(info['metrics']['loss'], 0.)
        self.assertNotIn('Most Correct Shape', labels)  # Same validation winner, deduplicated.
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            a.print_frontier(self.models, ['x'], ['y'], [None],
                             recommendations=(labels, choices), evaluation=evaluation)
        output = text.getvalue()
        self.assertIn('Recommended models (validation scores)', output)
        self.assertIn('mean loss=0, losses=y=0', output)
        self.assertIn('age=9', output)
        self.assertIn('1 structural family', output)
        self.assertEqual([(m.objectives, m.scales, m.age) for m in self.models], original)

    def test_distinct_affine_fits_of_one_tree_are_not_dropped(self):
        other = self.general.clone()
        other.scales = [(2., 0.)]
        a.assess(other, self.X, self.training, False, [None], fit_affine=False)
        selected, _ = a.select_best_model([other, self.general], self.X, self.validation, [None])
        self.assertIs(selected, self.general)

    def test_nonfinite_and_infeasible_scores_are_excluded(self):
        bad = model(('c', 0.))
        for loss in (float('inf'), float('nan')):
            bad.objectives = (loss, 0., 1., 0)
            self.assertIs(a.select_best_model([bad, self.general])[0], self.general)
            self.assertIs(a.select_pareto_knee([bad, self.general])[0], self.general)
        bad.feasible = False
        with self.assertRaisesRegex(ValueError, 'No feasible model'):
            a.model_options([bad])
        invalid_on_validation = model(('x', 10))
        invalid_on_validation.objectives = (0., 0., 1., 0)
        with self.assertRaisesRegex(ValueError, 'finite validation'):
            a.select_best_model([invalid_on_validation], self.X, self.validation, [None])
        self.assertTrue(invalid_on_validation.feasible)
        invalid_scale = self.general.clone()
        invalid_scale.scales = [(float('nan'), 0.)]
        with self.assertRaisesRegex(ValueError, 'No feasible model'):
            a.select_best_model([invalid_scale], self.X, self.validation, [None])

    def test_loss_tolerance_still_selects_shortest_equivalent_model(self):
        self.general.objectives = (1., .1, 200., 9)
        self.overfit.objectives = (1.005, .2, 100., 7)
        selected, info = a.select_best_model(self.models)
        self.assertIs(selected, self.overfit)
        self.assertEqual(info['source'], 'training')
        self.assertEqual(info['loss_tolerance'], .01)
        self.assertIs(a.select_best_model(self.models, loss_tolerance=0.)[0], self.general)

    def test_incomplete_validation_and_invalid_tolerance_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'both X and Y'):
            a.select_best_model(self.models, X=self.X)
        for tolerance in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
                a.select_best_model(self.models, loss_tolerance=tolerance)

    def test_frozen_evaluation_preserves_model_even_when_scoring_raises(self):
        before = self.general.clone()
        def fail(m, *args, **kwargs):
            m.objectives = (123.,)*4
            m.scales = [(42., 42.)]
            raise RuntimeError('scoring interrupted')
        with patch.object(a, 'assess', side_effect=fail):
            with self.assertRaises(RuntimeError):
                a.frozen_objectives(self.general, self.X, self.validation, [None])
        self.assertEqual(self.general.objectives, before.objectives)
        self.assertEqual(self.general.scales, before.scales)

    def test_knee_ignores_dominated_extremes_and_shape_magnitude(self):
        entries = []
        for i, (loss, bits, shape) in enumerate([(4., 10., 1.), (.2, 80., .2), (.1, 200., .1), (1e12, 164., 0.)]):
            m = model(('c', float(i)))
            m.objectives = (loss, shape, bits, 0)
            entries.append(m)
        chosen, info = a.select_pareto_knee(entries)
        self.assertIs(chosen, entries[1])
        self.assertTrue(info['interior_knee'])
        self.assertIs(a.select_pareto_knee(entries[:3])[0], chosen)
        self.assertEqual(len(a.selection_frontier(a.selection_evaluation(entries)[1])), 3)
        for pool in ([entries[0]], entries[:2]):
            picked, info = a.select_pareto_knee(pool)
            self.assertIs(picked, pool[-1])
            self.assertFalse(info['interior_knee'])

    def test_archive_display_uses_elapsed_age_not_neutral_objective(self):
        self.general.objectives = (*self.general.objectives[:-1], 0)
        self.assertEqual(a.model_age(self.general), 9)

    def test_progress_reports_validation_winner_without_mutating_training_state(self):
        state = island(self.X, [None], ['+', '*'])
        state.archive.items = [self.general]
        before = [(m.objectives, list(m.scales)) for m in self.models]
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            a.evolution_progress(100, [self.overfit], np.arange(len(self.X)), started=a.time.time(),
                Xt=self.X, Yt=self.training, Xv=self.X, Yv=self.validation,
                names=['x'], out_names=['y'], cats=[None], constraints=None,
                coev=False, cases=state.cases, archive=state.archive,
                semantic_qd=state.semantic_qd, structural_qd=state.structural_qd,
                qd_controller=state.qd_controller, pressure=state.pressure, bayes=state.bayes,
                population=self.models)
        self.assertIn('best validation mean loss 0', text.getvalue())
        self.assertIn('same model, full training', text.getvalue())
        self.assertIn('Recommended models (validation scores)', text.getvalue())
        self.assertEqual([(m.objectives, m.scales) for m in self.models], before)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        a.rng.seed(12)
        np.random.seed(12)
        a.INVALID_DIAGNOSTICS.clear()
        self.X = np.arange(1., 6.)[:, None]
        self.Y = 2*self.X+3

    def evaluator(self, X=None, Y=None, cats=None, workers=1, affine=True, constraints=None):
        evaluator = a.ModelEvaluator(workers, {"train": (
            self.X if X is None else X, self.Y if Y is None else Y)},
            affine, [None] if cats is None else cats, constraints, ["y"])
        self.addCleanup(evaluator.close)
        return evaluator

    def test_affine_bounds_preserve_cancellation_and_beat_constant_fit(self):
        x = 3.+1e-8*np.arange(20.)
        y = 5.*np.arange(20.)+2.
        slope, intercept = a.affine(x, y)
        constant = a.affine(np.ones(len(x)), y)[1]
        self.assertLessEqual(abs(slope), 1e9)
        self.assertLessEqual(abs(intercept), 1e9)
        self.assertLess(a.robust_loss(slope*x+intercept, y), a.robust_loss(np.full(len(x), constant), y))
        self.assertLess(np.max(np.abs(slope*x+intercept-y)), 20.)

    def test_exact_affine_beats_bitshift_winner_on_held_out_widths(self):
        X = (16.*np.arange(1,401))[:,None]
        train = np.arange(400)%5 != 0
        Y = X*9/16
        direct, = a.direct_feature_baselines(X[train],Y[train],[None],tuple(a.OPS),True)
        winner = model(("rshift", ("+", ("log_base", ("root4",
            ("gaussian", ("x",0))), ("c",-0.)), ("x",0)), ("c",1.)),
            ops=tuple(a.OPS))
        winner.scales = [(1.125,7031253.375)]
        a.assess(winner,X[train],Y[train],False,[None],fit_affine=False)
        self.assertEqual(direct.scales,[(.5625,0.)])
        self.assertEqual(a.aggregate_loss(direct),0.)
        self.assertEqual(a.model_complexity(direct),22.)  # flat-cost constants (see exact_float_code)
        selected,_ = a.select_best_model([winner,direct],X[~train],Y[~train],[None])
        self.assertIs(selected,direct)
        unseen = np.array([1921.,1920.5])[:,None]
        np.testing.assert_array_equal(a.predict_model(selected,unseen),unseen*9/16)

    def test_affine_simplification_preserves_meaningful_small_coefficients(self):
        x = np.linspace(1e8,2e8,41)
        y = 1.234567891234e-8*x+1.234567891234e-5
        original = (1.234567891234e-8,1.234567891234e-5)
        simplified = a.simplify_affine(x,y,*original)
        np.testing.assert_array_equal(simplified[0]*x+simplified[1],y)
        self.assertNotEqual(simplified[0],0.)
        self.assertNotEqual(simplified[1],0.)

    def test_affine_simplification_does_not_trade_noisy_fit_for_bits(self):
        x = np.linspace(-3.,3.,51)
        y = .56251*x+.00013+np.sin(np.arange(51))*.001
        original = (.56251,.00013)
        simplified = a.simplify_affine(x,y,*original)
        self.assertLessEqual(a.robust_loss(simplified[0]*x+simplified[1],y),
                             a.robust_loss(original[0]*x+original[1],y))
        np.testing.assert_allclose(simplified[0]*x+simplified[1],
                                   original[0]*x+original[1],rtol=0,atol=1e-14)

    def test_affine_constrained_solution_matches_known_boundary_optimum(self):
        x = np.linspace(-1e-9, 1e-9, 21)
        y = 2e9*x+3.
        slope, intercept = a.affine(x, y)
        self.assertAlmostEqual(slope, 1e9)
        self.assertAlmostEqual(intercept, 3.)

    def test_likelihood_is_invariant_to_affine_representation(self):
        implicit = model(("x", 0))
        explicit = model(("+", ("*", ("c", 2.), ("x", 0)), ("c", 3.)))
        a.assess(implicit, self.X, self.Y, True, [None])
        a.assess(explicit, self.X, self.Y, False, [None])
        bank = a.PosteriorParticlePopulation(complexity_prior=0)
        bank.configure_likelihood("student_t", 1.)
        np.testing.assert_allclose(a.predict_model(implicit, self.X), a.predict_model(explicit, self.X))
        self.assertAlmostEqual(bank._likelihood_energy(implicit, self.X, self.Y, [None]),
                               bank._likelihood_energy(explicit, self.X, self.Y, [None]))

    def test_noise_uses_training_prediction_once(self):
        m = model(("x", 0))
        a.assess(m, self.X, self.Y, True, [None])
        banks = a.PerOutputBayesianBanks(["+"], 1, cats=[None])
        banks.update([m], [None], self.X, self.Y)
        self.assertLess(banks[0].particles.noise_scale, 1e-10)

    def test_binary_training_parent_and_predictive_labels_agree(self):
        for scores in ([-2., -1., 2., 3.], [.1, .4, .6, .9]):
            X = np.array(scores)[:, None]
            Y = np.array([0., 0., 1., 1.])[:, None]
            m = model(("x", 0))
            cats = [["no", "yes"]]
            a.assess(m, X, Y, False, cats)
            predictions, probabilities = a.predict_targets(m, X, cats, True)
            np.testing.assert_array_equal(predictions, Y)
            # Binary loss is log loss (graded); the error rate lives in the shape objective.
            self.assertEqual(a.model_shapes(m)[0], 0.)
            np.testing.assert_array_equal(np.argmax(probabilities[0], axis=1), Y[:, 0])
            bank = a.PosteriorParticlePopulation()
            bank.particles = [m]; bank.weights = np.array([1.])
            check = bank.posterior_predictive_check(X, Y, cats)
            self.assertEqual(check["classification"]["accuracy"], 1.)
            wrong = model(("c", 0.))
            a.assess(wrong, X, Y, False, cats)
            self.assertLess(a.aggregate_loss(m), a.aggregate_loss(wrong))
            self.assertTrue(all(parent is m for parent in a.lexicase_parents([m, wrong], 20, X, Y, cats)))

    def check_cache(self, workers):
        evaluator = self.evaluator(workers=workers)
        old, young = model(("x", 0), 17), model(("x", 0), 0)
        evaluator.assess([old, young])
        self.assertEqual([a.model_age(old), a.model_age(young)], [17, 0])
        young.age = 3
        evaluator.assess([young])
        self.assertEqual(a.model_age(young), 3)
        for scored in (old, young):
            direct = scored.clone()
            a.assess(direct, self.X, self.Y, True, [None])
            np.testing.assert_allclose(scored.objectives, direct.objectives)

    def test_cache_preserves_individual_age(self):
        self.check_cache(1)

    def test_multiprocessing_cache_preserves_individual_age(self):
        self.check_cache(2)

    def test_cache_uses_active_grammar(self):
        small = model(("square", ("x", 0)), ops=("square",))
        large = model(("square", ("x", 0)), ops=tuple(a.OPS))
        evaluator = self.evaluator(affine=False)
        evaluator.assess([small, large])
        self.assertNotEqual(a.model_complexity(small), a.model_complexity(large))
        direct = large.clone(); a.assess(direct, self.X, self.Y, False, [None])
        self.assertEqual(large.objectives, direct.objectives)

    def test_constrained_complexity_and_final_selection(self):
        constraints = a.compile_constraints("biology", {"outputs": {"y": {"nonnegative": True}}})
        m = model(("x", 0))
        a.assess(m, self.X, self.Y, True, [None], constraints=constraints, output_names=["y"])
        self.assertEqual(a.model_complexity(m), a.model_description_bits(m))
        self.assertEqual(a._selection_metrics(m, m.objectives, [None])["mdl_bits"], a.model_description_bits(m))
        self.assertEqual(a.model_violations(m), (0.,))

    def test_huber_readout_resists_an_extreme_outlier(self):
        x = np.arange(100.); y = np.r_[x[:-1], 1e8]
        slope, intercept = a.affine(x, y)
        self.assertLess(abs(slope-1), .1)
        self.assertLess(a.robust_loss(slope*x+intercept, y),
                        a.robust_loss(np.full(len(y), np.median(y)), y))
        self.assertLessEqual(a.robust_loss(slope*x+intercept, y), a.robust_loss(x, y))

    def test_huber_readout_fits_clean_and_constant_data(self):
        slope, intercept = a.affine(self.X[:, 0], self.Y[:, 0])
        np.testing.assert_allclose(slope*self.X[:, 0]+intercept, self.Y[:, 0], atol=1e-12)
        slope, intercept = a.affine(np.ones(5), np.full(5, 7.))
        self.assertAlmostEqual(slope+intercept, 7.)

    def test_monotonicity_holds_other_covariates_fixed(self):
        X = np.column_stack((np.arange(4.), -np.arange(4.)))
        predict = lambda inputs: (inputs[:, 0]+100*inputs[:, 1])[:, None]
        constraints = a.compile_constraints("biology", {"outputs": {"y": {"monotonic": {"0": "increasing"}}}})
        self.assertEqual(constraints.violations(predict(X), X, ["y"], predict)[0], 0.)
        decreasing = lambda inputs: (-inputs[:, 0]+100*inputs[:, 1])[:, None]
        self.assertGreater(constraints.violations(decreasing(X), X, ["y"], decreasing)[0], 0.)

    def test_constant_monotonic_feature_is_finite(self):
        X = np.ones((5, 1)); Y = np.arange(5.)[:, None]
        constraints = a.compile_constraints("biology", {"outputs": {"y": {"monotonic": {"0": "increasing"}}}})
        m = model(("x", 0))
        a.assess(m, X, Y, True, [None], constraints=constraints, output_names=["y"])
        self.assertTrue(m.feasible)
        self.assertTrue(np.isfinite(m.objectives).all())
        self.assertEqual(a.model_violations(m), (0.,))

    def test_nonfinite_objectives_are_not_feasible(self):
        class InvalidConstraint:
            active = True
            def violations(self, *args): return (float("nan"),)
        m = model(("x", 0))
        a.assess(m, self.X, self.Y, True, [None], constraints=InvalidConstraint(), output_names=["y"])
        self.assertFalse(m.feasible)
        self.assertEqual(m.invalid_reason, "nonfinite_objectives")

    def test_random_and_mutated_trees_obey_node_and_depth_limits(self):
        for budget in (1, 2, 3, 7, 15, 31):
            for _ in range(100):
                tree = a.random_tree(2, a.DEFAULT_OPS, budget, 4)
                self.assertLessEqual(a.node_size(tree), budget)
                self.assertLessEqual(a.node_depth(tree), 4)
                mutated = a.mutate(tree, 2, a.DEFAULT_OPS, budget, 4)
                self.assertLessEqual(a.node_size(mutated), budget)
                self.assertLessEqual(a.node_depth(mutated), 4)

    def test_constant_parent_can_mutate_and_cross_to_a_feature(self):
        class FeatureMutation:
            def apply(self, *args, **kwargs): return ("x", 0), "subtree"
        child, _, _ = a.semantic_mutate(("c", 0.), self.X, FeatureMutation(), 1, ["+"], 15, 4)
        self.assertEqual(child, ("x", 0))
        self.assertEqual(a.semantic_crossover(("c", 0.), ("x", 0), self.X, 15, 4), ("x", 0))

    def test_repeated_catalogue_update_does_not_recount_data(self):
        X = np.linspace(-1, 1, 11)[:, None]; Y = X.copy()
        entries = [model(("x", 0)), model(("c", 0.))]
        for m in entries: a.assess(m, X, Y, False, [None])
        bank = a.PosteriorParticlePopulation(capacity=2, complexity_prior=0)
        bank.configure_likelihood("student_t", 1.)
        bank.update(entries, X, Y, [None]); expected = bank.catalog_weights.copy()
        for _ in range(4):
            bank.update(entries, X, Y, [None])
            np.testing.assert_allclose(bank.catalog_weights, expected)
        before = [(m.trees, weight) for m, weight in zip(bank.catalog, bank.catalog_weights)]
        bank._resample()
        self.assertEqual([(m.trees, weight) for m, weight in zip(bank.catalog, bank.catalog_weights)], before)
        bank.update([], X, Y, [None])
        np.testing.assert_allclose(bank.catalog_weights, expected)

    def test_particle_projection_has_its_own_complexity_and_ancestry(self):
        m = a.Model([("x", 0), ("square", ("x", 0))], [(1., 0.)]*2,
                    age=8, birth_generation=2, mdl_feature_count=1, mdl_operators=("square",))
        Y = np.column_stack((self.X[:, 0], self.X[:, 0]**2))
        a.assess(m, self.X, Y, False, [None, None])
        banks = a.PerOutputBayesianBanks(["square"], 1, cats=[None, None])
        banks.update([m], [None, None], self.X, Y, affine_on=False)
        for bank in banks.banks:
            for particle in bank.particles.catalog:
                self.assertEqual(a.model_complexity(particle), a.model_description_bits(particle))
                self.assertEqual(particle.birth_generation, 2)
                self.assertEqual(particle.founder_ids, m.founder_ids)

    def test_diversity_reserve_cannot_exceed_catalogue_capacity(self):
        good = model(("x", 0)); diverse = model(("c", 0.))
        for m in (good, diverse): a.assess(m, self.X, self.Y, False, [None])
        bank = a.PosteriorParticlePopulation(capacity=1)
        bank.update([good], diverse=[diverse])
        self.assertEqual(len(bank.catalog), 1)
        self.assertEqual(len(bank.particles), 1)

    def test_particle_injection_returns_inherited_source(self):
        m = model(("x", 0), age=17); m.birth_generation = 3
        a.assess(m, self.X, self.Y, True, [None])
        banks = a.PerOutputBayesianBanks(["+"], 1, cats=[None])
        banks.update([m], [None], self.X, self.Y)
        saw_reuse = False
        for _ in range(30):
            _, _, sources = a.bayesian_injection_trees(banks, 1, 1, ["+"], 15, 4, return_sources=True)
            if sources:
                saw_reuse = True
                self.assertEqual(sources[0].birth_generation, 3)
                self.assertEqual(sources[0].age, 17)
                self.assertEqual(sources[0].founder_ids, m.founder_ids)
        self.assertTrue(saw_reuse)

    def test_archive_ignores_age_for_quality_dominance(self):
        old = model(("x", 0), age=20); young = model(("c", 0.))
        old.objectives = (.1, .1, 5., 20)
        young.objectives = (1., 1., 10., 0)
        archive = a.ParetoArchive(); archive.update([young]); archive.update([old])
        self.assertEqual(len(archive.items), 1)
        self.assertEqual(archive.items[0].trees, old.trees)
        self.assertEqual(archive.items[0].age, 20)

    def test_archived_copy_ages_with_elapsed_generations(self):
        archived = model(("x", 0), age=7)
        a.synchronize_model_ages([archived], 10)
        a.synchronize_model_ages([archived], 30)
        self.assertEqual(archived.age, 27)
        self.assertEqual(archived.birth_generation, 3)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        a.rng.seed(12); np.random.seed(12); a.INVALID_DIAGNOSTICS.clear()
        self.ops = ["+", "-", "*", "square"]

    def evaluator(self, X, Y, cats, workers=1, affine=True):
        ev = a.ModelEvaluator(workers, {"train": (X, Y)}, affine, cats,
                              a.compile_constraints(), [f"y{j}" for j in range(Y.shape[1])])
        self.addCleanup(ev.close)
        return ev

    def assert_persistent_scores(self, state, X, Y, cats):
        for m in [*state.archive.items, *state.semantic_qd.cells.values(),
                  *state.structural_qd.cells.values(), state.best_models.model]:
            if m is None: continue
            actual = a.frozen_metrics(m, X, Y, cats)
            self.assertAlmostEqual(a.aggregate_loss(m), actual["loss"], places=10)

    def test_degenerate_population_stays_at_configured_capacity(self):
        X = np.ones((12, 1)); Y = np.arange(12.)[:, None]; cats = [None]
        state = island(X, cats, self.ops); ev = self.evaluator(X, Y, cats)
        for generation in range(3):
            advance(state, generation, X, Y, cats, self.ops, ev)
            self.assertEqual(len(state.population), state.population_size)
            self.assertTrue(all(m.feasible for m in state.population))

    def test_direct_feature_baseline_survives_first_generation(self):
        X = (16.*np.arange(1,41))[:,None]; Y = X*9/16; cats = [None]
        state = island(X,cats,self.ops)
        self.assertEqual(state.population[0].trees,[("x",0)])
        ev = self.evaluator(X,Y,cats)
        advance(state,0,X,Y,cats,self.ops,ev)
        selected,_ = a.select_best_model([*state.population,*state.archive.items])
        self.assertEqual(selected.trees,[("x",0)])
        self.assertEqual(selected.scales,[(.5625,0.)])

    def test_minibatch_archives_use_full_training_scores(self):
        X = np.linspace(-3, 3, 600)[:, None]; Y = np.sin(X)+.3*X**2; cats = [None]
        state = island(X, cats, self.ops, coev=True); ev = self.evaluator(X, Y, cats)
        for generation in range(3):
            advance(state, generation, X, Y, cats, self.ops, ev, coev=True)
            self.assert_persistent_scores(state, X, Y, cats)

    def test_minibatch_island_migration_compares_full_training_scores(self):
        X = np.linspace(-3, 3, 600)[:, None]; Y = np.sin(X)+.3*X**2; cats = [None]
        states = [island(X, cats, self.ops, coev=True, index=j) for j in range(2)]
        ev = self.evaluator(X, Y, cats)
        for state in states: advance(state, 0, X, Y, cats, self.ops, ev, coev=True)
        a.migrate_islands(states, 1, X=X, nsga_normalization="intercept", parsimony_quality_tolerance=.01, evaluator=ev)
        for state in states:
            self.assert_persistent_scores(state, X, Y, cats)
            for m in state.population:
                self.assertAlmostEqual(a.aggregate_loss(m), a.frozen_metrics(m, X, Y, cats)["loss"])

    def test_old_minibatch_record_is_repaired_before_final_selection(self):
        X = np.arange(10.)[:, None]; Y = X.copy(); Y[5:] += 20; cats = [None]
        state = island(X, cats, self.ops); ev = self.evaluator(X, Y, cats)
        old = model(("x", 0)); a.assess(old, X[:5], Y[:5], True, cats)
        state.archive.update([old], X); state.best_models.update([old])
        a.refresh_persistent_scores(state.archive, state.best_models, state.semantic_qd, state.structural_qd, ev)
        self.assert_persistent_scores(state, X, Y, cats)
        self.assertGreater(a.aggregate_loss(state.archive.items[0]), .001)

    def test_exact_quadratic_fit_survives_evolution(self):
        X = np.linspace(-2, 2, 64)[:, None]; Y = X*X; cats = [None]
        state = island(X, cats, self.ops); ev = self.evaluator(X, Y, cats)
        state.population[0] = model(("square", ("x", 0)))
        for generation in range(3): advance(state, generation, X, Y, cats, self.ops, ev)
        self.assertLess(a.aggregate_loss(state.best_models.model), 1e-20)
        self.assert_persistent_scores(state, X, Y, cats)

    def test_binary_and_multiclass_generations(self):
        X = np.linspace(-2, 2, 36)[:, None]
        for cats, Y in (([["a", "b"]], (X>0).astype(float)),
                        ([["a", "b", "c"]], np.digitize(X, [-.5, .5]).astype(float))):
            state = island(X, cats, self.ops); ev = self.evaluator(X, Y, cats, affine=False)
            for generation in range(2): advance(state, generation, X, Y, cats, self.ops, ev)
            self.assertEqual(len(state.population), state.population_size)
            self.assert_persistent_scores(state, X, Y, cats)

    def test_migrated_adf_is_available_to_descendants(self):
        X = np.linspace(-2, 2, 24)[:, None]; Y = X*X; cats = [None]
        source = island(X, cats, self.ops, adfs=True)
        target = island(X, cats, self.ops, adfs=True, index=1)
        name = "adf_foreign"
        definition = dict(tree=("*", ("arg", 0), ("arg", 0)), arity=1,
                          dependencies=[], supporting_founders=[1, 2, 3], activated_generation=0,
                          retired_generation=None, elite_uses=0, validation=[])
        source.adf_registry.definitions[name] = definition
        source.adf_registry.active = [name]; source.adf_registry.last_used[name] = 0
        m = model((name, ("x", 0)), ops=tuple(self.ops+[name])); m.adfs = {name: definition}
        a.assess(m, X, Y, True, cats); source.population = [m.clone() for _ in source.population]
        ev = self.evaluator(X, Y, cats); ev.assess(target.population)
        a.migrate_islands([source, target], 1, X=X, nsga_normalization="intercept", parsimony_quality_tolerance=.01)
        self.assertIn(name, target.adf_registry.definitions)
        for generation in range(1, 4): advance(target, generation, X, Y, cats, self.ops, ev)
        self.assertFalse(any("adf_foreign" in reason for reason in a.INVALID_DIAGNOSTICS))
        for m in target.population:
            self.assertTrue(m.feasible)

    def test_checkpoint_resume_matches_uninterrupted_generation(self):
        X = np.linspace(-2, 2, 24)[:, None]; Y = X*X+.1*X; cats = [None]
        state = island(X, cats, self.ops); ev = self.evaluator(X, Y, cats)
        advance(state, 0, X, Y, cats, self.ops, ev)
        payload = {}
        a.snapshot_islands(payload, [state], {"count": 1})
        with tempfile.TemporaryDirectory(prefix="afpo-test-") as directory:
            path = Path(directory)/"checkpoint.json"
            a.save_checkpoint(path, 1, state.population, state.bayes, state.archive, payload)
            advance(state, 1, X, Y, cats, self.ops, ev)
            expected = a._json_checkpoint_value(a.island_snapshot(state))
            generation, _, _, _, loaded = a.load_checkpoint(path, False)
            restored = a.island_from_snapshot(loaded["island_states"][0], len(X), .01)
            advance(restored, generation, X, Y, cats, self.ops, ev)
            self.assertEqual(a._json_checkpoint_value(a.island_snapshot(restored)), expected)


if __name__ == "__main__":
    unittest.main()
