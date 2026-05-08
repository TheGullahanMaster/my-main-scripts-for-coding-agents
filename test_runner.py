import os
import evo13
import test_conditional

evo13.EVOLUTION_MODEL = "bayesian_cgp"
evo13.QUALITY_DIVERSITY_ENABLED = True
evo13.BAYESIAN_VARIANT = "afpo_islands"
evo13.BAYESIAN_INITIAL_SAMPLES = 20
evo13.BAYESIAN_BATCH_SIZE = 10
evo13.BAYESIAN_N_CANDIDATES = 20
evo13.BAYESIAN_MAX_GP_POINTS = 50

# Run diode
print("Testing diode with Bayesian QD...")
test_conditional.test_diode(generations=10, seed=42)

# Run quadratic
print("Testing quadratic with Bayesian QD...")
test_conditional.test_quadratic_disc(generations=10, seed=42)
