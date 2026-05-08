import re

with open('evo13.py', 'r') as f:
    content = f.read()

# Add new globals
global_str = """BAYESIAN_EXPLORATION     = 0.01   # exploration–exploitation trade-off (xi for EI)
BAYESIAN_MAX_GP_POINTS   = 500    # cap on GP training set (oldest evicted first)
BAYESIAN_CONST_OPT_FREQ  = 10     # run light const-opt on HoF every N iterations
BAYESIAN_VARIANT         = "regular" # bayesian variant

QUALITY_DIVERSITY_ENABLED = False # QD algorithms
"""

content = content.replace(
    'BAYESIAN_EXPLORATION     = 0.01   # exploration–exploitation trade-off (xi for EI)\nBAYESIAN_MAX_GP_POINTS   = 500    # cap on GP training set (oldest evicted first)\nBAYESIAN_CONST_OPT_FREQ  = 10     # run light const-opt on HoF every N iterations',
    global_str
)

global_decl = """    global BAYESIAN_EXPLORATION, BAYESIAN_GP_REFIT_FREQ, BAYESIAN_MAX_GP_POINTS
    global BAYESIAN_CONST_OPT_FREQ, BAYESIAN_VARIANT
    global QUALITY_DIVERSITY_ENABLED
    global _INIT_PHASE"""

content = content.replace(
    '    global BAYESIAN_EXPLORATION, BAYESIAN_GP_REFIT_FREQ, BAYESIAN_MAX_GP_POINTS\n    global BAYESIAN_CONST_OPT_FREQ\n    global _INIT_PHASE',
    global_decl
)

bayesian_opts = """            EVOLUTION_MODEL = "bayesian_cgp"
            NUM_ISLANDS_GLOBAL = 1

            print("\\n  Bayesian Variants:")
            print("    [1] Bayesian regular")
            print("    [2] Bayesian with AFPO Generator")
            print("    [3] Bayesian with Islands")
            print("    [4] Bayesian with Islanded AFPO")
            print("    [5] Bayesian with AFPO Islands")
            print("    [6] Bayesian with Quality-Diversity (QD) algorithms")
            b_var = input("  Choose variant [1]: ").strip()

            if b_var == '2':
                BAYESIAN_VARIANT = "afpo"
            elif b_var == '3':
                BAYESIAN_VARIANT = "islands"
            elif b_var == '4':
                BAYESIAN_VARIANT = "islanded_afpo"
            elif b_var == '5':
                BAYESIAN_VARIANT = "afpo_islands"
            elif b_var == '6':
                BAYESIAN_VARIANT = "qd"
            else:
                BAYESIAN_VARIANT = "regular"

            # Bayesian CGP hyperparameters"""

content = content.replace(
    '            EVOLUTION_MODEL = "bayesian_cgp"\n            NUM_ISLANDS_GLOBAL = 1\n            # Bayesian CGP hyperparameters',
    bayesian_opts
)

qd_prompt = """    # ---- Quality-Diversity algorithms ----
    print("\\n" + "─" * 70)
    print("QUALITY-DIVERSITY (QD) ALGORITHMS")
    print("─" * 70)
    print("  When enabled, maintains an archive of high-performing individuals")
    print("  spread across a behavioral descriptor space, optimizing for both")
    print("  performance and diversity.")
    print("─" * 70)
    qd_in = input("Enable Quality-Diversity (QD) algorithms? [y/N]: ").strip().lower()
    if qd_in.startswith('y'):
        QUALITY_DIVERSITY_ENABLED = True
        print("  ✓  Quality-Diversity algorithms ENABLED.")
        if BAYESIAN_VARIANT != "qd":
            pass # Keep user's bayesian variant if they set it. But if they enable QD generally it applies.
    else:
        QUALITY_DIVERSITY_ENABLED = False
        print("  Quality-Diversity algorithms disabled.")

    # ---- Population size (evolutionary modes only) ----"""

content = content.replace(
    '    # ---- Population size (evolutionary modes only) ----',
    qd_prompt
)

with open('evo13.py', 'w') as f:
    f.write(content)
