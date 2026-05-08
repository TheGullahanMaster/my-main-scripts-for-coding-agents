import re

with open('evo13.py', 'r') as f:
    content = f.read()

# We need to add Bayesian options to the evolution model selection.
# Currently it's:
# [5] Bayesian CGP (surrogate-assisted)

new_options = """    print("  [5] Bayesian CGP  (surrogate-assisted)")
    print("      Uses a Gaussian Process surrogate to model the fitness landscape")
    print("      over CGP genotype space.  Expected Improvement selects which")
    print("      candidate mutations to evaluate, dramatically reducing the number")
    print("      of expensive fitness evaluations.  Single-process, best for small")
    print("      datasets where each evaluation is fast but the search space is large.")
    print("      Requires scikit-learn.")
    print()
    print("  [6] Quality-Diversity (QD) Algorithms")
    print("      Maintains an archive of high-performing individuals spread across")
    print("      a behavioral descriptor space, optimizing for both performance")
    print("      and diversity.")"""

content = content.replace('    print("  [5] Bayesian CGP  (surrogate-assisted)")\n    print("      Uses a Gaussian Process surrogate to model the fitness landscape")\n    print("      over CGP genotype space.  Expected Improvement selects which")\n    print("      candidate mutations to evaluate, dramatically reducing the number")\n    print("      of expensive fitness evaluations.  Single-process, best for small")\n    print("      datasets where each evaluation is fast but the search space is large.")\n    print("      Requires scikit-learn.")', new_options)

with open('evo13.py', 'w') as f:
    f.write(content)
