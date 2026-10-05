The optional Cython fitter

`fitcore.pyx` implements the flat-tree evaluator, damped finite-difference
Levenberg–Marquardt constant fit, and streaming Huber affine readout used by
`afpo.py --fit-backend auto` (the default).

Install Cython in the same Python environment as AFPO and have a C compiler
available. AFPO builds the extension on first use and caches it in
`~/.pyxbld`. No NumPy development headers are needed. If that directory is
unwritable, set `AFPO_PYXBUILD_DIR` to a writable build/cache directory.
Missing build dependencies retain the Python fallback; `--fit-backend python`
explicitly selects it.

The compiled evaluator supports arithmetic, comparisons, basic rounding,
conditionals, and common smooth functions. `fitcore.OPERATOR_CODES` lists the
supported operators. Other trees, ADFs, sequence operators, and non-Huber
loss modes use the Python fitter. Transcendental evaluation retains NumPy's
ufuncs; arithmetic and readout/residual reductions use C loops. The small LM
linear system uses NumPy's solver. Jacobian probes currently evaluate the
whole flat tree. Floating-point reductions can differ from Python to round-off.

Validate with:

```sh
python -B -m unittest test_fitcore test_afpo.CompiledFitterTests test_afpo.BlockedEvaluationTests
```
