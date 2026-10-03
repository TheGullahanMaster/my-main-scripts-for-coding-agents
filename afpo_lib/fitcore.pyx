# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled constant fitter for afpo.py (built on first use through pyximport).

afpo.fit_tree_constants hands over a flat pre-order program (see
afpo._compiled_program): per node an operator code, an argument (feature
index for code 0, constant slot for code 1) and up to three child indices,
children always after their parent.  Every operator here reproduces
afpo._OP_TABLE followed by afpo.clean (clamp to +/-1e12, NaN -> 0); a tree
using any other operator gets None from _compiled_program and stays on the
Python fitter.  Arithmetic operators agree with numpy bit for bit;
transcendental ones come from libm, which can differ from numpy's own kernels
in the last bit (LIBM_OPERATORS), so the two fitters agree to round-off.

fit() is the Levenberg-Marquardt / variable-projection loop of
afpo.fit_tree_constants (same readout, damping, step and stopping rules);
robust_affine() is the Huber IRLS of afpo._affine for large data.
"""
import numpy as np
from libc.math cimport (fabs, sqrt, exp, expm1, log, log1p, sin, cos, tan, tanh, atan, sinh, cosh,
                        hypot, pow, isnan, isfinite)


cdef double EPS = 1e-12
cdef double CLIP = 1e12
cdef double NAN_VALUE = float("nan")

OPERATOR_CODES = {
    "+": 2, "-": 3, "*": 4, "/": 5, "delta": 6, "max": 7, "min": 8, "neg": 9, "abs": 10,
    "square": 11, "cube": 12, "sqrt": 13, "inv": 14, "relu": 15, "leaky_relu": 16,
    "exp": 17, "expm1": 18, "log": 19, "log1p": 20, "sin": 21, "cos": 22, "tan": 23,
    "tanh": 24, "sigmoid": 25, "gaussian": 26, "atan": 27, "sinh": 28, "cosh": 29,
    "pow": 30, "hypot": 31, "distance_2": 32, "exp_decay": 33, "rbf": 34, "if_else": 35,
}
LIBM_OPERATORS = frozenset(("cube", "exp", "expm1", "log", "log1p", "sin", "cos", "tan", "tanh", "sigmoid", "gaussian",
                            "atan", "sinh", "cosh", "pow", "hypot", "distance_2", "exp_decay", "rbf"))

cdef inline double clean(double v) nogil:
    if isnan(v): return 0.
    if v > CLIP: return CLIP
    if v < -CLIP: return -CLIP
    return v

cdef inline double clip(double v, double lo, double hi) nogil:
    # np.clip: NaN passes through, -0. stays -0.
    if v < lo: return lo
    if v > hi: return hi
    return v

cdef inline double maximum(double a, double b) nogil:
    # np.maximum: NaN propagates; on a tie (0. vs -0.) the second operand wins.
    if isnan(a): return a
    if isnan(b): return b
    return a if a > b else b

cdef inline double minimum(double a, double b) nogil:
    if isnan(a): return a
    if isnan(b): return b
    return a if a < b else b

cdef inline double sign(double v) nogil:
    if isnan(v): return v
    if v > 0: return 1.
    if v < 0: return -1.
    return 0.

cdef inline double sigmoid(double v) nogil:
    return 1./(1.+exp(-clip(v, -50., 50.)))

cdef inline double apply(int op, double x, double y, double z) nogil:
    cdef double d
    if op == 2: return x+y
    if op == 3: return x-y
    if op == 4: return x*y
    if op == 5: return x/(EPS if fabs(y) < EPS else y)
    if op == 6: return fabs(x-y)
    if op == 7: return maximum(x, y)
    if op == 8: return minimum(x, y)
    if op == 9: return -x
    if op == 10: return fabs(x)
    if op == 11: return x*x
    if op == 12: return pow(x, 3.)
    if op == 13: return sign(x)*sqrt(fabs(x))
    if op == 14: return 1./(fabs(x)+EPS)
    if op == 15: return maximum(0., x)
    if op == 16: return x if x >= 0 else .01*x
    if op == 17: return exp(clip(x, -50., 50.))
    if op == 18: return expm1(clip(x, -50., 50.))
    if op == 19: return log(fabs(x)+EPS)
    if op == 20: return log1p(fabs(x))
    if op == 21: return sin(x)
    if op == 22: return cos(x)
    if op == 23: return tan(clip(x, -1.55, 1.55))
    if op == 24: return tanh(x)
    if op == 25: return sigmoid(x)
    if op == 26: return exp(-clip(x*x, 0., 50.))
    if op == 27: return atan(x)
    if op == 28: return sinh(clip(x, -20., 20.))
    if op == 29: return cosh(clip(x, -20., 20.))
    if op == 30: return sign(x)*pow(fabs(x), clip(y, -12., 12.))
    if op == 31 or op == 32: return hypot(x, y)
    if op == 33: return exp(-clip(x*y, -50., 50.))
    if op == 34:
        d = x-y
        return exp(-clip(d*d, 0., 50.))
    if op == 35: return y if x > .5 else z
    return NAN_VALUE

cdef void run(const int[::1] code, const int[::1] arg, const int[:, ::1] kid, const double[::1] constants,
              const double[:, ::1] X, double[:, ::1] values) noexcept nogil:
    """Every node's values; node 0 is the root."""
    cdef Py_ssize_t node, row, n = X.shape[0]
    cdef int op, a, b, c
    cdef double y, z
    for node in range(code.shape[0]-1, -1, -1):
        op = code[node]
        if op == 0:
            for row in range(n): values[node, row] = X[row, arg[node]]
        elif op == 1:
            for row in range(n): values[node, row] = constants[arg[node]]
        else:
            a = kid[node, 0]; b = kid[node, 1]; c = kid[node, 2]
            for row in range(n):
                y = values[b, row] if b >= 0 else 0.
                z = values[c, row] if c >= 0 else 0.
                values[node, row] = clean(apply(op, values[a, row], y, z))

def evaluate(code, arg, kid, constants, X):
    """Root values of the program on X (afpo.evaluate of the same tree)."""
    cdef const double[:, ::1] Xc = np.ascontiguousarray(X, dtype=float)
    result = np.empty((len(code), Xc.shape[0]))
    cdef double[:, ::1] values = result
    run(np.ascontiguousarray(code, np.int32), np.ascontiguousarray(arg, np.int32), np.ascontiguousarray(kid, np.int32),
        np.ascontiguousarray(constants, dtype=float).reshape(-1), Xc, values)
    return result[0].copy()

cdef bint weighted_line(const double[::1] u, const double[::1] y, const double[::1] w, double* c0, double* c1) noexcept nogil:
    """afpo._weighted_line: weighted least-squares y ~ c0*u+c1; False when degenerate."""
    cdef Py_ssize_t i, n = u.shape[0]
    cdef double s = 0., su = 0., sy = 0., suu = 0., sdy = 0., uu = 0., mu, my, du
    for i in range(n):
        s += w[i]; su += w[i]*u[i]; sy += w[i]*y[i]
    if not s > 0: return False
    mu = su/s; my = sy/s
    for i in range(n):
        du = u[i]-mu; suu += w[i]*du*du; sdy += w[i]*du*(y[i]-my); uu += w[i]*u[i]*u[i]
    if not isfinite(suu) or suu <= 1e-12*(uu if uu > EPS else EPS): return False
    c0[0] = sdy/suu
    if not isfinite(c0[0]): return False
    c1[0] = my-c0[0]*mu
    return True

cdef bint readout(const double[::1] pred, const double[::1] y, double scale, bint fit_readout, bint robust,
                  double bound, double huber, double[::1] w, double[::1] r) noexcept nogil:
    """The readout() of afpo.fit_tree_constants: residuals into r; False when not finite."""
    cdef Py_ssize_t i, n = pred.shape[0]
    cdef int round_, rounds = 3 if robust else 1
    cdef double slope, intercept, s, t, mean, magnitude, hw
    cdef bint tail
    for i in range(n): w[i] = 1.
    if fit_readout:
        for round_ in range(rounds):
            if not weighted_line(pred, y, w, &slope, &intercept):
                mean = 0.
                for i in range(n): mean += y[i]
                mean /= n
                for i in range(n): r[i] = mean
                break
            if fabs(slope) > bound:
                slope = bound if slope > 0 else -bound
                s = 0.; t = 0.
                for i in range(n): s += w[i]; t += w[i]*(y[i]-slope*pred[i])
                intercept = t/s
            intercept = clip(intercept, -bound, bound)
            for i in range(n): r[i] = slope*pred[i]+intercept
            if robust:
                tail = False
                for i in range(n):
                    magnitude = fabs(r[i]-y[i])
                    hw = huber*scale/(magnitude if magnitude > EPS else EPS)
                    if hw < 1.: tail = True
                    else: hw = 1.
                    w[i] = hw
                if not tail: break
        for i in range(n): r[i] = (r[i]-y[i])/scale
    else:
        for i in range(n): r[i] = (pred[i]-y[i])/scale
    for i in range(n):
        if robust:
            magnitude = fabs(r[i])
            if not magnitude <= huber:
                t = 2*huber*magnitude-huber*huber
                r[i] = sign(r[i])*sqrt(t if t > 0 else 0.)
        if not isfinite(r[i]): return False
    return True

cdef class _Problem:
    cdef const int[::1] code
    cdef const int[::1] arg
    cdef const int[:, ::1] kid
    cdef const double[:, ::1] X
    cdef const double[::1] y
    cdef double[:, ::1] values
    cdef double[::1] w
    cdef double scale, bound, huber
    cdef bint fit_readout, robust
    def __init__(self, code, arg, kid, X, y, scale, fit_readout, robust, bound, huber):
        self.code = code; self.arg = arg; self.kid = kid; self.X = X; self.y = y
        self.values = np.empty((len(code), X.shape[0])); self.w = np.empty(X.shape[0])
        self.scale = scale; self.fit_readout = fit_readout; self.robust = robust; self.bound = bound; self.huber = huber
    def residual(self, constants):
        cdef double[::1] c = np.ascontiguousarray(constants, dtype=float)
        r = np.empty(self.X.shape[0])
        cdef double[::1] rv = r
        cdef double[::1] root = self.values[0]
        cdef bint ok
        with nogil:
            run(self.code, self.arg, self.kid, c, self.X, self.values)
            ok = readout(root, self.y, self.scale, self.fit_readout, self.robust, self.bound, self.huber, self.w, rv)
        return r if ok else None

def fit(code, arg, kid, start, X, y, scale, fit_readout, robust, iterations, bound, huber, limit):
    """(constants, cost, initial cost) after Levenberg-Marquardt, or None when the start is not finite."""
    if not np.isscalar(scale): return None
    problem = _Problem(np.ascontiguousarray(code, np.int32), np.ascontiguousarray(arg, np.int32), np.ascontiguousarray(kid, np.int32),
                       np.ascontiguousarray(X, dtype=float), np.ascontiguousarray(y, dtype=float), float(scale),
                       bool(fit_readout), bool(robust), float(bound), float(huber))
    current = np.array(start, dtype=float)
    r = problem.residual(current)
    if r is None: return None
    cost = initial = float(r@r); damping = 1e-3; k = len(current)
    for _ in range(int(iterations)):
        if cost <= 1e-24: break
        J = np.empty((len(r), k))
        for index in range(k):
            step = 1e-6*max(1., abs(current[index])); probe = current.copy(); probe[index] += step
            rk = problem.residual(probe)
            if rk is None: J = None; break
            J[:, index] = (rk-r)/step
        if J is None or not J.any(): break
        g = J.T@r; H = J.T@J; improved = False; gain = 0.
        regulariser = np.diag(np.diag(H))+1e-12*np.eye(k); descent = -g
        for _ in range(8):
            try: delta = np.linalg.solve(H+damping*regulariser, descent)
            except np.linalg.LinAlgError: damping *= 10; continue
            candidate = np.minimum(np.maximum(current+delta, -limit), limit)
            rc = problem.residual(candidate)
            if rc is not None:
                candidate_cost = float(rc@rc)
                if candidate_cost < cost:
                    gain = cost-candidate_cost; current, r, cost = candidate, rc, candidate_cost
                    damping = max(damping/3, 1e-9); improved = True; break
            damping *= 4
        if not improved or gain <= 1e-10*max(cost, 1e-30): break
    return current, cost, initial

cdef bint irls_affine(const double[::1] p, const double[::1] t, double centre, double spread, double cutoff, double bound,
                      double[::1] u, double[::1] w, double* a_out, double* b_out) noexcept nogil:
    cdef Py_ssize_t i, n = p.shape[0]
    cdef double c0, c1, a, b, updated_a, updated_b, residual, change, change_norm, reference_norm
    cdef double tolerance = sqrt(2.220446049250313e-16)
    cdef int iteration
    for i in range(n): u[i] = (p[i]-centre)/spread; w[i] = 1.
    if not weighted_line(u, t, w, &c0, &c1): return False
    a = c0/spread; b = c1-a*centre
    if fabs(a) > bound or fabs(b) > bound: return False
    for iteration in range(200):
        for i in range(n):
            residual = fabs(a*p[i]+b-t[i])
            residual = cutoff/(residual if residual > EPS else EPS)
            w[i] = 1. if residual >= 1. else residual   # afpo squares its sqrt weights: min(1, cutoff/|r|)
        if not weighted_line(u, t, w, &c0, &c1): return False
        updated_a = c0/spread; updated_b = c1-updated_a*centre
        if fabs(updated_a) > bound or fabs(updated_b) > bound: return False
        change_norm = 0.; reference_norm = 0.
        for i in range(n):
            change = (updated_a-a)*p[i]+updated_b-b; change_norm += change*change
            residual = a*p[i]+b; reference_norm += residual*residual
        a = updated_a; b = updated_b
        if sqrt(change_norm) <= tolerance*(1.+sqrt(reference_norm)): break
    a_out[0] = a; b_out[0] = b
    return True

def robust_affine(pred, y, double centre, double spread, double cutoff, double bound):
    """afpo._affine's Huber IRLS without its bounded or degenerate branches: (a, b), or None to fall back."""
    cdef const double[::1] p = np.ascontiguousarray(pred, dtype=float)
    cdef const double[::1] t = np.ascontiguousarray(y, dtype=float)
    cdef double[::1] u = np.empty(p.shape[0]), w = np.empty(p.shape[0])
    cdef double a = 0., b = 0.
    cdef bint ok
    with nogil: ok = irls_affine(p, t, centre, spread, cutoff, bound, u, w, &a, &b)
    return (float(a), float(b)) if ok else None
