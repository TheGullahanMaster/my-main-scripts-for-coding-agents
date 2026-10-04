# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# distutils: extra_compile_args = -ffp-contract=off
"""Compiled constant fitter for afpo.py (built on first use through pyximport).

afpo.fit_tree_constants hands over a flat pre-order program (see
afpo._compiled_program): per node an operator code, an argument (feature
index for code 0, constant slot for code 1) and up to three child indices,
children always after their parent.  Every operator here reproduces
afpo._OP_TABLE followed by afpo.clean (clamp to +/-1e12, NaN -> 0); a tree
using any other operator gets None from _compiled_program and stays on the
Python fitter.  Every operator except seqsum/seqprod (which need the
sequence layout) is compiled.  Arithmetic, rounding, comparison and integer
operators agree with numpy bit for bit (integer ones with numpy's int64
wraparound and the platform's float-to-int cast); transcendental and
pow-based ones come from libm, which can differ from numpy's own kernels in
the last bit (LIBM_OPERATORS), so the two fitters agree to round-off.

fit() is the Levenberg-Marquardt / variable-projection loop of
afpo.fit_tree_constants (same readout, damping, step and stopping rules),
entirely in C: the small normal-equation solve is Gaussian elimination with
partial pivoting instead of LAPACK, so it too agrees to round-off;
node_values(), scan_costs() and prediction_costs() are the jump-constant scan
(afpo.scan_jump_constants); robust_affine() is the Huber IRLS of afpo._affine;
numeric_inverse() is afpo.numeric_inverse (the per-row root finding of
semantic backpropagation) on the same operator kernels.

Floating-point contraction is off so a*b+c rounds twice, as numpy does.
"""
import numpy as np
from libc.math cimport (fabs, sqrt, exp, expm1, log, log1p, log10, sin, cos, tan, tanh, atan, atan2, sinh, cosh,
                        hypot, pow, fmod, floor, ceil, trunc, rint, erf, copysign, signbit, isnan, isfinite, INFINITY)
from libc.stdlib cimport malloc, free


cdef double EPS = 1e-12
cdef double CLIP = 1e12

def set_limits(double clip, double eps):
    """afpo's --clamp: the value clamp and the divisor/log guard used by every kernel below."""
    global CLIP, EPS
    CLIP = clip
    EPS = eps
cdef double NAN_VALUE = float("nan")
cdef double PI = 3.141592653589793
cdef double LOGE2 = 0.693147180559945309417232121458176568
cdef extern from "<limits.h>":
    long long LLONG_MIN
cdef long long INT64_MIN = LLONG_MIN

OPERATOR_CODES = {
    "+": 2, "-": 3, "*": 4, "/": 5, "delta": 6, "max": 7, "min": 8, "neg": 9, "abs": 10,
    "square": 11, "cube": 12, "sqrt": 13, "inv": 14, "relu": 15, "leaky_relu": 16,
    "exp": 17, "expm1": 18, "log": 19, "log1p": 20, "sin": 21, "cos": 22, "tan": 23,
    "tanh": 24, "sigmoid": 25, "gaussian": 26, "atan": 27, "sinh": 28, "cosh": 29,
    "pow": 30, "hypot": 31, "distance_2": 32, "exp_decay": 33, "rbf": 34, "if_else": 35,
    "log_base": 36, "atan2": 37, "harmonic": 38, "geometric": 39, "mod": 40, "copysign": 41, "floordiv": 42,
    "gt": 43, "lt": 44, "gte": 45, "lte": 46, "eq": 47, "ne": 48, "round2": 49, "floor2": 50, "ceil2": 51,
    "quantize": 52, "perceptronSigma2": 53, "perceptronReLU2": 54, "perceptronCustom2": 55,
    "bitwise_and": 56, "bitwise_or": 57, "bitwise_xor": 58, "lshift": 59, "rshift": 60, "gcd": 61, "lcm": 62,
    "cat": 63, "x_at_pos_y": 64, "10^x": 65, "log10": 66,
    **{f"pow{n}": 63+n for n in range(4, 11)},       # 67..73
    **{f"root{n}": 71+n for n in range(3, 11)},      # 74..81
    "frac": 82, "round": 83, "floor": 84, "ceil": 85, "int": 86, "softplus": 87, "sign": 88, "signbit": 89,
    "sinc": 90, "xlogx": 91, "erf": 92, "deg2rad": 93, "rad2deg": 94, "perceptronSigma1": 95,
    "perceptronReLU1": 96, "perceptronCustom1": 97, "oom": 98, "bitwise_not": 99, "if_in_range": 100,
    "if_out_of_range": 101, "lerp": 102, "distance_3": 103, "python_rng": 104, "perlin_noise": 105,
}
LIBM_OPERATORS = frozenset(("cube", "exp", "expm1", "log", "log1p", "sin", "cos", "tan", "tanh", "sigmoid", "gaussian",
                            "atan", "sinh", "cosh", "pow", "hypot", "distance_2", "exp_decay", "rbf",
                            "log_base", "atan2", "round2", "floor2", "ceil2", "perceptronSigma2", "perceptronCustom2",
                            "cat", "x_at_pos_y", "10^x", "log10", *[f"pow{n}" for n in range(4, 11)],
                            *[f"root{n}" for n in range(3, 11)], "softplus", "sinc", "xlogx", "erf",
                            "perceptronSigma1", "perceptronCustom1", "oom", "python_rng", "perlin_noise"))

cdef inline double clean(double v) noexcept nogil:
    if isnan(v): return 0.
    if v > CLIP: return CLIP
    if v < -CLIP: return -CLIP
    return v

cdef inline double clip(double v, double lo, double hi) noexcept nogil:
    # np.clip: NaN passes through, -0. stays -0.
    if v < lo: return lo
    if v > hi: return hi
    return v

cdef inline double maximum(double a, double b) noexcept nogil:
    # np.maximum: NaN propagates; on a tie (0. vs -0.) the second operand wins.
    if isnan(a): return a
    if isnan(b): return b
    return a if a > b else b

cdef inline double minimum(double a, double b) noexcept nogil:
    if isnan(a): return a
    if isnan(b): return b
    return a if a < b else b

cdef inline double sign(double v) noexcept nogil:
    if isnan(v): return v
    if v > 0: return 1.
    if v < 0: return -1.
    return 0.

cdef inline double sigmoid(double v) noexcept nogil:
    return 1./(1.+exp(-clip(v, -50., 50.)))

cdef inline double guard(double v, double fill) noexcept nogil:
    # afpo._guard_zero: np.where(np.abs(v) < EPS, fill, v)
    return fill if fabs(v) < EPS else v

cdef inline double np_mod(double a, double b) noexcept nogil:
    """np.mod for floats (npy_remainder): the result takes the divisor's sign."""
    cdef double m = fmod(a, b)
    if b == 0: return m
    if m != 0:
        if (b < 0) != (m < 0): m += b
    else: m = copysign(0., b)
    return m

cdef inline double np_clip(double x, double lo, double hi) noexcept nogil:
    """np.clip with array bounds: _NPY_MIN(_NPY_MAX(x, lo), hi), NaN in any operand propagates."""
    cdef double t = x if isnan(x) else (x if x > lo else lo)
    return t if isnan(t) else (t if t < hi else hi)

cdef inline long long to_int64(double v) noexcept nogil:
    # ndarray.astype(np.int64): the platform's C cast, as numpy uses.
    return <long long>v

cdef inline long long wrap_abs(long long a) noexcept nogil:
    # np.abs on int64 wraps: abs(INT64_MIN) is INT64_MIN.
    return <long long>(0ULL-<unsigned long long>a) if a < 0 else a

cdef inline long long wrap_mul(long long a, long long b) noexcept nogil:
    return <long long>(<unsigned long long>a*<unsigned long long>b)

cdef inline long long floor_div64(long long a, long long b) noexcept nogil:
    """np.floor_divide on int64 (b != 0 here)."""
    cdef long long q
    if b == -1 and a == INT64_MIN: return INT64_MIN
    q = a//b                                  # C division (cdivision): truncates toward zero
    if a % b != 0 and ((a < 0) != (b < 0)): q -= 1
    return q

cdef inline long long gcd64(long long a, long long b) noexcept nogil:
    """np.gcd on int64 (npy_gcdll): Euclid on the wrapped absolute values."""
    cdef unsigned long long u = <unsigned long long>(0ULL-<unsigned long long>a) if a < 0 else <unsigned long long>a
    cdef unsigned long long v = <unsigned long long>(0ULL-<unsigned long long>b) if b < 0 else <unsigned long long>b
    cdef unsigned long long c
    while u != 0:
        c = u; u = v % u; v = c
    return <long long>v

cdef inline long long shift_left64(long long a, long long b) noexcept nogil:
    # npy_lshiftll: zero once the shift reaches the word size (b is unsigned there).
    if <unsigned long long>b < 64: return <long long>(<unsigned long long>a << b)
    return 0

cdef inline long long shift_right64(long long a, long long b) noexcept nogil:
    if <unsigned long long>b < 64: return a >> b
    return -1 if a < 0 else 0

cdef inline long long int_operand(double v) noexcept nogil:
    return to_int64(clip(trunc(v), -1e12, 1e12))

cdef inline double softplus(double v) noexcept nogil:
    """np.logaddexp(0, v) (npy_logaddexp)."""
    cdef double tmp
    if v == 0: return LOGE2
    tmp = 0.-v
    if tmp > 0: return log1p(exp(-tmp))
    if tmp <= 0: return v+log1p(exp(tmp))
    return tmp

cdef inline double sinc(double v) noexcept nogil:
    """np.sinc(v/np.pi): sin(pi*t)/(pi*t) with t=v/pi, and t=1e-20 at zero."""
    cdef double t = v/PI, q
    if t == 0: t = 1e-20
    q = PI*t
    return sin(q)/q

cdef inline double apply(int op, double x, double y, double z) noexcept nogil:
    cdef double d
    if op == 2: return x+y
    elif op == 3: return x-y
    elif op == 4: return x*y
    elif op == 5: return x/(EPS if fabs(y) < EPS else y)
    elif op == 6: return fabs(x-y)
    elif op == 7: return maximum(x, y)
    elif op == 8: return minimum(x, y)
    elif op == 9: return -x
    elif op == 10: return fabs(x)
    elif op == 11: return x*x
    elif op == 12: return pow(x, 3.)
    elif op == 13: return sign(x)*sqrt(fabs(x))
    elif op == 14: return 1./(fabs(x)+EPS)
    elif op == 15: return maximum(0., x)
    elif op == 16: return x if x >= 0 else .01*x
    elif op == 17: return exp(clip(x, -50., 50.))
    elif op == 18: return expm1(clip(x, -50., 50.))
    elif op == 19: return log(fabs(x)+EPS)
    elif op == 20: return log1p(fabs(x))
    elif op == 21: return sin(x)
    elif op == 22: return cos(x)
    elif op == 23: return tan(clip(x, -1.55, 1.55))
    elif op == 24: return tanh(x)
    elif op == 25: return sigmoid(x)
    elif op == 26: return exp(-clip(x*x, 0., 50.))
    elif op == 27: return atan(x)
    elif op == 28: return sinh(clip(x, -20., 20.))
    elif op == 29: return cosh(clip(x, -20., 20.))
    elif op == 30: return sign(x)*pow(fabs(x), clip(y, -12., 12.))
    elif op == 31 or op == 32: return hypot(x, y)
    elif op == 33: return exp(-clip(x*y, -50., 50.))
    elif op == 34:
        d = x-y
        return exp(-clip(d*d, 0., 50.))
    elif op == 35: return y if x > .5 else z
    if op <= 66: return apply_binary(op, x, y)
    return apply_more(op, x, y, z)

cdef inline double apply_binary(int op, double x, double y) noexcept nogil:
    cdef double f, e
    cdef long long u, v, g
    if op == 36: return log(fabs(x)+EPS)/log(fabs(y)+1.000001)
    elif op == 37: return atan2(x, y)
    elif op == 38: return 2*x*y/(fabs(x+y)+EPS)
    elif op == 39: return sqrt(fabs(x*y))
    elif op == 40: return np_mod(x, guard(y, 1.))
    elif op == 41: return fabs(x)*sign(y)
    elif op == 42: return floor(x/guard(y, 1.))
    elif op == 43: return 1. if x > y else 0.
    elif op == 44: return 1. if x < y else 0.
    elif op == 45: return 1. if x >= y else 0.
    elif op == 46: return 1. if x <= y else 0.
    elif op == 47: return 1. if x == y else 0.
    elif op == 48: return 1. if x != y else 0.
    elif op == 49 or op == 50 or op == 51:
        f = pow(10., clip(rint(y), -10., 10.))
        if op == 49: return rint(x*f)/f
        if op == 50: return floor(x*f)/f
        return ceil(x*f)/f
    elif op == 52: return rint(x/guard(y, 1.))*y
    elif op == 53: return 1./(1.+exp(-clip(x+y, -50., 50.)))
    elif op == 54: return maximum(0., x+y)
    elif op == 55: return (x+y)/(1.+exp(-clip(x+y, -50., 50.)))
    elif op == 56: return <double>(int_operand(x) & int_operand(y))
    elif op == 57: return <double>(int_operand(x) | int_operand(y))
    elif op == 58: return <double>(int_operand(x) ^ int_operand(y))
    elif op == 59: return <double>shift_left64(int_operand(x), to_int64(clip(trunc(y), 0., 62.)))
    elif op == 60: return <double>shift_right64(int_operand(x), to_int64(clip(trunc(y), 0., 62.)))
    elif op == 61 or op == 62:
        u = wrap_abs(to_int64(trunc(x))); v = wrap_abs(to_int64(trunc(y))); g = gcd64(u, v)
        if op == 61: return <double>g
        return <double>wrap_abs(wrap_mul(floor_div64(u, 1 if g == 0 else g), v))
    elif op == 63:
        f = fabs(trunc(x)); e = fabs(trunc(y))
        return f*pow(10., 1. if e == 0 else minimum(9., floor(log10(e+EPS))+1))+e
    elif op == 64: return <double>to_int64(np_mod(floor(fabs(x))/pow(10., clip(trunc(y), 0., 12.)), 10.))
    elif op == 65: return pow(10., clip(x, -12., 12.))
    elif op == 66: return log10(fabs(x)+EPS)
    return NAN_VALUE

cdef inline double apply_more(int op, double x, double y, double z) noexcept nogil:
    cdef double lo, hi
    if op <= 73: return pow(x, <double>(op-63))
    if op <= 81: return sign(x)*pow(fabs(x), 1./(op-71))
    if op == 82: return x-floor(x)
    elif op == 83: return rint(x)
    elif op == 84: return floor(x)
    elif op == 85: return ceil(x)
    elif op == 86: return trunc(x)
    elif op == 87: return softplus(x)
    elif op == 88: return sign(x)
    elif op == 89: return 1. if signbit(x) else 0.
    elif op == 90: return sinc(x)
    elif op == 91: return x*log(fabs(x)+EPS)
    elif op == 92: return erf(x)
    elif op == 93: return x*(PI/180.)
    elif op == 94: return x*(180./PI)
    elif op == 95: return sigmoid(x)
    elif op == 96: return maximum(0., x)
    elif op == 97: return x/(1.+exp(-clip(x, -50., 50.)))
    elif op == 98: return floor(log10(fabs(x)+EPS))
    elif op == 99: return <double>(~int_operand(x))
    elif op == 100: return np_clip(x, minimum(y, z), maximum(y, z))
    elif op == 101:
        lo = minimum(y, z); hi = maximum(y, z)
        if x < lo or x > hi: return x
        return lo if x < (lo+hi)/2 else hi
    elif op == 102: return x+(y-x)*z
    elif op == 103: return sqrt(x*x+y*y+z*z)
    elif op == 104: return np_mod(sin(floor(x)*12.9898)*43758.5453, 1.)
    elif op == 105: return sin(x*12.9898)*.5+sin(x*78.233)*.25
    if 67 <= op <= 73: return pow(x, <double>(op-63))
    if 74 <= op <= 81: return sign(x)*pow(fabs(x), 1./(op-71))
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
    cdef bint residual_into(self, const double[::1] constants, double[::1] r) noexcept nogil:
        run(self.code, self.arg, self.kid, constants, self.X, self.values)
        return readout(self.values[0], self.y, self.scale, self.fit_readout, self.robust, self.bound, self.huber, self.w, r)
    def residual(self, constants):
        cdef double[::1] c = np.ascontiguousarray(constants, dtype=float)
        r = np.empty(self.X.shape[0])
        cdef double[::1] rv = r
        cdef bint ok
        with nogil: ok = self.residual_into(c, rv)
        return r if ok else None

cdef bint solve_in_place(double* A, double* b, Py_ssize_t k) noexcept nogil:
    """A x = b by Gaussian elimination with partial pivoting (A is k x k, row major);
    x overwrites b.  False on an exactly zero pivot, where LAPACK's gesv (which
    numpy.linalg.solve calls) reports a singular matrix."""
    cdef Py_ssize_t i, j, col, pivot
    cdef double best, factor, t
    for col in range(k):
        pivot = col; best = fabs(A[col*k+col])
        for i in range(col+1, k):
            if fabs(A[i*k+col]) > best: best = fabs(A[i*k+col]); pivot = i
        if A[pivot*k+col] == 0.: return False
        if pivot != col:
            for j in range(k):
                t = A[col*k+j]; A[col*k+j] = A[pivot*k+j]; A[pivot*k+j] = t
            t = b[col]; b[col] = b[pivot]; b[pivot] = t
        for i in range(col+1, k):
            factor = A[i*k+col]/A[col*k+col]
            if factor != 0.:
                for j in range(col, k): A[i*k+j] -= factor*A[col*k+j]
                b[i] -= factor*b[col]
    for i in range(k-1, -1, -1):
        t = b[i]
        for j in range(i+1, k): t -= A[i*k+j]*b[j]
        b[i] = t/A[i*k+i]
    return True

cdef int lm_fit(_Problem problem, double[::1] current, double[::1] probe, double[::1] candidate,
                double[::1] r, double[::1] rk, double[::1] rc, double[:, ::1] J, double* H, double* A,
                double* g, double* delta, int iterations, double limit, double* cost_out, double* initial_out) noexcept nogil:
    """The Levenberg-Marquardt loop of afpo.fit_tree_constants (same damping,
    step and stopping rules), entirely in C.  Returns 0 when the start is not
    finite, else 1 with the result in current/cost_out/initial_out."""
    cdef Py_ssize_t n = r.shape[0], k = current.shape[0], i, j, index, row
    cdef int iteration, attempt
    cdef double cost, initial, damping = 1e-3, step, gain, candidate_cost, s
    cdef bint improved, nonzero, failed
    if not problem.residual_into(current, r): return 0
    cost = 0.
    for row in range(n): cost += r[row]*r[row]
    initial = cost
    for iteration in range(iterations):
        if cost <= 1e-24: break
        failed = False; nonzero = False
        for index in range(k):
            step = 1e-6*(fabs(current[index]) if fabs(current[index]) > 1. else 1.)
            for i in range(k): probe[i] = current[i]
            probe[index] += step
            if not problem.residual_into(probe, rk): failed = True; break
            for row in range(n):
                J[row, index] = (rk[row]-r[row])/step
                if J[row, index] != 0.: nonzero = True
        if failed or not nonzero: break
        for i in range(k):
            s = 0.
            for row in range(n): s += J[row, i]*r[row]
            g[i] = s
            for j in range(i, k):
                s = 0.
                for row in range(n): s += J[row, i]*J[row, j]
                H[i*k+j] = s; H[j*k+i] = s
        improved = False; gain = 0.
        for attempt in range(8):
            for i in range(k):
                for j in range(k): A[i*k+j] = H[i*k+j]
                A[i*k+i] = H[i*k+i]+damping*(H[i*k+i]+1e-12)
                delta[i] = -g[i]
            if not solve_in_place(A, delta, k): damping *= 10; continue
            for i in range(k): candidate[i] = clip(current[i]+delta[i], -limit, limit)   # NaN passes, as in numpy
            if problem.residual_into(candidate, rc):
                candidate_cost = 0.
                for row in range(n): candidate_cost += rc[row]*rc[row]
                if candidate_cost < cost:
                    gain = cost-candidate_cost; cost = candidate_cost
                    for i in range(k): current[i] = candidate[i]
                    for row in range(n): r[row] = rc[row]
                    damping = damping/3 if damping/3 > 1e-9 else 1e-9
                    improved = True; break
            damping *= 4
        if not improved or gain <= 1e-10*(cost if cost > 1e-30 else 1e-30): break
    cost_out[0] = cost; initial_out[0] = initial
    return 1

def fit(code, arg, kid, start, X, y, scale, fit_readout, robust, iterations, bound, huber, limit):
    """(constants, cost, initial cost) after Levenberg-Marquardt, or None when the start is not finite."""
    if not np.isscalar(scale): return None
    problem = _Problem(np.ascontiguousarray(code, np.int32), np.ascontiguousarray(arg, np.int32), np.ascontiguousarray(kid, np.int32),
                       np.ascontiguousarray(X, dtype=float), np.ascontiguousarray(y, dtype=float), float(scale),
                       bool(fit_readout), bool(robust), float(bound), float(huber))
    current = np.array(start, dtype=float)
    cdef Py_ssize_t k = current.shape[0], n = problem.X.shape[0]
    cdef double[::1] cv = current
    cdef double[::1] probe = np.empty(k), candidate = np.empty(k), r = np.empty(n), rk = np.empty(n), rc = np.empty(n)
    cdef double[:, ::1] J = np.empty((n, k))
    cdef double[::1] work = np.empty(2*k*k+2*k+1)
    cdef double cost = 0., initial = 0.
    cdef int ok, steps = int(iterations)
    cdef double bound_limit = float(limit)
    with nogil:
        ok = lm_fit(problem, cv, probe, candidate, r, rk, rc, J, &work[0], &work[k*k], &work[2*k*k], &work[2*k*k+k],
                    steps, bound_limit, &cost, &initial)
    if not ok: return None
    return current, cost, initial

def node_values(code, arg, kid, constants, X):
    """Every node's values on X (row 0 is the root): afpo._FlatTree.values on the compiled operators."""
    cdef const double[:, ::1] Xc = np.ascontiguousarray(X, dtype=float)
    result = np.empty((len(code), Xc.shape[0]))
    cdef double[:, ::1] values = result
    run(np.ascontiguousarray(code, np.int32), np.ascontiguousarray(arg, np.int32), np.ascontiguousarray(kid, np.int32),
        np.ascontiguousarray(constants, dtype=float).reshape(-1), Xc, values)
    return result

cdef double scan_cost(const double* P, const double[::1] y, double scale, bint fit_readout, double huber, double bound) noexcept nogil:
    """afpo._scan_costs for one prediction row: Huber cost after a least-squares readout."""
    cdef Py_ssize_t i, n = y.shape[0]
    cdef double mu = 0., my = 0., suu = 0., spp = 0., sdy = 0., du, slope = 0., p, r, cost = 0.
    cdef bint degenerate
    if fit_readout:
        for i in range(n): mu += P[i]; my += y[i]
        mu /= n; my /= n
        for i in range(n):
            du = P[i]-mu; suu += du*du; spp += P[i]*P[i]; sdy += du*(y[i]-my)
        degenerate = not (suu > 1e-12*(spp if spp > EPS else EPS))
        slope = sdy/(1. if degenerate else suu)
        if degenerate or not isfinite(slope): slope = 0.
        else: slope = clip(slope, -bound, bound)
    for i in range(n):
        p = slope*(P[i]-mu)+my if fit_readout else P[i]
        r = fabs((p-y[i])/scale)
        cost += .5*r*r if r <= huber else huber*(r-.5*huber)
    return cost if isfinite(cost) else INFINITY

def prediction_costs(P, y, double scale, bint fit_readout, double huber, double bound):
    """afpo._scan_costs of a (K, n) prediction batch."""
    cdef const double[:, ::1] Pc = np.ascontiguousarray(P, dtype=float)
    cdef const double[::1] yc = np.ascontiguousarray(y, dtype=float)
    costs = np.empty(Pc.shape[0])
    cdef double[::1] out = costs
    cdef Py_ssize_t k
    with nogil:
        for k in range(Pc.shape[0]): out[k] = scan_cost(&Pc[k, 0], yc, scale, fit_readout, huber, bound)
    return costs

def scan_costs(code, kid, values, int slot_node, path, candidates, y, double scale, bint fit_readout, double huber, double bound):
    """Costs of K values of one constant (node slot_node): only the nodes on
    its path to the root (path: parent first, root last) are recomputed, row
    by row, from the other nodes' stored values (afpo._nudged_roots)."""
    cdef const int[::1] codes = np.ascontiguousarray(code, np.int32)
    cdef const int[:, ::1] kids = np.ascontiguousarray(kid, np.int32)
    cdef const double[:, ::1] V = np.ascontiguousarray(values, dtype=float)
    cdef const int[::1] up = np.ascontiguousarray(path, np.int32)
    cdef const double[::1] cand = np.ascontiguousarray(candidates, dtype=float)
    cdef const double[::1] yc = np.ascontiguousarray(y, dtype=float)
    cdef Py_ssize_t K = cand.shape[0], n = V.shape[1], c, row, step
    cdef double[:, ::1] P = np.empty((K, n))
    costs = np.empty(K)
    cdef double[::1] out = costs
    cdef int node, changed, a, b, e
    cdef double current, u, v, w
    with nogil:
        for c in range(K):
            for row in range(n):
                current = cand[c]; changed = slot_node
                for step in range(up.shape[0]):
                    node = up[step]; a = kids[node, 0]; b = kids[node, 1]; e = kids[node, 2]
                    u = current if a == changed else V[a, row]
                    v = (current if b == changed else V[b, row]) if b >= 0 else 0.
                    w = (current if e == changed else V[e, row]) if e >= 0 else 0.
                    current = clean(apply(codes[node], u, v, w)); changed = node
                P[c, row] = current
            out[c] = scan_cost(&P[c, 0], yc, scale, fit_readout, huber, bound)
    return costs

cdef inline double apply_exact(int op, double x, double y, double z) noexcept nogil:
    """afpo._UNGUARDED: the operators' mathematics without their input clips."""
    if op == 30: return sign(x)*pow(fabs(x), y)
    elif op == 17: return exp(x)
    elif op == 18: return expm1(x)
    elif op == 65: return pow(10., x)
    elif op == 33: return exp(-x*y)
    elif op == 26: return exp(-x*x)
    elif op == 25 or op == 95: return 1./(1.+exp(-x))
    elif op == 53: return 1./(1.+exp(-(x+y)))
    elif op == 97: return x/(1.+exp(-x))
    elif op == 55: return (x+y)/(1.+exp(-(x+y)))
    elif op == 28: return sinh(x)
    elif op == 29: return cosh(x)
    elif op == 23: return tan(x)
    return apply(op, x, y, z)

cdef int guard_block(const int[::1] code, const int[::1] arg, const int[:, ::1] kid, const double[::1] constants,
                     const double[:, ::1] X, Py_ssize_t start, Py_ssize_t stop, double[:, ::1] value, double[:, ::1] exact,
                     double[::1] root_value, double[::1] root_exact) noexcept nogil:
    """Guarded and exact values of rows start..stop; 1 when a value reaches the clamp."""
    cdef Py_ssize_t node, row, i
    cdef int op, a, b, c
    cdef double v, y, z, ey, ez
    for node in range(code.shape[0]-1, -1, -1):
        op = code[node]
        for row in range(start, stop):
            i = row-start
            if op == 0: value[node, i] = X[row, arg[node]]; exact[node, i] = value[node, i]
            elif op == 1: value[node, i] = constants[arg[node]]; exact[node, i] = value[node, i]
            else:
                a = kid[node, 0]; b = kid[node, 1]; c = kid[node, 2]
                y = value[b, i] if b >= 0 else 0.; z = value[c, i] if c >= 0 else 0.
                ey = exact[b, i] if b >= 0 else 0.; ez = exact[c, i] if c >= 0 else 0.
                v = clean(apply(op, value[a, i], y, z))
                if fabs(v) >= CLIP: return 1
                value[node, i] = v; exact[node, i] = apply_exact(op, exact[a, i], ey, ez)
    for row in range(start, stop):
        root_value[row] = value[0, row-start]; root_exact[row] = exact[0, row-start]
    return 0

def guard_verdict(code, arg, kid, constants, X, double tolerance, int block=4096):
    """afpo.guard_engagement for one compiled tree: 0 when no numeric guard
    shapes its values on X, 1 at the value clamp, 2 when an input clip changes
    the output beyond tolerance relative to the output's spread."""
    cdef const int[::1] codes = np.ascontiguousarray(code, np.int32)
    cdef const int[::1] args = np.ascontiguousarray(arg, np.int32)
    cdef const int[:, ::1] kids = np.ascontiguousarray(kid, np.int32)
    cdef const double[::1] c = np.ascontiguousarray(constants, dtype=float).reshape(-1)
    cdef const double[:, ::1] Xc = np.ascontiguousarray(X, dtype=float)
    cdef Py_ssize_t n = Xc.shape[0], width = n if n <= 2*block else block, start, row
    cdef double[:, ::1] value = np.empty((codes.shape[0], width)), exact = np.empty((codes.shape[0], width))
    cdef double[::1] root_value = np.empty(n), root_exact = np.empty(n)
    cdef double mean = 0., spread = 0., d, limit
    cdef int verdict = 0
    with nogil:
        start = 0
        while start < n and verdict == 0:
            verdict = guard_block(codes, args, kids, c, Xc, start, start+width if start+width < n else n, value, exact, root_value, root_exact)
            start += width
        if verdict == 0 and n > 0:
            if n > 1:
                for row in range(n): mean += root_value[row]
                mean /= n
                for row in range(n):
                    d = root_value[row]-mean; spread += d*d
                spread = sqrt(spread/n)
            for row in range(n):
                limit = tolerance*(spread+1e-9*(1.+fabs(root_value[row])))
                if not fabs(root_exact[row]-root_value[row]) <= limit: verdict = 2; break
    return verdict

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


cdef void merge3(const double* a, Py_ssize_t na, const double* b, Py_ssize_t nb, const double* c, Py_ssize_t nc,
                 double* out) noexcept nogil:
    """Merge three ascending runs into out (the same values np.sort would give)."""
    cdef Py_ssize_t i = 0, j = 0, l = 0, o
    cdef double x, y, z
    for o in range(na+nb+nc):
        x = a[i] if i < na else INFINITY
        y = b[j] if j < nb else INFINITY
        z = c[l] if l < nc else INFINITY
        if i < na and x <= y and x <= z: out[o] = x; i += 1
        elif j < nb and y <= z: out[o] = y; j += 1
        else: out[o] = z; l += 1

cdef inline double apply_at(int op, int k, double z, double a0, double a1, double a2) noexcept nogil:
    """Operator value with z in argument position k (afpo.numeric_inverse's apply)."""
    if k == 0: a0 = z
    elif k == 1: a1 = z
    else: a2 = z
    return clean(apply(op, a0, a1, a2))

cdef inline double finite_or_inf(double v) noexcept nogil:
    return v if isfinite(v) else INFINITY

cdef void invert_rows(int op, int k, const double[::1] desired, const double[:, ::1] args,
                      const double[::1] wide, const double[::1] narrow, const double[::1] fixed,
                      double* grid, double* f, double[::1] out) noexcept nogil:
    cdef Py_ssize_t row, j, best, index, n = desired.shape[0]
    cdef Py_ssize_t W = wide.shape[0], N = narrow.shape[0], F = fixed.shape[0], G = W+N+F
    cdef int iteration
    cdef double d, current, span, a0, a1, a2, distance, best_distance, near, best_near
    cdef double lo, hi, flo, fhi, mid, fmid, root, tolerance, left, right, magnitude, a_, b_, m1, m2, f1, f2, dip_root
    cdef bint found, crossing_ok, dip_ok
    for row in range(n):
        d = desired[row]; a0 = args[0, row]; a1 = args[1, row]; a2 = args[2, row]
        current = args[k, row]; span = maximum(fabs(current), 1.)
        # The three pieces are each ascending (span > 0): a three-way merge is the sort.
        for j in range(W): f[j] = current+span*wide[j]
        for j in range(N): f[W+j] = current+span*narrow[j]
        merge3(f, W, f+W, N, &fixed[0], F, grid)
        for j in range(G):
            f[j] = apply_at(op, k, grid[j], a0, a1, a2)-d
            if not isfinite(f[j]): f[j] = NAN_VALUE
        # Sign change nearest the current value (first on ties, like np.argmin).
        best = 0; best_distance = INFINITY
        for j in range(G-1):
            if isfinite(f[j]) and isfinite(f[j+1]) and sign(f[j])*sign(f[j+1]) <= 0:
                distance = fabs(.5*(grid[j]+grid[j+1])-current)
                if distance < best_distance: best_distance = distance; best = j
        found = isfinite(best_distance)
        lo = grid[best]; hi = grid[best+1]; flo = f[best]; fhi = f[best+1]
        for iteration in range(30):
            mid = .5*(lo+hi); fmid = apply_at(op, k, mid, a0, a1, a2)-d
            if sign(flo)*sign(fmid) <= 0: hi = mid; fhi = fmid
            else: lo = mid; flo = fmid
        root = lo if fabs(flo) <= fabs(fhi) else hi
        tolerance = 1e-6*maximum(fabs(d), 1.)
        # Nearest dip of |f|, then a ternary search around it.
        index = 0; best_near = INFINITY
        for j in range(G):
            magnitude = finite_or_inf(fabs(f[j]))
            left = finite_or_inf(fabs(f[j-1])) if j > 0 else INFINITY
            right = finite_or_inf(fabs(f[j+1])) if j < G-1 else INFINITY
            if magnitude <= left and magnitude < right and isfinite(magnitude):
                near = fabs(grid[j]-current)
                if near < best_near: best_near = near; index = j
        a_ = grid[index-1 if index > 0 else 0]; b_ = grid[index+1 if index < G-1 else G-1]
        for iteration in range(60):
            m1 = a_+(b_-a_)/3; m2 = b_-(b_-a_)/3
            f1 = finite_or_inf(fabs(apply_at(op, k, m1, a0, a1, a2)-d))
            f2 = finite_or_inf(fabs(apply_at(op, k, m2, a0, a1, a2)-d))
            if f1 <= f2: b_ = m2
            else: a_ = m1
        dip_root = .5*(a_+b_)
        crossing_ok = found and fabs(apply_at(op, k, root, a0, a1, a2)-d) <= tolerance
        dip_ok = fabs(apply_at(op, k, dip_root, a0, a1, a2)-d) <= tolerance
        if dip_ok and (not crossing_ok or fabs(dip_root-current) < fabs(root-current)): root = dip_root
        out[row] = root if fabs(apply_at(op, k, root, a0, a1, a2)-d) <= tolerance else NAN_VALUE

def numeric_inverse(int op, int k, desired, args, wide, narrow, fixed):
    """afpo.numeric_inverse for operator code op: per row, the value of argument k
    nearest its current value that makes the operator output desired (NaN if none)."""
    cdef const double[::1] d = np.ascontiguousarray(desired, dtype=float)
    cdef Py_ssize_t n = d.shape[0]
    stacked = np.zeros((3, n))
    for index, values in enumerate(args): stacked[index] = values
    cdef const double[:, ::1] a = stacked
    cdef const double[::1] w = np.ascontiguousarray(wide, dtype=float)
    cdef const double[::1] q = np.ascontiguousarray(narrow, dtype=float)
    cdef const double[::1] c = np.ascontiguousarray(fixed, dtype=float)
    result = np.empty(n)
    cdef double[::1] out = result
    cdef Py_ssize_t G = w.shape[0]+q.shape[0]+c.shape[0]
    cdef double* grid = <double*>malloc(G*sizeof(double))
    cdef double* f = <double*>malloc(G*sizeof(double))
    if grid == NULL or f == NULL:
        free(grid); free(f); raise MemoryError()
    try:
        with nogil: invert_rows(op, k, d, a, w, q, c, grid, f, out)
    finally:
        free(grid); free(f)
    return result
