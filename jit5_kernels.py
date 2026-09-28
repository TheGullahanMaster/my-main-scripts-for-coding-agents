"""Hand-written Triton kernels for jit5 models (Hyena long convolution).

Each entry point has a PyTorch fallback, used on CPU or when Triton cannot
build a loadable kernel (``kernels_available``, shared with linegen_kernels,
which also points Triton at a ptxas the installed driver accepts).
The ops are registered with torch.library so torch.compile treats them as
single opaque calls instead of splitting the graph around them.
"""
import torch

try:
    import triton
    import triton.language as tl
    from linegen_kernels import kernels_available
    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only / Triton-less installs
    triton = tl = None
    HAS_TRITON = False

    def kernels_available(tensor):
        return False


def _precision():
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


# ====================================================== Hyena 2D long conv
# y[b, c, oy, ox] = sum_{iy, ix} v[b, c, iy, ix] * f[c, oy - iy + H - 1, ox - ix + W - 1]
# with a centred (2H-1) x (2W-1) filter: a "same" linear convolution in which
# every output sees every input. Per channel this is a Toeplitz matrix-vector
# product over all N = H*W tokens; the kernels gather Toeplitz tiles from the
# filter on the fly and multiply them on tensor cores with the batch as rows.
if HAS_TRITON:

    @triton.jit
    def _toeplitz_conv_kernel(X, F, Y, B, C, H, W,
                              BLOCK_B: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_I: tl.constexpr,
                              PRECISION: tl.constexpr):
        c = tl.program_id(0)
        n = H * W
        fw = 2 * W - 1
        o = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
        bb = tl.program_id(2) * BLOCK_B + tl.arange(0, BLOCK_B)
        omask, bmask = o < n, bb < B
        oy, ox = o // W, o % W
        rows = (bb[:, None] * C + c) * n
        acc = tl.zeros([BLOCK_B, BLOCK_O], dtype=tl.float32)
        for i0 in range(0, n, BLOCK_I):
            i = i0 + tl.arange(0, BLOCK_I)
            imask = i < n
            iy, ix = i // W, i % W
            x = tl.load(X + rows + i[None, :], mask=bmask[:, None] & imask[None, :], other=0.0).to(tl.float32)
            tap = (oy[None, :] - iy[:, None] + H - 1) * fw + (ox[None, :] - ix[:, None] + W - 1)
            t = tl.load(F + c * (2 * H - 1) * fw + tap, mask=imask[:, None] & omask[None, :], other=0.0)
            acc += tl.dot(x, t.to(tl.float32), input_precision=PRECISION)
        tl.store(Y + rows + o[None, :], acc, mask=bmask[:, None] & omask[None, :])

    @triton.jit
    def _toeplitz_filter_grad_kernel(G, X, DF, B, C, H, W,
                                     W_PAD: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        # One program per (channel, vertical offset sy). Rows oy of the output
        # gradient meet input rows oy - sy; M[ox, ix] sums g[b, oy, ox] *
        # x[b, oy - sy, ix] over (b, oy), and each filter tap (sy, sx) is the
        # sum of M along its diagonal ox - ix = sx. Deterministic, no atomics.
        c = tl.program_id(0)
        dy = tl.program_id(1)
        sy = dy - (H - 1)
        n = H * W
        fw = 2 * W - 1
        col = tl.arange(0, W_PAD)
        cmask = col < W
        m = tl.zeros([W_PAD, W_PAD], dtype=tl.float32)
        for k0 in range(0, B * H, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            b, oy = k // H, k % H
            iy = oy - sy
            valid = (b < B) & (iy >= 0) & (iy < H)
            base = (b * C + c) * n
            g = tl.load(G + (base + oy * W)[:, None] + col[None, :],
                        mask=valid[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            x = tl.load(X + (base + iy * W)[:, None] + col[None, :],
                        mask=valid[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            m += tl.dot(tl.trans(g), x, input_precision=PRECISION)
        diag = col[:, None] - col[None, :] + W - 1
        for dx in range(0, fw):
            total = tl.sum(tl.sum(tl.where(diag == dx, m, 0.0), axis=1), axis=0)
            tl.store(DF + (c * (2 * H - 1) + dy) * fw + dx, total)


def _conv_launch(x, f):
    B, C, H, W = x.shape
    x = x.contiguous()
    f = f.float().contiguous()
    y = torch.empty(B, C, H, W, device=x.device, dtype=torch.float32)
    block_o = 64 if H * W >= 64 else 16
    grid = (C, triton.cdiv(H * W, block_o), triton.cdiv(B, 16))
    _toeplitz_conv_kernel[grid](x, f, y, B, C, H, W, BLOCK_B=16, BLOCK_O=block_o, BLOCK_I=32,
                                PRECISION=_precision())
    return y


@torch.library.custom_op("jit5::hyena_conv", mutates_args=())
def _hyena_conv_op(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return _conv_launch(v, f).to(v.dtype)


@_hyena_conv_op.register_fake
def _(v, f):
    return v.new_empty(v.shape)  # contiguous, like the real output (inputs may be strided)


@torch.library.custom_op("jit5::hyena_filter_grad", mutates_args=())
def _hyena_filter_grad_op(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, C, H, W = v.shape
    df = torch.empty(C, 2 * H - 1, 2 * W - 1, device=v.device, dtype=torch.float32)
    w_pad = max(16, triton.next_power_of_2(W))
    _toeplitz_filter_grad_kernel[(C, 2 * H - 1)](g.contiguous(), v.contiguous(), df, B, C, H, W,
                                                  W_PAD=w_pad, BLOCK_K=32, PRECISION=_precision())
    return df


@_hyena_filter_grad_op.register_fake
def _(g, v):
    B, C, H, W = v.shape
    return v.new_empty(C, 2 * H - 1, 2 * W - 1, dtype=torch.float32)


def _hyena_conv_setup(ctx, inputs, output):
    v, f = inputs
    ctx.save_for_backward(v, f)


def _hyena_conv_backward(ctx, grad):
    v, f = ctx.saved_tensors
    grad_v = grad_f = None
    if ctx.needs_input_grad[0]:
        # The transposed Toeplitz operator is the same convolution with the
        # filter flipped in both axes.
        grad_v = _hyena_conv_op(grad, f.flip(-1, -2))
    if ctx.needs_input_grad[1]:
        grad_f = _hyena_filter_grad_op(grad, v).to(f.dtype)
    return grad_v, grad_f


_hyena_conv_op.register_autograd(_hyena_conv_backward, setup_context=_hyena_conv_setup)

# The direct form costs O(N^2) per channel; cuFFT's O(N log N) wins on large
# grids. Measured on an RTX 3090 (B=16, C=64, fwd+bwd): direct is 1.3x faster
# at 8x8-24x24, 4.3x at 32x32, and still 1.3x at 64x64 tokens.
HYENA_DIRECT_MAX_TOKENS = 4096


@torch.compiler.assume_constant_result
def _triton_ok(device_index: int) -> bool:
    # Evaluated once while tracing, so torch.compile does not step into the
    # probe launch inside kernels_available.
    return kernels_available(torch.empty(1, device=torch.device('cuda', device_index)))


def hyena_direct_available(v):
    H, W = v.shape[-2:]
    return v.is_cuda and H * W <= HYENA_DIRECT_MAX_TOKENS and _triton_ok(v.device.index or 0)


def hyena_conv(v, f):
    """Centred "same" 2D linear convolution of v (B, C, H, W) with f (C, 2H-1, 2W-1).

    Matches Hyena2D.conv_fft exactly, including its scale: that path applies
    norm='ortho' to all three transforms over the (3H-2) x (3W-2) padded grid,
    which divides the true convolution by sqrt((3H-2) * (3W-2)).
    Triton direct form; returns v's dtype. Callers check
    ``hyena_direct_available`` first and use their FFT path otherwise.
    """
    H, W = v.shape[-2:]
    return _hyena_conv_op(v, f * ((3 * H - 2) * (3 * W - 2)) ** -0.5)


# =============================================== FCDM block elementwise ops
# Every tl.sum below reduces a tl.where(mask, value, 0.0), never a loaded
# value directly: with Triton 3.0 and a CUDA 11.7 ptxas (the older-driver
# workaround), tl.sum over a raw tl.load inside a loop miscompiled for some
# tile shapes (16x64, 32x32, 64x16...) while the tl.where form stayed exact.
# Channels-last (B, H, W, C) activations are rows of C contiguous values; the
# grid's second axis tiles rows within one batch item so per-batch reductions
# (adaLN shift/scale gradients, GRN norms) stay in that program.
if HAS_TRITON:

    @triton.jit
    def _ln_modulate_fwd_kernel(X, SHIFT, SCALE, Y, MEAN, RSTD, HW, C, eps, MOD_STRIDE,
                                BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        b = tl.program_id(0)
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, BLOCK_C)
        rmask, cmask = r < HW, cols < C
        mask = rmask[:, None] & cmask[None, :]
        offs = (b * HW + r)[:, None] * C + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / C
        xc = tl.where(mask, x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(tl.where(mask, xc * xc, 0.0), axis=1) / C + eps)
        scale = tl.load(SCALE + b * MOD_STRIDE + cols, mask=cmask, other=0.0).to(tl.float32)
        shift = tl.load(SHIFT + b * MOD_STRIDE + cols, mask=cmask, other=0.0).to(tl.float32)
        y = xc * rstd[:, None] * (1.0 + scale[None, :]) + shift[None, :]
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(MEAN + b * HW + r, mean, mask=rmask)
        tl.store(RSTD + b * HW + r, rstd, mask=rmask)

    @triton.jit
    def _ln_modulate_bwd_kernel(DY, X, SCALE, MEAN, RSTD, DX, DSCALE_P, DSHIFT_P, HW, C, NBLK, MOD_STRIDE,
                                BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        b = tl.program_id(0)
        blk = tl.program_id(1)
        r = blk * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, BLOCK_C)
        rmask, cmask = r < HW, cols < C
        mask = rmask[:, None] & cmask[None, :]
        offs = (b * HW + r)[:, None] * C + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(MEAN + b * HW + r, mask=rmask, other=0.0)
        rstd = tl.load(RSTD + b * HW + r, mask=rmask, other=0.0)
        scale = tl.load(SCALE + b * MOD_STRIDE + cols, mask=cmask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        dxhat = dy * (1.0 + scale[None, :])
        c1 = tl.sum(tl.where(mask, dxhat * xhat, 0.0), axis=1) / C
        c2 = tl.sum(tl.where(mask, dxhat, 0.0), axis=1) / C
        dx = (dxhat - xhat * c1[:, None] - c2[:, None]) * rstd[:, None]
        tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)
        part = (b * NBLK + blk) * C + cols
        tl.store(DSCALE_P + part, tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0), mask=cmask)
        tl.store(DSHIFT_P + part, tl.sum(tl.where(mask, dy, 0.0), axis=0), mask=cmask)

    @triton.jit
    def _gelu(a):
        return 0.5 * a * (1.0 + tl.math.erf(a * 0.7071067811865476))

    @triton.jit
    def _gelu_sumsq_kernel(A, G, SSQ_P, HW, E, NBLK, SPAN: tl.constexpr, BLOCK_R: tl.constexpr,
                           BLOCK_E: tl.constexpr):
        b = tl.program_id(0)
        blk = tl.program_id(1)
        e = tl.program_id(2) * BLOCK_E + tl.arange(0, BLOCK_E)
        emask = e < E
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        for r0 in range(0, SPAN, BLOCK_R):
            r = blk * SPAN + r0 + tl.arange(0, BLOCK_R)
            mask = (r < HW)[:, None] & emask[None, :]
            offs = (b * HW + r)[:, None] * E + e[None, :]
            g = _gelu(tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)).to(G.dtype.element_ty)
            tl.store(G + offs, g, mask=mask)
            g = tl.where(mask, g.to(tl.float32), 0.0)  # norms of the stored (rounded) values
            acc += tl.sum(tl.where(mask, g * g, 0.0), axis=0)
        tl.store(SSQ_P + (b * NBLK + blk) * E + e, acc, mask=emask)

    @triton.jit
    def _gelu_grn_bwd_reduce_kernel(A, DY, S_P, T_P, HW, E, NBLK, SPAN: tl.constexpr, BLOCK_R: tl.constexpr,
                                    BLOCK_E: tl.constexpr):
        b = tl.program_id(0)
        blk = tl.program_id(1)
        e = tl.program_id(2) * BLOCK_E + tl.arange(0, BLOCK_E)
        emask = e < E
        s_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        t_acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        for r0 in range(0, SPAN, BLOCK_R):
            r = blk * SPAN + r0 + tl.arange(0, BLOCK_R)
            mask = (r < HW)[:, None] & emask[None, :]
            offs = (b * HW + r)[:, None] * E + e[None, :]
            g = _gelu(tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)).to(A.dtype.element_ty).to(tl.float32)
            dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
            s_acc += tl.sum(tl.where(mask, dy * g, 0.0), axis=0)
            t_acc += tl.sum(tl.where(mask, dy, 0.0), axis=0)
        part = (b * NBLK + blk) * E + e
        tl.store(S_P + part, s_acc, mask=emask)
        tl.store(T_P + part, t_acc, mask=emask)

    @triton.jit
    def _gelu_grn_bwd_elem_kernel(A, DY, COEF, K, DA, HW, E, BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr):
        b = tl.program_id(0)
        e = tl.program_id(2) * BLOCK_E + tl.arange(0, BLOCK_E)
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        emask = e < E
        mask = (r < HW)[:, None] & emask[None, :]
        offs = (b * HW + r)[:, None] * E + e[None, :]
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        g = _gelu(a).to(A.dtype.element_ty).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        coef = tl.load(COEF + b * E + e, mask=emask, other=0.0).to(tl.float32)
        k = tl.load(K + b * E + e, mask=emask, other=0.0)
        cdf = 0.5 * (1.0 + tl.math.erf(a * 0.7071067811865476))
        pdf = 0.3989422804014327 * tl.exp(-0.5 * a * a)
        da = (dy * coef[None, :] + k[None, :] * g) * (cdf + a * pdf)
        tl.store(DA + offs, da.to(DA.dtype.element_ty), mask=mask)

    @triton.jit
    def _grn_coef_kernel(SSQ_P, GAMMA, NORM, MEAN, REL, COEF, E, NBLK, BLOCK_E: tl.constexpr):
        # Per batch item: n = sqrt(sum ssq), M = mean_E n + 1e-6, r = n / M, and
        # coef = 1 + gamma * r, rounded at the same points as the PyTorch code
        # (relative, the product and coef are all in the activation dtype).
        b = tl.program_id(0)
        e = tl.arange(0, BLOCK_E)
        emask = e < E
        ssq = tl.zeros([BLOCK_E], dtype=tl.float32)
        for blk in range(0, NBLK):
            ssq += tl.load(SSQ_P + (b * NBLK + blk) * E + e, mask=emask, other=0.0)
        norm = tl.sqrt(ssq)
        mean = tl.sum(tl.where(emask, norm, 0.0), axis=0) / E + 1e-6
        rel = (norm / mean).to(REL.dtype.element_ty)
        gamma = tl.load(GAMMA + e, mask=emask, other=0.0).to(REL.dtype.element_ty).to(tl.float32)
        prod = (gamma * rel.to(tl.float32)).to(REL.dtype.element_ty).to(tl.float32)
        tl.store(NORM + b * E + e, norm, mask=emask)
        tl.store(MEAN + b, mean)
        tl.store(REL + b * E + e, rel, mask=emask)
        tl.store(COEF + b * E + e, (1.0 + prod).to(COEF.dtype.element_ty), mask=emask)

    @triton.jit
    def _grn_grad_kernel(S_P, T_P, GAMMA, NORM, MEAN, REL, K, DG_P, DB_P, E, NBLK, BLOCK_E: tl.constexpr):
        # r = n / (mean_E n + eps): dL/dn = q / M - sum_E(q n) / (E M^2) with
        # q = gamma * sum(dy * g); the elementwise pass then adds (dL/dn / n) * g.
        b = tl.program_id(0)
        e = tl.arange(0, BLOCK_E)
        emask = e < E
        s = tl.zeros([BLOCK_E], dtype=tl.float32)
        t = tl.zeros([BLOCK_E], dtype=tl.float32)
        for blk in range(0, NBLK):
            s += tl.load(S_P + (b * NBLK + blk) * E + e, mask=emask, other=0.0)
            t += tl.load(T_P + (b * NBLK + blk) * E + e, mask=emask, other=0.0)
        gamma = tl.load(GAMMA + e, mask=emask, other=0.0).to(tl.float32)
        norm = tl.load(NORM + b * E + e, mask=emask, other=0.0)
        rel = tl.load(REL + b * E + e, mask=emask, other=0.0).to(tl.float32)
        mean = tl.load(MEAN + b)
        q = gamma * s
        dnorm = q / mean - tl.sum(tl.where(emask, q * norm, 0.0), axis=0) / (E * mean * mean)
        k = tl.where(norm > 0, dnorm / tl.maximum(norm, 1e-30), 0.0)
        tl.store(K + b * E + e, k, mask=emask)
        tl.store(DG_P + b * E + e, s * rel, mask=emask)
        tl.store(DB_P + b * E + e, t, mask=emask)


_ROWS = 32


def _rows_grid(x):
    B, H, W, _ = x.shape
    return B, H * W, triton.cdiv(H * W, _ROWS)


class _LNModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, eps, out_dtype):
        B, HW, nblk = _rows_grid(x)
        C = x.shape[-1]
        y = torch.empty(x.shape, device=x.device, dtype=out_dtype)
        mean = torch.empty(B, HW, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        if shift.stride(-1) != 1 or scale.stride(-1) != 1 or shift.stride(0) != scale.stride(0):
            shift, scale = shift.contiguous(), scale.contiguous()
        # shift/scale are usually column slices of one adaLN output: read them
        # in place via their row stride instead of copying.
        _ln_modulate_fwd_kernel[(B, nblk)](x, shift, scale, y, mean, rstd, HW, C, eps, scale.stride(0),
                                           BLOCK_R=_ROWS, BLOCK_C=triton.next_power_of_2(C))
        ctx.save_for_backward(x, scale, mean, rstd)
        ctx.dtypes = shift.dtype, scale.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale, mean, rstd = ctx.saved_tensors
        B, HW, nblk = _rows_grid(x)
        C = x.shape[-1]
        dx = torch.empty_like(x)
        dscale = torch.empty(B, nblk, C, device=x.device, dtype=torch.float32)
        dshift = torch.empty_like(dscale)
        _ln_modulate_bwd_kernel[(B, nblk)](dy.contiguous(), x, scale, mean, rstd, dx, dscale, dshift,
                                           HW, C, nblk, scale.stride(0),
                                           BLOCK_R=_ROWS, BLOCK_C=triton.next_power_of_2(C))
        shift_dtype, scale_dtype = ctx.dtypes
        return dx, dshift.sum(1).to(shift_dtype), dscale.sum(1).to(scale_dtype), None, None


_SPAN = 512  # rows summed per reduction program


class _GeluGRN(torch.autograd.Function):
    """y = g * (1 + gamma * r) + beta with g = gelu(a), r = |g|_HW / (mean_E |g|_HW + 1e-6).

    Matches FCDMGRN.forward_nhwc(gelu(a)), including where values are rounded
    to the activation dtype. Forward: GELU + norm pass, one per-batch finalize,
    one addcmul. Backward: one reduction, one per-batch finalize, one
    elementwise pass. Few launches matter: eager FCDM is launch-bound.
    """
    @staticmethod
    def forward(ctx, a, gamma, beta):
        B, H, W, E = a.shape
        HW = H * W
        nblk = triton.cdiv(HW, _SPAN)
        block_e = min(128, triton.next_power_of_2(E))
        g = torch.empty_like(a)
        ssq = torch.empty(B, nblk, E, device=a.device, dtype=torch.float32)
        _gelu_sumsq_kernel[(B, nblk, triton.cdiv(E, block_e))](a, g, ssq, HW, E, nblk, SPAN=_SPAN,
                                                               BLOCK_R=_ROWS, BLOCK_E=block_e)
        norm = torch.empty(B, E, device=a.device, dtype=torch.float32)
        mean = torch.empty(B, device=a.device, dtype=torch.float32)
        relative = torch.empty(B, E, device=a.device, dtype=a.dtype)
        coef = torch.empty_like(relative)
        _grn_coef_kernel[(B,)](ssq, gamma, norm, mean, relative, coef, E, nblk,
                               BLOCK_E=triton.next_power_of_2(E))
        y = torch.addcmul(beta.view(1, 1, 1, -1).to(a.dtype), g, coef.view(B, 1, 1, E))
        ctx.save_for_backward(a, gamma, norm, mean, relative, coef)
        return y

    @staticmethod
    def backward(ctx, dy):
        a, gamma, norm, mean, relative, coef = ctx.saved_tensors
        B, H, W, E = a.shape
        HW = H * W
        nblk = triton.cdiv(HW, _SPAN)
        block_e = min(128, triton.next_power_of_2(E))
        dy = dy.contiguous()
        s = torch.empty(B, nblk, E, device=a.device, dtype=torch.float32)
        t = torch.empty_like(s)
        _gelu_grn_bwd_reduce_kernel[(B, nblk, triton.cdiv(E, block_e))](a, dy, s, t, HW, E, nblk, SPAN=_SPAN,
                                                                        BLOCK_R=_ROWS, BLOCK_E=block_e)
        k = torch.empty(B, E, device=a.device, dtype=torch.float32)
        dg = torch.empty_like(k)
        db = torch.empty_like(k)
        _grn_grad_kernel[(B,)](s, t, gamma, norm, mean, relative, k, dg, db, E, nblk,
                               BLOCK_E=triton.next_power_of_2(E))
        da = torch.empty_like(a)
        _gelu_grn_bwd_elem_kernel[(B, triton.cdiv(HW, _ROWS), triton.cdiv(E, block_e))](
            a, dy, coef, k, da, HW, E, BLOCK_R=_ROWS, BLOCK_E=block_e)
        return da, dg.sum(0).view_as(gamma).to(gamma.dtype), db.sum(0).view_as(gamma).to(gamma.dtype)


def fcdm_kernels_available(x):
    return x.is_cuda and not torch.compiler.is_compiling() and _triton_ok(x.device.index or 0)


def fcdm_ln_modulate(h, shift, scale, eps):
    """LayerNorm over C of channels-last h (B, H, W, C), then h * (1 + scale) + shift.

    shift/scale are (B, C). Output dtype follows autocast (the dtype the next
    linear layer would cast to), else h's dtype.
    """
    out_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else h.dtype
    return _LNModulate.apply(h.contiguous(), shift, scale, eps, out_dtype)


def fcdm_gelu_grn(a, gamma, beta):
    """FCDMGRN.forward_nhwc(gelu(a)) for channels-last a (B, H, W, E)."""
    return _GeluGRN.apply(a.contiguous(), gamma, beta)
