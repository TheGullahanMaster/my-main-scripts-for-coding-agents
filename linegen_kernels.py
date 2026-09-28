"""Fused recurrent kernels for linegen's IndRNN, JANET, sLSTM and mLSTM cores.

IndRNN, JANET and sLSTM use Triton kernels that keep the whole time loop on
the GPU and run every timestep inside one launch.  IndRNN programs own a
block of (batch row, channel) pairs; sLSTM programs own one (batch row, head)
and stream that head's recurrent matrices from L2; JANET programs own a tile
of 16 batch rows and run each step's dense recurrent matmul through tl.dot.
Each has a matching reverse-time backward kernel; weight gradients are then
one large GEMM.

mLSTM does not need a hand-written kernel: with one gate per head it has an
exact chunkwise-parallel form made of large matmuls, implemented here in
PyTorch and differentiated by autograd.

Every entry point falls back to the caller's reference loop when Triton or a
CUDA tensor is unavailable (see ``kernels_available``).
"""
import ctypes
import glob
import math
import os
import re
import subprocess

import torch
import torch.nn.functional as F


def _configure_ptxas():
    """Use a system ptxas the driver can load when it predates CUDA 12.

    Triton ships a CUDA 12 ptxas; a CUDA 11 driver rejects its binaries with
    "device kernel image is invalid".  Triton then needs a ptxas no newer than
    the driver, and derives the PTX version from it.
    """
    if os.environ.get("TRITON_PTXAS_PATH"):
        return
    try:
        version = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
        driver = version.value // 1000, (version.value % 1000) // 10
    except Exception:
        return
    if driver >= (12, 0):
        return
    candidates = []
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if home:
            candidates.append(os.path.join(home, "bin", "ptxas"))
    candidates += glob.glob("/usr/local/cuda-*/bin/ptxas")
    best = None
    for path in dict.fromkeys(candidates):
        if not os.path.isfile(path):
            continue
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            continue
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            found = int(match.group(1)), int(match.group(2))
            if found <= driver and (best is None or found > best[0]):
                best = (found, path)
    if best is not None:
        os.environ["TRITON_PTXAS_PATH"] = best[1]


_configure_ptxas()

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only / Triton-less installs
    triton = None
    tl = None
    HAS_TRITON = False

_KERNELS_WORK = None


def kernels_available(tensor: torch.Tensor) -> bool:
    """Whether the fused Triton path can run for this tensor.

    The first CUDA call runs one tiny kernel; if the toolchain cannot produce
    a loadable binary, every caller falls back to its reference loop.
    """
    global _KERNELS_WORK
    if not (HAS_TRITON and tensor.is_cuda):
        return False
    if _KERNELS_WORK is None:
        try:
            p = torch.zeros(1, 1, 1, device=tensor.device)
            hf = torch.zeros(1, 2, 1, device=tensor.device)
            _indrnn_fwd_kernel[(1, 1)](p, p.view(1), p.view(1), hf, 1, 1, BLOCK=1, ACT=0)
            torch.cuda.synchronize(tensor.device)
            _KERNELS_WORK = True
        except Exception as exc:
            print(f"[linegen_kernels] Triton kernels unavailable ({exc}); using PyTorch loops")
            _KERNELS_WORK = False
    return _KERNELS_WORK


def _contig(t: torch.Tensor) -> torch.Tensor:
    """Contiguous view of a tensor inside an autograd.Function backward.

    Under torch.compile (seen on PyTorch 2.4.1), a ``.contiguous()`` call in a
    traced backward can collide with the FX names of tensors the forward saved
    via ``.contiguous()``, shifting the kernel arguments by one and silently
    producing wrong gradients.  A no-op return or ``clone`` emits no such node.
    _IRNNScan, _LRUScan and _RRUScan keep ``.contiguous()``: with this helper
    their compiled backward fails autograd's in-place version check.
    """
    return t if t.is_contiguous() else t.clone(memory_format=torch.contiguous_format)


def _row_block(width_p2: int) -> int:
    """Rows per recurrent-matrix tile, keeping a tile near 8K fp32 values."""
    return max(1, min(width_p2, 8192 // width_p2))


if HAS_TRITON:

    @triton.jit
    def _tanh(x):
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _logsigmoid(x):
        return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))

    # ------------------------------------------------------------------ IndRNN
    @triton.jit
    def _indrnn_fwd_kernel(P, U, BIAS, HF, T, H, BLOCK: tl.constexpr, ACT: tl.constexpr = 0):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < H
        u = tl.load(U + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
        # HF holds h_0 at index 0 and h_t at index t.
        h = tl.load(HF + b * (T + 1) * H + cols, mask=mask, other=0.0)
        for t in range(T):
            p = tl.load(P + (b * T + t) * H + cols, mask=mask, other=0.0).to(tl.float32)
            if ACT == 1:
                h = tl.maximum(p + u * h + bias, 0.0)
            else:
                h = _tanh(p + u * h + bias)
            tl.store(HF + (b * (T + 1) + t + 1) * H + cols, h, mask=mask)

    @triton.jit
    def _indrnn_bwd_kernel(DY, HF, U, DP, DU, DH0, T, H, BLOCK: tl.constexpr, ACT: tl.constexpr = 0):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < H
        u = tl.load(U + cols, mask=mask, other=0.0).to(tl.float32)
        dh = tl.zeros([BLOCK], dtype=tl.float32)
        du = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            h = tl.load(HF + (b * (T + 1) + t + 1) * H + cols, mask=mask, other=0.0)
            h_prev = tl.load(HF + (b * (T + 1) + t) * H + cols, mask=mask, other=0.0)
            g = tl.load(DY + (b * T + t) * H + cols, mask=mask, other=0.0).to(tl.float32) + dh
            if ACT == 1:
                dz = tl.where(h > 0.0, g, 0.0)
            else:
                dz = g * (1.0 - h * h)
            tl.store(DP + (b * T + t) * H + cols, dz, mask=mask)
            du += dz * h_prev
            dh = dz * u
        tl.store(DU + b * H + cols, du, mask=mask)
        tl.store(DH0 + b * H + cols, dh, mask=mask)

    # ------------------------------------------------------------------- JANET
    # One program owns a tile of BB batch rows, so each recurrent-weight tile
    # is read once per BB rows and the step's matmul runs through tl.dot.
    @triton.jit
    def _janet_fwd_kernel(PX, UT, CF, A, B, T, H, beta,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            for n0 in range(0, H, BN):
                rows = n0 + tl.arange(0, BN)
                rmask = rows < H
                acc_s = tl.zeros([BB, BN], dtype=tl.float32)
                acc_c = tl.zeros([BB, BN], dtype=tl.float32)
                for k0 in range(0, H, BK):
                    ks = k0 + tl.arange(0, BK)
                    kmask = ks < H
                    c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + ks[None, :],
                                     mask=bmask[:, None] & kmask[None, :], other=0.0)
                    wmask = kmask[:, None] & rmask[None, :]
                    # UT is (H, 2H): UT[k, r] = U[r, k], forget rows then candidate rows.
                    w_f = tl.load(UT + ks[:, None] * (2 * H) + rows[None, :], mask=wmask, other=0.0)
                    w_c = tl.load(UT + ks[:, None] * (2 * H) + H + rows[None, :], mask=wmask, other=0.0)
                    acc_s = tl.dot(c_prev, w_f, acc_s, input_precision=PREC)
                    acc_c = tl.dot(c_prev, w_c, acc_c, input_precision=PREC)
                tile = bmask[:, None] & rmask[None, :]
                px = PX + (brow[:, None] * T + t) * (2 * H) + rows[None, :]
                s = acc_s + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                cand_pre = acc_c + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                f = tl.sigmoid(s)
                keep_cand = 1.0 - tl.sigmoid(s - beta)
                cand = _tanh(cand_pre)
                c_rows = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + rows[None, :], mask=tile, other=0.0)
                tl.store(CF + (brow[:, None] * (T + 1) + t + 1) * H + rows[None, :],
                         f * c_rows + keep_cand * cand, mask=tile)
                a = A + (brow[:, None] * T + t) * (2 * H) + rows[None, :]
                tl.store(a, s, mask=tile)
                tl.store(a + H, cand_pre, mask=tile)
            # Every column of c_t must be written before the next step reads it.
            tl.debug_barrier()

    @triton.jit
    def _janet_bwd_kernel(DY, CF, A, U, DA, CARRY, DIRECT, B, T, H, beta,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for i in range(T):
            t = T - 1 - i
            # Pass 1: gate pre-activation gradients for this step.
            for n0 in range(0, H, BN):
                rows = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (rows < H)[None, :]
                dc = tl.load(DY + (brow[:, None] * T + t) * H + rows[None, :], mask=tile, other=0.0).to(tl.float32) \
                    + tl.load(CARRY + brow[:, None] * H + rows[None, :], mask=tile, other=0.0)
                a = A + (brow[:, None] * T + t) * (2 * H) + rows[None, :]
                s = tl.load(a, mask=tile, other=0.0)
                cand_pre = tl.load(a + H, mask=tile, other=0.0)
                c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + rows[None, :], mask=tile, other=0.0)
                f = tl.sigmoid(s)
                shifted = tl.sigmoid(s - beta)
                cand = _tanh(cand_pre)
                ds = dc * c_prev * f * (1.0 - f) - dc * cand * shifted * (1.0 - shifted)
                dcand = dc * (1.0 - shifted) * (1.0 - cand * cand)
                da = DA + (brow[:, None] * T + t) * (2 * H) + rows[None, :]
                tl.store(da, ds, mask=tile)
                tl.store(da + H, dcand, mask=tile)
                tl.store(DIRECT + brow[:, None] * H + rows[None, :], dc * f, mask=tile)
            tl.debug_barrier()
            # Pass 2: carry to c_{t-1} = direct path + [ds, dcand] @ [U_f; U_c].
            for n0 in range(0, H, BN):
                cols = n0 + tl.arange(0, BN)
                cmask = cols < H
                acc = tl.zeros([BB, BN], dtype=tl.float32)
                for k0 in range(0, 2 * H, BK):
                    rs = k0 + tl.arange(0, BK)
                    rsmask = rs < 2 * H
                    d = tl.load(DA + (brow[:, None] * T + t) * (2 * H) + rs[None, :],
                                mask=bmask[:, None] & rsmask[None, :], other=0.0)
                    w = tl.load(U + rs[:, None] * H + cols[None, :],
                                mask=rsmask[:, None] & cmask[None, :], other=0.0)
                    acc = tl.dot(d, w, acc, input_precision=PREC)
                tile = bmask[:, None] & cmask[None, :]
                direct = tl.load(DIRECT + brow[:, None] * H + cols[None, :], mask=tile, other=0.0)
                tl.store(CARRY + brow[:, None] * H + cols[None, :], acc + direct, mask=tile)
            tl.debug_barrier()

    # ------------------------------------------------------------------- sLSTM
    @triton.jit
    def _slstm_fwd_kernel(G, R, HF, CF, NF, MF, A, T, D, NH, DH,
                          EXP_FORGET: tl.constexpr, HP2: tl.constexpr, BR: tl.constexpr):
        b = tl.program_id(0)
        head = tl.program_id(1)
        kcols = tl.arange(0, HP2)
        kmask = kcols < DH
        base = head * DH
        for t in range(T):
            state = (b * (T + 1) + t) * D + base
            nxt = (b * (T + 1) + t + 1) * D + base
            h_prev = tl.load(HF + state + kcols, mask=kmask, other=0.0)
            for r0 in range(0, DH, BR):
                rows = r0 + tl.arange(0, BR)
                rmask = rows < DH
                wmask = rmask[:, None] & kmask[None, :]
                gin = G + ((b * T + t) * 4) * D + base
                ptr_a = A + ((b * T + t) * 4) * D + base
                # Gate order: z, i, f, o.
                r_z = tl.load(R + ((0 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_i = tl.load(R + ((1 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_f = tl.load(R + ((2 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_o = tl.load(R + ((3 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                pz = tl.sum(r_z * h_prev[None, :], axis=1) + tl.load(gin + 0 * D + rows, mask=rmask, other=0.0).to(tl.float32)
                pi = tl.sum(r_i * h_prev[None, :], axis=1) + tl.load(gin + 1 * D + rows, mask=rmask, other=0.0).to(tl.float32)
                pf = tl.sum(r_f * h_prev[None, :], axis=1) + tl.load(gin + 2 * D + rows, mask=rmask, other=0.0).to(tl.float32)
                po = tl.sum(r_o * h_prev[None, :], axis=1) + tl.load(gin + 3 * D + rows, mask=rmask, other=0.0).to(tl.float32)
                tl.store(ptr_a + 0 * D + rows, pz, mask=rmask)
                tl.store(ptr_a + 1 * D + rows, pi, mask=rmask)
                tl.store(ptr_a + 2 * D + rows, pf, mask=rmask)
                tl.store(ptr_a + 3 * D + rows, po, mask=rmask)
                z = _tanh(pz)
                if EXP_FORGET:
                    logf = pf
                else:
                    logf = _logsigmoid(pf)
                o = tl.sigmoid(po)
                m_prev = tl.load(MF + state + rows, mask=rmask, other=0.0)
                c_prev = tl.load(CF + state + rows, mask=rmask, other=0.0)
                n_prev = tl.load(NF + state + rows, mask=rmask, other=0.0)
                m_new = tl.maximum(logf + m_prev, pi)
                i_hat = tl.exp(pi - m_new)
                f_hat = tl.exp(logf + m_prev - m_new)
                c = f_hat * c_prev + i_hat * z
                n = f_hat * n_prev + i_hat
                h = o * c / tl.maximum(n, 1e-12)
                tl.store(CF + nxt + rows, c, mask=rmask)
                tl.store(NF + nxt + rows, n, mask=rmask)
                tl.store(MF + nxt + rows, m_new, mask=rmask)
                tl.store(HF + nxt + rows, h, mask=rmask)
            tl.debug_barrier()

    @triton.jit
    def _slstm_bwd_kernel(DY, R, HF, CF, NF, MF, A, DA, DHC, DCC, DNC, T, D, NH, DH,
                          EXP_FORGET: tl.constexpr, HP2: tl.constexpr, BR: tl.constexpr):
        b = tl.program_id(0)
        head = tl.program_id(1)
        kcols = tl.arange(0, HP2)
        kmask = kcols < DH
        base = head * DH
        for i in range(T):
            t = T - 1 - i
            state = (b * (T + 1) + t) * D + base
            nxt = (b * (T + 1) + t + 1) * D + base
            acc = tl.zeros([HP2], dtype=tl.float32)
            for r0 in range(0, DH, BR):
                rows = r0 + tl.arange(0, BR)
                rmask = rows < DH
                pa = A + ((b * T + t) * 4) * D + base
                pz = tl.load(pa + 0 * D + rows, mask=rmask, other=0.0)
                pi = tl.load(pa + 1 * D + rows, mask=rmask, other=0.0)
                pf = tl.load(pa + 2 * D + rows, mask=rmask, other=0.0)
                po = tl.load(pa + 3 * D + rows, mask=rmask, other=0.0)
                z = _tanh(pz)
                if EXP_FORGET:
                    logf = pf
                    dlogf_dpf = tl.full([BR], 1.0, tl.float32)
                else:
                    logf = _logsigmoid(pf)
                    dlogf_dpf = 1.0 - tl.sigmoid(pf)
                o = tl.sigmoid(po)
                m_prev = tl.load(MF + state + rows, mask=rmask, other=0.0)
                m_new = tl.load(MF + nxt + rows, mask=rmask, other=0.0)
                c_prev = tl.load(CF + state + rows, mask=rmask, other=0.0)
                n_prev = tl.load(NF + state + rows, mask=rmask, other=0.0)
                c = tl.load(CF + nxt + rows, mask=rmask, other=0.0)
                n = tl.maximum(tl.load(NF + nxt + rows, mask=rmask, other=1.0), 1e-12)
                i_hat = tl.exp(pi - m_new)
                f_hat = tl.exp(logf + m_prev - m_new)
                dh = tl.load(DY + (b * T + t) * D + base + rows, mask=rmask, other=0.0).to(tl.float32) \
                    + tl.load(DHC + b * D + base + rows, mask=rmask, other=0.0)
                dpo = dh * (c / n) * o * (1.0 - o)
                dc = dh * o / n + tl.load(DCC + b * D + base + rows, mask=rmask, other=0.0)
                dn = -dh * o * c / (n * n) + tl.load(DNC + b * D + base + rows, mask=rmask, other=0.0)
                dpz = dc * i_hat * (1.0 - z * z)
                dpi = (dc * z + dn) * i_hat
                dpf = (dc * c_prev + dn * n_prev) * f_hat * dlogf_dpf
                dpz = tl.where(rmask, dpz, 0.0)
                dpi = tl.where(rmask, dpi, 0.0)
                dpf = tl.where(rmask, dpf, 0.0)
                dpo = tl.where(rmask, dpo, 0.0)
                pd = DA + ((b * T + t) * 4) * D + base
                tl.store(pd + 0 * D + rows, dpz, mask=rmask)
                tl.store(pd + 1 * D + rows, dpi, mask=rmask)
                tl.store(pd + 2 * D + rows, dpf, mask=rmask)
                tl.store(pd + 3 * D + rows, dpo, mask=rmask)
                # The stabilizer is treated as a constant: h does not depend on it.
                tl.store(DCC + b * D + base + rows, dc * f_hat, mask=rmask)
                tl.store(DNC + b * D + base + rows, dn * f_hat, mask=rmask)
                wmask = rmask[:, None] & kmask[None, :]
                r_z = tl.load(R + ((0 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_i = tl.load(R + ((1 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_f = tl.load(R + ((2 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                r_o = tl.load(R + ((3 * NH + head) * DH + rows[:, None]) * DH + kcols[None, :], mask=wmask, other=0.0)
                acc += tl.sum(r_z * dpz[:, None], axis=0) + tl.sum(r_i * dpi[:, None], axis=0) \
                    + tl.sum(r_f * dpf[:, None], axis=0) + tl.sum(r_o * dpo[:, None], axis=0)
            tl.debug_barrier()
            tl.store(DHC + b * D + base + kcols, acc, mask=kmask)
            tl.debug_barrier()


# ============================================================ autograd wrappers
class _IndRNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, preact, u, bias, h0, act):
        preact = preact.contiguous()
        B, T, H = preact.shape
        hf = torch.empty(B, T + 1, H, device=preact.device, dtype=torch.float32)
        hf[:, 0] = h0.float()
        u32 = u.float().contiguous()
        block = min(1024, triton.next_power_of_2(H))
        _indrnn_fwd_kernel[(B, triton.cdiv(H, block))](preact, u32, bias.float().contiguous(), hf, T, H,
                                                       BLOCK=block, ACT=act)
        ctx.save_for_backward(hf, u32)
        ctx.shapes = (preact.dtype, h0.dtype, block, act)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        hf, u32 = ctx.saved_tensors
        p_dtype, h_dtype, block, act = ctx.shapes
        B, T1, H = hf.shape
        T = T1 - 1
        dy = _contig(dy)
        dp = torch.empty(B, T, H, device=hf.device, dtype=torch.float32)
        du = torch.empty(B, H, device=hf.device, dtype=torch.float32)
        dh0 = torch.empty(B, H, device=hf.device, dtype=torch.float32)
        _indrnn_bwd_kernel[(B, triton.cdiv(H, block))](dy, hf, u32, dp, du, dh0, T, H, BLOCK=block, ACT=act)
        return dp.to(p_dtype), du.sum(0), dp.sum((0, 1)), dh0.to(h_dtype), None


def indrnn_scan(preact, u, bias, h0, activation="tanh"):
    """h_t = act(preact_t + u * h_{t-1} + bias) over the whole sequence,
    with ``activation`` "tanh" or "relu"."""
    if bias is None:
        bias = torch.zeros_like(u)
    return _IndRNNScan.apply(preact, u, bias, h0, {"tanh": 0, "relu": 1}[activation])


def _dot_precision() -> str:
    """Follow PyTorch's fp32 matmul setting for the in-kernel tl.dot calls."""
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


def _janet_tiles(H: int):
    hp2 = triton.next_power_of_2(max(H, 16))
    return 16, min(64, hp2), min(64, hp2)


class _JanetScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, u, c0, beta):
        px = px.contiguous()
        B, T, H2 = px.shape
        H = H2 // 2
        u32 = u.float().contiguous()
        cf = torch.empty(B, T + 1, H, device=px.device, dtype=torch.float32)
        cf[:, 0] = c0.float()
        a = torch.empty(B, T, 2 * H, device=px.device, dtype=torch.float32)
        bb, bn, bk = _janet_tiles(H)
        prec = _dot_precision()
        _janet_fwd_kernel[(triton.cdiv(B, bb),)](px, u32.t().contiguous(), cf, a, B, T, H, float(beta),
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4)
        ctx.save_for_backward(cf, a, u32)
        ctx.meta = (px.dtype, c0.dtype, float(beta), prec)
        return cf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        cf, a, u32 = ctx.saved_tensors
        px_dtype, c_dtype, beta, prec = ctx.meta
        B, T1, H = cf.shape
        T = T1 - 1
        da = torch.empty(B, T, 2 * H, device=cf.device, dtype=torch.float32)
        carry = torch.zeros(B, H, device=cf.device, dtype=torch.float32)
        direct = torch.empty(B, H, device=cf.device, dtype=torch.float32)
        bb, bn, bk = _janet_tiles(H)
        _janet_bwd_kernel[(triton.cdiv(B, bb),)](_contig(dy), cf, a, u32, da, carry, direct, B, T, H, beta,
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4)
        du = da.reshape(-1, 2 * H).t() @ cf[:, :-1].reshape(-1, H)
        return da.to(px_dtype), du, carry.to(c_dtype), None


def janet_scan(px, u, c0, beta):
    """JANET over a sequence.

    ``px``: (B, T, 2H) input projections with all biases, forget part first;
    ``u``: (2H, H) stacked [U_f; U_c] recurrent weights.  Returns c_1..c_T.
    """
    return _JanetScan.apply(px, u, c0, beta)



class _SLSTMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gin, r, h0, c0, n0, m0, exp_forget):
        gin = gin.contiguous()
        B, T, _, D = gin.shape
        _, NH, DH, _ = r.shape
        r32 = r.float().contiguous()
        kw = dict(device=gin.device, dtype=torch.float32)
        hf = torch.empty(B, T + 1, D, **kw); hf[:, 0] = h0.float()
        cf = torch.empty(B, T + 1, D, **kw); cf[:, 0] = c0.float()
        nf = torch.empty(B, T + 1, D, **kw); nf[:, 0] = n0.float()
        mf = torch.empty(B, T + 1, D, **kw); mf[:, 0] = m0.float()
        a = torch.empty(B, T, 4, D, **kw)
        hp2 = triton.next_power_of_2(DH)
        br = _row_block(hp2)
        _slstm_fwd_kernel[(B, NH)](gin, r32, hf, cf, nf, mf, a, T, D, NH, DH,
                                   EXP_FORGET=bool(exp_forget), HP2=hp2, BR=br, num_warps=4)
        ctx.save_for_backward(r32, hf, cf, nf, mf, a)
        ctx.meta = (gin.dtype, bool(exp_forget), hp2, br)
        return hf[:, 1:], cf[:, -1], nf[:, -1], mf[:, -1]

    @staticmethod
    def backward(ctx, dy, dc_last, dn_last, dm_last):
        r32, hf, cf, nf, mf, a = ctx.saved_tensors
        g_dtype, exp_forget, hp2, br = ctx.meta
        B, T1, D = hf.shape
        T = T1 - 1
        _, NH, DH, _ = r32.shape
        kw = dict(device=hf.device, dtype=torch.float32)
        da = torch.empty(B, T, 4, D, **kw)
        dhc = torch.zeros(B, D, **kw)
        dcc = torch.zeros(B, D, **kw) if dc_last is None else _contig(dc_last.float()).clone()
        dnc = torch.zeros(B, D, **kw) if dn_last is None else _contig(dn_last.float()).clone()
        dy = torch.zeros(B, T, D, **kw) if dy is None else _contig(dy)
        _slstm_bwd_kernel[(B, NH)](dy, r32, hf, cf, nf, mf, a, da, dhc, dcc, dnc, T, D, NH, DH,
                                   EXP_FORGET=exp_forget, HP2=hp2, BR=br, num_warps=4)
        h_prev = hf[:, :-1].reshape(B, T, NH, DH)
        dr = torch.einsum("btghr,bthk->ghrk", da.view(B, T, 4, NH, DH), h_prev)
        return da.to(g_dtype), dr, dhc, dcc, dnc, None, None


def slstm_scan(gin, r, h0, c0, n0, m0, exp_forget=True):
    """sLSTM over a sequence.

    ``gin``: (B, T, 4, D) input-side pre-activations (z, i, f, o) with every
    bias folded in; ``r``: (4, NH, DH, DH) recurrent matrices, row-major so
    that gate pre-activation = r[g, head] @ h_head.  Returns
    (h_1..h_T, c_T, n_T, m_T).
    """
    return _SLSTMScan.apply(gin, r, h0, c0, n0, m0, exp_forget)


# ================================================================ mLSTM (exact)
def mlstm_chunkwise(q, k, v, log_i, log_f, state=None, chunk_size=64, eps=0.0):
    """Exact stabilized mLSTM with one input/forget gate per head.

    q, k, v: (B, NH, T, DH) with k already scaled by 1/sqrt(DH);
    log_i, log_f: (B, NH, T) log input-gate and log forget-gate values.
    state: optional dict with C (B, NH, DH, DH), n (B, NH, DH), m (B, NH),
    stored in the stabilized scale (true values = stored * exp(m)).

    Returns h_tilde (B, NH, T, DH) and the final state.  Equals the recurrent
    update C = f C + i v k^T, n = f n + i k,
    h = C q / max(|n.q|, exp(-m)) step for step (``eps`` is added to the
    stabilized denominator, as in the official xLSTM kernels).
    """
    B, NH, T, DH = q.shape
    # At least fp32: the exponentials and the running normalizer need it.
    dtype = torch.promote_types(q.dtype, torch.float32)
    if state is None:
        C = q.new_zeros(B, NH, DH, DH, dtype=dtype)
        n = q.new_zeros(B, NH, DH, dtype=dtype)
        m = q.new_zeros(B, NH, dtype=dtype)
    else:
        C, n, m = state["C"].to(dtype), state["n"].to(dtype), state["m"].to(dtype)
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    log_i, log_f = log_i.to(dtype), log_f.to(dtype)
    outs = []
    for start in range(0, T, chunk_size):
        stop = min(T, start + chunk_size)
        L = stop - start
        qc, kc, vc = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
        li, lf = log_i[:, :, start:stop], log_f[:, :, start:stop]
        cum_f = torch.cumsum(lf, dim=-1)                                   # F_t within chunk
        # log weight of input s at output t (s <= t): F_t - F_s + log i_s
        log_d = cum_f[..., :, None] - cum_f[..., None, :] + li[..., None, :]
        causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        log_d = log_d.masked_fill(~causal, float("-inf"))
        log_state = cum_f + m[..., None]                                   # weight of carried state
        m_t = torch.maximum(log_d.amax(dim=-1), log_state)                 # (B, NH, L)
        d = torch.exp(log_d - m_t[..., None])
        state_w = torch.exp(log_state - m_t)
        scores = (qc @ kc.transpose(-1, -2)) * d
        num = scores @ vc + state_w[..., None] * torch.einsum("bhde,bhte->bhtd", C, qc)
        den = scores.sum(-1) + state_w * torch.einsum("bhe,bhte->bht", n, qc)
        den = torch.maximum(den.abs(), torch.exp(-m_t)) + eps
        outs.append(num / den[..., None])

        # Carry the chunk's final state (stabilized at its own maximum).
        log_last = cum_f[..., -1:] - cum_f + li                            # (B, NH, L)
        log_keep = cum_f[..., -1] + m
        m_next = torch.maximum(log_last.amax(dim=-1), log_keep)
        w = torch.exp(log_last - m_next[..., None])
        keep = torch.exp(log_keep - m_next)
        C = keep[..., None, None] * C + torch.einsum("bht,bhtd,bhte->bhde", w, vc, kc)
        n = keep[..., None] * n + torch.einsum("bht,bhte->bhe", w, kc)
        m = m_next
    return torch.cat(outs, dim=2), {"C": C, "n": n, "m": m}


# ======================================================== IndyGRU / ATanU-LSTM
if HAS_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _indygru_fwd_kernel(GI, CI, UG, UC, HF, T, H, BLOCK: tl.constexpr, RELU_GATES: tl.constexpr = False):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < H
        u_r = tl.load(UG + cols, mask=mask, other=0.0).to(tl.float32)
        u_z = tl.load(UG + H + cols, mask=mask, other=0.0).to(tl.float32)
        u_c = tl.load(UC + cols, mask=mask, other=0.0).to(tl.float32)
        h = tl.load(HF + b * (T + 1) * H + cols, mask=mask, other=0.0)
        for t in range(T):
            g = GI + (b * T + t) * 2 * H
            r_pre = tl.load(g + cols, mask=mask, other=0.0).to(tl.float32) + h * u_r
            z_pre = tl.load(g + H + cols, mask=mask, other=0.0).to(tl.float32) + h * u_z
            if RELU_GATES:
                r = tl.maximum(r_pre, 0.0)
                z = tl.maximum(z_pre, 0.0)
            else:
                r = tl.sigmoid(r_pre)
                z = tl.sigmoid(z_pre)
            cand = _tanh(tl.load(CI + (b * T + t) * H + cols, mask=mask, other=0.0).to(tl.float32) + r * h * u_c)
            h = (1.0 - z) * h + z * cand
            tl.store(HF + (b * (T + 1) + t + 1) * H + cols, h, mask=mask)

    @triton.jit
    def _indygru_bwd_kernel(DY, GI, CI, UG, UC, HF, DGI, DCI, DU, DH0, T, H, BLOCK: tl.constexpr,
                            RELU_GATES: tl.constexpr = False):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < H
        u_r = tl.load(UG + cols, mask=mask, other=0.0).to(tl.float32)
        u_z = tl.load(UG + H + cols, mask=mask, other=0.0).to(tl.float32)
        u_c = tl.load(UC + cols, mask=mask, other=0.0).to(tl.float32)
        dh = tl.zeros([BLOCK], dtype=tl.float32)
        du_r = tl.zeros([BLOCK], dtype=tl.float32)
        du_z = tl.zeros([BLOCK], dtype=tl.float32)
        du_c = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            h_prev = tl.load(HF + (b * (T + 1) + t) * H + cols, mask=mask, other=0.0)
            g = GI + (b * T + t) * 2 * H
            r_pre = tl.load(g + cols, mask=mask, other=0.0).to(tl.float32) + h_prev * u_r
            z_pre = tl.load(g + H + cols, mask=mask, other=0.0).to(tl.float32) + h_prev * u_z
            if RELU_GATES:
                r = tl.maximum(r_pre, 0.0)
                z = tl.maximum(z_pre, 0.0)
                r_grad = tl.where(r_pre > 0.0, 1.0, 0.0)
                z_grad = tl.where(z_pre > 0.0, 1.0, 0.0)
            else:
                r = tl.sigmoid(r_pre)
                z = tl.sigmoid(z_pre)
                r_grad = r * (1.0 - r)
                z_grad = z * (1.0 - z)
            cand = _tanh(tl.load(CI + (b * T + t) * H + cols, mask=mask, other=0.0).to(tl.float32) + r * h_prev * u_c)
            grad = tl.load(DY + (b * T + t) * H + cols, mask=mask, other=0.0).to(tl.float32) + dh
            dz = grad * (cand - h_prev)
            d_cand_pre = grad * z * (1.0 - cand * cand)
            dh_prev = grad * (1.0 - z) + d_cand_pre * u_c * r
            d_r_pre = d_cand_pre * u_c * h_prev * r_grad
            d_z_pre = dz * z_grad
            dh_prev += d_r_pre * u_r + d_z_pre * u_z
            du_c += d_cand_pre * r * h_prev
            du_r += d_r_pre * h_prev
            du_z += d_z_pre * h_prev
            tl.store(DCI + (b * T + t) * H + cols, d_cand_pre, mask=mask)
            tl.store(DGI + (b * T + t) * 2 * H + cols, d_r_pre, mask=mask)
            tl.store(DGI + (b * T + t) * 2 * H + H + cols, d_z_pre, mask=mask)
            dh = dh_prev
        tl.store(DU + b * 3 * H + cols, du_r, mask=mask)
        tl.store(DU + b * 3 * H + H + cols, du_z, mask=mask)
        tl.store(DU + b * 3 * H + 2 * H + cols, du_c, mask=mask)
        tl.store(DH0 + b * H + cols, dh, mask=mask)

    # lamb.py's ATanU activations: atan_u(x) = (2/pi) atan(pi x / 2) and
    # asig_u(x) = (1 + atan_u(2 x)) / 2 = 1/2 + atan(pi x) / pi.
    @triton.jit
    def _atan_unit(x):
        return 0.6366197723675814 * libdevice.atan(1.5707963267948966 * x)

    @triton.jit
    def _asig_unit(x):
        return 0.5 + 0.3183098861837907 * libdevice.atan(3.141592653589793 * x)

    @triton.jit
    def _atanu_fwd_kernel(PX, WT, HF, CF, A, B, T, H,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            for n0 in range(0, H, BN):
                rows = n0 + tl.arange(0, BN)
                rmask = rows < H
                acc_i = tl.zeros([BB, BN], dtype=tl.float32)
                acc_f = tl.zeros([BB, BN], dtype=tl.float32)
                acc_g = tl.zeros([BB, BN], dtype=tl.float32)
                acc_o = tl.zeros([BB, BN], dtype=tl.float32)
                for k0 in range(0, H, BK):
                    ks = k0 + tl.arange(0, BK)
                    kmask = ks < H
                    h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ks[None, :],
                                     mask=bmask[:, None] & kmask[None, :], other=0.0)
                    wmask = kmask[:, None] & rmask[None, :]
                    # WT is W_hh^T: (H, 4H), gate order i, f, g, o.
                    w = WT + ks[:, None] * (4 * H) + rows[None, :]
                    acc_i = tl.dot(h_prev, tl.load(w, mask=wmask, other=0.0), acc_i, input_precision=PREC)
                    acc_f = tl.dot(h_prev, tl.load(w + H, mask=wmask, other=0.0), acc_f, input_precision=PREC)
                    acc_g = tl.dot(h_prev, tl.load(w + 2 * H, mask=wmask, other=0.0), acc_g, input_precision=PREC)
                    acc_o = tl.dot(h_prev, tl.load(w + 3 * H, mask=wmask, other=0.0), acc_o, input_precision=PREC)
                tile = bmask[:, None] & rmask[None, :]
                px = PX + (brow[:, None] * T + t) * (4 * H) + rows[None, :]
                p_i = acc_i + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                p_f = acc_f + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                p_g = acc_g + tl.load(px + 2 * H, mask=tile, other=0.0).to(tl.float32)
                p_o = acc_o + tl.load(px + 3 * H, mask=tile, other=0.0).to(tl.float32)
                c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + rows[None, :], mask=tile, other=0.0)
                c = _asig_unit(p_f) * c_prev + _asig_unit(p_i) * _atan_unit(p_g)
                h = _asig_unit(p_o) * _atan_unit(c)
                nxt = (brow[:, None] * (T + 1) + t + 1) * H + rows[None, :]
                tl.store(CF + nxt, c, mask=tile)
                tl.store(HF + nxt, h, mask=tile)
                a = A + (brow[:, None] * T + t) * (4 * H) + rows[None, :]
                tl.store(a, p_i, mask=tile)
                tl.store(a + H, p_f, mask=tile)
                tl.store(a + 2 * H, p_g, mask=tile)
                tl.store(a + 3 * H, p_o, mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _atanu_bwd_kernel(DY, CF, A, W, DA, CH, CC, B, T, H,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for i in range(T):
            t = T - 1 - i
            for n0 in range(0, H, BN):
                rows = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (rows < H)[None, :]
                a = A + (brow[:, None] * T + t) * (4 * H) + rows[None, :]
                p_i = tl.load(a, mask=tile, other=0.0)
                p_f = tl.load(a + H, mask=tile, other=0.0)
                p_g = tl.load(a + 2 * H, mask=tile, other=0.0)
                p_o = tl.load(a + 3 * H, mask=tile, other=0.0)
                gi = _asig_unit(p_i)
                gf = _asig_unit(p_f)
                go = _asig_unit(p_o)
                gg = _atan_unit(p_g)
                c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + rows[None, :], mask=tile, other=0.0)
                c = tl.load(CF + (brow[:, None] * (T + 1) + t + 1) * H + rows[None, :], mask=tile, other=0.0)
                dh = tl.load(DY + (brow[:, None] * T + t) * H + rows[None, :], mask=tile, other=0.0).to(tl.float32) \
                    + tl.load(CH + brow[:, None] * H + rows[None, :], mask=tile, other=0.0)
                # d atan_u(x)/dx = 1 / (1 + (pi x / 2)^2); d asig_u(x)/dx = 1 / (1 + (pi x)^2)
                half_pi_c = 1.5707963267948966 * c
                half_pi_g = 1.5707963267948966 * p_g
                pi_i = 3.141592653589793 * p_i
                pi_f = 3.141592653589793 * p_f
                pi_o = 3.141592653589793 * p_o
                dc = dh * go / (1.0 + half_pi_c * half_pi_c) \
                    + tl.load(CC + brow[:, None] * H + rows[None, :], mask=tile, other=0.0)
                da = DA + (brow[:, None] * T + t) * (4 * H) + rows[None, :]
                tl.store(da, dc * gg / (1.0 + pi_i * pi_i), mask=tile)
                tl.store(da + H, dc * c_prev / (1.0 + pi_f * pi_f), mask=tile)
                tl.store(da + 2 * H, dc * gi / (1.0 + half_pi_g * half_pi_g), mask=tile)
                tl.store(da + 3 * H, dh * _atan_unit(c) / (1.0 + pi_o * pi_o), mask=tile)
                tl.store(CC + brow[:, None] * H + rows[None, :], dc * gf, mask=tile)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                cols = n0 + tl.arange(0, BN)
                cmask = cols < H
                acc = tl.zeros([BB, BN], dtype=tl.float32)
                for k0 in range(0, 4 * H, BK):
                    rs = k0 + tl.arange(0, BK)
                    rsmask = rs < 4 * H
                    d = tl.load(DA + (brow[:, None] * T + t) * (4 * H) + rs[None, :],
                                mask=bmask[:, None] & rsmask[None, :], other=0.0)
                    w = tl.load(W + rs[:, None] * H + cols[None, :], mask=rsmask[:, None] & cmask[None, :], other=0.0)
                    acc = tl.dot(d, w, acc, input_precision=PREC)
                tl.store(CH + brow[:, None] * H + cols[None, :], acc, mask=bmask[:, None] & cmask[None, :])
            tl.debug_barrier()


class _IndyGRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_in, cand_in, u_gate, u_cand, h0, relu_gates):
        gate_in, cand_in = gate_in.contiguous(), cand_in.contiguous()
        B, T, H = cand_in.shape
        hf = torch.empty(B, T + 1, H, device=cand_in.device, dtype=torch.float32)
        hf[:, 0] = h0.float()
        ug, uc = u_gate.float().contiguous(), u_cand.float().contiguous()
        block = min(1024, triton.next_power_of_2(H))
        _indygru_fwd_kernel[(B, triton.cdiv(H, block))](gate_in, cand_in, ug, uc, hf, T, H, BLOCK=block,
                                                       RELU_GATES=relu_gates)
        ctx.save_for_backward(gate_in, cand_in, ug, uc, hf)
        ctx.meta = (h0.dtype, block, relu_gates)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        gate_in, cand_in, ug, uc, hf = ctx.saved_tensors
        h_dtype, block, relu_gates = ctx.meta
        B, T, H = cand_in.shape
        kw = dict(device=hf.device, dtype=torch.float32)
        dgi, dci = torch.empty(B, T, 2 * H, **kw), torch.empty(B, T, H, **kw)
        du, dh0 = torch.empty(B, 3 * H, **kw), torch.empty(B, H, **kw)
        _indygru_bwd_kernel[(B, triton.cdiv(H, block))](_contig(dy), gate_in, cand_in, ug, uc, hf,
                                                       dgi, dci, du, dh0, T, H, BLOCK=block, RELU_GATES=relu_gates)
        du = du.sum(0)
        return dgi.to(gate_in.dtype), dci.to(cand_in.dtype), du[:2 * H], du[2 * H:], dh0.to(h_dtype), None


def indygru_scan(gate_in, cand_in, u_gate, u_cand, h0, relu_gates=False):
    """IndyGRU over a sequence: r, z = sigmoid(gate_in + [h, h] * u_gate)
    (ReLU instead of sigmoid with ``relu_gates``); h~ = tanh(cand_in +
    (r * h) * u_cand); h = (1 - z) h + z h~."""
    return _IndyGRUScan.apply(gate_in, cand_in, u_gate, u_cand, h0, bool(relu_gates))


class _ATanULSTMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, w_hh, h0, c0):
        px = px.contiguous()
        B, T, H4 = px.shape
        H = H4 // 4
        w32 = w_hh.float().contiguous()
        kw = dict(device=px.device, dtype=torch.float32)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        cf = torch.empty(B, T + 1, H, **kw); cf[:, 0] = c0.float()
        a = torch.empty(B, T, 4 * H, **kw)
        bb, bn, bk = _janet_tiles(H)
        prec = _dot_precision()
        _atanu_fwd_kernel[(triton.cdiv(B, bb),)](px, w32.t().contiguous(), hf, cf, a, B, T, H,
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(w32, hf, cf, a)
        ctx.meta = (px.dtype, h0.dtype, prec)
        return hf[:, 1:], cf[:, -1]

    @staticmethod
    def backward(ctx, dy, dc_last):
        w32, hf, cf, a = ctx.saved_tensors
        px_dtype, s_dtype, prec = ctx.meta
        B, T1, H = hf.shape
        T = T1 - 1
        kw = dict(device=hf.device, dtype=torch.float32)
        da = torch.empty(B, T, 4 * H, **kw)
        ch = torch.zeros(B, H, **kw)
        cc = torch.zeros(B, H, **kw) if dc_last is None else _contig(dc_last.float()).clone()
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        bb, bn, bk = _janet_tiles(H)
        _atanu_bwd_kernel[(triton.cdiv(B, bb),)](dy, cf, a, w32, da, ch, cc, B, T, H,
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        dw = da.reshape(-1, 4 * H).t() @ hf[:, :-1].reshape(-1, H)
        return da.to(px_dtype), dw, ch.to(s_dtype), cc.to(s_dtype)


def atanu_lstm_scan(px, w_hh, h0, c0):
    """ATanU-LSTM over a sequence.  ``px``: (B, T, 4H) input projections with
    both biases (gate order i, f, g, o).  Returns (h_1..h_T, c_T)."""
    return _ATanULSTMScan.apply(px, w_hh, h0, c0)


# ================================================================ RWKV-4 WKV
if HAS_TRITON:

    @triton.jit
    def _wkv4_fwd_kernel(W, U, K, V, Y, AA, BB, PP, T, C, BLOCK: tl.constexpr):
        # State at index t is the state *before* step t (index 0 = initial).
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < C
        w = tl.load(W + cols, mask=mask, other=0.0)
        u = tl.load(U + cols, mask=mask, other=0.0)
        aa = tl.load(AA + b * (T + 1) * C + cols, mask=mask, other=0.0)
        bb = tl.load(BB + b * (T + 1) * C + cols, mask=mask, other=0.0)
        pp = tl.load(PP + b * (T + 1) * C + cols, mask=mask, other=0.0)
        for t in range(T):
            k = tl.load(K + (b * T + t) * C + cols, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(V + (b * T + t) * C + cols, mask=mask, other=0.0).to(tl.float32)
            ww = u + k
            p = tl.maximum(pp, ww)
            e1 = tl.exp(pp - p)
            e2 = tl.exp(ww - p)
            tl.store(Y + (b * T + t) * C + cols, (e1 * aa + e2 * v) / (e1 * bb + e2), mask=mask)
            ww = w + pp
            p = tl.maximum(ww, k)
            e1 = tl.exp(ww - p)
            e2 = tl.exp(k - p)
            aa = e1 * aa + e2 * v
            bb = e1 * bb + e2
            pp = p
            nxt = (b * (T + 1) + t + 1) * C + cols
            tl.store(AA + nxt, aa, mask=mask)
            tl.store(BB + nxt, bb, mask=mask)
            tl.store(PP + nxt, pp, mask=mask)

    @triton.jit
    def _wkv4_bwd_kernel(W, U, K, V, DY, AA, BB, PP, DAA, DBB, DK, DV, DW, DU, T, C, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < C
        w = tl.load(W + cols, mask=mask, other=0.0)
        u = tl.load(U + cols, mask=mask, other=0.0)
        daa = tl.load(DAA + b * C + cols, mask=mask, other=0.0)
        dbb = tl.load(DBB + b * C + cols, mask=mask, other=0.0)
        dw = tl.zeros([BLOCK], dtype=tl.float32)
        du = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            st = (b * (T + 1) + t) * C + cols
            aa = tl.load(AA + st, mask=mask, other=0.0)
            bb = tl.load(BB + st, mask=mask, other=0.0)
            pp = tl.load(PP + st, mask=mask, other=0.0)
            k = tl.load(K + (b * T + t) * C + cols, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(V + (b * T + t) * C + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + (b * T + t) * C + cols, mask=mask, other=0.0).to(tl.float32)
            # Output y = (e1 aa + e2 v) / (e1 bb + e2).  The running maximum is a
            # pure stabilizer (y and the true state do not depend on it), so it
            # is treated as a constant here, exactly as in the official kernel.
            ww = u + k
            p = tl.maximum(pp, ww)
            e1 = tl.exp(pp - p)
            e2 = tl.exp(ww - p)
            den = e1 * bb + e2
            y = (e1 * aa + e2 * v) / den
            dnum = dy / den
            dden = -dy * y / den
            # State update aa' = f1 aa + f2 v, bb' = f1 bb + f2.
            ww2 = w + pp
            p2 = tl.maximum(ww2, k)
            f1 = tl.exp(ww2 - p2)
            f2 = tl.exp(k - p2)
            g_out = (dnum * v + dden) * e2
            g_state = (daa * v + dbb) * f2
            tl.store(DV + (b * T + t) * C + cols, dnum * e2 + daa * f2, mask=mask)
            tl.store(DK + (b * T + t) * C + cols, g_out + g_state, mask=mask)
            du += g_out
            dw += (daa * aa + dbb * bb) * f1
            daa = dnum * e1 + daa * f1
            dbb = dden * e1 + dbb * f1
        tl.store(DW + b * C + cols, dw, mask=mask)
        tl.store(DU + b * C + cols, du, mask=mask)
        tl.store(DAA + b * C + cols, daa, mask=mask)
        tl.store(DBB + b * C + cols, dbb, mask=mask)


class _WKV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v, aa0, bb0, pp0):
        k, v = k.contiguous(), v.contiguous()
        B, T, C = k.shape
        kw = dict(device=k.device, dtype=torch.float32)
        y = torch.empty(B, T, C, **kw)
        aa = torch.empty(B, T + 1, C, **kw); aa[:, 0] = aa0.float()
        bb = torch.empty(B, T + 1, C, **kw); bb[:, 0] = bb0.float()
        pp = torch.empty(B, T + 1, C, **kw); pp[:, 0] = pp0.float()
        w32, u32 = w.float().contiguous(), u.float().contiguous()
        block = min(1024, triton.next_power_of_2(C))
        _wkv4_fwd_kernel[(B, triton.cdiv(C, block))](w32, u32, k, v, y, aa, bb, pp, T, C, BLOCK=block)
        ctx.save_for_backward(w32, u32, k, v, aa, bb, pp)
        ctx.meta = (block, aa0.dtype)
        return y, aa[:, -1], bb[:, -1], pp[:, -1]

    @staticmethod
    def backward(ctx, dy, daa_last, dbb_last, dpp_last):
        w32, u32, k, v, aa, bb, pp = ctx.saved_tensors
        block, s_dtype = ctx.meta
        B, T, C = k.shape
        kw = dict(device=k.device, dtype=torch.float32)
        dy = torch.zeros(B, T, C, **kw) if dy is None else _contig(dy)
        daa = torch.zeros(B, C, **kw) if daa_last is None else _contig(daa_last.float()).clone()
        dbb = torch.zeros(B, C, **kw) if dbb_last is None else _contig(dbb_last.float()).clone()
        dk, dv = torch.empty(B, T, C, **kw), torch.empty(B, T, C, **kw)
        dw, du = torch.empty(B, C, **kw), torch.empty(B, C, **kw)
        _wkv4_bwd_kernel[(B, triton.cdiv(C, block))](w32, u32, k, v, dy, aa, bb, pp, daa, dbb,
                                                    dk, dv, dw, du, T, C, BLOCK=block)
        return dw.sum(0), du.sum(0), dk.to(k.dtype), dv.to(v.dtype), daa.to(s_dtype), dbb.to(s_dtype), None


def rwkv4_wkv(w, u, k, v, aa, bb, pp):
    """RWKV-4 WKV operator (stabilized, as in BlinkDL's wkv_cuda kernel).

    ``w``: per-channel log decay (= -exp(time_decay)); ``u``: time_first;
    ``k``, ``v``: (B, T, C); state (aa, bb, pp): (B, C) each.
    Returns (wkv, aa, bb, pp).
    """
    return _WKV4.apply(w, u, k, v, aa, bb, pp)


def rwkv4_wkv_reference(w, u, k, v, aa, bb, pp):
    """Loop implementation of ``rwkv4_wkv`` for CPU and testing."""
    k, v = k.float(), v.float()
    ys = []
    for t in range(k.size(1)):
        kt, vt = k[:, t], v[:, t]
        ww = u + kt
        p = torch.maximum(pp, ww)
        e1, e2 = torch.exp(pp - p), torch.exp(ww - p)
        ys.append((e1 * aa + e2 * vt) / (e1 * bb + e2))
        ww = w + pp
        p = torch.maximum(ww, kt)
        e1, e2 = torch.exp(ww - p), torch.exp(kt - p)
        aa, bb, pp = e1 * aa + e2 * vt, e1 * bb + e2, p
    return torch.stack(ys, 1), aa, bb, pp


# ================================================================ RWKV-7 WKV
_WKV7_CHUNK = 32

if HAS_TRITON:

    @triton.jit
    def _wkv7_fwd_kernel(R, W, K, V, A, Bv, S0, Y, SC, T, H,
                         N: tl.constexpr, P: tl.constexpr, CK: tl.constexpr):
        # Per (batch, head): state rows = value channels (P), columns = key
        # channels (N).  S = S diag(w) + (S a) b^T + v k^T,  y = S r.
        b = tl.program_id(0)
        h = tl.program_id(1)
        i = tl.arange(0, P)
        j = tl.arange(0, N)
        mat = i[:, None] * N + j[None, :]
        state = tl.load(S0 + (b * H + h) * P * N + mat)
        nchk = (T + CK - 1) // CK
        for t in range(T):
            if t % CK == 0:
                tl.store(SC + ((b * H + h) * (nchk + 1) + t // CK) * P * N + mat, state)
            kb = (b * T + t) * H * N + h * N
            vb = (b * T + t) * H * P + h * P
            r = tl.load(R + kb + j).to(tl.float32)
            w = tl.load(W + kb + j).to(tl.float32)
            k = tl.load(K + kb + j).to(tl.float32)
            v = tl.load(V + vb + i).to(tl.float32)
            a = tl.load(A + kb + j).to(tl.float32)
            bb = tl.load(Bv + kb + j).to(tl.float32)
            sa = tl.sum(state * a[None, :], axis=1)
            state = state * w[None, :] + sa[:, None] * bb[None, :] + v[:, None] * k[None, :]
            tl.store(Y + vb + i, tl.sum(state * r[None, :], axis=1))
        tl.store(SC + ((b * H + h) * (nchk + 1) + nchk) * P * N + mat, state)

    @triton.jit
    def _wkv7_bwd_kernel(R, W, K, V, A, Bv, DY, SC, SCR, DS,
                         DR, DW, DK, DV, DA, DB, T, H,
                         N: tl.constexpr, P: tl.constexpr, CK: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        i = tl.arange(0, P)
        j = tl.arange(0, N)
        mat = i[:, None] * N + j[None, :]
        nchk = (T + CK - 1) // CK
        dstate = tl.load(DS + (b * H + h) * P * N + mat)
        scratch = SCR + (b * H + h) * CK * P * N
        for ci in range(nchk):
            c = nchk - 1 - ci
            t0 = c * CK
            t1 = tl.minimum(t0 + CK, T)
            # Recompute the states entering each step of this chunk.
            state = tl.load(SC + ((b * H + h) * (nchk + 1) + c) * P * N + mat)
            for t in range(t0, t1):
                tl.store(scratch + (t - t0) * P * N + mat, state)
                kb = (b * T + t) * H * N + h * N
                vb = (b * T + t) * H * P + h * P
                w = tl.load(W + kb + j).to(tl.float32)
                k = tl.load(K + kb + j).to(tl.float32)
                v = tl.load(V + vb + i).to(tl.float32)
                a = tl.load(A + kb + j).to(tl.float32)
                bb = tl.load(Bv + kb + j).to(tl.float32)
                sa = tl.sum(state * a[None, :], axis=1)
                state = state * w[None, :] + sa[:, None] * bb[None, :] + v[:, None] * k[None, :]
            tl.debug_barrier()
            for ti in range(t1 - t0):
                t = t1 - 1 - ti
                kb = (b * T + t) * H * N + h * N
                vb = (b * T + t) * H * P + h * P
                prev = tl.load(scratch + (t - t0) * P * N + mat)
                r = tl.load(R + kb + j).to(tl.float32)
                w = tl.load(W + kb + j).to(tl.float32)
                k = tl.load(K + kb + j).to(tl.float32)
                v = tl.load(V + vb + i).to(tl.float32)
                a = tl.load(A + kb + j).to(tl.float32)
                bb = tl.load(Bv + kb + j).to(tl.float32)
                dy = tl.load(DY + vb + i).to(tl.float32)
                sa = tl.sum(prev * a[None, :], axis=1)
                new = prev * w[None, :] + sa[:, None] * bb[None, :] + v[:, None] * k[None, :]
                tl.store(DR + kb + j, tl.sum(new * dy[:, None], axis=0))
                dstate += dy[:, None] * r[None, :]
                tl.store(DW + kb + j, tl.sum(dstate * prev, axis=0))
                dsa = tl.sum(dstate * bb[None, :], axis=1)
                tl.store(DB + kb + j, tl.sum(dstate * sa[:, None], axis=0))
                tl.store(DV + vb + i, tl.sum(dstate * k[None, :], axis=1))
                tl.store(DK + kb + j, tl.sum(dstate * v[:, None], axis=0))
                tl.store(DA + kb + j, tl.sum(prev * dsa[:, None], axis=0))
                dstate = dstate * w[None, :] + dsa[:, None] * a[None, :]
            tl.debug_barrier()
        tl.store(DS + (b * H + h) * P * N + mat, dstate)


class _WKV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, s0):
        r, w, k, v, a, b = (t.contiguous() for t in (r, w, k, v, a, b))
        B, T, C = r.shape
        H, P, N = s0.shape[1], s0.shape[2], s0.shape[3]
        kw = dict(device=r.device, dtype=torch.float32)
        y = torch.empty(B, T, H * P, **kw)
        nchk = triton.cdiv(T, _WKV7_CHUNK)
        sc = torch.empty(B, H, nchk + 1, P, N, **kw)
        s0 = s0.float().contiguous()
        warps = 4 if P * N <= 1024 else 8
        _wkv7_fwd_kernel[(B, H)](r, w, k, v, a, b, s0, y, sc, T, H, N=N, P=P, CK=_WKV7_CHUNK, num_warps=warps)
        ctx.save_for_backward(r, w, k, v, a, b, sc)
        ctx.warps = warps
        return y, sc[:, :, -1]

    @staticmethod
    def backward(ctx, dy, ds_last):
        r, w, k, v, a, b, sc = ctx.saved_tensors
        B, T, _ = r.shape
        H, P, N = sc.shape[1], sc.shape[3], sc.shape[4]
        kw = dict(device=r.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H * P, **kw) if dy is None else _contig(dy)
        ds = torch.zeros(B, H, P, N, **kw) if ds_last is None else _contig(ds_last.float()).clone()
        gk = [torch.empty(B, T, H * N, **kw) for _ in range(5)]
        dv = torch.empty(B, T, H * P, **kw)
        scratch = torch.empty(B, H, _WKV7_CHUNK, P, N, **kw)
        dr, dw, dk, da, db = gk
        _wkv7_bwd_kernel[(B, H)](r, w, k, v, a, b, dy, sc, scratch, ds, dr, dw, dk, dv, da, db, T, H,
                                 N=N, P=P, CK=_WKV7_CHUNK, num_warps=ctx.warps)
        return (dr.to(r.dtype), dw.to(w.dtype), dk.to(k.dtype), dv.to(v.dtype),
                da.to(a.dtype), db.to(b.dtype), ds)


def rwkv7_wkv(r, w, k, v, a, b, state):
    """RWKV-7 state evolution (BlinkDL RWKV7_OP), per head of size N:
    S = S diag(w) + (S a) b^T + v k^T,  y = S r.

    r, w, k, a, b: (B, T, H*N) and v: (B, T, H*P), with ``w`` the per-step
    decay factor; state: (B, H, P, N) with rows = value channels (P may differ
    from the key width N; both powers of two).  Returns (y, final state).
    """
    return _WKV7.apply(r, w, k, v, a, b, state)


def rwkv7_wkv_reference(r, w, k, v, a, b, state):
    """Loop implementation of ``rwkv7_wkv`` for CPU and testing."""
    B, T, _ = r.shape
    H, P, N = state.shape[1], state.shape[2], state.shape[3]
    vk = lambda t: t.float().view(B, T, H, N)
    r, w, k, a, b = map(vk, (r, w, k, a, b))
    v = v.float().view(B, T, H, P)
    s = state.float()
    ys = []
    for t in range(T):
        sa = torch.einsum("bhij,bhj->bhi", s, a[:, t])
        s = s * w[:, t, :, None, :] + sa[..., None] * b[:, t, :, None, :] + v[:, t, :, :, None] * k[:, t, :, None, :]
        ys.append(torch.einsum("bhij,bhj->bhi", s, r[:, t]).reshape(B, H * P))
    return torch.stack(ys, 1), s


# ============================================================ Mamba selective scan
_SSM_CHUNK = 32

if HAS_TRITON:

    @triton.jit
    def _selective_scan_fwd_kernel(U, DELTA, A, Bm, Cm, DSKIP, X0, Y, XC, T, D, NCHK,
                                   N: tl.constexpr, BLOCK_D: tl.constexpr, CK: tl.constexpr):
        b = tl.program_id(0)
        ds = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = ds < D
        n = tl.arange(0, N)
        m2 = dmask[:, None] & (n[None, :] < N)
        a = tl.load(A + ds[:, None] * N + n[None, :], mask=m2, other=0.0)
        dskip = tl.load(DSKIP + ds, mask=dmask, other=0.0)
        x = tl.load(X0 + (b * D + ds[:, None]) * N + n[None, :], mask=m2, other=0.0)
        for t in range(T):
            if t % CK == 0:
                tl.store(XC + ((b * (NCHK + 1) + t // CK) * D + ds[:, None]) * N + n[None, :], x, mask=m2)
            u = tl.load(U + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
            dt = tl.load(DELTA + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
            bt = tl.load(Bm + (b * T + t) * N + n).to(tl.float32)
            ct = tl.load(Cm + (b * T + t) * N + n).to(tl.float32)
            x = tl.exp(dt[:, None] * a) * x + (dt * u)[:, None] * bt[None, :]
            y = tl.sum(x * ct[None, :], axis=1) + dskip * u
            tl.store(Y + (b * T + t) * D + ds, y, mask=dmask)
        tl.store(XC + ((b * (NCHK + 1) + NCHK) * D + ds[:, None]) * N + n[None, :], x, mask=m2)

    @triton.jit
    def _selective_scan_bwd_kernel(U, DELTA, A, Bm, Cm, DSKIP, DY, XC, SCR, DX,
                                   DU, DDELTA, DBP, DCP, DAP, DDP, T, D, NCHK, NDB,
                                   N: tl.constexpr, BLOCK_D: tl.constexpr, CK: tl.constexpr):
        b = tl.program_id(0)
        db = tl.program_id(1)
        ds = db * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = ds < D
        n = tl.arange(0, N)
        m2 = dmask[:, None] & (n[None, :] < N)
        a = tl.load(A + ds[:, None] * N + n[None, :], mask=m2, other=0.0)
        dskip = tl.load(DSKIP + ds, mask=dmask, other=0.0)
        dx = tl.load(DX + (b * D + ds[:, None]) * N + n[None, :], mask=m2, other=0.0)
        da_acc = tl.zeros([BLOCK_D, N], dtype=tl.float32)
        dd_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        scratch = SCR + ((b * NDB + db) * CK) * BLOCK_D * N
        loc = tl.arange(0, BLOCK_D)[:, None] * N + n[None, :]
        for ci in range(NCHK):
            c = NCHK - 1 - ci
            t0 = c * CK
            t1 = tl.minimum(t0 + CK, T)
            x = tl.load(XC + ((b * (NCHK + 1) + c) * D + ds[:, None]) * N + n[None, :], mask=m2, other=0.0)
            for t in range(t0, t1):
                tl.store(scratch + (t - t0) * BLOCK_D * N + loc, x)
                u = tl.load(U + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
                dt = tl.load(DELTA + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
                bt = tl.load(Bm + (b * T + t) * N + n).to(tl.float32)
                x = tl.exp(dt[:, None] * a) * x + (dt * u)[:, None] * bt[None, :]
            tl.debug_barrier()
            for ti in range(t1 - t0):
                t = t1 - 1 - ti
                prev = tl.load(scratch + (t - t0) * BLOCK_D * N + loc)
                u = tl.load(U + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
                dt = tl.load(DELTA + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
                bt = tl.load(Bm + (b * T + t) * N + n).to(tl.float32)
                ct = tl.load(Cm + (b * T + t) * N + n).to(tl.float32)
                dy = tl.load(DY + (b * T + t) * D + ds, mask=dmask, other=0.0).to(tl.float32)
                decay = tl.exp(dt[:, None] * a)
                x = decay * prev + (dt * u)[:, None] * bt[None, :]
                # y = C.x + D u
                tl.store(DCP + ((b * NDB + db) * T + t) * N + n, tl.sum(tl.where(m2, x * dy[:, None], 0.0), axis=0))
                dd_acc += dy * u
                dx += dy[:, None] * ct[None, :]
                # x = exp(dt A) prev + dt u B
                g_decay = dx * prev * decay
                d_dt = tl.sum(g_decay * a, axis=1) + tl.sum(dx * bt[None, :], axis=1) * u
                d_u = tl.sum(dx * bt[None, :], axis=1) * dt + dy * dskip
                da_acc += g_decay * dt[:, None]
                tl.store(DBP + ((b * NDB + db) * T + t) * N + n,
                         tl.sum(tl.where(m2, dx * (dt * u)[:, None], 0.0), axis=0))
                tl.store(DU + (b * T + t) * D + ds, d_u, mask=dmask)
                tl.store(DDELTA + (b * T + t) * D + ds, d_dt, mask=dmask)
                dx = dx * decay
            tl.debug_barrier()
        tl.store(DX + (b * D + ds[:, None]) * N + n[None, :], dx, mask=m2)
        tl.store(DAP + (b * D + ds[:, None]) * N + n[None, :], da_acc, mask=m2)
        tl.store(DDP + b * D + ds, dd_acc, mask=dmask)


def _ssm_block_d(N: int) -> int:
    return max(16, min(128, 2048 // N))


class _SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, Bm, Cm, Dskip, x0):
        u, delta, Bm, Cm = u.contiguous(), delta.contiguous(), Bm.contiguous(), Cm.contiguous()
        B, T, D = u.shape
        N = A.shape[1]
        kw = dict(device=u.device, dtype=torch.float32)
        a32, d32 = A.float().contiguous(), Dskip.float().contiguous()
        y = torch.empty(B, T, D, **kw)
        nchk = triton.cdiv(T, _SSM_CHUNK)
        xc = torch.empty(B, nchk + 1, D, N, **kw)
        bd = _ssm_block_d(N)
        _selective_scan_fwd_kernel[(B, triton.cdiv(D, bd))](u, delta, a32, Bm, Cm, d32, x0.float().contiguous(),
                                                           y, xc, T, D, nchk, N=N, BLOCK_D=bd, CK=_SSM_CHUNK)
        ctx.save_for_backward(u, delta, a32, Bm, Cm, d32, xc)
        ctx.meta = (bd, nchk, x0.dtype)
        return y, xc[:, -1]

    @staticmethod
    def backward(ctx, dy, dx_last):
        u, delta, a32, Bm, Cm, d32, xc = ctx.saved_tensors
        bd, nchk, x_dtype = ctx.meta
        B, T, D = u.shape
        N = a32.shape[1]
        ndb = triton.cdiv(D, bd)
        kw = dict(device=u.device, dtype=torch.float32)
        dy = torch.zeros(B, T, D, **kw) if dy is None else _contig(dy)
        dx = torch.zeros(B, D, N, **kw) if dx_last is None else _contig(dx_last.float()).clone()
        du, ddelta = torch.empty(B, T, D, **kw), torch.empty(B, T, D, **kw)
        dbp, dcp = torch.empty(B, ndb, T, N, **kw), torch.empty(B, ndb, T, N, **kw)
        dap, ddp = torch.empty(B, D, N, **kw), torch.empty(B, D, **kw)
        scratch = torch.empty(B, ndb, _SSM_CHUNK, bd, N, **kw)
        _selective_scan_bwd_kernel[(B, ndb)](u, delta, a32, Bm, Cm, d32, dy, xc, scratch, dx,
                                             du, ddelta, dbp, dcp, dap, ddp, T, D, nchk, ndb,
                                             N=N, BLOCK_D=bd, CK=_SSM_CHUNK)
        return (du.to(u.dtype), ddelta.to(delta.dtype), dap.sum(0), dbp.sum(1).to(Bm.dtype),
                dcp.sum(1).to(Cm.dtype), ddp.sum(0), dx.to(x_dtype))


def selective_scan(u, delta, A, Bm, Cm, Dskip, x0):
    """Mamba selective scan (selective_scan_ref semantics, without z):
    x_t = exp(delta_t A) x_{t-1} + delta_t u_t B_t,  y_t = C_t . x_t + D u_t.

    u, delta: (B, T, D) with delta already softplus-ed; A: (D, N) (negative);
    Bm, Cm: (B, T, N); Dskip: (D,); x0: (B, D, N).  Returns (y, x_T).
    """
    return _SelectiveScan.apply(u, delta, A, Bm, Cm, Dskip, x0)


def selective_scan_reference(u, delta, A, Bm, Cm, Dskip, x0):
    """Loop implementation of ``selective_scan`` for CPU and testing."""
    x = x0.float()
    ys = []
    for t in range(u.size(1)):
        dt, ut = delta[:, t].float(), u[:, t].float()
        x = torch.exp(dt[..., None] * A) * x + (dt * ut)[..., None] * Bm[:, t, None, :].float()
        ys.append((x * Cm[:, t, None, :].float()).sum(-1) + Dskip * ut)
    return torch.stack(ys, 1), x


# ================================================================ LTC (liquid)
# Liquid time-constant cell (Hasani et al., 2021) with the ncps LTCCell fused
# ODE solver.  Synapse activations are sigmoid((v_i - mu_ij) * sigma_ij)
# scaled by a positive weight; every step runs ``unfolds`` semi-implicit
# updates of all neurons.
if HAS_TRITON:

    @triton.jit
    def _ltc_sensory_fwd_kernel(XM, WP, MU, SG, ER, NUMS, DENS, I, H,
                                BI: tl.constexpr, BJ: tl.constexpr):
        row = tl.program_id(0)
        js = tl.program_id(1) * BJ + tl.arange(0, BJ)
        jm = js < H
        num = tl.zeros([BJ], dtype=tl.float32)
        den = tl.zeros([BJ], dtype=tl.float32)
        for i0 in range(0, I, BI):
            ids = i0 + tl.arange(0, BI)
            im = ids < I
            x = tl.load(XM + row * I + ids, mask=im, other=0.0).to(tl.float32)
            m2 = im[:, None] & jm[None, :]
            off = ids[:, None] * H + js[None, :]
            act = tl.load(WP + off, mask=m2, other=0.0) * tl.sigmoid(
                tl.load(SG + off, mask=m2, other=0.0) * (x[:, None] - tl.load(MU + off, mask=m2, other=0.0)))
            num += tl.sum(act * tl.load(ER + off, mask=m2, other=0.0), axis=0)
            den += tl.sum(act, axis=0)
        tl.store(NUMS + row * H + js, num, mask=jm)
        tl.store(DENS + row * H + js, den, mask=jm)

    @triton.jit
    def _ltc_input_grad_kernel(XV, DNV, DDV, WP, MU, SG, ER, DX, I, H,
                               BI: tl.constexpr, BJ: tl.constexpr):
        # dx_i = sum_j (dN_j erev_ij + dD_j) w_ij s'(z_ij) sigma_ij
        row = tl.program_id(0)
        ids = tl.program_id(1) * BI + tl.arange(0, BI)
        im = ids < I
        x = tl.load(XV + row * I + ids, mask=im, other=0.0).to(tl.float32)
        acc = tl.zeros([BI], dtype=tl.float32)
        for j0 in range(0, H, BJ):
            js = j0 + tl.arange(0, BJ)
            jm = js < H
            m2 = im[:, None] & jm[None, :]
            off = ids[:, None] * H + js[None, :]
            dn = tl.load(DNV + row * H + js, mask=jm, other=0.0)
            dd = tl.load(DDV + row * H + js, mask=jm, other=0.0)
            sg = tl.load(SG + off, mask=m2, other=0.0)
            s = tl.sigmoid(sg * (x[:, None] - tl.load(MU + off, mask=m2, other=0.0)))
            g = (dn[None, :] * tl.load(ER + off, mask=m2, other=0.0) + dd[None, :]) \
                * tl.load(WP + off, mask=m2, other=0.0) * s * (1.0 - s)
            acc += tl.sum(g * sg, axis=1)
        tl.store(DX + row * I + ids, acc, mask=im)

    @triton.jit
    def _ltc_pair_grad_kernel(XV, DNV, DDV, WP, MU, SG, ER, GW, GMU, GSG, GER, S, I, H, PER,
                              BI: tl.constexpr, BJ: tl.constexpr):
        # Per-synapse parameter gradients summed over samples [s0, s1).
        ids = tl.program_id(0) * BI + tl.arange(0, BI)
        js = tl.program_id(1) * BJ + tl.arange(0, BJ)
        split = tl.program_id(2)
        im = ids < I
        jm = js < H
        m2 = im[:, None] & jm[None, :]
        off = ids[:, None] * H + js[None, :]
        wp = tl.load(WP + off, mask=m2, other=0.0)
        mu = tl.load(MU + off, mask=m2, other=0.0)
        sg = tl.load(SG + off, mask=m2, other=0.0)
        er = tl.load(ER + off, mask=m2, other=0.0)
        g_w = tl.zeros([BI, BJ], dtype=tl.float32)
        g_mu = tl.zeros([BI, BJ], dtype=tl.float32)
        g_sg = tl.zeros([BI, BJ], dtype=tl.float32)
        g_er = tl.zeros([BI, BJ], dtype=tl.float32)
        s0 = split * PER
        s1 = tl.minimum(S, s0 + PER)
        for s in range(s0, s1):
            x = tl.load(XV + s * I + ids, mask=im, other=0.0).to(tl.float32)
            dn = tl.load(DNV + s * H + js, mask=jm, other=0.0)
            dd = tl.load(DDV + s * H + js, mask=jm, other=0.0)
            diff = x[:, None] - mu
            sig = tl.sigmoid(sg * diff)
            dact = dn[None, :] * er + dd[None, :]
            g_w += dact * sig
            g_er += dn[None, :] * wp * sig
            g = dact * wp * sig * (1.0 - sig)
            g_sg += g * diff
            g_mu -= g * sg
        out = (split * I + ids[:, None]) * H + js[None, :]
        tl.store(GW + out, g_w, mask=m2)
        tl.store(GMU + out, g_mu, mask=m2)
        tl.store(GSG + out, g_sg, mask=m2)
        tl.store(GER + out, g_er, mask=m2)

    @triton.jit
    def _ltc_fwd_kernel(NUMS, DENS, WP, MU, SG, ER, CMT, GLP, VL, VS, DENF, T, H, eps,
                        U: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
        # VS: (B, T*U + 1, H); unfold u of step t maps state index s -> s + 1.
        b = tl.program_id(0)
        for t in range(T):
            for u in range(U):
                s = t * U + u
                vin = VS + (b * (T * U + 1) + s) * H
                for j0 in range(0, H, BJ):
                    js = j0 + tl.arange(0, BJ)
                    jm = js < H
                    num = tl.zeros([BJ], dtype=tl.float32)
                    den = tl.zeros([BJ], dtype=tl.float32)
                    for i0 in range(0, H, BI):
                        ids = i0 + tl.arange(0, BI)
                        im = ids < H
                        v_i = tl.load(vin + ids, mask=im, other=0.0)
                        m2 = im[:, None] & jm[None, :]
                        off = ids[:, None] * H + js[None, :]
                        act = tl.load(WP + off, mask=m2, other=0.0) * tl.sigmoid(
                            tl.load(SG + off, mask=m2, other=0.0) * (v_i[:, None] - tl.load(MU + off, mask=m2, other=0.0)))
                        num += tl.sum(act * tl.load(ER + off, mask=m2, other=0.0), axis=0)
                        den += tl.sum(act, axis=0)
                    cm = tl.load(CMT + js, mask=jm, other=1.0)
                    gl = tl.load(GLP + js, mask=jm, other=0.0)
                    vl = tl.load(VL + js, mask=jm, other=0.0)
                    v_j = tl.load(vin + js, mask=jm, other=0.0)
                    num += cm * v_j + gl * vl + tl.load(NUMS + (b * T + t) * H + js, mask=jm, other=0.0)
                    den += cm + gl + tl.load(DENS + (b * T + t) * H + js, mask=jm, other=0.0) + eps
                    tl.store(vin + H + js, num / den, mask=jm)
                    tl.store(DENF + (b * T * U + s) * H + js, den, mask=jm)
                tl.debug_barrier()

    @triton.jit
    def _ltc_bwd_kernel(DY, WP, MU, SG, ER, CMT, GLP, VL, VS, DENF, CARRY, DIRECT,
                        DNO, DDO, DNUMS, DDENS, DCM, DGL, DVL, T, H,
                        U: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
        b = tl.program_id(0)
        for ti in range(T):
            t = T - 1 - ti
            for j0 in range(0, H, BJ):
                js = j0 + tl.arange(0, BJ)
                jm = js < H
                c = tl.load(CARRY + b * H + js, mask=jm, other=0.0)
                tl.store(CARRY + b * H + js,
                         c + tl.load(DY + (b * T + t) * H + js, mask=jm, other=0.0).to(tl.float32), mask=jm)
            tl.debug_barrier()
            for ui in range(U):
                u = U - 1 - ui
                s = t * U + u
                vin = VS + (b * (T * U + 1) + s) * H
                # Pass A: gradients of the rational update's numerator/denominator.
                for j0 in range(0, H, BJ):
                    js = j0 + tl.arange(0, BJ)
                    jm = js < H
                    dvp = tl.load(CARRY + b * H + js, mask=jm, other=0.0)
                    vp = tl.load(vin + H + js, mask=jm, other=0.0)
                    v = tl.load(vin + js, mask=jm, other=0.0)
                    den = tl.load(DENF + (b * T * U + s) * H + js, mask=jm, other=1.0)
                    cm = tl.load(CMT + js, mask=jm, other=0.0)
                    gl = tl.load(GLP + js, mask=jm, other=0.0)
                    vl = tl.load(VL + js, mask=jm, other=0.0)
                    dn = dvp / den
                    dd = -dvp * vp / den
                    tl.store(DNO + (b * T * U + s) * H + js, dn, mask=jm)
                    tl.store(DDO + (b * T * U + s) * H + js, dd, mask=jm)
                    tl.store(DIRECT + b * H + js, dn * cm, mask=jm)
                    tl.store(DCM + b * H + js, tl.load(DCM + b * H + js, mask=jm, other=0.0) + dn * v + dd, mask=jm)
                    tl.store(DGL + b * H + js, tl.load(DGL + b * H + js, mask=jm, other=0.0) + dn * vl + dd, mask=jm)
                    tl.store(DVL + b * H + js, tl.load(DVL + b * H + js, mask=jm, other=0.0) + dn * gl, mask=jm)
                    p = (b * T + t) * H + js
                    tl.store(DNUMS + p, tl.load(DNUMS + p, mask=jm, other=0.0) + dn, mask=jm)
                    tl.store(DDENS + p, tl.load(DDENS + p, mask=jm, other=0.0) + dd, mask=jm)
                tl.debug_barrier()
                # Pass B: carry to the unfold's input state through the synapses.
                for i0 in range(0, H, BI):
                    ids = i0 + tl.arange(0, BI)
                    im = ids < H
                    v_i = tl.load(vin + ids, mask=im, other=0.0)
                    acc = tl.zeros([BI], dtype=tl.float32)
                    for j0 in range(0, H, BJ):
                        js = j0 + tl.arange(0, BJ)
                        jm = js < H
                        m2 = im[:, None] & jm[None, :]
                        off = ids[:, None] * H + js[None, :]
                        dn = tl.load(DNO + (b * T * U + s) * H + js, mask=jm, other=0.0)
                        dd = tl.load(DDO + (b * T * U + s) * H + js, mask=jm, other=0.0)
                        sg = tl.load(SG + off, mask=m2, other=0.0)
                        sig = tl.sigmoid(sg * (v_i[:, None] - tl.load(MU + off, mask=m2, other=0.0)))
                        g = (dn[None, :] * tl.load(ER + off, mask=m2, other=0.0) + dd[None, :]) \
                            * tl.load(WP + off, mask=m2, other=0.0) * sig * (1.0 - sig)
                        acc += tl.sum(g * sg, axis=1)
                    tl.store(CARRY + b * H + ids,
                             tl.load(DIRECT + b * H + ids, mask=im, other=0.0) + acc, mask=im)
                tl.debug_barrier()


_LTC_TILE = 32


def _ltc_pair_grads(xv, dnv, ddv, wp, mu, sg, er):
    """Sum per-synapse gradients over all samples (rows of xv / dnv / ddv)."""
    S, I = xv.shape
    H = dnv.shape[1]
    splits = max(1, min(64, triton.cdiv(S, 256)))
    per = triton.cdiv(S, splits)
    kw = dict(device=xv.device, dtype=torch.float32)
    outs = [torch.empty(splits, I, H, **kw) for _ in range(4)]
    grid = (triton.cdiv(I, _LTC_TILE), triton.cdiv(H, _LTC_TILE), splits)
    _ltc_pair_grad_kernel[grid](xv, dnv, ddv, wp, mu, sg, er, *outs, S, I, H, per,
                                BI=_LTC_TILE, BJ=_LTC_TILE)
    return [o.sum(0) for o in outs]      # d wp, d mu, d sigma, d erev


class _LTCSensory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xm, wp, mu, sg, er):
        B, T, I = xm.shape
        H = wp.shape[1]
        rows = xm.reshape(B * T, I).float().contiguous()
        params = [p.float().contiguous() for p in (wp, mu, sg, er)]
        kw = dict(device=xm.device, dtype=torch.float32)
        nums, dens = torch.empty(B * T, H, **kw), torch.empty(B * T, H, **kw)
        _ltc_sensory_fwd_kernel[(B * T, triton.cdiv(H, _LTC_TILE))](rows, *params, nums, dens, I, H,
                                                                    BI=_LTC_TILE, BJ=_LTC_TILE)
        ctx.save_for_backward(rows, *params)
        ctx.shape = (B, T, xm.dtype)
        return nums.view(B, T, H), dens.view(B, T, H)

    @staticmethod
    def backward(ctx, dnums, ddens):
        rows, wp, mu, sg, er = ctx.saved_tensors
        B, T, x_dtype = ctx.shape
        I, H = wp.shape
        dn = _contig(dnums.reshape(B * T, H).float())
        dd = _contig(ddens.reshape(B * T, H).float())
        dx = torch.empty(B * T, I, device=rows.device, dtype=torch.float32)
        _ltc_input_grad_kernel[(B * T, triton.cdiv(I, _LTC_TILE))](rows, dn, dd, wp, mu, sg, er, dx, I, H,
                                                                   BI=_LTC_TILE, BJ=_LTC_TILE)
        return (dx.view(B, T, I).to(x_dtype), *_ltc_pair_grads(rows, dn, dd, wp, mu, sg, er))


class _LTCScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nums, dens, v0, wp, mu, sg, er, cmt, glp, vleak, unfolds, eps):
        B, T, H = nums.shape
        U = int(unfolds)
        kw = dict(device=nums.device, dtype=torch.float32)
        params = [p.float().contiguous() for p in (wp, mu, sg, er, cmt, glp, vleak)]
        vs = torch.empty(B, T * U + 1, H, **kw)
        vs[:, 0] = v0.float()
        denf = torch.empty(B, T * U, H, **kw)
        nums, dens = nums.float().contiguous(), dens.float().contiguous()
        _ltc_fwd_kernel[(B,)](nums, dens, *params, vs, denf, T, H, float(eps),
                              U=U, BI=_LTC_TILE, BJ=_LTC_TILE)
        ctx.save_for_backward(vs, denf, *params)
        ctx.meta = (U, v0.dtype)
        return vs[:, U::U]

    @staticmethod
    def backward(ctx, dy):
        vs, denf, wp, mu, sg, er, cmt, glp, vleak = ctx.saved_tensors
        U, v_dtype = ctx.meta
        B, TU1, H = vs.shape
        T = (TU1 - 1) // U
        kw = dict(device=vs.device, dtype=torch.float32)
        carry = torch.zeros(B, H, **kw)
        direct = torch.empty(B, H, **kw)
        dno, ddo = torch.empty(B, T * U, H, **kw), torch.empty(B, T * U, H, **kw)
        dnums, ddens = torch.zeros(B, T, H, **kw), torch.zeros(B, T, H, **kw)
        dcm, dgl, dvl = (torch.zeros(B, H, **kw) for _ in range(3))
        _ltc_bwd_kernel[(B,)](_contig(dy), wp, mu, sg, er, cmt, glp, vleak, vs, denf, carry, direct,
                              dno, ddo, dnums, ddens, dcm, dgl, dvl, T, H,
                              U=U, BI=_LTC_TILE, BJ=_LTC_TILE)
        pair = _ltc_pair_grads(_contig(vs[:, :-1].reshape(-1, H)), dno.view(-1, H), ddo.view(-1, H),
                               wp, mu, sg, er)
        return (dnums, ddens, carry.to(v_dtype), *pair, dcm.sum(0), dgl.sum(0), dvl.sum(0), None, None)


def ltc_sensory(xm, wp, mu, sg, er):
    """Sensory synapse drive of an LTC layer: (numerator, denominator), each
    (B, T, H), summed over input features."""
    return _LTCSensory.apply(xm, wp, mu, sg, er)


def ltc_scan(nums, dens, v0, wp, mu, sg, er, cmt, glp, vleak, unfolds=6, eps=1e-8):
    """LTC state over a sequence (ncps fused ODE solver).  Returns the state
    after every step, (B, T, H)."""
    return _LTCScan.apply(nums, dens, v0, wp, mu, sg, er, cmt, glp, vleak, unfolds, eps)


# ============================================== IndyLSTM / UnICORNN / SRU (diagonal)
if HAS_TRITON:

    @triton.jit
    def _gate(x, RELU: tl.constexpr):
        if RELU:
            return tl.maximum(x, 0.0)
        return tl.sigmoid(x)

    @triton.jit
    def _gate_grad(pre, g, RELU: tl.constexpr):
        if RELU:
            return tl.where(pre > 0.0, 1.0, 0.0)
        return g * (1.0 - g)

    @triton.jit
    def _indylstm_fwd_kernel(GX, U, HF, CF, T, H, BLOCK: tl.constexpr, RELU_GATES: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        u_i = tl.load(U + cols, mask=m, other=0.0)
        u_f = tl.load(U + H + cols, mask=m, other=0.0)
        u_g = tl.load(U + 2 * H + cols, mask=m, other=0.0)
        u_o = tl.load(U + 3 * H + cols, mask=m, other=0.0)
        h = tl.load(HF + b * (T + 1) * H + cols, mask=m, other=0.0)
        c = tl.load(CF + b * (T + 1) * H + cols, mask=m, other=0.0)
        for t in range(T):
            g = GX + (b * T + t) * 4 * H + cols
            i = _gate(tl.load(g, mask=m, other=0.0).to(tl.float32) + u_i * h, RELU_GATES)
            f = _gate(tl.load(g + H, mask=m, other=0.0).to(tl.float32) + u_f * h, RELU_GATES)
            cand = _tanh(tl.load(g + 2 * H, mask=m, other=0.0).to(tl.float32) + u_g * h)
            o = _gate(tl.load(g + 3 * H, mask=m, other=0.0).to(tl.float32) + u_o * h, RELU_GATES)
            c = f * c + i * cand
            h = o * _tanh(c)
            nxt = (b * (T + 1) + t + 1) * H + cols
            tl.store(HF + nxt, h, mask=m)
            tl.store(CF + nxt, c, mask=m)

    @triton.jit
    def _indylstm_bwd_kernel(DY, GX, U, HF, CF, DGX, DU, DH, DC, T, H, BLOCK: tl.constexpr,
                             RELU_GATES: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        u_i = tl.load(U + cols, mask=m, other=0.0)
        u_f = tl.load(U + H + cols, mask=m, other=0.0)
        u_g = tl.load(U + 2 * H + cols, mask=m, other=0.0)
        u_o = tl.load(U + 3 * H + cols, mask=m, other=0.0)
        dh = tl.load(DH + b * H + cols, mask=m, other=0.0)
        dc = tl.load(DC + b * H + cols, mask=m, other=0.0)
        du_i = tl.zeros([BLOCK], dtype=tl.float32)
        du_f = tl.zeros([BLOCK], dtype=tl.float32)
        du_g = tl.zeros([BLOCK], dtype=tl.float32)
        du_o = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(T):
            t = T - 1 - k
            h_prev = tl.load(HF + (b * (T + 1) + t) * H + cols, mask=m, other=0.0)
            c_prev = tl.load(CF + (b * (T + 1) + t) * H + cols, mask=m, other=0.0)
            c = tl.load(CF + (b * (T + 1) + t + 1) * H + cols, mask=m, other=0.0)
            g = GX + (b * T + t) * 4 * H + cols
            p_i = tl.load(g, mask=m, other=0.0).to(tl.float32) + u_i * h_prev
            p_f = tl.load(g + H, mask=m, other=0.0).to(tl.float32) + u_f * h_prev
            p_g = tl.load(g + 2 * H, mask=m, other=0.0).to(tl.float32) + u_g * h_prev
            p_o = tl.load(g + 3 * H, mask=m, other=0.0).to(tl.float32) + u_o * h_prev
            i = _gate(p_i, RELU_GATES)
            f = _gate(p_f, RELU_GATES)
            cand = _tanh(p_g)
            o = _gate(p_o, RELU_GATES)
            tc = _tanh(c)
            dh = dh + tl.load(DY + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32)
            dc = dc + dh * o * (1.0 - tc * tc)
            d_i = dc * cand * _gate_grad(p_i, i, RELU_GATES)
            d_f = dc * c_prev * _gate_grad(p_f, f, RELU_GATES)
            d_g = dc * i * (1.0 - cand * cand)
            d_o = dh * tc * _gate_grad(p_o, o, RELU_GATES)
            dg = DGX + (b * T + t) * 4 * H + cols
            tl.store(dg, d_i, mask=m)
            tl.store(dg + H, d_f, mask=m)
            tl.store(dg + 2 * H, d_g, mask=m)
            tl.store(dg + 3 * H, d_o, mask=m)
            du_i += d_i * h_prev
            du_f += d_f * h_prev
            du_g += d_g * h_prev
            du_o += d_o * h_prev
            dh = d_i * u_i + d_f * u_f + d_g * u_g + d_o * u_o
            dc = dc * f
        du = DU + b * 4 * H + cols
        tl.store(du, du_i, mask=m)
        tl.store(du + H, du_f, mask=m)
        tl.store(du + 2 * H, du_g, mask=m)
        tl.store(du + 3 * H, du_o, mask=m)
        tl.store(DH + b * H + cols, dh, mask=m)
        tl.store(DC + b * H + cols, dc, mask=m)

    @triton.jit
    def _unicornn_fwd_kernel(VX, W, DT, YF, ZF, T, H, alpha, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        w = tl.load(W + cols, mask=m, other=0.0)
        dt = tl.load(DT + cols, mask=m, other=0.0)
        y = tl.load(YF + b * (T + 1) * H + cols, mask=m, other=0.0)
        z = tl.load(ZF + b * (T + 1) * H + cols, mask=m, other=0.0)
        for t in range(T):
            a = w * y + tl.load(VX + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32)
            z = z - dt * (_tanh(a) + alpha * y)
            y = y + dt * z
            nxt = (b * (T + 1) + t + 1) * H + cols
            tl.store(YF + nxt, y, mask=m)
            tl.store(ZF + nxt, z, mask=m)

    @triton.jit
    def _unicornn_bwd_kernel(DYo, VX, W, DT, YF, ZF, DVX, DW, DDT, DY, DZ, T, H, alpha,
                             BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        w = tl.load(W + cols, mask=m, other=0.0)
        dt = tl.load(DT + cols, mask=m, other=0.0)
        dy = tl.load(DY + b * H + cols, mask=m, other=0.0)
        dz = tl.load(DZ + b * H + cols, mask=m, other=0.0)
        dw = tl.zeros([BLOCK], dtype=tl.float32)
        ddt = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(T):
            t = T - 1 - k
            y_prev = tl.load(YF + (b * (T + 1) + t) * H + cols, mask=m, other=0.0)
            z_new = tl.load(ZF + (b * (T + 1) + t + 1) * H + cols, mask=m, other=0.0)
            dy = dy + tl.load(DYo + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32)
            # y_t = y_{t-1} + dt z_t
            dz = dz + dy * dt
            ddt += dy * z_new
            # z_t = z_{t-1} - dt (tanh(a) + alpha y_{t-1}),  a = w y_{t-1} + vx_t
            a = w * y_prev + tl.load(VX + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32)
            ta = _tanh(a)
            ddt += -dz * (ta + alpha * y_prev)
            da = -dz * dt * (1.0 - ta * ta)
            tl.store(DVX + (b * T + t) * H + cols, da, mask=m)
            dw += da * y_prev
            dy = dy + da * w - dz * dt * alpha
        tl.store(DW + b * H + cols, dw, mask=m)
        tl.store(DDT + b * H + cols, ddt, mask=m)
        tl.store(DY + b * H + cols, dy, mask=m)
        tl.store(DZ + b * H + cols, dz, mask=m)

    @triton.jit
    def _sru_fwd_kernel(U3, X, VF, VR, CF, HOUT, T, H, alpha, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        vf = tl.load(VF + cols, mask=m, other=0.0)
        vr = tl.load(VR + cols, mask=m, other=0.0)
        c = tl.load(CF + b * (T + 1) * H + cols, mask=m, other=0.0)
        for t in range(T):
            u = U3 + (b * T + t) * 3 * H + cols
            xt = tl.load(u, mask=m, other=0.0).to(tl.float32)
            f = tl.sigmoid(tl.load(u + H, mask=m, other=0.0).to(tl.float32) + vf * c)
            r = tl.sigmoid(tl.load(u + 2 * H, mask=m, other=0.0).to(tl.float32) + vr * c)
            c = f * c + (1.0 - f) * xt
            skip = tl.load(X + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32) * alpha
            tl.store(HOUT + (b * T + t) * H + cols, r * c + (1.0 - r) * skip, mask=m)
            tl.store(CF + (b * (T + 1) + t + 1) * H + cols, c, mask=m)

    @triton.jit
    def _sru_bwd_kernel(DH, U3, X, VF, VR, CF, DU3, DX, DV, DC, T, H, alpha, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        vf = tl.load(VF + cols, mask=m, other=0.0)
        vr = tl.load(VR + cols, mask=m, other=0.0)
        dc = tl.load(DC + b * H + cols, mask=m, other=0.0)
        dvf = tl.zeros([BLOCK], dtype=tl.float32)
        dvr = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(T):
            t = T - 1 - k
            c_prev = tl.load(CF + (b * (T + 1) + t) * H + cols, mask=m, other=0.0)
            c = tl.load(CF + (b * (T + 1) + t + 1) * H + cols, mask=m, other=0.0)
            u = U3 + (b * T + t) * 3 * H + cols
            xt = tl.load(u, mask=m, other=0.0).to(tl.float32)
            f = tl.sigmoid(tl.load(u + H, mask=m, other=0.0).to(tl.float32) + vf * c_prev)
            r = tl.sigmoid(tl.load(u + 2 * H, mask=m, other=0.0).to(tl.float32) + vr * c_prev)
            skip = tl.load(X + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32) * alpha
            dh = tl.load(DH + (b * T + t) * H + cols, mask=m, other=0.0).to(tl.float32)
            d_r = dh * (c - skip) * r * (1.0 - r)
            tl.store(DX + (b * T + t) * H + cols, dh * (1.0 - r) * alpha, mask=m)
            dc = dc + dh * r
            d_f = dc * (c_prev - xt) * f * (1.0 - f)
            du = DU3 + (b * T + t) * 3 * H + cols
            tl.store(du, dc * (1.0 - f), mask=m)
            tl.store(du + H, d_f, mask=m)
            tl.store(du + 2 * H, d_r, mask=m)
            dvf += d_f * c_prev
            dvr += d_r * c_prev
            dc = dc * f + d_f * vf + d_r * vr
        tl.store(DV + b * 2 * H + cols, dvf, mask=m)
        tl.store(DV + b * 2 * H + H + cols, dvr, mask=m)
        tl.store(DC + b * H + cols, dc, mask=m)


def _diag_block(H):
    return min(1024, triton.next_power_of_2(H))


class _IndyLSTMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gx, u, h0, c0, relu_gates):
        gx = gx.contiguous()
        B, T, H4 = gx.shape
        H = H4 // 4
        kw = dict(device=gx.device, dtype=torch.float32)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        cf = torch.empty(B, T + 1, H, **kw); cf[:, 0] = c0.float()
        u32 = u.float().contiguous()
        blk = _diag_block(H)
        _indylstm_fwd_kernel[(B, triton.cdiv(H, blk))](gx, u32, hf, cf, T, H, BLOCK=blk, RELU_GATES=relu_gates)
        ctx.save_for_backward(gx, u32, hf, cf)
        ctx.meta = (relu_gates, blk, h0.dtype)
        return hf[:, 1:], cf[:, -1]

    @staticmethod
    def backward(ctx, dy, dc_last):
        gx, u32, hf, cf = ctx.saved_tensors
        relu_gates, blk, s_dtype = ctx.meta
        B, T, H4 = gx.shape
        H = H4 // 4
        kw = dict(device=gx.device, dtype=torch.float32)
        dgx, du = torch.empty(B, T, 4 * H, **kw), torch.empty(B, 4 * H, **kw)
        dh = torch.zeros(B, H, **kw)
        dc = torch.zeros(B, H, **kw) if dc_last is None else _contig(dc_last.float()).clone()
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        _indylstm_bwd_kernel[(B, triton.cdiv(H, blk))](dy, gx, u32, hf, cf, dgx, du, dh, dc, T, H,
                                                       BLOCK=blk, RELU_GATES=relu_gates)
        return dgx.to(gx.dtype), du.sum(0), dh.to(s_dtype), dc.to(s_dtype), None


def indylstm_scan(gx, u, h0, c0, relu_gates=False):
    """IndyLSTM (Gonnet & Deselaers, 2019): an LSTM whose recurrent weights are
    diagonal.  ``gx``: (B, T, 4H) input projections with bias, gate order
    i, f, g, o; ``u``: (4H,).  Returns (h_1..h_T, c_T)."""
    return _IndyLSTMScan.apply(gx, u, h0, c0, bool(relu_gates))


def indylstm_reference(gx, u, h0, c0, relu_gates=False):
    gate = torch.relu if relu_gates else torch.sigmoid
    u_i, u_f, u_g, u_o = u.chunk(4)
    h, c, hs = h0.float(), c0.float(), []
    for t in range(gx.size(1)):
        g_i, g_f, g_g, g_o = gx[:, t].float().chunk(4, -1)
        c = gate(g_f + u_f * h) * c + gate(g_i + u_i * h) * torch.tanh(g_g + u_g * h)
        h = gate(g_o + u_o * h) * torch.tanh(c)
        hs.append(h)
    return torch.stack(hs, 1), c


class _UnICORNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vx, w, dt, y0, z0, alpha):
        vx = vx.contiguous()
        B, T, H = vx.shape
        kw = dict(device=vx.device, dtype=torch.float32)
        yf = torch.empty(B, T + 1, H, **kw); yf[:, 0] = y0.float()
        zf = torch.empty(B, T + 1, H, **kw); zf[:, 0] = z0.float()
        w32, dt32 = w.float().contiguous(), dt.float().contiguous()
        blk = _diag_block(H)
        _unicornn_fwd_kernel[(B, triton.cdiv(H, blk))](vx, w32, dt32, yf, zf, T, H, float(alpha), BLOCK=blk)
        ctx.save_for_backward(vx, w32, dt32, yf, zf)
        ctx.meta = (float(alpha), blk, y0.dtype)
        return yf[:, 1:], zf[:, -1]

    @staticmethod
    def backward(ctx, dy_out, dz_last):
        vx, w32, dt32, yf, zf = ctx.saved_tensors
        alpha, blk, s_dtype = ctx.meta
        B, T, H = vx.shape
        kw = dict(device=vx.device, dtype=torch.float32)
        dvx = torch.empty(B, T, H, **kw)
        dw, ddt = torch.empty(B, H, **kw), torch.empty(B, H, **kw)
        dy = torch.zeros(B, H, **kw)
        dz = torch.zeros(B, H, **kw) if dz_last is None else _contig(dz_last.float()).clone()
        dy_out = torch.zeros(B, T, H, **kw) if dy_out is None else _contig(dy_out)
        _unicornn_bwd_kernel[(B, triton.cdiv(H, blk))](dy_out, vx, w32, dt32, yf, zf, dvx, dw, ddt, dy, dz,
                                                       T, H, alpha, BLOCK=blk)
        return dvx.to(vx.dtype), dw.sum(0), ddt.sum(0), dy.to(s_dtype), dz.to(s_dtype), None


def unicornn_scan(vx, w, dt, y0, z0, alpha):
    """UnICORNN (Rusch & Mishra, 2021) symplectic-Euler oscillators:
    z_t = z_{t-1} - dt (tanh(w y_{t-1} + vx_t) + alpha y_{t-1}), y_t = y_{t-1} + dt z_t,
    with ``dt`` the per-neuron effective step (dt * sigmoid(c)).
    Returns (y_1..y_T, z_T)."""
    return _UnICORNNScan.apply(vx, w, dt, y0, z0, alpha)


def unicornn_reference(vx, w, dt, y0, z0, alpha):
    y, z, ys = y0.float(), z0.float(), []
    for t in range(vx.size(1)):
        z = z - dt * (torch.tanh(w * y + vx[:, t].float()) + alpha * y)
        y = y + dt * z
        ys.append(y)
    return torch.stack(ys, 1), z


class _SRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u3, x, vf, vr, c0, alpha):
        u3, x = u3.contiguous(), x.contiguous()
        B, T, H = x.shape
        kw = dict(device=x.device, dtype=torch.float32)
        cf = torch.empty(B, T + 1, H, **kw); cf[:, 0] = c0.float()
        h = torch.empty(B, T, H, **kw)
        vf32, vr32 = vf.float().contiguous(), vr.float().contiguous()
        blk = _diag_block(H)
        _sru_fwd_kernel[(B, triton.cdiv(H, blk))](u3, x, vf32, vr32, cf, h, T, H, float(alpha), BLOCK=blk)
        ctx.save_for_backward(u3, x, vf32, vr32, cf)
        ctx.meta = (float(alpha), blk, c0.dtype)
        return h, cf[:, -1]

    @staticmethod
    def backward(ctx, dh, dc_last):
        u3, x, vf32, vr32, cf = ctx.saved_tensors
        alpha, blk, s_dtype = ctx.meta
        B, T, H = x.shape
        kw = dict(device=x.device, dtype=torch.float32)
        du3, dx = torch.empty(B, T, 3 * H, **kw), torch.empty(B, T, H, **kw)
        dv = torch.empty(B, 2 * H, **kw)
        dc = torch.zeros(B, H, **kw) if dc_last is None else _contig(dc_last.float()).clone()
        dh = torch.zeros(B, T, H, **kw) if dh is None else _contig(dh)
        _sru_bwd_kernel[(B, triton.cdiv(H, blk))](dh, u3, x, vf32, vr32, cf, du3, dx, dv, dc, T, H, alpha,
                                                  BLOCK=blk)
        dv = dv.sum(0)
        return du3.to(u3.dtype), dx.to(x.dtype), dv[:H], dv[H:], dc.to(s_dtype), None


def sru_scan(u3, x, vf, vr, c0, alpha):
    """SRU (Lei et al., 2018) light recurrence and scaled highway:
    f = sigmoid(u_f + vf * c_{t-1}), r = sigmoid(u_r + vr * c_{t-1}),
    c = f c_{t-1} + (1 - f) u_x, h = r c + (1 - r) alpha x.
    ``u3``: (B, T, 3H) = [W x, W_f x + b_f, W_r x + b_r].  Returns (h, c_T)."""
    return _SRUScan.apply(u3, x, vf, vr, c0, alpha)


def sru_reference(u3, x, vf, vr, c0, alpha):
    c, hs = c0.float(), []
    for t in range(x.size(1)):
        u_x, u_f, u_r = u3[:, t].float().chunk(3, -1)
        f, r = torch.sigmoid(u_f + vf * c), torch.sigmoid(u_r + vr * c)
        c = f * c + (1 - f) * u_x
        hs.append(r * c + (1 - r) * alpha * x[:, t].float())
    return torch.stack(hs, 1), c


# ================================================ dense-recurrence building block
# Dense cells run one program per tile of BB batch rows for the whole
# sequence.  Each step is a short chain of row-tile matmuls (through tl.dot)
# and elementwise passes, separated by barriers, with intermediates in small
# global scratch buffers owned by the program.
if HAS_TRITON:

    @triton.jit
    def _mm_rows(A, lda, W, ldw, OUT, ldo, brow, bmask, K, N,
                 ACC: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PREC: tl.constexpr):
        """OUT[brow, :N] (+)= A[brow, :K] @ W[:K, :N]  (row-major, fp32)."""
        for n0 in range(0, N, BN):
            ns = n0 + tl.arange(0, BN)
            nm = ns < N
            acc = tl.zeros([BB, BN], dtype=tl.float32)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                km = ks < K
                a = tl.load(A + brow[:, None] * lda + ks[None, :], mask=bmask[:, None] & km[None, :], other=0.0)
                w = tl.load(W + ks[:, None] * ldw + ns[None, :], mask=km[:, None] & nm[None, :], other=0.0)
                acc = tl.dot(a, w, acc, input_precision=PREC)
            out = OUT + brow[:, None] * ldo + ns[None, :]
            om = bmask[:, None] & nm[None, :]
            if ACC:
                acc += tl.load(out, mask=om, other=0.0)
            tl.store(out, acc, mask=om)

    # ------------------------------------------------ Intersection RNN (+RNN)
    @triton.jit
    def _irnn_fwd_kernel(PX, X, UT, HF, Y, A, R, B, T, H,
                         BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            # R = h_{t-1} @ [W_yh; W_hh; W_gyh; W_ghh]^T  -> (B, 4H)
            _mm_rows(HF + t * H, (T + 1) * H, UT, 4 * H, R, 4 * H, brow, bmask, H, 4 * H,
                     False, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                r = R + brow[:, None] * 4 * H + ns[None, :]
                px = PX + (brow[:, None] * T + t) * 4 * H + ns[None, :]
                p_y = tl.load(r, mask=tile, other=0.0) + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                p_h = tl.load(r + H, mask=tile, other=0.0) + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                p_gy = tl.load(r + 2 * H, mask=tile, other=0.0) + tl.load(px + 2 * H, mask=tile, other=0.0).to(tl.float32)
                p_gh = tl.load(r + 3 * H, mask=tile, other=0.0) + tl.load(px + 3 * H, mask=tile, other=0.0).to(tl.float32)
                a = A + (brow[:, None] * T + t) * 4 * H + ns[None, :]
                tl.store(a, p_y, mask=tile)
                tl.store(a + H, p_h, mask=tile)
                tl.store(a + 2 * H, p_gy, mask=tile)
                tl.store(a + 3 * H, p_gh, mask=tile)
                x = tl.load(X + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                gy = tl.sigmoid(p_gy)
                gh = tl.sigmoid(p_gh)
                tl.store(Y + (brow[:, None] * T + t) * H + ns[None, :],
                         gy * x + (1.0 - gy) * tl.maximum(p_y, 0.0), mask=tile)
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         gh * h_prev + (1.0 - gh) * _tanh(p_h), mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _irnn_bwd_kernel(DY, DH, X, U, HF, A, DA, DX, CARRY, B, T, H,
                         BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                a = A + (brow[:, None] * T + t) * 4 * H + ns[None, :]
                p_y = tl.load(a, mask=tile, other=0.0)
                p_h = tl.load(a + H, mask=tile, other=0.0)
                gy = tl.sigmoid(tl.load(a + 2 * H, mask=tile, other=0.0))
                gh = tl.sigmoid(tl.load(a + 3 * H, mask=tile, other=0.0))
                x = tl.load(X + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                dy = tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                dh = tl.load(CARRY + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                if t == T - 1:
                    dh += tl.load(DH + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                y_in = tl.maximum(p_y, 0.0)
                h_in = _tanh(p_h)
                da = DA + (brow[:, None] * T + t) * 4 * H + ns[None, :]
                tl.store(da, tl.where(p_y > 0.0, dy * (1.0 - gy), 0.0), mask=tile)
                tl.store(da + H, dh * (1.0 - gh) * (1.0 - h_in * h_in), mask=tile)
                tl.store(da + 2 * H, dy * (x - y_in) * gy * (1.0 - gy), mask=tile)
                tl.store(da + 3 * H, dh * (h_prev - h_in) * gh * (1.0 - gh), mask=tile)
                tl.store(DX + (brow[:, None] * T + t) * H + ns[None, :], dy * gy, mask=tile)
                tl.store(CARRY + brow[:, None] * H + ns[None, :], dh * gh, mask=tile)
            tl.debug_barrier()
            _mm_rows(DA + t * 4 * H, T * 4 * H, U, H, CARRY, H, brow, bmask, 4 * H, H,
                     True, BB, BN, BK, PREC)
            tl.debug_barrier()

    # ------------------------------------------------ Light Recurrent Unit
    @triton.jit
    def _lru_fwd_kernel(PF, CAND, UT, HF, A, R, B, T, H,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            _mm_rows(HF + t * H, (T + 1) * H, UT, H, R, H, brow, bmask, H, H, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                idx = (brow[:, None] * T + t) * H + ns[None, :]
                p = tl.load(R + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(PF + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(A + idx, p, mask=tile)
                f = tl.sigmoid(p)
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                cand = tl.load(CAND + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         (1.0 - f) * h_prev + f * cand, mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _lru_bwd_kernel(DY, CAND, U, HF, A, DP, DCAND, CARRY, B, T, H,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                idx = (brow[:, None] * T + t) * H + ns[None, :]
                f = tl.sigmoid(tl.load(A + idx, mask=tile, other=0.0))
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                cand = tl.load(CAND + idx, mask=tile, other=0.0).to(tl.float32)
                dh = tl.load(CARRY + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(DP + idx, dh * (cand - h_prev) * f * (1.0 - f), mask=tile)
                tl.store(DCAND + idx, dh * f, mask=tile)
                tl.store(CARRY + brow[:, None] * H + ns[None, :], dh * (1.0 - f), mask=tile)
            tl.debug_barrier()
            _mm_rows(DP + t * H, T * H, U, H, CARRY, H, brow, bmask, H, H, True, BB, BN, BK, PREC)
            tl.debug_barrier()

    # ------------------------------------------------ expRNN (orthogonal, modReLU)
    @triton.jit
    def _exprnn_fwd_kernel(PX, WT, BIAS, HF, A, R, B, T, H,
                           BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            _mm_rows(HF + t * H, (T + 1) * H, WT, H, R, H, brow, bmask, H, H, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                idx = (brow[:, None] * T + t) * H + ns[None, :]
                z = tl.load(R + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(PX + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(A + idx, z, mask=tile)
                mag = tl.abs(z) + tl.load(BIAS + ns, mask=nm, other=0.0)[None, :]
                sign = tl.where(z >= 0.0, 1.0, -1.0)
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         sign * tl.maximum(mag, 0.0), mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _exprnn_bwd_kernel(DY, W, BIAS, A, DZ, DB, CARRY, B, T, H,
                           BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                idx = (brow[:, None] * T + t) * H + ns[None, :]
                z = tl.load(A + idx, mask=tile, other=0.0)
                active = (tl.abs(z) + tl.load(BIAS + ns, mask=nm, other=0.0)[None, :]) > 0.0
                dh = tl.load(CARRY + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + idx, mask=tile, other=0.0).to(tl.float32)
                g = tl.where(active, dh, 0.0)
                tl.store(DZ + idx, g, mask=tile)
                db = DB + brow[:, None] * H + ns[None, :]
                tl.store(db, tl.load(db, mask=tile, other=0.0) + g * tl.where(z >= 0.0, 1.0, -1.0), mask=tile)
            tl.debug_barrier()
            _mm_rows(DZ + t * H, T * H, W, H, CARRY, H, brow, bmask, H, H, False, BB, BN, BK, PREC)
            tl.debug_barrier()


def _dense_tiles(width):
    hp2 = triton.next_power_of_2(max(width, 16))
    return 16, min(64, hp2), min(64, hp2)


class _IRNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, x, u, h0):
        px, x = px.contiguous(), x.contiguous()
        B, T, H = x.shape
        kw = dict(device=x.device, dtype=torch.float32)
        u32 = u.float().contiguous()
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        y, a, r = torch.empty(B, T, H, **kw), torch.empty(B, T, 4 * H, **kw), torch.empty(B, 4 * H, **kw)
        bb, bn, bk = _dense_tiles(H)
        prec = _dot_precision()
        _irnn_fwd_kernel[(triton.cdiv(B, bb),)](px, x, u32.t().contiguous(), hf, y, a, r, B, T, H,
                                                BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(x, u32, hf, a)
        ctx.meta = (prec, px.dtype, h0.dtype)
        return y, hf[:, -1]

    @staticmethod
    def backward(ctx, dy, dh_last):
        x, u32, hf, a = ctx.saved_tensors
        prec, px_dtype, s_dtype = ctx.meta
        B, T, H = x.shape
        kw = dict(device=x.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H, **kw) if dy is None else dy.contiguous()
        dh = torch.zeros(B, H, **kw) if dh_last is None else dh_last.float().contiguous()
        da, dx, carry = torch.empty(B, T, 4 * H, **kw), torch.empty(B, T, H, **kw), torch.zeros(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        _irnn_bwd_kernel[(triton.cdiv(B, bb),)](dy, dh, x, u32, hf, a, da, dx, carry, B, T, H,
                                                BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        du = da.reshape(-1, 4 * H).t() @ hf[:, :-1].reshape(-1, H)
        return da.to(px_dtype), dx.to(x.dtype), du, carry.to(s_dtype)


def irnn_scan(px, x, u, h0):
    """Intersection RNN (+RNN; Collins et al., 2017) over a sequence.
    ``px``: (B, T, 4H) input-side pre-activations [y_in, h_in, g_y, g_h] with
    biases; ``x``: (B, T, H) depth input; ``u``: (4H, H) recurrent weights.
    y = g_y x + (1 - g_y) ReLU(.), h = g_h h_{t-1} + (1 - g_h) tanh(.).
    Returns (y_1..y_T, h_T)."""
    return _IRNNScan.apply(px, x, u, h0)


def irnn_reference(px, x, u, h0):
    h, ys = h0.float(), []
    for t in range(x.size(1)):
        p = px[:, t].float() + h @ u.t()
        p_y, p_h, p_gy, p_gh = p.chunk(4, -1)
        gy, gh = torch.sigmoid(p_gy), torch.sigmoid(p_gh)
        ys.append(gy * x[:, t] + (1 - gy) * torch.relu(p_y))
        h = gh * h + (1 - gh) * torch.tanh(p_h)
    return torch.stack(ys, 1), h


class _LRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pf, cand, u, h0):
        pf, cand = pf.contiguous(), cand.contiguous()
        B, T, H = pf.shape
        kw = dict(device=pf.device, dtype=torch.float32)
        u32 = u.float().contiguous()
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        a, r = torch.empty(B, T, H, **kw), torch.empty(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        prec = _dot_precision()
        _lru_fwd_kernel[(triton.cdiv(B, bb),)](pf, cand, u32.t().contiguous(), hf, a, r, B, T, H,
                                               BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(cand, u32, hf, a)
        ctx.meta = (prec, pf.dtype, h0.dtype)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        cand, u32, hf, a = ctx.saved_tensors
        prec, p_dtype, s_dtype = ctx.meta
        B, T, H = cand.shape
        kw = dict(device=cand.device, dtype=torch.float32)
        dp, dcand, carry = torch.empty(B, T, H, **kw), torch.empty(B, T, H, **kw), torch.zeros(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        _lru_bwd_kernel[(triton.cdiv(B, bb),)](dy.contiguous(), cand, u32, hf, a, dp, dcand, carry, B, T, H,
                                               BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        du = dp.reshape(-1, H).t() @ hf[:, :-1].reshape(-1, H)
        return dp.to(p_dtype), dcand.to(cand.dtype), du, carry.to(s_dtype)


def lru_scan(pf, cand, u, h0):
    """Light Recurrent Unit (Electronics 2024, 13, 3204):
    f = sigmoid(U_f h_{t-1} + pf_t), h = (1 - f) h_{t-1} + f cand_t.
    ``pf``: W_f x + b_f; ``cand``: tanh(W_h x) (or the previous layer's
    output in the highway stacking).  Returns h_1..h_T."""
    return _LRUScan.apply(pf, cand, u, h0)


def lru_reference(pf, cand, u, h0):
    h, hs = h0.float(), []
    for t in range(pf.size(1)):
        f = torch.sigmoid(pf[:, t].float() + h @ u.t())
        h = (1 - f) * h + f * cand[:, t].float()
        hs.append(h)
    return torch.stack(hs, 1)


# ================================================ UGRNN (update-gate RNN)
if HAS_TRITON:

    @triton.jit
    def _ugrnn_fwd_kernel(PX, UT, HF, A, R, B, T, H,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            # R = h_{t-1} @ [U_c; U_g]^T  -> (B, 2H)
            _mm_rows(HF + t * H, (T + 1) * H, UT, 2 * H, R, 2 * H, brow, bmask, H, 2 * H,
                     False, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                r = R + brow[:, None] * 2 * H + ns[None, :]
                px = PX + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                p_c = tl.load(r, mask=tile, other=0.0) + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                p_g = tl.load(r + H, mask=tile, other=0.0) + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                a = A + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                tl.store(a, p_c, mask=tile)
                tl.store(a + H, p_g, mask=tile)
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                g = tl.sigmoid(p_g)
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         g * h_prev + (1.0 - g) * _tanh(p_c), mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _ugrnn_bwd_kernel(DY, U, HF, A, DA, CARRY, B, T, H,
                          BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                a = A + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                c = _tanh(tl.load(a, mask=tile, other=0.0))
                g = tl.sigmoid(tl.load(a + H, mask=tile, other=0.0))
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                dh = tl.load(CARRY + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                da = DA + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                tl.store(da, dh * (1.0 - g) * (1.0 - c * c), mask=tile)
                tl.store(da + H, dh * (h_prev - c) * g * (1.0 - g), mask=tile)
                tl.store(CARRY + brow[:, None] * H + ns[None, :], dh * g, mask=tile)
            tl.debug_barrier()
            _mm_rows(DA + t * 2 * H, T * 2 * H, U, H, CARRY, H, brow, bmask, 2 * H, H,
                     True, BB, BN, BK, PREC)
            tl.debug_barrier()


class _UGRNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, u, h0):
        px = px.contiguous()
        B, T, H2 = px.shape
        H = H2 // 2
        kw = dict(device=px.device, dtype=torch.float32)
        u32 = u.float().contiguous()
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        a, r = torch.empty(B, T, 2 * H, **kw), torch.empty(B, 2 * H, **kw)
        bb, bn, bk = _dense_tiles(H)
        prec = _dot_precision()
        _ugrnn_fwd_kernel[(triton.cdiv(B, bb),)](px, u32.t().contiguous(), hf, a, r, B, T, H,
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(u32, hf, a)
        ctx.meta = (prec, px.dtype, h0.dtype)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        u32, hf, a = ctx.saved_tensors
        prec, px_dtype, s_dtype = ctx.meta
        B, T, H2 = a.shape
        H = H2 // 2
        kw = dict(device=a.device, dtype=torch.float32)
        da, carry = torch.empty(B, T, 2 * H, **kw), torch.zeros(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        _ugrnn_bwd_kernel[(triton.cdiv(B, bb),)](dy.contiguous(), u32, hf, a, da, carry, B, T, H,
                                                 BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        du = da.reshape(-1, 2 * H).t() @ hf[:, :-1].reshape(-1, H)
        return da.to(px_dtype), du, carry.to(s_dtype)


def ugrnn_scan(px, u, h0):
    """UGRNN (update-gate RNN; Collins, Sohl-Dickstein & Sussillo, 2017).
    ``px``: (B, T, 2H) input-side pre-activations [c_in, g_in] with biases;
    ``u``: (2H, H) recurrent weights [U_c; U_g].
    c = tanh(.), g = sigmoid(.), h = g h_{t-1} + (1 - g) c.  Returns h_1..h_T."""
    return _UGRNNScan.apply(px, u, h0)


def ugrnn_reference(px, u, h0):
    h, hs = h0.float(), []
    for t in range(px.size(1)):
        p_c, p_g = (px[:, t].float() + h @ u.t()).chunk(2, -1)
        g = torch.sigmoid(p_g)
        h = g * h + (1 - g) * torch.tanh(p_c)
        hs.append(h)
    return torch.stack(hs, 1)


class _ExpRNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, w, bias, h0):
        px = px.contiguous()
        B, T, H = px.shape
        kw = dict(device=px.device, dtype=torch.float32)
        w32, b32 = w.float().contiguous(), bias.float().contiguous()
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        a, r = torch.empty(B, T, H, **kw), torch.empty(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        prec = _dot_precision()
        _exprnn_fwd_kernel[(triton.cdiv(B, bb),)](px, w32.t().contiguous(), b32, hf, a, r, B, T, H,
                                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(w32, b32, hf, a)
        ctx.meta = (prec, px.dtype, h0.dtype)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        w32, b32, hf, a = ctx.saved_tensors
        prec, p_dtype, s_dtype = ctx.meta
        B, T1, H = hf.shape
        T = T1 - 1
        kw = dict(device=hf.device, dtype=torch.float32)
        dz, db, carry = torch.empty(B, T, H, **kw), torch.zeros(B, H, **kw), torch.zeros(B, H, **kw)
        bb, bn, bk = _dense_tiles(H)
        _exprnn_bwd_kernel[(triton.cdiv(B, bb),)](_contig(dy), w32, b32, a, dz, db, carry, B, T, H,
                                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        dw = dz.reshape(-1, H).t() @ hf[:, :-1].reshape(-1, H)
        return dz.to(p_dtype), dw, db.sum(0), carry.to(s_dtype)


def exprnn_scan(px, w, bias, h0):
    """expRNN recurrence (Lezcano-Casado & Martinez-Rubio, 2019):
    h = modReLU(W h_{t-1} + px_t), modReLU(z) = sign(z) ReLU(|z| + b), with W
    orthogonal (the caller builds it as a matrix exponential).  Returns h_1..h_T."""
    return _ExpRNNScan.apply(px, w, bias, h0)


def exprnn_reference(px, w, bias, h0):
    h, hs = h0.float(), []
    for t in range(px.size(1)):
        z = px[:, t].float() + h @ w.t()
        h = torch.sign(z) * torch.relu(z.abs() + bias)
        hs.append(h)
    return torch.stack(hs, 1)


# ================================================ RRU (Residual Recurrent Unit)
if HAS_TRITON:

    @triton.jit
    def _rru_fwd_kernel(PXJ, WHT, WCT, BC, MASK, SV, ZV, HF, A, RSTD, D, CS, R1, RC, B, T, H, G, eps,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for t in range(T):
            _mm_rows(HF + t * H, (T + 1) * H, WHT, G, R1, G, brow, bmask, H, G, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            # j = ReLU(RMSNorm(W_h h + W_x x + b_j)), then the dropout mask.
            ss = tl.zeros([BB], dtype=tl.float32)
            for n0 in range(0, G, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < G)[None, :]
                idx = (brow[:, None] * T + t) * G + ns[None, :]
                a = tl.load(R1 + brow[:, None] * G + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(PXJ + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(A + idx, a, mask=tile)
                ss += tl.sum(a * a, axis=1)
            rstd = 1.0 / tl.sqrt(ss / G + eps)
            tl.store(RSTD + brow * T + t, rstd, mask=bmask)
            for n0 in range(0, G, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < G)[None, :]
                idx = (brow[:, None] * T + t) * G + ns[None, :]
                a = tl.load(A + idx, mask=tile, other=0.0)
                d = tl.maximum(a * rstd[:, None], 0.0) * tl.load(MASK + idx, mask=tile, other=0.0).to(tl.float32)
                tl.store(D + idx, d, mask=tile)
            tl.debug_barrier()
            _mm_rows(D + t * G, T * G, WCT, H, RC, H, brow, bmask, G, H, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            # h = sigmoid(S) h + Z c,  c = W_c d + b_c
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                c = tl.load(RC + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(BC + ns, mask=nm, other=0.0)[None, :]
                tl.store(CS + (brow[:, None] * T + t) * H + ns[None, :], c, mask=tile)
                h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         tl.load(SV + ns, mask=nm, other=0.0)[None, :] * h_prev
                         + tl.load(ZV + ns, mask=nm, other=0.0)[None, :] * c, mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _rru_bwd_kernel(DDO, WH, WC, MASK, SV, ZV, A, RSTD, DC, DHS, DA, RD, CARRY, B, T, H, G,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                dh = tl.load(CARRY + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                idx = (brow[:, None] * T + t) * H + ns[None, :]
                tl.store(DHS + idx, dh, mask=tile)
                tl.store(DC + idx, dh * tl.load(ZV + ns, mask=nm, other=0.0)[None, :], mask=tile)
                tl.store(CARRY + brow[:, None] * H + ns[None, :],
                         dh * tl.load(SV + ns, mask=nm, other=0.0)[None, :], mask=tile)
            tl.debug_barrier()
            # dd = dc @ W_c + (gradient of d from the output projection)
            _mm_rows(DC + t * H, T * H, WC, G, RD, G, brow, bmask, H, G, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            rstd = tl.load(RSTD + brow * T + t, mask=bmask, other=0.0)
            dot = tl.zeros([BB], dtype=tl.float32)
            for n0 in range(0, G, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < G)[None, :]
                idx = (brow[:, None] * T + t) * G + ns[None, :]
                a = tl.load(A + idx, mask=tile, other=0.0)
                dd = (tl.load(RD + brow[:, None] * G + ns[None, :], mask=tile, other=0.0)
                      + tl.load(DDO + idx, mask=tile, other=0.0).to(tl.float32)) \
                    * tl.load(MASK + idx, mask=tile, other=0.0).to(tl.float32)
                dy = tl.where(a * rstd[:, None] > 0.0, dd, 0.0)
                tl.store(RD + brow[:, None] * G + ns[None, :], dy, mask=tile)
                dot += tl.sum(dy * a, axis=1)
            coef = rstd * rstd * rstd * dot / G
            for n0 in range(0, G, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < G)[None, :]
                idx = (brow[:, None] * T + t) * G + ns[None, :]
                a = tl.load(A + idx, mask=tile, other=0.0)
                dy = tl.load(RD + brow[:, None] * G + ns[None, :], mask=tile, other=0.0)
                tl.store(DA + idx, rstd[:, None] * dy - coef[:, None] * a, mask=tile)
            tl.debug_barrier()
            _mm_rows(DA + t * G, T * G, WH, H, CARRY, H, brow, bmask, G, H, True, BB, BN, BK, PREC)
            tl.debug_barrier()


class _RRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pxj, w_h, w_c, b_c, mask, s_vec, z_vec, h0, eps):
        pxj, mask = pxj.contiguous(), mask.contiguous()
        B, T, G = pxj.shape
        H = w_c.shape[0]
        kw = dict(device=pxj.device, dtype=torch.float32)
        wh, wc = w_h.float().contiguous(), w_c.float().contiguous()   # (G, H), (H, G)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        a, d, cs = torch.empty(B, T, G, **kw), torch.empty(B, T, G, **kw), torch.empty(B, T, H, **kw)
        rstd = torch.empty(B, T, **kw)
        r1, rc = torch.empty(B, G, **kw), torch.empty(B, H, **kw)
        bb, bn, bk = _dense_tiles(max(H, G))
        prec = _dot_precision()
        _rru_fwd_kernel[(triton.cdiv(B, bb),)](pxj, wh.t().contiguous(), wc.t().contiguous(), b_c.float().contiguous(),
                                               mask, s_vec.float().contiguous(), z_vec.float().contiguous(),
                                               hf, a, rstd, d, cs, r1, rc, B, T, H, G, float(eps),
                                               BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(wh, wc, mask, s_vec.float().contiguous(), z_vec.float().contiguous(), hf, a, rstd, d, cs)
        ctx.meta = (prec, pxj.dtype, h0.dtype)
        return d, hf[:, -1]

    @staticmethod
    def backward(ctx, dd_out, dh_last):
        wh, wc, mask, sv, zv, hf, a, rstd, d, cs = ctx.saved_tensors
        prec, p_dtype, s_dtype = ctx.meta
        B, T, G = a.shape
        H = wc.shape[0]
        kw = dict(device=a.device, dtype=torch.float32)
        dd_out = torch.zeros(B, T, G, **kw) if dd_out is None else dd_out.contiguous()
        carry = torch.zeros(B, H, **kw) if dh_last is None else dh_last.float().contiguous().clone()
        dc, dhs, da = torch.empty(B, T, H, **kw), torch.empty(B, T, H, **kw), torch.empty(B, T, G, **kw)
        rd = torch.empty(B, G, **kw)
        bb, bn, bk = _dense_tiles(max(H, G))
        _rru_bwd_kernel[(triton.cdiv(B, bb),)](dd_out, wh, wc, mask, sv, zv, a, rstd, dc, dhs, da, rd, carry,
                                               B, T, H, G, BB=bb, BN=bn, BK=bk, PREC=prec,
                                               num_warps=4, num_stages=1)
        h_prev = hf[:, :-1].reshape(-1, H)
        d_wh = da.reshape(-1, G).t() @ h_prev
        d_wc = dc.reshape(-1, H).t() @ d.reshape(-1, G)
        d_s = (dhs * hf[:, :-1]).sum((0, 1))
        d_z = (dhs * cs).sum((0, 1))
        return (da.to(p_dtype), d_wh, d_wc, dc.sum((0, 1)), None, d_s, d_z, carry.to(s_dtype), None)


def rru_scan(pxj, w_h, w_c, b_c, mask, s_vec, z_vec, h0, eps=1e-6):
    """RRU recurrence (Zakovskis et al., 2021; official RRUCell):
    d = ReLU(RMSNorm(W_h h_{t-1} + pxj_t)) * mask,  c = W_c d + b_c,
    h = s * h_{t-1} + z * c  (s = sigmoid(S), z = Z).
    Returns (d_1..d_T, h_T); the cell output is W_o d + b_o."""
    return _RRUScan.apply(pxj, w_h, w_c, b_c, mask, s_vec, z_vec, h0, eps)


def rru_reference(pxj, w_h, w_c, b_c, mask, s_vec, z_vec, h0, eps=1e-6):
    h, ds, hs = h0.float(), [], []
    for t in range(pxj.size(1)):
        a = pxj[:, t].float() + h @ w_h.t()
        d = torch.relu(a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + eps)) * mask[:, t]
        h = s_vec * h + z_vec * (d @ w_c.t() + b_c)
        ds.append(d)
    return torch.stack(ds, 1), h


# ================================================ Mogrifier LSTM / GRU
# Melis et al. (2020): before each step the input and state gate each other
# for ``rounds`` rounds (x <- 2 sigmoid(Q h) x on odd rounds, h <- 2 sigmoid(R x) h
# on even ones), then a standard LSTM (MODE 0) or GRU (MODE 1) step runs on
# the mogrified pair.  All round intermediates are kept for the backward pass.
if HAS_TRITON:

    @triton.jit
    def _mog_fwd_kernel(X, QT, RT, WT, UT, BI, BH, HF, CF, XR, HR, GQ, GR, AG, GH, S, S2, S3,
                        B, T, H, ROUNDS, NQ, NR,
                        MODE: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 4 * H if MODE == 0 else 3 * H
        lxr = T * (NQ + 1) * H
        lhr = T * (NR + 1) * H
        for t in range(T):
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                tl.store(XR + (brow[:, None] * T + t) * (NQ + 1) * H + ns[None, :],
                         tl.load(X + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32),
                         mask=tile)
                tl.store(HR + (brow[:, None] * T + t) * (NR + 1) * H + ns[None, :],
                         tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0),
                         mask=tile)
            tl.debug_barrier()
            jx = 0
            jh = 0
            for i in range(ROUNDS):
                if i % 2 == 0:
                    _mm_rows(HR + (t * (NR + 1) + jh) * H, lhr, QT + jx * H * H, H, S, H, brow, bmask,
                             H, H, False, BB, BN, BK, PREC)
                    tl.debug_barrier()
                    for n0 in range(0, H, BN):
                        ns = n0 + tl.arange(0, BN)
                        tile = bmask[:, None] & (ns < H)[None, :]
                        g = 2.0 * tl.sigmoid(tl.load(S + brow[:, None] * H + ns[None, :], mask=tile, other=0.0))
                        base = XR + ((brow[:, None] * T + t) * (NQ + 1) + jx) * H + ns[None, :]
                        tl.store(base + H, g * tl.load(base, mask=tile, other=0.0), mask=tile)
                        tl.store(GQ + ((brow[:, None] * T + t) * NQ + jx) * H + ns[None, :], g, mask=tile)
                    jx += 1
                else:
                    _mm_rows(XR + (t * (NQ + 1) + jx) * H, lxr, RT + jh * H * H, H, S, H, brow, bmask,
                             H, H, False, BB, BN, BK, PREC)
                    tl.debug_barrier()
                    for n0 in range(0, H, BN):
                        ns = n0 + tl.arange(0, BN)
                        tile = bmask[:, None] & (ns < H)[None, :]
                        g = 2.0 * tl.sigmoid(tl.load(S + brow[:, None] * H + ns[None, :], mask=tile, other=0.0))
                        base = HR + ((brow[:, None] * T + t) * (NR + 1) + jh) * H + ns[None, :]
                        tl.store(base + H, g * tl.load(base, mask=tile, other=0.0), mask=tile)
                        tl.store(GR + ((brow[:, None] * T + t) * NR + jh) * H + ns[None, :], g, mask=tile)
                    jh += 1
                tl.debug_barrier()
            # The recurrent step on the mogrified (x, h).
            _mm_rows(XR + (t * (NQ + 1) + NQ) * H, lxr, WT, G, S2, G, brow, bmask, H, G, False, BB, BN, BK, PREC)
            if MODE == 0:
                _mm_rows(HR + (t * (NR + 1) + NR) * H, lhr, UT, G, S2, G, brow, bmask, H, G, True, BB, BN, BK, PREC)
            else:
                _mm_rows(HR + (t * (NR + 1) + NR) * H, lhr, UT, G, S3, G, brow, bmask, H, G, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                s2 = S2 + brow[:, None] * G + ns[None, :]
                ag = AG + (brow[:, None] * T + t) * G + ns[None, :]
                h_fin = tl.load(HR + ((brow[:, None] * T + t) * (NR + 1) + NR) * H + ns[None, :], mask=tile, other=0.0)
                if MODE == 0:
                    p_i = tl.load(s2, mask=tile, other=0.0) + tl.load(BI + ns, mask=nm, other=0.0)[None, :]
                    p_f = tl.load(s2 + H, mask=tile, other=0.0) + tl.load(BI + H + ns, mask=nm, other=0.0)[None, :]
                    p_g = tl.load(s2 + 2 * H, mask=tile, other=0.0) + tl.load(BI + 2 * H + ns, mask=nm, other=0.0)[None, :]
                    p_o = tl.load(s2 + 3 * H, mask=tile, other=0.0) + tl.load(BI + 3 * H + ns, mask=nm, other=0.0)[None, :]
                    tl.store(ag, p_i, mask=tile)
                    tl.store(ag + H, p_f, mask=tile)
                    tl.store(ag + 2 * H, p_g, mask=tile)
                    tl.store(ag + 3 * H, p_o, mask=tile)
                    c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                    c = tl.sigmoid(p_f) * c_prev + tl.sigmoid(p_i) * _tanh(p_g)
                    tl.store(CF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :], c, mask=tile)
                    h = tl.sigmoid(p_o) * _tanh(c)
                else:
                    s3 = S3 + brow[:, None] * G + ns[None, :]
                    gh = GH + (brow[:, None] * T + t) * G + ns[None, :]
                    i_r = tl.load(s2, mask=tile, other=0.0) + tl.load(BI + ns, mask=nm, other=0.0)[None, :]
                    i_z = tl.load(s2 + H, mask=tile, other=0.0) + tl.load(BI + H + ns, mask=nm, other=0.0)[None, :]
                    i_n = tl.load(s2 + 2 * H, mask=tile, other=0.0) + tl.load(BI + 2 * H + ns, mask=nm, other=0.0)[None, :]
                    h_r = tl.load(s3, mask=tile, other=0.0) + tl.load(BH + ns, mask=nm, other=0.0)[None, :]
                    h_z = tl.load(s3 + H, mask=tile, other=0.0) + tl.load(BH + H + ns, mask=nm, other=0.0)[None, :]
                    h_n = tl.load(s3 + 2 * H, mask=tile, other=0.0) + tl.load(BH + 2 * H + ns, mask=nm, other=0.0)[None, :]
                    tl.store(ag, i_r, mask=tile)
                    tl.store(ag + H, i_z, mask=tile)
                    tl.store(ag + 2 * H, i_n, mask=tile)
                    tl.store(gh, h_r, mask=tile)
                    tl.store(gh + H, h_z, mask=tile)
                    tl.store(gh + 2 * H, h_n, mask=tile)
                    r = tl.sigmoid(i_r + h_r)
                    z = tl.sigmoid(i_z + h_z)
                    n = _tanh(i_n + r * h_n)
                    # Like the LSTM's cell, the carried state is not mogrified
                    # (only the gates see the mogrified h); this keeps h bounded.
                    h_prev = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                    h = (1.0 - z) * n + z * h_prev
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :], h, mask=tile)
            tl.debug_barrier()

    @triton.jit
    def _mog_bwd_kernel(DY, Q, R, W, U, HF, CF, XR, HR, GQ, GR, AG, GH, DAG, DGH, DSQ, DSR, DXO,
                        DXF, DHF, CH, CC, B, T, H, ROUNDS, NQ, NR,
                        MODE: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        PREC: tl.constexpr):
        brow = tl.program_id(0) * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 4 * H if MODE == 0 else 3 * H
        for k in range(T):
            t = T - 1 - k
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                dh = tl.load(CH + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                ag = AG + (brow[:, None] * T + t) * G + ns[None, :]
                dag = DAG + (brow[:, None] * T + t) * G + ns[None, :]
                if MODE == 0:
                    gi = tl.sigmoid(tl.load(ag, mask=tile, other=0.0))
                    gf = tl.sigmoid(tl.load(ag + H, mask=tile, other=0.0))
                    gg = _tanh(tl.load(ag + 2 * H, mask=tile, other=0.0))
                    go = tl.sigmoid(tl.load(ag + 3 * H, mask=tile, other=0.0))
                    c_prev = tl.load(CF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                    tc = _tanh(tl.load(CF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :], mask=tile, other=0.0))
                    dc = tl.load(CC + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) + dh * go * (1.0 - tc * tc)
                    tl.store(dag, dc * gg * gi * (1.0 - gi), mask=tile)
                    tl.store(dag + H, dc * c_prev * gf * (1.0 - gf), mask=tile)
                    tl.store(dag + 2 * H, dc * gi * (1.0 - gg * gg), mask=tile)
                    tl.store(dag + 3 * H, dh * tc * go * (1.0 - go), mask=tile)
                    tl.store(CC + brow[:, None] * H + ns[None, :], dc * gf, mask=tile)
                else:
                    gh = GH + (brow[:, None] * T + t) * G + ns[None, :]
                    dgh = DGH + (brow[:, None] * T + t) * G + ns[None, :]
                    h_fin = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                    h_n = tl.load(gh + 2 * H, mask=tile, other=0.0)
                    r = tl.sigmoid(tl.load(ag, mask=tile, other=0.0) + tl.load(gh, mask=tile, other=0.0))
                    z = tl.sigmoid(tl.load(ag + H, mask=tile, other=0.0) + tl.load(gh + H, mask=tile, other=0.0))
                    n = _tanh(tl.load(ag + 2 * H, mask=tile, other=0.0) + r * h_n)
                    d_npre = dh * (1.0 - z) * (1.0 - n * n)
                    d_r = d_npre * h_n * r * (1.0 - r)
                    d_z = dh * (h_fin - n) * z * (1.0 - z)
                    tl.store(dag, d_r, mask=tile)
                    tl.store(dag + H, d_z, mask=tile)
                    tl.store(dag + 2 * H, d_npre, mask=tile)
                    tl.store(dgh, d_r, mask=tile)
                    tl.store(dgh + H, d_z, mask=tile)
                    tl.store(dgh + 2 * H, d_npre * r, mask=tile)
                    # Direct path to h_{t-1}, bypassing the mogrifier rounds.
                    tl.store(CC + brow[:, None] * H + ns[None, :], dh * z, mask=tile)
            tl.debug_barrier()
            _mm_rows(DAG + t * G, T * G, W, H, DXF, H, brow, bmask, G, H, False, BB, BN, BK, PREC)
            if MODE == 0:
                _mm_rows(DAG + t * G, T * G, U, H, DHF, H, brow, bmask, G, H, False, BB, BN, BK, PREC)
            else:
                _mm_rows(DGH + t * G, T * G, U, H, DHF, H, brow, bmask, G, H, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            jx = NQ
            jh = NR
            for kk in range(ROUNDS):
                i = ROUNDS - 1 - kk
                if i % 2 == 1:
                    jh -= 1
                    for n0 in range(0, H, BN):
                        ns = n0 + tl.arange(0, BN)
                        tile = bmask[:, None] & (ns < H)[None, :]
                        g = tl.load(GR + ((brow[:, None] * T + t) * NR + jh) * H + ns[None, :], mask=tile, other=0.0)
                        h_cur = tl.load(HR + ((brow[:, None] * T + t) * (NR + 1) + jh) * H + ns[None, :], mask=tile, other=0.0)
                        dhn = tl.load(DHF + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                        tl.store(DSR + ((brow[:, None] * T + t) * NR + jh) * H + ns[None, :],
                                 dhn * h_cur * g * (1.0 - 0.5 * g), mask=tile)
                        tl.store(DHF + brow[:, None] * H + ns[None, :], dhn * g, mask=tile)
                    tl.debug_barrier()
                    _mm_rows(DSR + (t * NR + jh) * H, T * NR * H, R + jh * H * H, H, DXF, H, brow, bmask,
                             H, H, True, BB, BN, BK, PREC)
                else:
                    jx -= 1
                    for n0 in range(0, H, BN):
                        ns = n0 + tl.arange(0, BN)
                        tile = bmask[:, None] & (ns < H)[None, :]
                        g = tl.load(GQ + ((brow[:, None] * T + t) * NQ + jx) * H + ns[None, :], mask=tile, other=0.0)
                        x_cur = tl.load(XR + ((brow[:, None] * T + t) * (NQ + 1) + jx) * H + ns[None, :], mask=tile, other=0.0)
                        dxn = tl.load(DXF + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                        tl.store(DSQ + ((brow[:, None] * T + t) * NQ + jx) * H + ns[None, :],
                                 dxn * x_cur * g * (1.0 - 0.5 * g), mask=tile)
                        tl.store(DXF + brow[:, None] * H + ns[None, :], dxn * g, mask=tile)
                    tl.debug_barrier()
                    _mm_rows(DSQ + (t * NQ + jx) * H, T * NQ * H, Q + jx * H * H, H, DHF, H, brow, bmask,
                             H, H, True, BB, BN, BK, PREC)
                tl.debug_barrier()
            for n0 in range(0, H, BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                tl.store(DXO + (brow[:, None] * T + t) * H + ns[None, :],
                         tl.load(DXF + brow[:, None] * H + ns[None, :], mask=tile, other=0.0), mask=tile)
                carry = tl.load(DHF + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                if MODE == 1:
                    carry += tl.load(CC + brow[:, None] * H + ns[None, :], mask=tile, other=0.0)
                tl.store(CH + brow[:, None] * H + ns[None, :], carry, mask=tile)
            tl.debug_barrier()


class _MogrifierScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, r, w, u, b_i, b_h, h0, c0, rounds, mode):
        x = x.contiguous()
        B, T, H = x.shape
        NQ, NR = (rounds + 1) // 2, rounds // 2
        G = (4 if mode == 0 else 3) * H
        kw = dict(device=x.device, dtype=torch.float32)
        f32 = lambda t: t.float().contiguous()
        q32 = f32(q) if NQ else torch.zeros(1, H, H, **kw)
        r32 = f32(r) if NR else torch.zeros(1, H, H, **kw)
        w32, u32 = f32(w), f32(u)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        cf = torch.empty(B, T + 1, H, **kw); cf[:, 0] = c0.float()
        xr, hr = torch.empty(B, T, NQ + 1, H, **kw), torch.empty(B, T, NR + 1, H, **kw)
        gq, gr = torch.empty(B, T, max(NQ, 1), H, **kw), torch.empty(B, T, max(NR, 1), H, **kw)
        ag, gh = torch.empty(B, T, G, **kw), torch.empty(B, T, G, **kw)
        s, s2, s3 = torch.empty(B, H, **kw), torch.empty(B, G, **kw), torch.empty(B, G, **kw)
        bb, bn, bk = _dense_tiles(H)
        prec = _dot_precision()
        _mog_fwd_kernel[(triton.cdiv(B, bb),)](
            x, q32.transpose(1, 2).contiguous(), r32.transpose(1, 2).contiguous(), w32.t().contiguous(),
            u32.t().contiguous(), f32(b_i), f32(b_h), hf, cf, xr, hr, gq, gr, ag, gh, s, s2, s3,
            B, T, H, rounds, NQ, NR, MODE=mode, BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(q32, r32, w32, u32, hf, cf, xr, hr, gq, gr, ag, gh)
        ctx.meta = (rounds, mode, prec, x.dtype, h0.dtype)
        # A GRU has no cell state; its c passes through unchanged.
        return hf[:, 1:], cf[:, -1] if mode == 0 else cf[:, 0]

    @staticmethod
    def backward(ctx, dy, dc_last):
        q32, r32, w32, u32, hf, cf, xr, hr, gq, gr, ag, gh = ctx.saved_tensors
        rounds, mode, prec, x_dtype, s_dtype = ctx.meta
        B, T, NQ1, H = xr.shape
        NQ, NR = NQ1 - 1, hr.shape[2] - 1
        G = ag.shape[2]
        kw = dict(device=hf.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        dag, dgh = torch.empty(B, T, G, **kw), torch.empty(B, T, G, **kw)
        dsq, dsr = torch.zeros(B, T, max(NQ, 1), H, **kw), torch.zeros(B, T, max(NR, 1), H, **kw)
        dxo, dxf, dhf = torch.empty(B, T, H, **kw), torch.empty(B, H, **kw), torch.empty(B, H, **kw)
        ch = torch.zeros(B, H, **kw)
        cc = torch.zeros(B, H, **kw) if dc_last is None else _contig(dc_last.float()).clone()
        bb, bn, bk = _dense_tiles(H)
        _mog_bwd_kernel[(triton.cdiv(B, bb),)](
            dy, q32, r32, w32, u32, hf, cf, xr, hr, gq, gr, ag, gh, dag, dgh, dsq, dsr, dxo, dxf, dhf, ch, cc,
            B, T, H, rounds, NQ, NR, MODE=mode, BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        flat = lambda t: t.reshape(-1, t.shape[-1])
        x_fin, h_fin = xr[:, :, NQ], hr[:, :, NR]
        dw = flat(dag).t() @ flat(x_fin)
        if mode == 0:
            du, db_i, db_h = flat(dag).t() @ flat(h_fin), dag.sum((0, 1)), None
        else:
            du, db_i, db_h = flat(dgh).t() @ flat(h_fin), dag.sum((0, 1)), dgh.sum((0, 1))
        # Q_j gates x with h after j state rounds; R_j gates h with x after j+1 input rounds.
        dq = torch.stack([flat(dsq[:, :, j]).t() @ flat(hr[:, :, j]) for j in range(NQ)]) if NQ else None
        dr = torch.stack([flat(dsr[:, :, j]).t() @ flat(xr[:, :, j + 1]) for j in range(NR)]) if NR else None
        return (dxo.to(x_dtype), dq, dr, dw, du, db_i, db_h, ch.to(s_dtype),
                cc.to(s_dtype) if mode == 0 else None, None, None)


def mogrifier_scan(x, q, r, w, u, b_i, b_h, h0, c0, rounds, cell="lstm"):
    """Mogrifier LSTM / GRU over a sequence.
    q: (ceil(r/2), H, H) and r: (floor(r/2), H, H) gating matrices (as Linear
    weights: out x in); w, u: input and recurrent weights of the LSTM (4H, H)
    or GRU (3H, H); b_i: combined LSTM bias or GRU input bias; b_h: GRU hidden
    bias (ignored for LSTM).  As the LSTM's cell, the GRU's carried state is
    the original h; only the gate pre-activations see the mogrified h.
    Returns (h_1..h_T, c_T)."""
    mode = 0 if cell == "lstm" else 1
    if b_h is None:
        b_h = torch.zeros_like(b_i)
    return _MogrifierScan.apply(x, q, r, w, u, b_i, b_h, h0, c0, int(rounds), mode)


def mogrifier_reference(x, q, r, w, u, b_i, b_h, h0, c0, rounds, cell="lstm"):
    h, c, hs = h0.float(), c0.float(), []
    for t in range(x.size(1)):
        xt, hm = x[:, t].float(), h
        qi = ri = 0
        for i in range(rounds):
            if i % 2 == 0:
                xt = 2 * torch.sigmoid(hm @ q[qi].t()) * xt; qi += 1
            else:
                hm = 2 * torch.sigmoid(xt @ r[ri].t()) * hm; ri += 1
        if cell == "lstm":
            p_i, p_f, p_g, p_o = (xt @ w.t() + hm @ u.t() + b_i).chunk(4, -1)
            c = torch.sigmoid(p_f) * c + torch.sigmoid(p_i) * torch.tanh(p_g)
            h = torch.sigmoid(p_o) * torch.tanh(c)
        else:
            i_r, i_z, i_n = (xt @ w.t() + b_i).chunk(3, -1)
            h_r, h_z, h_n = (hm @ u.t() + b_h).chunk(3, -1)
            rr, zz = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
            h = (1 - zz) * torch.tanh(i_n + rr * h_n) + zz * h
        hs.append(h)
    return torch.stack(hs, 1), c


# ======================================================== Mamba-3 (SISO / MIMO)
def mamba3_chunkwise(q, k, v, k_prev0, v_prev0, adt, dt, trap, s0, chunk_size=64):
    """Exact Mamba-3 recurrence (Lahoti et al., 2026; state-spaces/mamba
    ``mamba3_siso_step_ref`` / ``selective_state_update_fused_ref_v2``) in
    chunkwise-parallel form.

    SISO: q, k: (B, H, T, N) rotated C / B; v: (B, H, T, P) head inputs;
    k_prev0, v_prev0: (B, H, N), (B, H, P) previous step's k, v.
    MIMO (rank R): q, k: (B, H, T, R, N); v: (B, H, T, R, P);
    k_prev0, v_prev0: (B, H, R, N), (B, H, R, P).
    adt = Delta * A (<= 0), dt = Delta, trap = sigmoid(lambda logits): (B, H, T);
    s0: (B, H, P, N).  With alpha = exp(adt), beta = (1 - trap) dt alpha,
    gamma = trap dt:
        S_t = alpha_t S_{t-1} + beta_t sum_r v_{t-1,r} k_{t-1,r}^T + gamma_t sum_r v_{t,r} k_{t,r}^T,
        y_{t,r} = S_t q_{t,r}.
    MIMO is SISO over the flattened (step, rank) axis with every rank of one
    step sharing that step's decay.  ``chunk_size`` counts steps for both: the
    paper's chunk_size // R (Section 3.3) bounds a fused kernel's shared
    memory, while these batched matmuls ran ~3x faster with 64-step chunks
    than with 16 at R = 4 (RTX 3090).
    Returns y (shaped like v) and the final (S, k_T, v_T).
    """
    mimo = q.dim() == 5
    if not mimo:
        q, k, v = q.unsqueeze(3), k.unsqueeze(3), v.unsqueeze(3)
        k_prev0, v_prev0 = k_prev0.unsqueeze(2), v_prev0.unsqueeze(2)
    B, H, T, R, N = q.shape
    P = v.shape[-1]
    dtype = torch.promote_types(q.dtype, torch.float32)
    q, k, v, adt, dt, trap = (t.to(dtype) for t in (q, k, v, adt, dt, trap))
    k_prev = torch.cat((k_prev0.to(dtype).unsqueeze(2), k[:, :, :-1]), 2)
    v_prev = torch.cat((v_prev0.to(dtype).unsqueeze(2), v[:, :, :-1]), 2)
    beta = (1 - trap) * dt * torch.exp(adt)
    gamma = trap * dt
    S = s0.to(dtype)
    outs = []
    for start in range(0, T, chunk_size):
        stop = min(T, start + chunk_size)
        L = stop - start
        sl = slice(start, stop)
        cum = torch.cumsum(adt[:, :, sl], -1)                            # log decay to step t
        causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        decay = torch.exp((cum[..., :, None] - cum[..., None, :]).masked_fill(~causal, float("-inf")))
        w_end = torch.exp(cum[..., -1:] - cum)                           # decay from s to chunk end
        cum_last = cum[..., -1]
        beta_c, gamma_c = beta[:, :, sl], gamma[:, :, sl]
        if R > 1:
            decay = decay.repeat_interleave(R, -1).repeat_interleave(R, -2)
            cum, w_end, beta_c, gamma_c = (t.repeat_interleave(R, -1) for t in (cum, w_end, beta_c, gamma_c))
        qc, kc, kpc = (t[:, :, sl].reshape(B, H, L * R, N) for t in (q, k, k_prev))
        vc, vpc = (t[:, :, sl].reshape(B, H, L * R, P) for t in (v, v_prev))
        a1 = (qc @ kpc.transpose(-1, -2)) * decay * beta_c[:, :, None, :]
        a2 = (qc @ kc.transpose(-1, -2)) * decay * gamma_c[:, :, None, :]
        y = a1 @ vpc + a2 @ vc
        y = y + torch.exp(cum)[..., None] * torch.einsum("bhpn,bhtn->bhtp", S, qc)
        outs.append(y.view(B, H, L, R, P))
        S = torch.exp(cum_last)[..., None, None] * S \
            + torch.einsum("bht,bhtp,bhtn->bhpn", w_end * beta_c, vpc, kpc) \
            + torch.einsum("bht,bhtp,bhtn->bhpn", w_end * gamma_c, vc, kc)
    y, k_last, v_last = torch.cat(outs, 2), k[:, :, -1], v[:, :, -1]
    if not mimo:
        y, k_last, v_last = y.squeeze(3), k_last.squeeze(2), v_last.squeeze(2)
    return y, (S, k_last, v_last)


def mamba3_reference(q, k, v, k_prev0, v_prev0, adt, dt, trap, s0):
    """Step-by-step Mamba-3 recurrence, for checking ``mamba3_chunkwise``
    (same shapes; SISO or MIMO)."""
    mimo = q.dim() == 5
    if not mimo:
        q, k, v = q.unsqueeze(3), k.unsqueeze(3), v.unsqueeze(3)
        k_prev0, v_prev0 = k_prev0.unsqueeze(2), v_prev0.unsqueeze(2)
    S, kp, vp, ys = s0.double(), k_prev0.double(), v_prev0.double(), []
    for t in range(q.size(2)):
        alpha = torch.exp(adt[:, :, t].double())
        beta = (1 - trap[:, :, t].double()) * dt[:, :, t].double() * alpha
        gamma = trap[:, :, t].double() * dt[:, :, t].double()
        kt, vt = k[:, :, t].double(), v[:, :, t].double()
        S = alpha[..., None, None] * S + beta[..., None, None] * torch.einsum("bhrp,bhrn->bhpn", vp, kp) \
            + gamma[..., None, None] * torch.einsum("bhrp,bhrn->bhpn", vt, kt)
        ys.append(torch.einsum("bhpn,bhrn->bhrp", S, q[:, :, t].double()))
        kp, vp = kt, vt
    y = torch.stack(ys, 2)
    return (y if mimo else y.squeeze(3)), S


# ======================================================================= M2RNN
# One program per (batch row, value head) keeps the K x V matrix state and the
# V x V transition in registers for the whole sequence; each step is one
# (K x V) @ (V x V) tl.dot.  As in the paper (Section 4), the forward stores
# no intermediate state; the backward first re-runs the forward to cache every
# H_t, then walks time in reverse.
if HAS_TRITON:

    @triton.jit
    def _m2rnn_fwd_kernel(Q, KK, V, F, W, H0, Y, HT, HS, T, NH,
                          KD: tl.constexpr, VD: tl.constexpr, SAVE: tl.constexpr, PREC: tl.constexpr):
        b = tl.program_id(0)
        n = tl.program_id(1)
        ks = tl.arange(0, KD)
        vs = tl.arange(0, VD)
        mat = ks[:, None] * VD + vs[None, :]
        w = tl.load(W + n * VD * VD + vs[:, None] * VD + vs[None, :])
        h = tl.load(H0 + (b * NH + n) * KD * VD + mat)
        for t in range(T):
            q = tl.load(Q + (b * T + t) * KD + ks).to(tl.float32)
            k = tl.load(KK + (b * T + t) * KD + ks).to(tl.float32)
            v = tl.load(V + ((b * T + t) * NH + n) * VD + vs).to(tl.float32)
            f = tl.load(F + (b * T + t) * NH + n).to(tl.float32)
            z = _tanh(tl.dot(h, w, input_precision=PREC) + k[:, None] * v[None, :])
            h = f * h + (1.0 - f) * z
            tl.store(Y + ((b * T + t) * NH + n) * VD + vs, tl.sum(h * q[:, None], axis=0))
            if SAVE:
                tl.store(HS + ((b * NH + n) * (T + 1) + t + 1) * KD * VD + mat, h)
        tl.store(HT + (b * NH + n) * KD * VD + mat, h)

    @triton.jit
    def _m2rnn_bwd_kernel(Q, KK, V, F, W, HS, DY, DHT, DQ, DK, DV, DF, DW, DH0, T, NH,
                          KD: tl.constexpr, VD: tl.constexpr, PREC: tl.constexpr):
        b = tl.program_id(0)
        n = tl.program_id(1)
        ks = tl.arange(0, KD)
        vs = tl.arange(0, VD)
        mat = ks[:, None] * VD + vs[None, :]
        w = tl.load(W + n * VD * VD + vs[:, None] * VD + vs[None, :])
        dh = tl.load(DHT + (b * NH + n) * KD * VD + mat)
        dw = tl.zeros([VD, VD], dtype=tl.float32)
        for step in range(T):
            t = T - 1 - step
            row = (b * T + t) * NH + n
            q = tl.load(Q + (b * T + t) * KD + ks).to(tl.float32)
            k = tl.load(KK + (b * T + t) * KD + ks).to(tl.float32)
            v = tl.load(V + row * VD + vs).to(tl.float32)
            f = tl.load(F + row).to(tl.float32)
            dy = tl.load(DY + row * VD + vs).to(tl.float32)
            h_t = tl.load(HS + ((b * NH + n) * (T + 1) + t + 1) * KD * VD + mat)
            h_p = tl.load(HS + ((b * NH + n) * (T + 1) + t) * KD * VD + mat)
            tl.store(DQ + row * KD + ks, tl.sum(h_t * dy[None, :], axis=1))   # y = H^T q
            dh += q[:, None] * dy[None, :]
            z = _tanh(tl.dot(h_p, w, input_precision=PREC) + k[:, None] * v[None, :])
            tl.store(DF + row, tl.sum(tl.sum(dh * (h_p - z), axis=1), axis=0))
            dp = (1.0 - f) * dh * (1.0 - z * z)
            tl.store(DK + row * KD + ks, tl.sum(dp * v[None, :], axis=1))
            tl.store(DV + row * VD + vs, tl.sum(dp * k[:, None], axis=0))
            dw += tl.dot(tl.trans(h_p), dp, input_precision=PREC)
            dh = f * dh + tl.dot(dp, tl.trans(w), input_precision=PREC)
        tl.store(DW + (b * NH + n) * VD * VD + vs[:, None] * VD + vs[None, :], dw)
        tl.store(DH0 + (b * NH + n) * KD * VD + mat, dh)


def _m2rnn_launch(q, k, v, f, w, h0, save):
    B, T, NH, VD = v.shape
    KD = q.shape[-1]
    kw = dict(device=v.device, dtype=torch.float32)
    y, ht = torch.empty(B, T, NH, VD, **kw), torch.empty(B, NH, KD, VD, **kw)
    hs = torch.empty(B, NH, T + 1, KD, VD, **kw) if save else ht
    if save:
        hs[:, :, 0] = h0
    _m2rnn_fwd_kernel[(B, NH)](q, k, v, f, w, h0, y, ht, hs, T, NH, KD=KD, VD=VD, SAVE=save,
                               PREC=_dot_precision(), num_warps=4, num_stages=1)
    return y, ht, hs


class _M2RNNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, f, w, h0):
        q, k, v, f = (t.contiguous() for t in (q, k, v, f))
        w32, h32 = w.float().contiguous(), h0.float().contiguous()
        y, ht, _ = _m2rnn_launch(q, k, v, f, w32, h32, save=False)
        ctx.save_for_backward(q, k, v, f, w32, h32)
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, f.dtype, w.dtype, h0.dtype)
        return y, ht

    @staticmethod
    def backward(ctx, dy, dht):
        q, k, v, f, w32, h32 = ctx.saved_tensors
        B, T, NH, VD = v.shape
        KD = q.shape[-1]
        kw = dict(device=v.device, dtype=torch.float32)
        _, _, hs = _m2rnn_launch(q, k, v, f, w32, h32, save=True)
        dy = torch.zeros(B, T, NH, VD, **kw) if dy is None else _contig(dy.float())
        dht = torch.zeros(B, NH, KD, VD, **kw) if dht is None else _contig(dht.float())
        dq, dk = torch.empty(B, T, NH, KD, **kw), torch.empty(B, T, NH, KD, **kw)
        dv, df = torch.empty(B, T, NH, VD, **kw), torch.empty(B, T, NH, **kw)
        dw, dh0 = torch.empty(B, NH, VD, VD, **kw), torch.empty(B, NH, KD, VD, **kw)
        _m2rnn_bwd_kernel[(B, NH)](q, k, v, f, w32, hs, dy, dht, dq, dk, dv, df, dw, dh0, T, NH,
                                   KD=KD, VD=VD, PREC=_dot_precision(), num_warps=4, num_stages=1)
        del hs
        qd, kd, vd, fd, wd, hd = ctx.dtypes
        return (dq.sum(2).to(qd), dk.sum(2).to(kd), dv.to(vd), df.to(fd), dw.sum(0).to(wd), dh0.to(hd))


def m2rnn_supported(key_dim, value_dim):
    """The kernel needs power-of-two head sizes of at least 16 (tl.dot tiles)."""
    return all(d >= 16 and d & (d - 1) == 0 for d in (key_dim, value_dim))


def m2rnn_scan(q, k, v, f, w, h0):
    """M2RNN recurrence (Mishra et al., 2026), multi-value form: one query /
    key head shared by every value head.
    q, k: (B, T, K); v: (B, T, NH, V); f: (B, T, NH) forget gate in [0, 1];
    w: (NH, V, V) transitions; h0: (B, NH, K, V).
        Z_t = tanh(H_{t-1} W + k_t v_t^T),  H_t = f_t H_{t-1} + (1 - f_t) Z_t,
        y_t = H_t^T q_t.
    Returns y (B, T, NH, V) and H_T."""
    return _M2RNNScan.apply(q, k, v, f, w, h0)


def m2rnn_reference(q, k, v, f, w, h0):
    h, ys = h0.float(), []
    for t in range(v.size(1)):
        z = torch.tanh(h @ w.float() + k[:, t, None, :, None].float() * v[:, t, :, None, :].float())
        ft = f[:, t, :, None, None].float()
        h = ft * h + (1 - ft) * z
        ys.append(torch.einsum("bnkv,bk->bnv", h, q[:, t].float()))
    return torch.stack(ys, 1), h


# ============================================================ diagonal affine scan
if HAS_TRITON:

    @triton.jit
    def _affine_fwd_kernel(A, Bv, HF, T, D, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < D
        h = tl.load(HF + b * (T + 1) * D + cols, mask=m, other=0.0)
        for t in range(T):
            idx = (b * T + t) * D + cols
            h = tl.load(A + idx, mask=m, other=0.0).to(tl.float32) * h \
                + tl.load(Bv + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(HF + (b * (T + 1) + t + 1) * D + cols, h, mask=m)

    @triton.jit
    def _affine_bwd_kernel(DY, A, HF, DA, DB, DH0, T, D, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < D
        dh = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(T):
            t = T - 1 - k
            idx = (b * T + t) * D + cols
            g = tl.load(DY + idx, mask=m, other=0.0).to(tl.float32) + dh
            tl.store(DB + idx, g, mask=m)
            tl.store(DA + idx, g * tl.load(HF + (b * (T + 1) + t) * D + cols, mask=m, other=0.0), mask=m)
            dh = g * tl.load(A + idx, mask=m, other=0.0).to(tl.float32)
        tl.store(DH0 + b * D + cols, dh, mask=m)


class _AffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, h0):
        a, b = a.contiguous(), b.contiguous()
        B, T, D = a.shape
        hf = torch.empty(B, T + 1, D, device=a.device, dtype=torch.float32)
        hf[:, 0] = 0.0 if h0 is None else h0.float()
        blk = _diag_block(D)
        _affine_fwd_kernel[(B, triton.cdiv(D, blk))](a, b, hf, T, D, BLOCK=blk)
        ctx.save_for_backward(a, hf)
        ctx.meta = (blk, a.dtype, b.dtype, h0 is not None, None if h0 is None else h0.dtype)
        return hf[:, 1:]

    @staticmethod
    def backward(ctx, dy):
        a, hf = ctx.saved_tensors
        blk, a_dtype, b_dtype, has_h0, h_dtype = ctx.meta
        B, T, D = a.shape
        kw = dict(device=a.device, dtype=torch.float32)
        da, db, dh0 = torch.empty(B, T, D, **kw), torch.empty(B, T, D, **kw), torch.empty(B, D, **kw)
        _affine_bwd_kernel[(B, triton.cdiv(D, blk))](_contig(dy), a, hf, da, db, dh0, T, D, BLOCK=blk)
        return da.to(a_dtype), db.to(b_dtype), (dh0.to(h_dtype) if has_h0 else None)


def affine_scan(a, b, h0=None):
    """h_t = a_t * h_{t-1} + b_t elementwise over (B, T, D) (real inputs);
    one program per (batch, channel block) runs the whole time loop."""
    return _AffineScan.apply(a, b, h0)


# ============================================================ LMU (Legendre Memory Unit)
# Voelker et al. (2019): a nonlinear hidden state h reads a fixed linear
# Legendre delay memory m of U units x N orders.  Per step:
#   d  = px_d + h W_hd^T + m W_md^T          (memory drive, U)
#   m' = m A^T + d (x) B                     (each unit's N coefficients)
#   h' = tanh(px_h + h W_rec^T + m' R^T)
# The memory is N times wider than h, so every step streams two (U*N x H)
# matrices; one program per batch tile would leave most SMs idle.  Each batch
# tile therefore gets NS programs that split the output columns (and units) of
# every phase and meet at a counter barrier between phases.  All programs of a
# tile must be resident at once, so the launcher keeps the grid within the SM
# count.  The unit x order mix uses exact fp32 dots: A is close to identity and
# its rounding error compounds along the whole delay window.
if HAS_TRITON:

    @triton.jit
    def _group_barrier(CNT, target):
        """Wait until the tile's phase counter reaches ``target``."""
        tl.debug_barrier()
        tl.atomic_add(CNT, 1, sem="release")
        while tl.atomic_add(CNT, 0, sem="acquire") < target:
            pass
        tl.debug_barrier()

    @triton.jit
    def _mm_split(A, lda, W, ldw, OUT, ldo, brow, bmask, K, N, split, NS,
                  ACC: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  PREC: tl.constexpr):
        """_mm_rows over this program's column tiles; A comes from other programs."""
        for n0 in range(split * BN, N, NS * BN):
            ns = n0 + tl.arange(0, BN)
            nm = ns < N
            acc = tl.zeros([BB, BN], dtype=tl.float32)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                km = ks < K
                a = tl.load(A + brow[:, None] * lda + ks[None, :], mask=bmask[:, None] & km[None, :], other=0.0,
                            cache_modifier=".cg")
                w = tl.load(W + ks[:, None] * ldw + ns[None, :], mask=km[:, None] & nm[None, :], other=0.0)
                acc = tl.dot(a, w, acc, input_precision=PREC)
            out = OUT + brow[:, None] * ldo + ns[None, :]
            om = bmask[:, None] & nm[None, :]
            if ACC:
                acc += tl.load(out, mask=om, other=0.0, cache_modifier=".cg")
            tl.store(out, acc, mask=om)

    @triton.jit
    def _lmu_fwd_kernel(PXH, PXD, WHDT, WMDT, WRECT, RMT, AT, BV, HF, MF, S, CNT,
                        B, T, H, U, N, NS,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        BU: tl.constexpr, NP: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        UN = U * N
        rr = tl.arange(0, BB * BU)
        rb = pid * BB + rr // BU
        ns = tl.arange(0, NP)
        nm = ns < N
        at = tl.load(AT + ns[:, None] * N + ns[None, :], mask=nm[:, None] & nm[None, :], other=0.0)
        bv = tl.load(BV + ns, mask=nm, other=0.0)
        phase = 0
        for t in range(T):
            _mm_split(HF + t * H, (T + 1) * H, WHDT, U, S, U, brow, bmask, H, U, split, NS,
                      False, BB, BN, BK, PREC)
            tl.debug_barrier()
            _mm_split(MF + t * UN, (T + 1) * UN, WMDT, U, S, U, brow, bmask, UN, U, split, NS,
                      True, BB, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for u0 in range(split * BU, U, NS * BU):
                us = u0 + rr % BU
                rm = (rb < B) & (us < U)
                d = tl.load(S + rb * U + us, mask=rm, other=0.0, cache_modifier=".cg") \
                    + tl.load(PXD + (rb * T + t) * U + us, mask=rm, other=0.0).to(tl.float32)
                tile = rm[:, None] & nm[None, :]
                off = us[:, None] * N + ns[None, :]
                m = tl.load(MF + (rb[:, None] * (T + 1) + t) * UN + off, mask=tile, other=0.0, cache_modifier=".cg")
                m = tl.dot(m, at, input_precision="ieee") + d[:, None] * bv[None, :]
                tl.store(MF + (rb[:, None] * (T + 1) + t + 1) * UN + off, m, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # h_t lands in its own slot, so the two products accumulate there.
            _mm_split(HF + t * H, (T + 1) * H, WRECT, H, HF + (t + 1) * H, (T + 1) * H, brow, bmask, H, H,
                      split, NS, False, BB, BN, BK, PREC)
            tl.debug_barrier()
            _mm_split(MF + (t + 1) * UN, (T + 1) * UN, RMT, H, HF + (t + 1) * H, (T + 1) * H, brow, bmask,
                      UN, H, split, NS, True, BB, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(split * BN, H, NS * BN):
                cs = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (cs < H)[None, :]
                ptr = HF + (brow[:, None] * (T + 1) + t + 1) * H + cs[None, :]
                z = tl.load(ptr, mask=tile, other=0.0) \
                    + tl.load(PXH + (brow[:, None] * T + t) * H + cs[None, :], mask=tile, other=0.0).to(tl.float32)
                tl.store(ptr, _tanh(z), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    @triton.jit
    def _lmu_bwd_kernel(DY, WHD, WMD, WREC, RM, A, BV, HF, DZ, DD, CH, CM, CNT,
                        B, T, H, U, N, NS,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        BU: tl.constexpr, NP: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        UN = U * N
        rr = tl.arange(0, BB * BU)
        rb = pid * BB + rr // BU
        ns = tl.arange(0, NP)
        nm = ns < N
        a = tl.load(A + ns[:, None] * N + ns[None, :], mask=nm[:, None] & nm[None, :], other=0.0)
        bv = tl.load(BV + ns, mask=nm, other=0.0)
        phase = 0
        for k in range(T):
            t = T - 1 - k
            for n0 in range(split * BN, H, NS * BN):
                cs = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (cs < H)[None, :]
                dh = tl.load(CH + brow[:, None] * H + cs[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + cs[None, :], mask=tile, other=0.0).to(tl.float32)
                h = tl.load(HF + (brow[:, None] * (T + 1) + t + 1) * H + cs[None, :], mask=tile, other=0.0)
                tl.store(DZ + (brow[:, None] * T + t) * H + cs[None, :], dh * (1.0 - h * h), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # m_t feeds h_t through R.
            _mm_split(DZ + t * H, T * H, RM, UN, CM, UN, brow, bmask, H, UN, split, NS,
                      True, BB, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for u0 in range(split * BU, U, NS * BU):
                us = u0 + rr % BU
                rm = (rb < B) & (us < U)
                tile = rm[:, None] & nm[None, :]
                ptr = CM + rb[:, None] * UN + us[:, None] * N + ns[None, :]
                dm = tl.load(ptr, mask=tile, other=0.0, cache_modifier=".cg")
                tl.store(DD + (rb * T + t) * U + us, tl.sum(dm * bv[None, :], axis=1), mask=rm)
                tl.store(ptr, tl.dot(dm, a, input_precision="ieee"), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            _mm_split(DZ + t * H, T * H, WREC, H, CH, H, brow, bmask, H, H, split, NS,
                      False, BB, BN, BK, PREC)
            tl.debug_barrier()
            _mm_split(DD + t * U, T * U, WHD, H, CH, H, brow, bmask, U, H, split, NS,
                      True, BB, BN, BK, PREC)
            _mm_split(DD + t * U, T * U, WMD, UN, CM, UN, brow, bmask, U, UN, split, NS,
                      True, BB, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)


def _lmu_launch(B, H, U, N, device):
    """(grid, BB, BN, BK, BU, NP, NS): NS column-splitting programs per batch
    tile, with the whole grid co-resident (at most one program per SM)."""
    bb, bk = 16, min(64, triton.next_power_of_2(max(H, 16)))
    bn = 16
    tiles = triton.cdiv(B, bb)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ns = max(1, min(sms // tiles, triton.cdiv(max(H, U), bn)))
    return (tiles, ns), bb, bn, bk, max(1, 64 // bb), max(16, triton.next_power_of_2(N)), ns


class _LMUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pxh, pxd, whd, wmd, wrec, rm, a, bv, h0, m0):
        pxh, pxd = pxh.contiguous(), pxd.contiguous()
        B, T, H = pxh.shape
        U, N = m0.shape[1], m0.shape[2]
        kw = dict(device=pxh.device, dtype=torch.float32)
        f32 = lambda t: t.float().contiguous()
        whd32, wmd32, wrec32, rm32, a32, bv32 = f32(whd), f32(wmd), f32(wrec), f32(rm), f32(a), f32(bv)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        mf = torch.empty(B, T + 1, U * N, **kw); mf[:, 0] = m0.float().reshape(B, U * N)
        s = torch.empty(B, U, **kw)
        grid, bb, bn, bk, bu, np2, ns = _lmu_launch(B, H, U, N, pxh.device)
        cnt = torch.zeros(grid[0], device=pxh.device, dtype=torch.int32)
        prec = _dot_precision()
        _lmu_fwd_kernel[grid](
            pxh, pxd, whd32.t().contiguous(), wmd32.t().contiguous(), wrec32.t().contiguous(),
            rm32.t().contiguous(), a32.t().contiguous(), bv32, hf, mf, s, cnt, B, T, H, U, N, ns,
            BB=bb, BN=bn, BK=bk, BU=bu, NP=np2, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(whd32, wmd32, wrec32, rm32, a32, bv32, hf, mf)
        ctx.meta = (prec, pxh.dtype, pxd.dtype, h0.dtype, m0.dtype)
        return hf[:, 1:], hf[:, -1], mf[:, -1].view(B, U, N)

    @staticmethod
    def backward(ctx, dy, dh_last, dm_last):
        whd32, wmd32, wrec32, rm32, a32, bv32, hf, mf = ctx.saved_tensors
        prec, pxh_dtype, pxd_dtype, h_dtype, m_dtype = ctx.meta
        B, T1, H = hf.shape
        T, UN = T1 - 1, mf.shape[2]
        U = whd32.shape[0]
        N = UN // U
        kw = dict(device=hf.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        ch = torch.zeros(B, H, **kw) if dh_last is None else dh_last.float().clone()
        cm = torch.zeros(B, UN, **kw) if dm_last is None else dm_last.float().reshape(B, UN).clone()
        dz, dd = torch.empty(B, T, H, **kw), torch.empty(B, T, U, **kw)
        grid, bb, bn, bk, bu, np2, ns = _lmu_launch(B, H, U, N, hf.device)
        cnt = torch.zeros(grid[0], device=hf.device, dtype=torch.int32)
        _lmu_bwd_kernel[grid](
            dy, whd32, wmd32, wrec32, rm32, a32, bv32, hf, dz, dd, ch, cm, cnt, B, T, H, U, N, ns,
            BB=bb, BN=bn, BK=bk, BU=bu, NP=np2, PREC=prec, num_warps=4, num_stages=1)
        flat = lambda t: t.reshape(-1, t.shape[-1])
        h_prev, m_prev, m_cur = flat(hf[:, :-1]), flat(mf[:, :-1]), flat(mf[:, 1:])
        dwhd, dwmd = flat(dd).t() @ h_prev, flat(dd).t() @ m_prev
        dwrec, drm = flat(dz).t() @ h_prev, flat(dz).t() @ m_cur
        return (dz.to(pxh_dtype), dd.to(pxd_dtype), dwhd, dwmd, dwrec, drm, None, None,
                ch.to(h_dtype), cm.view(B, U, N).to(m_dtype))


@torch.compiler.disable
def lmu_scan(pxh, pxd, whd, wmd, wrec, rm, a, bv, h0, m0):
    """One LMU layer over a sequence.
    ``pxh``: (B, T, H) input part of the hidden pre-activation (with bias);
    ``pxd``: (B, T, U) input part of the memory drive (with bias); ``whd``
    (U, H) and ``wmd`` (U, U*N): drive weights reading h and m; ``wrec`` (H, H)
    and ``rm`` (H, U*N): hidden weights reading h and the updated m; ``a``
    (N, N) and ``bv`` (N,): the fixed discrete Legendre delay system (no
    gradient); ``h0`` (B, H), ``m0`` (B, U, N).  Returns (h_1..h_T, h_T, m_T)."""
    return _LMUScan.apply(pxh, pxd, whd, wmd, wrec, rm, a, bv, h0, m0)


def lmu_reference(pxh, pxd, whd, wmd, wrec, rm, a, bv, h0, m0):
    h, m, hs = h0.float(), m0.float(), []
    B, U, N = m.shape
    for t in range(pxh.size(1)):
        d = pxd[:, t].float() + h @ whd.t() + m.reshape(B, -1) @ wmd.t()
        m = m @ a.t() + d.unsqueeze(-1) * bv
        h = torch.tanh(pxh[:, t].float() + h @ wrec.t() + m.reshape(B, -1) @ rm.t())
        hs.append(h)
    return torch.stack(hs, 1), h, m


# ============================================================ RIN latent core
# Recurrent Interface Network block (Jabri et al., 2022), autoregressive form.
# Per token the latent bank L (M x D) is updated by
#   X0 = L_{t-1} + r_t                  (read: a single data key, so softmax = 1)
#   X1 = X0 + MHA(LN1(X0))               (latent self-attention)
#   L_t = X1 + W2 gelu(W1 LN2(X1))       (latent MLP)
# The read value r_t and the data-side write depend only on the input tokens
# and on L_t, so the caller computes them for the whole sequence at once; only
# this latent step is sequential.  A batch tile is 16 latent rows (a few batch
# rows' latents, padded to a power of two); attention stays inside a batch
# row.  As in the LMU kernel, NS programs per tile split the columns (and
# heads) of every phase and meet at _group_barrier; each recomputes the cheap
# LayerNorm row statistics itself instead of waiting for another phase.
if HAS_TRITON:

    @triton.jit
    def _mm_off(A, a_off, W, ldw, BIAS, OUT, o_off, rmask, K, N, split, NS,
                ACC: tl.constexpr, HAS_BIAS: tl.constexpr, GELU_A: tl.constexpr,
                R: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        """OUT[rows, cols] (+)= f(A[rows, :K]) @ W[:K, cols] (+ bias) for this
        program's column tiles; rows are given by offsets."""
        for n0 in range(split * BN, N, NS * BN):
            ns = n0 + tl.arange(0, BN)
            nm = ns < N
            acc = tl.zeros([R, BN], dtype=tl.float32)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                km = ks < K
                a = tl.load(A + a_off[:, None] + ks[None, :], mask=rmask[:, None] & km[None, :], other=0.0,
                            cache_modifier=".cg")
                if GELU_A:
                    a = 0.5 * a * (1.0 + tl.math.erf(a * 0.7071067811865476))
                w = tl.load(W + ks[:, None] * ldw + ns[None, :], mask=km[:, None] & nm[None, :], other=0.0)
                acc = tl.dot(a, w, acc, input_precision=PREC)
            if HAS_BIAS:
                acc += tl.load(BIAS + ns, mask=nm, other=0.0)[None, :]
            out = OUT + o_off[:, None] + ns[None, :]
            om = rmask[:, None] & nm[None, :]
            if ACC:
                acc += tl.load(out, mask=om, other=0.0, cache_modifier=".cg")
            tl.store(out, acc, mask=om)

    @triton.jit
    def _rin_rows(B, M, MP: tl.constexpr, R: tl.constexpr):
        rr = tl.arange(0, R)
        rb = tl.program_id(0) * (R // MP) + rr // MP
        rj = rr % MP
        return rr, rb, rj, (rb < B) & (rj < M)

    @triton.jit
    def _rin_attn_scores(QKV, row3, rb, rv, h, D, HD, scale, HDP: tl.constexpr):
        cs = tl.arange(0, HDP)
        cm = rv[:, None] & (cs < HD)[None, :]
        base = QKV + row3[:, None] + h * HD + cs[None, :]
        q = tl.load(base, mask=cm, other=0.0, cache_modifier=".cg")
        k = tl.load(base + D, mask=cm, other=0.0, cache_modifier=".cg")
        v = tl.load(base + 2 * D, mask=cm, other=0.0, cache_modifier=".cg")
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        same = (rb[:, None] == rb[None, :]) & rv[None, :]
        s = tl.where(same, s, -1.0e30)
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        return q, k, v, p, cs, cm

    @triton.jit
    def _rin_ln_rows(X, x_off, X2, x2_off, HAS_X2: tl.constexpr, G, Bb, OUT, out_off,
                     rv, D, eps, R: tl.constexpr, BN: tl.constexpr):
        """OUT = LayerNorm(X (+ X2)) over full rows; returns (mean, rstd)."""
        s1 = tl.zeros([R], dtype=tl.float32)
        for n0 in range(0, D, BN):
            cs = n0 + tl.arange(0, BN)
            tile = rv[:, None] & (cs < D)[None, :]
            x = tl.load(X + x_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            if HAS_X2:
                x += tl.load(X2 + x2_off[:, None] + cs[None, :], mask=tile, other=0.0)
            s1 += tl.sum(x, axis=1)
        mu = s1 / D
        s2 = tl.zeros([R], dtype=tl.float32)
        for n0 in range(0, D, BN):
            cs = n0 + tl.arange(0, BN)
            tile = rv[:, None] & (cs < D)[None, :]
            x = tl.load(X + x_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            if HAS_X2:
                x += tl.load(X2 + x2_off[:, None] + cs[None, :], mask=tile, other=0.0)
            x = tl.where(tile, x - mu[:, None], 0.0)
            s2 += tl.sum(x * x, axis=1)
        rs = 1.0 / tl.sqrt(s2 / D + eps)
        for n0 in range(0, D, BN):
            cs = n0 + tl.arange(0, BN)
            cmask = cs < D
            tile = rv[:, None] & cmask[None, :]
            x = tl.load(X + x_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            if HAS_X2:
                x += tl.load(X2 + x2_off[:, None] + cs[None, :], mask=tile, other=0.0)
            y = (x - mu[:, None]) * rs[:, None] * tl.load(G + cs, mask=cmask, other=0.0)[None, :] \
                + tl.load(Bb + cs, mask=cmask, other=0.0)[None, :]
            tl.store(OUT + out_off[:, None] + cs[None, :], y, mask=tile)
        return mu, rs

    @triton.jit
    def _rin_fwd_kernel(RD, LF, WINT, BIN, WOT, BO, G1, B1, G2, B2, W1T, BM1, W2T, BM2,
                        QKV, O, X1, UU, MU1, RS1, MU2, RS2, SN, CNT,
                        B, T, M, D, NH, HD, NS, scale, eps,
                        MP: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        HDP: tl.constexpr, PREC: tl.constexpr):
        rr, rb, rj, rv = _rin_rows(B, M, MP, R)
        split = tl.program_id(1)
        cnt = CNT + tl.program_id(0)
        lead = rv & (split == 0)
        srow = ((tl.program_id(0) * NS + split) * R + rr) * D      # private scratch rows
        phase = 0
        for t in range(T):
            sv = (rb * T + t) * M + rj                              # saved (b, t, j) row
            lprev = ((rb * (T + 1) + t) * M + rj) * D
            lnext = lprev + M * D
            rrow = (rb * T + t) * D
            # ---- q, k, v = LN1(X0), X0 = L_{t-1} + r_t
            mu, rs = _rin_ln_rows(LF, lprev, RD, rrow, True, G1, B1, SN, srow, rv, D, eps, R, BN)
            tl.store(MU1 + sv, mu, mask=lead)
            tl.store(RS1 + sv, rs, mask=lead)
            tl.debug_barrier()
            _mm_off(SN, srow, WINT, 3 * D, BIN, QKV, sv * 3 * D, rv, D, 3 * D, split, NS,
                    False, True, False, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- attention among one batch row's latents
            for h in range(split, NH, NS):
                q, k, v, p, cs, cm = _rin_attn_scores(QKV, sv * 3 * D, rb, rv, h, D, HD, scale, HDP)
                tl.store(O + (sv * D)[:, None] + h * HD + cs[None, :], tl.dot(p, v, input_precision="ieee"),
                         mask=cm)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- X1 = X0 + O Wo^T + bo, also the start of L_t
            _mm_off(O, sv * D, WOT, D, BO, X1, sv * D, rv, D, D, split, NS,
                    False, True, False, R, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(split * BN, D, NS * BN):
                cs = n0 + tl.arange(0, BN)
                tile = rv[:, None] & (cs < D)[None, :]
                x = tl.load(X1 + (sv * D)[:, None] + cs[None, :], mask=tile, other=0.0) \
                    + tl.load(LF + lprev[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg") \
                    + tl.load(RD + rrow[:, None] + cs[None, :], mask=tile, other=0.0)
                tl.store(X1 + (sv * D)[:, None] + cs[None, :], x, mask=tile)
                tl.store(LF + lnext[:, None] + cs[None, :], x, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- latent MLP: L_t = X1 + W2 gelu(W1 LN2(X1) + b1) + b2
            mu, rs = _rin_ln_rows(X1, sv * D, X1, sv * D, False, G2, B2, SN, srow, rv, D, eps, R, BN)
            tl.store(MU2 + sv, mu, mask=lead)
            tl.store(RS2 + sv, rs, mask=lead)
            tl.debug_barrier()
            _mm_off(SN, srow, W1T, 4 * D, BM1, UU, sv * 4 * D, rv, D, 4 * D, split, NS,
                    False, True, False, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            _mm_off(UU, sv * 4 * D, W2T, D, BM2, LF, lnext, rv, 4 * D, D, split, NS,
                    True, True, True, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)

    @triton.jit
    def _rin_ln_bwd(DN, dn_off, XS, x_off, X2, x2_off, HAS_X2: tl.constexpr, mu, rs, G,
                    RES, res_off, OUT, out_off, rv, D, split, NS, R: tl.constexpr, BN: tl.constexpr):
        """OUT = RES + LayerNorm backward of DN at input XS (+ X2), own columns."""
        c1 = tl.zeros([R], dtype=tl.float32)
        c2 = tl.zeros([R], dtype=tl.float32)
        for n0 in range(0, D, BN):
            cs = n0 + tl.arange(0, BN)
            cmask = cs < D
            tile = rv[:, None] & cmask[None, :]
            x = tl.load(XS + x_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            if HAS_X2:
                x += tl.load(X2 + x2_off[:, None] + cs[None, :], mask=tile, other=0.0)
            xh = tl.where(tile, (x - mu[:, None]) * rs[:, None], 0.0)
            g = tl.load(DN + dn_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg") \
                * tl.load(G + cs, mask=cmask, other=0.0)[None, :]
            c1 += tl.sum(g, axis=1)
            c2 += tl.sum(g * xh, axis=1)
        c1 = c1 / D
        c2 = c2 / D
        for n0 in range(split * BN, D, NS * BN):
            cs = n0 + tl.arange(0, BN)
            cmask = cs < D
            tile = rv[:, None] & cmask[None, :]
            x = tl.load(XS + x_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            if HAS_X2:
                x += tl.load(X2 + x2_off[:, None] + cs[None, :], mask=tile, other=0.0)
            xh = (x - mu[:, None]) * rs[:, None]
            g = tl.load(DN + dn_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg") \
                * tl.load(G + cs, mask=cmask, other=0.0)[None, :]
            dx = tl.load(RES + res_off[:, None] + cs[None, :], mask=tile, other=0.0, cache_modifier=".cg") \
                + rs[:, None] * (g - c1[:, None] - xh * c2[:, None])
            tl.store(OUT + out_off[:, None] + cs[None, :], dx, mask=tile)

    @triton.jit
    def _rin_bwd_kernel(DLO, RD, LF, WIN, WO, G1, G2, W1, W2,
                        QKV, X1, UU, MU1, RS1, MU2, RS2,
                        DL, DU, DN2, DA, DQKV, DN1, DX0, DOB, CNT,
                        B, T, M, D, NH, HD, NS, scale,
                        MP: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        HDP: tl.constexpr, PREC: tl.constexpr):
        rr, rb, rj, rv = _rin_rows(B, M, MP, R)
        split = tl.program_id(1)
        cnt = CNT + tl.program_id(0)
        drow = (rb * M + rj) * D                                   # shared dO rows
        phase = 0
        for k in range(T):
            t = T - 1 - k
            sv = (rb * T + t) * M + rj
            # ---- dL_t = output grad + dX0 of step t+1 (the recurrence)
            has_next = t + 1 < T
            for n0 in range(split * BN, D, NS * BN):
                cs = n0 + tl.arange(0, BN)
                tile = rv[:, None] & (cs < D)[None, :]
                o = (sv * D)[:, None] + cs[None, :]
                tl.store(DL + o, tl.load(DLO + o, mask=tile, other=0.0).to(tl.float32)
                         + tl.load(DX0 + o + M * D, mask=tile & has_next, other=0.0, cache_modifier=".cg"),
                         mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- MLP
            _mm_off(DL, sv * D, W2, 4 * D, W2, DU, sv * 4 * D, rv, D, 4 * D, split, NS,
                    False, False, False, R, BN, BK, PREC)
            tl.debug_barrier()
            for n0 in range(split * BN, 4 * D, NS * BN):
                cs = n0 + tl.arange(0, BN)
                tile = rv[:, None] & (cs < 4 * D)[None, :]
                o = (sv * 4 * D)[:, None] + cs[None, :]
                u = tl.load(UU + o, mask=tile, other=0.0)
                cdf = 0.5 * (1.0 + tl.math.erf(u * 0.7071067811865476))
                pdf = 0.3989422804014327 * tl.exp(-0.5 * u * u)
                tl.store(DU + o, tl.load(DU + o, mask=tile, other=0.0) * (cdf + u * pdf), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            _mm_off(DU, sv * 4 * D, W1, D, W1, DN2, sv * D, rv, 4 * D, D, split, NS,
                    False, False, False, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            mu = tl.load(MU2 + sv, mask=rv, other=0.0)
            rs = tl.load(RS2 + sv, mask=rv, other=0.0)
            _rin_ln_bwd(DN2, sv * D, X1, sv * D, X1, sv * D, False, mu, rs, G2,
                        DL, sv * D, DA, sv * D, rv, D, split, NS, R, BN)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- attention: dO = dA Wo
            _mm_off(DA, sv * D, WO, D, WO, DOB, drow, rv, D, D, split, NS,
                    False, False, False, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for h in range(split, NH, NS):
                q, kk, v, p, cs, cm = _rin_attn_scores(QKV, sv * 3 * D, rb, rv, h, D, HD, scale, HDP)
                do = tl.load(DOB + drow[:, None] + h * HD + cs[None, :], mask=cm, other=0.0, cache_modifier=".cg")
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - tl.sum(dp * p, axis=1)[:, None])
                base = DQKV + (sv * 3 * D)[:, None] + h * HD + cs[None, :]
                tl.store(base, tl.dot(ds, kk, input_precision="ieee") * scale, mask=cm)
                tl.store(base + D, tl.dot(tl.trans(ds), q, input_precision="ieee") * scale, mask=cm)
                tl.store(base + 2 * D, tl.dot(tl.trans(p), do, input_precision="ieee"), mask=cm)
            phase += 1
            _group_barrier(cnt, phase * NS)
            _mm_off(DQKV, sv * 3 * D, WIN, D, WIN, DN1, sv * D, rv, 3 * D, D, split, NS,
                    False, False, False, R, BN, BK, PREC)
            phase += 1
            _group_barrier(cnt, phase * NS)
            # ---- dX0 = dX1 + LN1 backward; it is dL_{t-1} and (summed) dr_t
            mu = tl.load(MU1 + sv, mask=rv, other=0.0)
            rs = tl.load(RS1 + sv, mask=rv, other=0.0)
            _rin_ln_bwd(DN1, sv * D, LF, ((rb * (T + 1) + t) * M + rj) * D, RD, (rb * T + t) * D, True,
                        mu, rs, G1, DA, sv * D, DX0, sv * D, rv, D, split, NS, R, BN)
            phase += 1
            _group_barrier(cnt, phase * NS)


def _rin_launch(B, M, D, NH, device):
    """(grid, MP, R, BN, BK, HDP, NS) with the whole grid co-resident."""
    mp = triton.next_power_of_2(M)
    rows = max(16, mp)
    dp2 = triton.next_power_of_2(max(D, 16))
    bn, bk = min(32, dp2), min(64, dp2)
    tiles = triton.cdiv(B, rows // mp)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ns = max(1, min(sms // tiles, triton.cdiv(4 * D, bn)))
    hdp = max(16, triton.next_power_of_2(D // NH))
    return (tiles, ns), mp, rows, bn, bk, hdp, ns


class _RINLatentScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, l0, g1, b1, w_in, b_in, w_o, b_o, g2, b2, w1, bm1, w2, bm2, heads, eps):
        B, T, D = r.shape
        M = l0.shape[1]
        kw = dict(device=r.device, dtype=torch.float32)
        f32 = lambda t: t.float().contiguous()
        r32 = f32(r)
        p = [f32(x) for x in (g1, b1, w_in, b_in, w_o, b_o, g2, b2, w1, bm1, w2, bm2)]
        g1_, b1_, win, bin_, wo, bo, g2_, b2_, w1_, bm1_, w2_, bm2_ = p
        lf = torch.empty(B, T + 1, M, D, **kw); lf[:, 0] = l0.float()
        qkv, o, x1 = torch.empty(B, T, M, 3 * D, **kw), torch.empty(B, T, M, D, **kw), torch.empty(B, T, M, D, **kw)
        uu = torch.empty(B, T, M, 4 * D, **kw)
        mu1, rs1, mu2, rs2 = (torch.empty(B, T, M, **kw) for _ in range(4))
        grid, mp, rows, bn, bk, hdp, ns = _rin_launch(B, M, D, heads, r.device)
        sn = torch.empty(grid[0] * ns * rows, D, **kw)
        cnt = torch.zeros(grid[0], device=r.device, dtype=torch.int32)
        hd = D // heads
        prec = _dot_precision()
        _rin_fwd_kernel[grid](
            r32, lf, win.t().contiguous(), bin_, wo.t().contiguous(), bo, g1_, b1_, g2_, b2_,
            w1_.t().contiguous(), bm1_, w2_.t().contiguous(), bm2_,
            qkv, o, x1, uu, mu1, rs1, mu2, rs2, sn, cnt, B, T, M, D, heads, hd, ns, hd ** -0.5, eps,
            MP=mp, R=rows, BN=bn, BK=bk, HDP=hdp, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(r32, lf, qkv, o, x1, uu, mu1, rs1, mu2, rs2, *p)
        ctx.meta = (heads, prec, r.dtype, l0.dtype, [x.dtype for x in (g1, b1, w_in, b_in, w_o, b_o,
                                                                          g2, b2, w1, bm1, w2, bm2)])
        return lf[:, 1:]

    @staticmethod
    def backward(ctx, dlo):
        r32, lf, qkv, o, x1, uu, mu1, rs1, mu2, rs2, *p = ctx.saved_tensors
        g1_, b1_, win, bin_, wo, bo, g2_, b2_, w1_, bm1_, w2_, bm2_ = p
        heads, prec, r_dtype, l_dtype, p_dtypes = ctx.meta
        B, T, M, D = x1.shape
        kw = dict(device=x1.device, dtype=torch.float32)
        dlo = torch.zeros(B, T, M, D, **kw) if dlo is None else _contig(dlo)
        dl, dn2, da, dn1, dx0 = (torch.empty(B, T, M, D, **kw) for _ in range(5))
        du, dqkv = torch.empty(B, T, M, 4 * D, **kw), torch.empty(B, T, M, 3 * D, **kw)
        grid, mp, rows, bn, bk, hdp, ns = _rin_launch(B, M, D, heads, x1.device)
        dob = torch.empty(B, M, D, **kw)
        cnt = torch.zeros(grid[0], device=x1.device, dtype=torch.int32)
        hd = D // heads
        _rin_bwd_kernel[grid](
            dlo, r32, lf, win, wo, g1_, g2_, w1_, w2_, qkv, x1, uu, mu1, rs1, mu2, rs2,
            dl, du, dn2, da, dqkv, dn1, dx0, dob, cnt, B, T, M, D, heads, hd, ns, hd ** -0.5,
            MP=mp, R=rows, BN=bn, BK=bk, HDP=hdp, PREC=prec, num_warps=4, num_stages=1)
        flat = lambda t: t.reshape(-1, t.shape[-1])
        xh1 = (lf[:, :-1] + r32[:, :, None] - mu1[..., None]) * rs1[..., None]
        xh2 = (x1 - mu2[..., None]) * rs2[..., None]
        n1, n2 = xh1 * g1_ + b1_, xh2 * g2_ + b2_
        g = F.gelu(uu)
        grads = (
            (flat(dn1) * flat(xh1)).sum(0), flat(dn1).sum(0),
            flat(dqkv).t() @ flat(n1), flat(dqkv).sum(0),
            flat(da).t() @ flat(o), flat(da).sum(0),
            (flat(dn2) * flat(xh2)).sum(0), flat(dn2).sum(0),
            flat(du).t() @ flat(n2), flat(du).sum(0),
            flat(dl).t() @ flat(g), flat(dl).sum(0),
        )
        grads = tuple(gr.to(dt) for gr, dt in zip(grads, p_dtypes))
        return (dx0.sum(2).to(r_dtype), dx0[:, 0].to(l_dtype), *grads, None, None)


@torch.compiler.disable
def rin_latent_scan(r, l0, g1, b1, w_in, b_in, w_o, b_o, g2, b2, w1, bm1, w2, bm2, heads, eps=1e-5):
    """Sequential RIN latent update over a sequence.
    ``r``: (B, T, D) per-token read added to every latent; ``l0``: (B, M, D)
    initial latents; LN1 ``g1``/``b1``, ``nn.MultiheadAttention`` packed
    ``w_in`` (3D, D)/``b_in`` and ``w_o``/``b_o``; LN2 ``g2``/``b2``; MLP
    ``w1`` (4D, D), ``bm1``, ``w2`` (D, 4D), ``bm2`` with exact GELU.
    Returns the latents after every token, (B, T, M, D)."""
    return _RINLatentScan.apply(r, l0, g1, b1, w_in, b_in, w_o, b_o, g2, b2, w1, bm1, w2, bm2,
                                int(heads), float(eps))


def rin_latent_reference(r, l0, g1, b1, w_in, b_in, w_o, b_o, g2, b2, w1, bm1, w2, bm2, heads, eps=1e-5):
    lat, outs = l0.float(), []
    B, M, D = lat.shape
    hd = D // heads
    for t in range(r.size(1)):
        x0 = lat + r[:, t, None].float()
        q, k, v = F.linear(F.layer_norm(x0, (D,), g1, b1, eps), w_in, b_in).chunk(3, -1)
        split = lambda z: z.view(B, M, heads, hd).transpose(1, 2)
        att = torch.softmax(split(q) @ split(k).transpose(-1, -2) * hd ** -0.5, -1) @ split(v)
        x1 = x0 + F.linear(att.transpose(1, 2).reshape(B, M, D), w_o, b_o)
        lat = x1 + F.linear(F.gelu(F.linear(F.layer_norm(x1, (D,), g2, b2, eps), w1, bm1)), w2, bm2)
        outs.append(lat)
    return torch.stack(outs, 1)


# ============================================================ CfC / NRU / latent GRU
# Three dense-recurrence cells on the split-column pattern of the LMU kernel:
# every batch tile of 16 rows gets NS programs, each owning a set of hidden
# columns, and the programs meet at _group_barrier whenever a phase needs whole
# rows written by the others.  Weights arrive transposed for the forward
# (K x N, row-major) and as stored for the backward.
if HAS_TRITON:

    @triton.jit
    def _dot_cols(A, lda, W, ldw, brow, bmask, K, ns, nm,
                  BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        """A[brow, :K] @ W[:K, ns] with A written by other programs."""
        acc = tl.zeros([BB, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            km = ks < K
            a = tl.load(A + brow[:, None] * lda + ks[None, :], mask=bmask[:, None] & km[None, :], other=0.0,
                        cache_modifier=".cg")
            w = tl.load(W + ks[:, None] * ldw + ns[None, :], mask=km[:, None] & nm[None, :], other=0.0)
            acc = tl.dot(a, w, acc, input_precision=PREC)
        return acc

    # ---------------------------------------------------------------- CfC
    # h' = s * tanh(a) + (1 - s) * tanh(b) * h,  s = sigmoid(g),
    # [g, a, b] = px + h W^T.
    @triton.jit
    def _cfc_fwd_kernel(PX, WT, HF, PRE, CNT, B, T, H, NS,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for t in range(T):
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                hrow = HF + t * H
                px = PX + (brow[:, None] * T + t) * G + ns[None, :]
                pre = PRE + (brow[:, None] * T + t) * G + ns[None, :]
                pg = _dot_cols(hrow, (T + 1) * H, WT, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                pa = _dot_cols(hrow, (T + 1) * H, WT + H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                pb = _dot_cols(hrow, (T + 1) * H, WT + 2 * H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px + 2 * H, mask=tile, other=0.0).to(tl.float32)
                tl.store(pre, pg, mask=tile)
                tl.store(pre + H, pa, mask=tile)
                tl.store(pre + 2 * H, pb, mask=tile)
                s = tl.sigmoid(pg)
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0,
                            cache_modifier=".cg")
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         s * _tanh(pa) + (1.0 - s) * _tanh(pb) * h, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    @triton.jit
    def _cfc_bwd_kernel(DY, W, HF, PRE, DPRE, DHD, CH, CNT, B, T, H, NS,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for k in range(T):
            t = T - 1 - k
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                g = tl.load(CH + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                pre = PRE + (brow[:, None] * T + t) * G + ns[None, :]
                s = tl.sigmoid(tl.load(pre, mask=tile, other=0.0))
                a = _tanh(tl.load(pre + H, mask=tile, other=0.0))
                b = _tanh(tl.load(pre + 2 * H, mask=tile, other=0.0))
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                d = DPRE + (brow[:, None] * T + t) * G + ns[None, :]
                tl.store(d, g * (a - b * h) * s * (1.0 - s), mask=tile)
                tl.store(d + H, g * s * (1.0 - a * a), mask=tile)
                tl.store(d + 2 * H, g * (1.0 - s) * h * (1.0 - b * b), mask=tile)
                tl.store(DHD + brow[:, None] * H + ns[None, :], g * (1.0 - s) * b, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                acc = _dot_cols(DPRE + t * G, T * G, W, H, brow, bmask, G, ns, nm, BB, BN, BK, PREC)
                tl.store(CH + brow[:, None] * H + ns[None, :],
                         acc + tl.load(DHD + brow[:, None] * H + ns[None, :], mask=tile, other=0.0), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    # ---------------------------------------------------------------- NRU
    # [a, w, r] = px + h W^T;  h' = h + tanh(a) w/|w| - (h . r/|r|) r/|r|.
    @triton.jit
    def _nru_row_sums(PRE, pre_row, HF, h_row, G_SRC, g_row, DY, dy_row, HAS_G: tl.constexpr,
                      bmask, H, BB: tl.constexpr, BN: tl.constexpr):
        """Whole-row sums: |w|^2, |r|^2, h.r and, with a gradient, g.r and g.tanh(a).w."""
        sw = tl.zeros([BB], dtype=tl.float32)
        sr = tl.zeros([BB], dtype=tl.float32)
        hr = tl.zeros([BB], dtype=tl.float32)
        gr = tl.zeros([BB], dtype=tl.float32)
        gtw = tl.zeros([BB], dtype=tl.float32)
        for n0 in range(0, H, BN):
            ns = n0 + tl.arange(0, BN)
            tile = bmask[:, None] & (ns < H)[None, :]
            base = PRE + pre_row[:, None] + ns[None, :]
            a = tl.load(base, mask=tile, other=0.0, cache_modifier=".cg")
            w = tl.load(base + H, mask=tile, other=0.0, cache_modifier=".cg")
            r = tl.load(base + 2 * H, mask=tile, other=0.0, cache_modifier=".cg")
            h = tl.load(HF + h_row[:, None] + ns[None, :], mask=tile, other=0.0, cache_modifier=".cg")
            sw += tl.sum(w * w, axis=1)
            sr += tl.sum(r * r, axis=1)
            hr += tl.sum(h * r, axis=1)
            if HAS_G:
                g = tl.load(G_SRC + g_row[:, None] + ns[None, :], mask=tile, other=0.0, cache_modifier=".cg") \
                    + tl.load(DY + dy_row[:, None] + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                gr += tl.sum(g * r, axis=1)
                gtw += tl.sum(g * _tanh(a) * w, axis=1)
        return tl.sqrt(sw), tl.sqrt(sr), hr, gr, gtw

    @triton.jit
    def _nru_fwd_kernel(PX, WT, HF, PRE, CNT, B, T, H, NS, eps,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for t in range(T):
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                px = PX + (brow[:, None] * T + t) * G + ns[None, :]
                pre = PRE + (brow[:, None] * T + t) * G + ns[None, :]
                for c in range(3):
                    acc = _dot_cols(HF + t * H, (T + 1) * H, WT + c * H, G, brow, bmask, H, ns, nm,
                                    BB, BN, BK, PREC)
                    tl.store(pre + c * H, acc + tl.load(px + c * H, mask=tile, other=0.0).to(tl.float32),
                             mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            nw, nr, hr, gr, gtw = _nru_row_sums(PRE, (brow * T + t) * G, HF, (brow * (T + 1) + t) * H,
                                                PRE, brow, PRE, brow, False, bmask, H, BB, BN)
            nw = tl.maximum(nw, eps)
            nr = tl.maximum(nr, eps)
            s = hr / nr
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                pre = PRE + (brow[:, None] * T + t) * G + ns[None, :]
                a = tl.load(pre, mask=tile, other=0.0)
                w = tl.load(pre + H, mask=tile, other=0.0)
                r = tl.load(pre + 2 * H, mask=tile, other=0.0)
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0,
                            cache_modifier=".cg")
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :],
                         h + _tanh(a) * w / nw[:, None] - s[:, None] * r / nr[:, None], mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    @triton.jit
    def _nru_bwd_kernel(DY, W, HF, PRE, DPRE, DHD, CH, CNT, B, T, H, NS, eps,
                        BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for k in range(T):
            t = T - 1 - k
            nw_raw, nr_raw, hr, gr, gtw = _nru_row_sums(
                PRE, (brow * T + t) * G, HF, (brow * (T + 1) + t) * H,
                CH, brow * H, DY, (brow * T + t) * H, True, bmask, H, BB, BN)
            nw = tl.maximum(nw_raw, eps)
            nr = tl.maximum(nr_raw, eps)
            s = hr / nr               # h . r_hat
            g_r = gr / nr             # g . r_hat
            w_dw = gtw / nw           # w_hat . d(w_hat)
            r_dr = -2.0 * s * g_r     # r_hat . d(r_hat)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                g = tl.load(CH + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                pre = PRE + (brow[:, None] * T + t) * G + ns[None, :]
                ta = _tanh(tl.load(pre, mask=tile, other=0.0))
                w_hat = tl.load(pre + H, mask=tile, other=0.0) / nw[:, None]
                r_hat = tl.load(pre + 2 * H, mask=tile, other=0.0) / nr[:, None]
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                d_what = g * ta
                d_rhat = -(s[:, None] * g + g_r[:, None] * h)
                # F.normalize divides by max(|x|, eps); below eps it is a plain scale.
                dw = tl.where((nw_raw > eps)[:, None], (d_what - w_hat * w_dw[:, None]), d_what) / nw[:, None]
                dr = tl.where((nr_raw > eps)[:, None], (d_rhat - r_hat * r_dr[:, None]), d_rhat) / nr[:, None]
                d = DPRE + (brow[:, None] * T + t) * G + ns[None, :]
                tl.store(d, g * w_hat * (1.0 - ta * ta), mask=tile)
                tl.store(d + H, dw, mask=tile)
                tl.store(d + 2 * H, dr, mask=tile)
                tl.store(DHD + brow[:, None] * H + ns[None, :], g - g_r[:, None] * r_hat, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                acc = _dot_cols(DPRE + t * G, T * G, W, H, brow, bmask, G, ns, nm, BB, BN, BK, PREC)
                tl.store(CH + brow[:, None] * H + ns[None, :],
                         acc + tl.load(DHD + brow[:, None] * H + ns[None, :], mask=tile, other=0.0), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    # ---------------------------------------------------------------- latent GRU (VRNN / SRNN prior path)
    # [mu, ls] = h Wp^T + bp;  z = mu (VRNN) or tanh(mu) sigmoid(-ls) (SRNN);
    # h' = GRUCell([y, z], h) with the y part of the input projection in px.
    @triton.jit
    def _lat_fwd_kernel(PX, WPT, BP, WZT, WHT, BH, HF, Z, P, GI, GH, CNT, B, T, H, NS,
                        SRNN: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for t in range(T):
            hrow = HF + t * H
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                mu = _dot_cols(hrow, (T + 1) * H, WPT, 2 * H, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(BP + ns, mask=nm, other=0.0)[None, :]
                ls = _dot_cols(hrow, (T + 1) * H, WPT + H, 2 * H, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(BP + H + ns, mask=nm, other=0.0)[None, :]
                p = P + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                tl.store(p, mu, mask=tile)
                tl.store(p + H, ls, mask=tile)
                z = _tanh(mu) * tl.sigmoid(-ls) if SRNN else mu
                tl.store(Z + (brow[:, None] * T + t) * H + ns[None, :], z, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                px = PX + (brow[:, None] * T + t) * G + ns[None, :]
                gi = GI + (brow[:, None] * T + t) * G + ns[None, :]
                gh = GH + (brow[:, None] * T + t) * G + ns[None, :]
                zrow = Z + t * H
                i_r = _dot_cols(zrow, T * H, WZT, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px, mask=tile, other=0.0).to(tl.float32)
                i_z = _dot_cols(zrow, T * H, WZT + H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px + H, mask=tile, other=0.0).to(tl.float32)
                i_n = _dot_cols(zrow, T * H, WZT + 2 * H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(px + 2 * H, mask=tile, other=0.0).to(tl.float32)
                h_r = _dot_cols(hrow, (T + 1) * H, WHT, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(BH + ns, mask=nm, other=0.0)[None, :]
                h_z = _dot_cols(hrow, (T + 1) * H, WHT + H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(BH + H + ns, mask=nm, other=0.0)[None, :]
                h_n = _dot_cols(hrow, (T + 1) * H, WHT + 2 * H, G, brow, bmask, H, ns, nm, BB, BN, BK, PREC) \
                    + tl.load(BH + 2 * H + ns, mask=nm, other=0.0)[None, :]
                tl.store(gi, i_r, mask=tile)
                tl.store(gi + H, i_z, mask=tile)
                tl.store(gi + 2 * H, i_n, mask=tile)
                tl.store(gh, h_r, mask=tile)
                tl.store(gh + H, h_z, mask=tile)
                tl.store(gh + 2 * H, h_n, mask=tile)
                r = tl.sigmoid(i_r + h_r)
                u = tl.sigmoid(i_z + h_z)
                n = _tanh(i_n + r * h_n)
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0,
                            cache_modifier=".cg")
                tl.store(HF + (brow[:, None] * (T + 1) + t + 1) * H + ns[None, :], (1.0 - u) * n + u * h,
                         mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)

    @triton.jit
    def _lat_bwd_kernel(DY, WP, WZ, WH, HF, P, GI, GH, DGI, DGH, DP, DHD, CH, CNT, B, T, H, NS,
                        SRNN: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        PREC: tl.constexpr):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        cnt = CNT + pid
        brow = pid * BB + tl.arange(0, BB)
        bmask = brow < B
        G = 3 * H
        phase = 0
        for k in range(T):
            t = T - 1 - k
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                tile = bmask[:, None] & (ns < H)[None, :]
                g = tl.load(CH + brow[:, None] * H + ns[None, :], mask=tile, other=0.0) \
                    + tl.load(DY + (brow[:, None] * T + t) * H + ns[None, :], mask=tile, other=0.0).to(tl.float32)
                gi = GI + (brow[:, None] * T + t) * G + ns[None, :]
                gh = GH + (brow[:, None] * T + t) * G + ns[None, :]
                h_n = tl.load(gh + 2 * H, mask=tile, other=0.0)
                r = tl.sigmoid(tl.load(gi, mask=tile, other=0.0) + tl.load(gh, mask=tile, other=0.0))
                u = tl.sigmoid(tl.load(gi + H, mask=tile, other=0.0) + tl.load(gh + H, mask=tile, other=0.0))
                n = _tanh(tl.load(gi + 2 * H, mask=tile, other=0.0) + r * h_n)
                h = tl.load(HF + (brow[:, None] * (T + 1) + t) * H + ns[None, :], mask=tile, other=0.0)
                d_n = g * (1.0 - u) * (1.0 - n * n)
                d_u = g * (h - n) * u * (1.0 - u)
                d_r = d_n * h_n * r * (1.0 - r)
                dgi = DGI + (brow[:, None] * T + t) * G + ns[None, :]
                dgh = DGH + (brow[:, None] * T + t) * G + ns[None, :]
                tl.store(dgi, d_r, mask=tile)
                tl.store(dgi + H, d_u, mask=tile)
                tl.store(dgi + 2 * H, d_n, mask=tile)
                tl.store(dgh, d_r, mask=tile)
                tl.store(dgh + H, d_u, mask=tile)
                tl.store(dgh + 2 * H, d_n * r, mask=tile)
                tl.store(DHD + brow[:, None] * H + ns[None, :], g * u, mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                dz = _dot_cols(DGI + t * G, T * G, WZ, H, brow, bmask, G, ns, nm, BB, BN, BK, PREC)
                dp = DP + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                if SRNN:
                    p = P + (brow[:, None] * T + t) * 2 * H + ns[None, :]
                    tm = _tanh(tl.load(p, mask=tile, other=0.0))
                    sg = tl.sigmoid(-tl.load(p + H, mask=tile, other=0.0))
                    tl.store(dp, dz * (1.0 - tm * tm) * sg, mask=tile)
                    tl.store(dp + H, -dz * tm * sg * (1.0 - sg), mask=tile)
                else:
                    tl.store(dp, dz, mask=tile)
                    tl.store(dp + H, tl.zeros([BB, BN], dtype=tl.float32), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)
            for n0 in range(split * BN, H, NS * BN):
                ns = n0 + tl.arange(0, BN)
                nm = ns < H
                tile = bmask[:, None] & nm[None, :]
                acc = _dot_cols(DGH + t * G, T * G, WH, H, brow, bmask, G, ns, nm, BB, BN, BK, PREC) \
                    + _dot_cols(DP + t * 2 * H, T * 2 * H, WP, H, brow, bmask, 2 * H, ns, nm, BB, BN, BK, PREC)
                tl.store(CH + brow[:, None] * H + ns[None, :],
                         acc + tl.load(DHD + brow[:, None] * H + ns[None, :], mask=tile, other=0.0), mask=tile)
            phase += 1
            _group_barrier(cnt, phase * NS)


def _cell_launch(B, H, device):
    """(grid, BB, BN, BK, NS) for the split-column cells, grid co-resident."""
    bb, bn, bk = 16, 16, min(64, triton.next_power_of_2(max(H, 16)))
    tiles = triton.cdiv(B, bb)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ns = max(1, min(sms // tiles, triton.cdiv(H, bn)))
    return (tiles, ns), bb, bn, bk, ns


class _GatedCellScan(torch.autograd.Function):
    """CfC (kind 0) and NRU (kind 1): px (B,T,3H), w (3H,H), h0 (B,H)."""
    @staticmethod
    def forward(ctx, px, w, h0, kind):
        px = px.contiguous()
        B, T, G = px.shape
        H = G // 3
        kw = dict(device=px.device, dtype=torch.float32)
        w32 = w.float().contiguous()
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        pre = torch.empty(B, T, G, **kw)
        grid, bb, bn, bk, ns = _cell_launch(B, H, px.device)
        cnt = torch.zeros(grid[0], device=px.device, dtype=torch.int32)
        prec = _dot_precision()
        if kind == 0:
            _cfc_fwd_kernel[grid](px, w32.t().contiguous(), hf, pre, cnt, B, T, H, ns,
                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        else:
            _nru_fwd_kernel[grid](px, w32.t().contiguous(), hf, pre, cnt, B, T, H, ns, 1e-12,
                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(w32, hf, pre)
        ctx.meta = (kind, prec, px.dtype, h0.dtype)
        return hf[:, 1:], hf[:, -1]

    @staticmethod
    def backward(ctx, dy, dh_last):
        w32, hf, pre = ctx.saved_tensors
        kind, prec, px_dtype, h_dtype = ctx.meta
        B, T1, H = hf.shape
        T = T1 - 1
        kw = dict(device=hf.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        ch = torch.zeros(B, H, **kw) if dh_last is None else dh_last.float().clone()
        dpre, dhd = torch.empty(B, T, 3 * H, **kw), torch.empty(B, H, **kw)
        grid, bb, bn, bk, ns = _cell_launch(B, H, hf.device)
        cnt = torch.zeros(grid[0], device=hf.device, dtype=torch.int32)
        if kind == 0:
            _cfc_bwd_kernel[grid](dy, w32, hf, pre, dpre, dhd, ch, cnt, B, T, H, ns,
                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        else:
            _nru_bwd_kernel[grid](dy, w32, hf, pre, dpre, dhd, ch, cnt, B, T, H, ns, 1e-12,
                                  BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        dw = dpre.reshape(-1, 3 * H).t() @ hf[:, :-1].reshape(-1, H)
        return dpre.to(px_dtype), dw, ch.to(h_dtype), None


@torch.compiler.disable
def cfc_scan(px, w, h0):
    """CfC layer: [g, a, b] = px + h W^T,  s = sigmoid(g),
    h' = s tanh(a) + (1 - s) tanh(b) h.  ``px``: (B, T, 3H) input part with
    biases; ``w``: (3H, H) the h-halves of the gate / a / b weights.
    Returns (h_1..h_T, h_T)."""
    return _GatedCellScan.apply(px, w, h0, 0)


def cfc_reference(px, w, h0):
    h, hs = h0.float(), []
    for t in range(px.size(1)):
        g, a, b = (px[:, t].float() + h @ w.t()).chunk(3, -1)
        s = torch.sigmoid(g)
        h = s * torch.tanh(a) + (1.0 - s) * torch.tanh(b) * h
        hs.append(h)
    return torch.stack(hs, 1), h


@torch.compiler.disable
def nru_scan(px, w, h0):
    """NRU layer: [a, w, r] = px + h W^T,
    h' = h + tanh(a) normalize(w) - (h . normalize(r)) normalize(r).
    ``px``: (B, T, 3H) input part with bias; ``w``: (3H, H).  Returns (h_1..h_T, h_T)."""
    return _GatedCellScan.apply(px, w, h0, 1)


def nru_reference(px, w, h0):
    h, hs = h0.float(), []
    for t in range(px.size(1)):
        a, wv, r = (px[:, t].float() + h @ w.t()).chunk(3, -1)
        write, read = F.normalize(wv, dim=-1), F.normalize(r, dim=-1)
        h = h + torch.tanh(a) * write - (h * read).sum(-1, keepdim=True) * read
        hs.append(h)
    return torch.stack(hs, 1), h


class _LatentGRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, wp, bp, wz, wh, bh, h0, srnn):
        px = px.contiguous()
        B, T, G = px.shape
        H = G // 3
        kw = dict(device=px.device, dtype=torch.float32)
        f32 = lambda t: t.float().contiguous()
        wp32, bp32, wz32, wh32, bh32 = f32(wp), f32(bp), f32(wz), f32(wh), f32(bh)
        hf = torch.empty(B, T + 1, H, **kw); hf[:, 0] = h0.float()
        z, p = torch.empty(B, T, H, **kw), torch.empty(B, T, 2 * H, **kw)
        gi, gh = torch.empty(B, T, G, **kw), torch.empty(B, T, G, **kw)
        grid, bb, bn, bk, ns = _cell_launch(B, H, px.device)
        cnt = torch.zeros(grid[0], device=px.device, dtype=torch.int32)
        prec = _dot_precision()
        _lat_fwd_kernel[grid](px, wp32.t().contiguous(), bp32, wz32.t().contiguous(), wh32.t().contiguous(),
                              bh32, hf, z, p, gi, gh, cnt, B, T, H, ns,
                              SRNN=bool(srnn), BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        ctx.save_for_backward(wp32, wz32, wh32, hf, z, p, gi, gh)
        ctx.meta = (bool(srnn), prec, px.dtype, h0.dtype)
        return hf[:, 1:], hf[:, -1]

    @staticmethod
    def backward(ctx, dy, dh_last):
        wp32, wz32, wh32, hf, z, p, gi, gh = ctx.saved_tensors
        srnn, prec, px_dtype, h_dtype = ctx.meta
        B, T1, H = hf.shape
        T = T1 - 1
        kw = dict(device=hf.device, dtype=torch.float32)
        dy = torch.zeros(B, T, H, **kw) if dy is None else _contig(dy)
        ch = torch.zeros(B, H, **kw) if dh_last is None else dh_last.float().clone()
        dgi, dgh = torch.empty(B, T, 3 * H, **kw), torch.empty(B, T, 3 * H, **kw)
        dp, dhd = torch.empty(B, T, 2 * H, **kw), torch.empty(B, H, **kw)
        grid, bb, bn, bk, ns = _cell_launch(B, H, hf.device)
        cnt = torch.zeros(grid[0], device=hf.device, dtype=torch.int32)
        _lat_bwd_kernel[grid](dy, wp32, wz32, wh32, hf, p, gi, gh, dgi, dgh, dp, dhd, ch, cnt, B, T, H, ns,
                              SRNN=srnn, BB=bb, BN=bn, BK=bk, PREC=prec, num_warps=4, num_stages=1)
        flat = lambda t: t.reshape(-1, t.shape[-1])
        h_prev = flat(hf[:, :-1])
        return (dgi.to(px_dtype), flat(dp).t() @ h_prev, flat(dp).sum(0), flat(dgi).t() @ flat(z),
                flat(dgh).t() @ h_prev, flat(dgh).sum(0), ch.to(h_dtype), None)


@torch.compiler.disable
def latent_gru_scan(px, wp, bp, wz, wh, bh, h0, srnn=False):
    """VRNN / SRNN prior-path layer.  [mu, ls] = h Wp^T + bp; the latent is
    z = mu (VRNN) or tanh(mu) sigmoid(-ls) (SRNN); then h' = GRUCell([y, z], h).
    ``px``: (B, T, 3H) = y W_ih[:, :H]^T + b_ih; ``wp`` (2H, H), ``bp``;
    ``wz`` = W_ih[:, H:] (3H, H); ``wh`` = W_hh (3H, H), ``bh`` = b_hh.
    Returns (h_1..h_T, h_T)."""
    return _LatentGRUScan.apply(px, wp, bp, wz, wh, bh, h0, bool(srnn))


def latent_gru_reference(px, wp, bp, wz, wh, bh, h0, srnn=False):
    h, hs = h0.float(), []
    for t in range(px.size(1)):
        mu, ls = (h @ wp.t() + bp).chunk(2, -1)
        z = torch.tanh(mu) * torch.sigmoid(-ls) if srnn else mu
        i_r, i_z, i_n = (px[:, t].float() + z @ wz.t()).chunk(3, -1)
        h_r, h_z, h_n = (h @ wh.t() + bh).chunk(3, -1)
        r, u = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
        h = (1.0 - u) * torch.tanh(i_n + r * h_n) + u * h
        hs.append(h)
    return torch.stack(hs, 1), h


# ===================================================================== ParaRNN
# ParaRNN (Danieli et al., ICLR 2026, arXiv 2510.21450; apple/ml-pararnn)
# trains a nonlinear RNN in parallel over the sequence: the recurrence
# h_t = f(h_{t-1}, x_t) for all t is one nonlinear system, solved by a few
# Newton iterations.  Each iteration's linearisation is a linear recurrence
#     delta_t = J_t delta_{t-1} + r_t,   J_t = df/dh at h_{t-1},  r_t = f(h_{t-1}) - h_t,
# solved by a parallel scan.  Diagonal state matrices keep J_t diagonal
# (ParaGRU) or made of 2 x 2 diagonal blocks (ParaLSTM, state [c, h]).  The
# backward pass needs no Newton at all: dL/dh_t = g_t + J_{t+1}^T dL/dh_{t+1}
# is one linear recurrence solved in reverse.
#
# The scan kernels follow the paper's long-sequence scheme (App. D.2): each
# program owns a tile of channels and scans a chunk of steps in parallel
# (tl.associative_scan), carrying the state sequentially across chunks.
if HAS_TRITON:

    @triton.jit
    def _linrec_combine(a1, b1, a2, b2):
        return a1 * a2, b1 * a2 + b2

    @triton.jit
    def _linrec_diag_kernel(A, Bv, H0, H, T, D, CH: tl.constexpr, BD: tl.constexpr):
        b = tl.program_id(0)
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        cm = cols < D
        rows = tl.arange(0, CH)
        carry = tl.load(H0 + b * D + cols, mask=cm, other=0.0).to(tl.float32)
        for start in range(0, T, CH):
            t = start + rows
            m = (t < T)[:, None] & cm[None, :]
            idx = (b * T + t)[:, None] * D + cols[None, :]
            a = tl.load(A + idx, mask=m, other=1.0).to(tl.float32)
            v = tl.load(Bv + idx, mask=m, other=0.0).to(tl.float32)
            a_cum, h_loc = tl.associative_scan((a, v), 0, _linrec_combine)
            h = a_cum * carry[None, :] + h_loc
            tl.store(H + idx, h, mask=m)
            last = tl.minimum(T - start, CH) - 1
            carry = tl.sum(tl.where((rows == last)[:, None], h, 0.0), axis=0)

    @triton.jit
    def _linrec_2x2_combine(p1, q1, r1, s1, u1, w1, p2, q2, r2, s2, u2, w2):
        # (M, v) then (N, y):  (N M, N v + y) with M = [[p, q], [r, s]], v = [u, w]
        return (p2 * p1 + q2 * r1, p2 * q1 + q2 * s1, r2 * p1 + s2 * r1, r2 * q1 + s2 * s1,
                p2 * u1 + q2 * w1 + u2, r2 * u1 + s2 * w1 + w2)

    @triton.jit
    def _linrec_2x2_kernel(J, R, H0, H, T, D, CH: tl.constexpr, BD: tl.constexpr):
        # J: (B, T, 4, D) as [Jcc, Jch, Jhc, Jhh]; R, H: (B, T, 2, D); H0: (B, 2, D)
        b = tl.program_id(0)
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        cm = cols < D
        rows = tl.arange(0, CH)
        c_carry = tl.load(H0 + b * 2 * D + cols, mask=cm, other=0.0).to(tl.float32)
        h_carry = tl.load(H0 + b * 2 * D + D + cols, mask=cm, other=0.0).to(tl.float32)
        for start in range(0, T, CH):
            t = start + rows
            m = (t < T)[:, None] & cm[None, :]
            jb = (b * T + t)[:, None] * 4 * D + cols[None, :]
            rb = (b * T + t)[:, None] * 2 * D + cols[None, :]
            p = tl.load(J + jb, mask=m, other=1.0).to(tl.float32)
            q = tl.load(J + jb + D, mask=m, other=0.0).to(tl.float32)
            r = tl.load(J + jb + 2 * D, mask=m, other=0.0).to(tl.float32)
            s = tl.load(J + jb + 3 * D, mask=m, other=1.0).to(tl.float32)
            u = tl.load(R + rb, mask=m, other=0.0).to(tl.float32)
            w = tl.load(R + rb + D, mask=m, other=0.0).to(tl.float32)
            p, q, r, s, u, w = tl.associative_scan((p, q, r, s, u, w), 0, _linrec_2x2_combine)
            c = p * c_carry[None, :] + q * h_carry[None, :] + u
            h = r * c_carry[None, :] + s * h_carry[None, :] + w
            tl.store(H + rb, c, mask=m)
            tl.store(H + rb + D, h, mask=m)
            last = (rows == tl.minimum(T - start, CH) - 1)[:, None]
            c_carry = tl.sum(tl.where(last, c, 0.0), axis=0)
            h_carry = tl.sum(tl.where(last, h, 0.0), axis=0)


def _linrec_tiles(T, per_step):
    ch = min(256, triton.next_power_of_2(max(T, 16)))
    bd = max(1, min(32, 2048 // (ch * per_step)))
    return ch, 1 << (bd.bit_length() - 1)          # tl.arange needs a power of two


def linrec_diag(a, b, h0):
    """h_t = a_t * h_{t-1} + b_t over (B, T, D), fp32 out, no autograd."""
    B, T, D = b.shape
    if kernels_available(b):
        h = torch.empty(B, T, D, device=b.device, dtype=torch.float32)
        ch, bd = _linrec_tiles(T, 2)
        _linrec_diag_kernel[(B, triton.cdiv(D, bd))](a.contiguous(), b.contiguous(), h0.contiguous(), h, T, D,
                                                    CH=ch, BD=bd)
        return h
    # Hillis-Steele log-depth scan (same result, O(T log T) work).
    a, h = a.float().clone(), b.float().clone()
    h[:, 0] += a[:, 0] * h0.float()
    a[:, 0] = 0.0
    step = 1
    while step < T:
        h = torch.cat((h[:, :step], h[:, step:] + a[:, step:] * h[:, :-step]), 1)
        a = torch.cat((a[:, :step], a[:, step:] * a[:, :-step]), 1)
        step *= 2
    return h


def linrec_2x2(J, r, h0):
    """[c; h]_t = J_t [c; h]_{t-1} + r_t per channel, J_t made of 2 x 2
    diagonal blocks.  J: (B, T, 4, D) as [Jcc, Jch, Jhc, Jhh]; r: (B, T, 2, D);
    h0: (B, 2, D).  fp32 out, no autograd."""
    B, T, _, D = r.shape
    if kernels_available(r):
        h = torch.empty(B, T, 2, D, device=r.device, dtype=torch.float32)
        ch, bd = _linrec_tiles(T, 6)
        _linrec_2x2_kernel[(B, triton.cdiv(D, bd))](J.contiguous(), r.contiguous(), h0.contiguous(), h, T, D,
                                                   CH=ch, BD=bd)
        return h
    M = J.float().view(B, T, 2, 2, D).clone()
    v = r.float().clone()
    v[:, 0] += torch.einsum("bijd,bjd->bid", M[:, 0], h0.float())
    M[:, 0] = 0.0
    step = 1
    while step < T:
        v = torch.cat((v[:, :step], v[:, step:] + torch.einsum("btijd,btjd->btid", M[:, step:], v[:, :-step])), 1)
        M = torch.cat((M[:, :step], torch.einsum("btijd,btjkd->btikd", M[:, step:], M[:, :-step])), 1)
        step *= 2
    return v


def _dsig(s):
    return s * (1 - s)


class ParaGRUCell:
    """Diagonal GRU (ParaGRU, paper Eq. 3.1a with A = diag(a)).  xp: (B, T, 3, D)
    input projections B x + b for [z, r, c]; a: (3, D); state h: (B, T, D).
        z = s(a_z h + xp_z), r = s(a_r h + xp_r), c = tanh(a_c (h r) + xp_c),
        h' = (1 - z) h + z c."""
    blocks = 1

    @staticmethod
    def step(xp, h, a):
        z = torch.sigmoid(a[0] * h + xp[..., 0, :])
        r = torch.sigmoid(a[1] * h + xp[..., 1, :])
        c = torch.tanh(a[2] * h * r + xp[..., 2, :])
        return (1 - z) * h + z * c

    @staticmethod
    def jacobian(xp, h, a):
        z = torch.sigmoid(a[0] * h + xp[..., 0, :])
        r = torch.sigmoid(a[1] * h + xp[..., 1, :])
        c = torch.tanh(a[2] * h * r + xp[..., 2, :])
        return (1 - z) + (c - h) * _dsig(z) * a[0] + z * (1 - c * c) * a[2] * (r + h * _dsig(r) * a[1])


class ParaLSTMCell:
    """Diagonal CIFG LSTM with peepholes (ParaLSTM, paper Eq. 3.1b with
    A = diag(a), C = diag(c)).  xp: (B, T, 3, D) projections for [f, o, z];
    a: (3, D); p (peepholes): (2, D); state (B, T, 2, D) = [c, h].
        f = s(a_f h + xp_f + p_f c), z = tanh(a_z h + xp_z),
        c' = f c + (1 - f) z,  o = s(a_o h + xp_o + p_o c'),  h' = o tanh(c')."""
    blocks = 2

    @staticmethod
    def _gates(xp, st, a, p):
        c, h = st[..., 0, :], st[..., 1, :]
        f = torch.sigmoid(a[0] * h + xp[..., 0, :] + p[0] * c)
        z = torch.tanh(a[2] * h + xp[..., 2, :])
        c_new = f * c + (1 - f) * z
        o = torch.sigmoid(a[1] * h + xp[..., 1, :] + p[1] * c_new)
        return c, f, z, c_new, o

    @classmethod
    def step(cls, xp, st, a, p):
        _, _, _, c_new, o = cls._gates(xp, st, a, p)
        return torch.stack((c_new, o * torch.tanh(c_new)), -2)

    @classmethod
    def jacobian(cls, xp, st, a, p):
        c, f, z, c_new, o = cls._gates(xp, st, a, p)
        tc = torch.tanh(c_new)
        j_cc = f + (c - z) * _dsig(f) * p[0]
        j_ch = (c - z) * _dsig(f) * a[0] + (1 - f) * (1 - z * z) * a[2]
        do = tc * _dsig(o)
        dt = o * (1 - tc * tc)
        j_hc = (do * p[1] + dt) * j_cc
        j_hh = do * (a[1] + p[1] * j_ch) + dt * j_ch
        return torch.stack((j_cc, j_ch, j_hc, j_hh), -2)


def _para_shift(sol, h0):
    """h_{t-1} for every t, with the carried initial state at t = 0."""
    return torch.cat((h0.unsqueeze(1).to(sol.dtype), sol[:, :-1]), 1)


def _para_linsolve(cell, J, r, h0_zero):
    return linrec_diag(J, r, h0_zero) if cell.blocks == 1 else linrec_2x2(J, r, h0_zero)


def _para_transpose(cell, J):
    return J if cell.blocks == 1 else J[:, :, [0, 2, 1, 3]]


class _ParaRNNSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cell, newton_iters, xp, h0, *params):
        with torch.no_grad():
            xp32, h032 = xp.float(), h0.float()
            p32 = [q.float() for q in params]
            zero = torch.zeros_like(h032)
            # Initial guess: one step from the carried state / zeros (as in apple/ml-pararnn).
            guess_prev = h032.new_zeros(xp32.shape[0], xp32.shape[1], *h032.shape[1:])
            guess_prev[:, 0] = h032
            sol = cell.step(xp32, guess_prev, *p32)
            for _ in range(newton_iters):
                prev = _para_shift(sol, h032)
                r = cell.step(xp32, prev, *p32) - sol
                J = cell.jacobian(xp32, prev, *p32)
                sol = sol + _para_linsolve(cell, J, r, zero)
        ctx.cell = cell
        ctx.save_for_backward(xp, h0, sol, *params)
        return sol

    @staticmethod
    def backward(ctx, grad_sol):
        xp, h0, sol, *params = ctx.saved_tensors
        cell = ctx.cell
        with torch.no_grad():
            p32 = [q.float() for q in params]
            prev = _para_shift(sol, h0.float())
            J = cell.jacobian(xp.float(), prev, *p32)
            # lambda_t = g_t + J_{t+1}^T lambda_{t+1}: a linear recurrence in reversed time.
            Jn = torch.cat((torch.zeros_like(J[:, :1]), _para_transpose(cell, J[:, 1:]).flip(1)), 1)
            lam = _para_linsolve(cell, Jn, grad_sol.float().flip(1), torch.zeros_like(sol[:, 0])).flip(1)
        with torch.enable_grad():
            xp_ = xp.detach().float().requires_grad_(xp.requires_grad)
            prev_ = _para_shift(sol.detach(), h0.detach().float()).requires_grad_(h0.requires_grad)
            ps = [q.detach().float().requires_grad_(q.requires_grad) for q in params]
            inputs = [t for t in (xp_, prev_, *ps) if t.requires_grad]
            grads = iter(torch.autograd.grad(cell.step(xp_, prev_, *ps), inputs, lam, allow_unused=True))
            g_xp = next(grads) if xp_.requires_grad else None
            g_prev = next(grads) if prev_.requires_grad else None
            g_ps = [next(grads) if q.requires_grad else None for q in ps]
        g_h0 = None if g_prev is None else g_prev[:, 0].to(h0.dtype)
        return (None, None, None if g_xp is None else g_xp.to(xp.dtype), g_h0,
                *[None if g is None else g.to(q.dtype) for g, q in zip(g_ps, params)])


def pararnn_solve(cell, xp, h0, *params, newton_iters=3):
    """Apply ParaGRUCell / ParaLSTMCell over a whole sequence with ParaRNN.
    xp: (B, T, 3, D) input projections; h0: (B, D) or (B, 2, D).  Returns
    every state, (B, T, D) or (B, T, 2, D), in fp32.  The paper found three
    Newton iterations sufficient for these cells (Sec. 2.1)."""
    return _ParaRNNSolve.apply(cell, int(newton_iters), xp, h0, *params)


def pararnn_reference(cell, xp, h0, *params):
    """Sequential application, for checks and single-token decoding."""
    h, out = h0.float(), []
    ps = [q.float() for q in params]
    for t in range(xp.size(1)):
        h = cell.step(xp[:, t].float(), h, *ps)
        out.append(h)
    return torch.stack(out, 1)
