# JiT5 training workflows

Run `conda run --no-capture-output -n pytorch2 python jit5.py`.

The menu accepts `0/train`, `1/sample`, `2/retry`, `3/continue`, and
`4/finetune`. Train retains its existing Continue question for compatibility.

## Configuration and retry

New configurations are saved as readable `JiTDiff_Flow/config.json` and embedded
in checkpoints. Existing `config.pt` files and checkpoints remain readable.
Train and retry first offer a prepared JSON config path. Providing one bypasses
all remaining setup prompts and starts with random weights.

Without a prepared file, retry displays the previous configuration and accepts
only the edits you want, one `key=value` per line. Press Enter to train:

```text
model_type=rin
dim=128
depth=4
batch_size=8
grad_accum_steps=4
use_grad_ckpt=true
synthetic_params.noise_min_sigma=2
```

Values use JSON syntax (bare strings are also accepted). Lists use JSON arrays.
Every field is editable, including nested synthetic parameters. Geometry fields
must remain consistent: for example, changing image size may also require
changing resize/crop/grid fields. Invalid combinations fail before training.
Retry initializes all parameters afresh and removes a previous fine-tuning freeze
policy. It does not load optimizer state or weights.

`jit5_config.example.json` is a small complete starting configuration. Change
`dataset_path` before using it. Boolean fields require `true` or `false` in JSON;
strings such as `"0"` are rejected instead of silently acting as enabled flags.
Interactive boolean prompts accept Enter for their displayed default, including
off defaults.

## EMA warmup

`ema_warmup` chooses how the EMA decay (target 0.9999) ramps up:

- `smooth` (the default for new runs): the decay rises from 0 (copy the model) to
  0.9999 over `ema_warmup_steps` (default 10000) along an S-curve, arriving with
  zero slope, so there is no jump when warmup ends.
- `legacy`: `min(0.9999, (1+n)/(10+n))`, reaching 0.9999 only near step 90k.
  Configs and checkpoints without the field keep this behaviour.
- `none`: 0.9999 from the first update.

## torch.compile

`compile_mode` is `off` (default), `default`, or `max-autotune-no-cudagraphs`.
It is purely a runtime setting: checkpoints are identical with or without it,
and it can be changed under "Edit training and preview settings?" when
continuing. The first steps are slow while kernels compile (tens of seconds;
about 2 minutes for the Swin UNets). `max-autotune-no-cudagraphs` was only
6-13% faster than `default` in testing but compiled 4-6x longer. If compilation
of a graph fails, that graph runs eager instead of stopping training. Compile
workers are capped at 4 processes to limit RAM; set
`TORCHINDUCTOR_COMPILE_THREADS` to override.

Before training, a compiled model is checked once: a synthetic batch of the
real shape is run eager and compiled, and if the gradient norms differ by more
than 2x, training continues eager with a warning. RNG state, buffers and
gradients are restored, so the check does not change the run.

Triton bundles a CUDA 12 `ptxas`. With an older driver (for example 11.7), its
kernels fail with "device kernel image is invalid". When `TRITON_PTXAS_PATH` is
unset and the driver is older than CUDA 12, jit5 points it at a system `ptxas`
no newer than the driver (from `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda*`, or
`PATH`). If none is found, training runs eager.

With that workaround (driver 11.7, Triton 3.0, PyTorch 2.4), compiled Swin v2
under BF16 autocast produced gradients 100-1000x too large at some shapes
(32x32, dim 32, batch 4) while the loss looked normal; FP32 was correct and
the same model was correct at 64x64, dim 64, batch 16. The startup check above
exists for this. Updating the driver to CUDA 12 or newer is the real fix.

Compiling helped most models 1.3-2.2x at 64x64, batch 16 (median 1.5x). It
did not help GRU (cuDNN already fuses it) or Vim, which has its own kernel
(below) and is fastest eager.

## Custom Triton kernels

These run automatically in eager CUDA training, each with a PyTorch fallback
on CPU or when Triton cannot build a loadable kernel. Checkpoints are unchanged.

- **Vim**: the selective scan uses `linegen_kernels.selective_scan` instead of a
  Python loop over tokens: 878 -> 16 ms per step at 64x64.
- **Hyena**: the long convolution runs as a direct Toeplitz product on tensor
  cores (`jit5_kernels.hyena_conv`) for grids up to 4096 tokens, and cuFFT
  beyond. It is 1.3x faster than the FFT at 8x8-24x24 tokens and 4.3x at 32x32.
  It reproduces `conv_fft` exactly, including that path's scale: `norm='ortho'`
  on all three transforms divides the convolution by sqrt((3H-2)(3W-2)).
  Registered with torch.library, so compiled Hyena has no graph breaks.
- **FCDM**: LayerNorm + adaLN modulation and GELU + GRN are fused, forward and
  backward (`jit5_kernels.fcdm_ln_modulate`, `fcdm_gelu_grn`): FCDM-UNet 1.14x
  faster at 64x64 and 1.33x at 128x128, with about a third less memory. Under
  torch.compile the PyTorch path is used, since Inductor fuses it itself.

Gradients whose strides differ from their parameter's only on size-1
dimensions (cuDNN depthwise and 1x1 conv weights, in 16 model types) made
foreach optimizers fall back to one launch per parameter. The training loop
now re-labels them as zero-copy views first; Adan's step became 2.5x faster.

With Triton 3.0 and the CUDA 11.7 `ptxas`, `tl.sum` over a raw `tl.load` inside
a loop miscompiled for some tile shapes, while the source was correct under
`TRITON_INTERPRET=1`. Every reduction in `jit5_kernels.py` therefore sums a
`tl.where(mask, value, 0.0)`. The kernels were checked against their PyTorch
references over 2632 output and gradient tensors spanning channel counts, row
counts at tile boundaries, grid sizes and batch sizes, with no failures.

## ConvNeXt and FCDM models

ConvNeXt now uses channel LayerNorm and GELU. Its AdaLN toggle is honored;
its block-level drop-path argument is applied. Previous RMSNorm/Mish ConvNeXt
checkpoints (including HierMLP models using that mixer) are not parameter-compatible
with this change; use a new run. No automatic conversion is performed.

Two new interactive model choices are available:

| Menu | `model_type` | Architecture |
|---|---|---|
| 46 | `fcdm_unet` | Three-resolution FCDM-UNet |
| 47 | `fcdm_isotropic` | Fixed-resolution FCDM blocks in the JiT patch/stem shell |

Both use 7x7 depthwise convolutions, adaptive LayerNorm, GELU, channel expansion,
GRN, and zero-initialized residual gates. `fcdm_mlp_ratio` controls expansion
(default 3). Time conditioning is always active; class embeddings are added only
for class-conditioned training. Unconditional and Pix2Pix use time alone.
The existing flow wrapper handles label dropout and CFG once for the whole model.

`use_grn` optionally adds FCDM-style Global Response Normalization to the
residual update of every JiT, MLPMixer, gMLP, aMLP, or ViP block (including the
original and convolutional gMLP/aMLP/Mixer variants). It normalizes each channel's
L2 response across the token grid relative to the average channel response. Its
learned scale and bias start at zero, so a newly enabled model begins with the
same forward mapping as without GRN; it is a new architecture and cannot load a
checkpoint created with `use_grn=false`.

FCDM-UNet uses `dim` as base width C and `depth` as base block count L. Widths
are C, 2C, 4C and encoder/bottleneck/decoder depths are L, 2L, 4L, 2L, L.
Down/up paths use convolutions and pixel unshuffle/shuffle with concatenated skips.
Its full-resolution entry, rather than a pretrained VAE, makes it a pixel-space
adaptation. Interactive defaults are C=32, L=2; published latent-model widths
can be expensive at full pixel resolution. Image dimensions must be multiples
of four. It has its own conditioned encoder/decoder and does not accept external
`conv_stem` or pixel self-conditioning. Fine-tuning `last` selects final stages
and the output head.

FCDM-Isotropic uses `dim` and `depth` for constant-width blocks and retains
existing patch geometry, optional convolutional stems, stem conditioning and
pixel self-conditioning. Its final normalization is conditioned LayerNorm;
the image projection starts at zero. This is an FCDM-block adaptation, not an
exact reproduction of the paper's isotropic ablation or latent diffusion setup.

Reference: [paper](https://arxiv.org/abs/2603.09408) and
[official architecture](https://github.com/star-kwon/FCDM/blob/main/models/fcdm_models.py).
No VAE, new dependency, or pretrained teacher is required.

## Convolutional stem conditioning options

`stem_conditioning` is offered during interactive stem setup and is editable in
prepared/retry JSON configurations:

| Value | Conditioned parts |
|---|---|
| 0 | None (default, including older configs/checkpoints) |
| 1 | Encoder only |
| 2 | Decoder only |
| 3 | Skip branches only |
| 4 | Encoder and decoder, no separate skip modulation |
| 5 | Encoder, decoder, and skip branches |

Conditioning uses the timestep embedding plus the class embedding when present.
Encoder/decoder stages apply channel-wise scale and shift after normalization,
before activation. Skip modulation acts on copied branch values after activation;
it does not feed back into the encoder. Encoder conditioning naturally affects
the features from which skips are taken, even in modes 1 and 4.

Each stage has a separate zero-initialized projection, initially giving identity
modulation. Modes require the corresponding convolutional stems; modes 3 and 5
also require `stem_skips=true`. JiT/other patch processors, RIN, and HierMLP are
supported, including timestep-only conditioning for unconditional/Pix2Pix models.

Mode 0 retains the original parameter layout. Nonzero modes add parameters and
are architecture choices for new/retry runs, not runtime continuation switches;
existing strict checkpoint loading does not migrate an unconditioned checkpoint.

`stem_grn` adds FCDM-style Global Response Normalization after the activation of
each input and output stem stage. It is offered during stem setup and can be set
in prepared/retry JSON. Its scale and bias start at zero, so enabling it begins
as an identity mapping; it is nevertheless a new checkpoint architecture.

## Attention normalization

Older attention blocks normalized twice before attention: the block's RMSNorm
(then adaLN modulation), followed by a second RMSNorm inside the attention module
that partly undid the modulation. `attn_double_norm=false` removes the inner norm.
New interactive runs and `jit5_config.example.json` use `false`. Configurations
without the key, including every earlier checkpoint, default to `true` and load
and behave exactly as before. The two settings have different parameters, so
the choice is an architecture setting for new/retry runs; set
`attn_double_norm=false` in retry to adopt it. It affects `jit`, `fullattn`,
`mixer_attn` and HierMLP's `jit` global mixer. (`maxvit` changes with
`arch_version`, below.)

## Reference-faithful HAT, MaxViT, Swin and Vim

`arch_version=2` builds these four families as their papers' official code does;
`arch_version=1` keeps the earlier approximations. New interactive runs use 2;
configurations without the key, including every earlier checkpoint, use 1 and
load and behave exactly as before. Set `arch_version=2` in retry to adopt it.
Without adaLN, each version-2 block matches the official implementation to
floating-point precision (checked against the HAT repository, timm MaxViT,
Microsoft Swin v1/v2, and Vim's reference selective scan). With adaLN, each
pre-norm is modulated and each residual branch gated, as in the other blocks.

| Model | Version 2 |
|---|---|
| `hat` | Each layer is one residual hybrid attention group: `hat_group_depth` HABs (8x8 or 4x4 windows, alternating shifted windows with the attention mask, plus 0.01 x the channel-attention conv branch), an overlapping cross-attention block, and a 3x3 conv with a group residual. The token grid needs sides divisible by 4. The paper uses 6 HABs per group; the default is 2. |
| `maxvit` | MBConv (with BatchNorm and squeeze-excitation, as in the paper), then local 8x8 block attention and dilated 8x8 grid attention, each with relative position bias and an FFN. Grid sides must be multiples of 8. |
| `swin_v1`, `swin_v2` | Shifted windows use the attention mask, and a level no larger than the window uses one unshifted window. Version-2 `swin_v2` adds scaled cosine attention and the log-spaced continuous position bias. |
| `vim` | Vim's bidirectional Mamba (separate conv/SSM parameters per direction, Mamba initialization) in a norm + mixer + residual block with no MLP. The scan is sequential PyTorch, so it is slow. |

MaxViT's BatchNorm pools statistics across a batch's mixed noise levels; that is
faithful to the paper but can suit diffusion less well than the other blocks.

## Pix2Pix sampling scale

Sample mode prepares inputs at the scale training used: the source is resized
onto the training canvas, then scaled by model size / training crop (the crop
closest to the model size). With the default resize = crop = model size this is
an ordinary resize, as before. When the canvas is larger than the model (for
example resize 256, crop 64), choose `tile` to translate the whole image in
model-sized tiles, averaged where edge tiles overlap, or `center` for one centred
window. Tiles are sampled independently, so seams can be visible.

New runs reject `cond_residual=true` with `pred_mode='eps'`: the residual adds the
source image to a noise estimate. Existing checkpoints can still be sampled and
continued.

## Muon-family AdamW fallback learning rate

Muon-family optimizers route embeddings, pixel heads, the time MLP, norms and
biases to an AdamW fallback. Its learning rate is `lr * muon_adam_lr_ratio`.
The interactive prompt asks for the fallback rate directly, defaulting to
4.2e-4 for AdaGO/AdamGO/RMSGO/AdaDeltaGO (whose 0.05 matrix rate is far too
large for Adam) and to the main rate for Muon/AdaMuon/NorMuon. Configurations
without the key use ratio 1.0, the previous shared rate, so older runs resume
unchanged; set `muon_adam_lr_ratio` in retry/prepared JSON (e.g. 0.0084 for
4.2e-4 at lr 0.05) or edit it under Continue's optimizer parameters. Warmup,
cosine decay and its floor scale both groups together. With `muon_all=true`
there is no AdamW group and the ratio is ignored.

## Class sampling

Class-conditioned training defaults to `class_sampling="uniform"`. Each shuffled
round draws one image from every nonempty training class. Images within a class
are shuffled and exhausted before reuse; smaller classes repeat until every class
has contributed as many examples as the largest training class. A balanced epoch
therefore contains class-count times largest-class-size examples. Individual
batches need not be balanced when their size does not match the number of classes.
Repeated images receive separately seeded augmentations, with exact stream resume
across worker counts. Empty known classes are skipped. Corrupt-image recovery stays
within the selected class and split; an unreadable class fails instead of silently
substituting another class.

Set `class_sampling="natural"` in a prepared/retry config for the original image
shuffle. Unconditional and Pix2Pix sampling are unaffected. Validation still
visits each held-out image once, so its aggregate loss is image-weighted.
Balancing increases rare-class exposure, but cannot add diversity to a class with
only a few images; those examples can still overfit.

Continuing an older class checkpoint without this setting enables uniform sampling
and explicitly restarts the data stream, retaining weights and optimizer state.
New checkpoints preserve the balanced stream exactly. Changing sampling policy
on an existing stream requires `reset_data_stream=true`.

## Validation and best weights

Choose `validation_percent` (0 disables splitting) or `validation_path`, not both.
Explicit folders use the same dataset format as training: images, class folders,
paired `A`/`B` folders, combined pairs, or synthetic source images. Use `-` at the
interactive validation-folder prompt to clear a saved folder.

Percentage splitting is reproducible using `validation_seed`; class splits are
stratified and retain at least one training example per class. Singleton classes
stay in training. Pairs remain together, and corrupt-image fallback stays inside
its split. Explicit folders reject overlapping resolved paths and unknown class
labels. Content duplicates stored under different paths and related video frames
are not automatically detected; prepare separate folders for those cases.

`validation_every` counts successful optimizer updates. Set
`validation_batch_size` independently and `validation_max_batches=0` to evaluate
the whole held-out set. Crops, synthetic corruptions, diffusion times, and noise
repeat between evaluations. Validation preserves training random state and module
modes. Changing validation batch size changes the noise assignment, so it resets
the best-score comparison. Validation also runs at the final planned update.
Scheduled previews use held-out examples when available.

With `save_best=true`, validation saves the lowest finite loss to
`JiTDiff_Flow/best.pt`, using EMA weights when EMA is enabled. Without validation,
`best_training_loss=true` enables a smoothed training-loss fallback using raw model
weights. That fallback does not measure generalization. Best-score metadata is
reset when the evaluation context changes.

`best.pt` contains weights, configuration, and score metadata, not optimizer state.
Use it in sample mode or as a fine-tuning source. It costs one extra model's worth
of disk space. `model.pt` remains the resumable checkpoint, with the existing single
backup behavior. Numbered history and separate model directories are unchanged.
Until a new run produces a qualifying best score, an older `best.pt` may remain;
its embedded configuration and metadata describe the run that produced it.
Each existing preview run directory also contains `metrics.csv`.

## Accumulation and continuation

In Continue, answer yes to **Edit training and preview settings?** to change
**Preview CFG strength (1=unguided)**. This appears for class-conditioned models
trained with class dropout. It defaults to the checkpoint's `guidance_scale`,
affects preview generation, and is saved with the continued configuration.

`grad_accum_steps` batches contribute to one optimizer update; gradients are
weighted by the number of examples, including short batches at epoch boundaries.
Clipping, optimizer/scaler updates, EMA, scheduling, previews, and checkpoint
intervals operate on complete updates. A rejected accumulated update discards all
its gradients; consumed examples remain consumed. Ctrl+C completes the current
update attempt and saves. The parent process handles Ctrl+C; workers ignore it.
Persistent workers are explicitly shut down before the completion message, and
repeated Ctrl+C presses do not interrupt saving or worker cleanup.

New checkpoints save consumed shuffle position, epoch, augmentation seed,
dataset/transform fingerprint, and learning-rate schedule state. Crops and
synthetic effects are seeded per epoch/example, independently of worker prefetch.
Changing `num_workers` does not change the sequence. Exact data continuation needs
unchanged files, transforms, and splitting. The fingerprint checks resolved paths,
sizes, modification times, and data settings, not image-content hashes.

Continue prompts for a dataset, defaulting to the saved one. Choosing a different
root starts a fresh data stream and resets score/guard baselines while retaining
weights and optimizer state. For changed files at the same root, explicitly choose
Reset data stream. Old checkpoints without stream state start a new stream and
print a notice. Changing batch size or accumulation changes update grouping even
though the remaining example order is preserved. GPU kernels may still be
nondeterministic; exact data-stream resume does not promise bitwise GPU training.

## Fine-tuning

Fine-tune loads compatible raw or EMA weights and starts fresh optimizer, scaler,
schedule, data stream, EMA history, and best-score tracking. Architecture and class
IDs come from the source checkpoint. It does not silently reshape weights or map
new classes. A dataset may omit known classes; unknown classes are rejected.

Policies:

- `all`: train all parameters.
- `last`: train the last N architecture blocks and output layers. JiT uses mixer
  blocks, RIN routing blocks, ConvNet decoder levels, and HierMLP refinement stages
  (global mixer blocks when there are no refinements).
- `modules`: select named modules or glob patterns after viewing the model's
  module list. For example, `to_pixels,rin_blocks.3`.

Only trainable parameters enter the optimizer. Fully frozen modules stay in
evaluation mode. The trainable count is printed; an empty selection is rejected.
Continue reapplies the saved freeze policy before restoring optimizer state.
The shared `model.pt` is still the training destination, so fine-tuning eventually
replaces it just as a fresh training run does.

## Verification

```bash
OMP_NUM_THREADS=1 conda run --no-capture-output -n pytorch2 python -B -m unittest -v test_jit5_training test_jit5_rin test_jit5_stem_conditioning test_jit5_fcdm test_jit5_training_options
```

Tests cover split isolation and formats, corrupt-image fallback, fixed validation,
worker-prefetch resume, unequal microbatches, freeze policies, best-score saves,
JSON/legacy configs, and uninterrupted versus interrupted/resumed training.
`test_jit5_training_options` covers the sampler's final step, EMA warmup and
fused updates, torch.compile setup and its gradient check, gradient strides,
and the Triton kernels against their PyTorch references (GPU tests are skipped
without CUDA).
