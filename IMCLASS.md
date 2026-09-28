# Interactive image classification

Run `conda run --no-capture-output -n pytorch2 python imclass.py` from this folder.
Uses the installed PyTorch, torchvision and Pillow, plus local `lamb.py` for
Muon and CLion. CUDA is selected automatically when available; otherwise CPU.
Pretrain means supervised training from scratch, without downloaded weights.

The script defaults CUDA_MODULE_LOADING to LAZY before importing PyTorch or
checking CUDA availability, while respecting an explicit caller setting. This
avoids eagerly loading unused CUDA library modules into host RAM on supported
drivers/runtimes. PyTorch's later automatic setting can be too late if an
availability check already initialized the driver. If importing this script
into a process that has already initialized CUDA, restart that process with
`CUDA_MODULE_LOADING=LAZY` set before startup. This does not change model
precision, batch size, architecture or data sampling.
See [NVIDIA's lazy-loading documentation](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-c-programming-guide/lazy-loading.html).

RAM verification (2026-09-21): PyTorch 2.4.1/CUDA 11.8 on RTX 3090, default
DeiT configuration, CIFAR10 with 54,000 training/6,000 validation images,
batch 64, Adam, no augmentation, seed 123, two CPU threads. Each comparison
ran three complete epochs (2,532 updates), including validation and saves.
Sampled process resident RAM plateaued at 4,040 MiB before this startup fix,
versus 950 MiB after the first validation with the fix (about 76% less).
Neither run showed continued growth over those three epochs. This establishes
a CUDA-loading overhead reduction, not a guarantee against leaks in every
configuration or longer run. All 18 classifier tests passed on CPU and CUDA.
Temporary profiling checkpoints were removed.

The four input-driven modes are pretrain (`0/p/pretrain`), finetune
(`1/f/finetune`), sample (`2/s/sample`) and DeepDream (`3/d/deepdream`).
Training takes a directory with at least two nonempty class subdirectories.
Images are discovered recursively. Class names are sorted and saved with weights.

Positive image resolution resizes to a square, without preserving aspect ratio.
A smaller crop size extracts a square; a crop at least as large is ignored.
Resolution -1 retains native size, upscales proportionally only if a side is
smaller than the crop, then crops. The default crop is the image resolution,
or 256 for -1. Crops are random during training and centered for evaluation.
Channels 1/2/3/4 mean L/LA/RGB/RGBA; ordinary image files cannot supply arbitrary
multispectral channels. Pixels are normalized from [0,1] to [-1,1].

Models are configurable family implementations, not fixed published presets:

| ID | Encoder | Configuration |
|---|---|---|
| 0 | Basic convolutional network | Minimum/maximum stage widths |
| 1 | ResNet basic residual blocks | Minimum/maximum stage widths |
| 2 | EfficientNet family, inverted bottlenecks with squeeze/excitation | Minimum/maximum stage widths |
| 3 | Basic ViT, global average token pooling | Patch, width, depth, heads |
| 4 | DeiT III style ViT, LayerScale and stochastic depth | Patch, width, depth, heads |
| 5 | MLP-Mixer | Patch, width, depth |
| 6 | gMLP | Patch, width, depth |
| 7 | aMLP | Patch, width, depth; fixed one 64-dimensional attention head |
| 8 | Hierarchical ConvNeXt | Minimum/maximum stage widths |
| 9 | Isotropic ConvNeXt | Patch, width, depth |
| 10 | PatchRNN | Patch, width, recurrent depth, RNN/GRU/LSTM cell |
| 11 | ViP (Vision Permutator) | Patch, width, depth; width divisible by grid side |

Hierarchical widths double until the specified maximum (included exactly).
Each stage contains two blocks. Small-image stems and GroupNorm in options 0–2
allow tiny images and singleton batches. ConvNeXt uses channel LayerNorm.
Isotropic inputs must divide evenly into patches; the prompts enforce this.
ViTs use learned patch positions. Mixer/gMLP token counts are fixed by the
checkpoint crop. Option 4 is an architecture variant, not a reproduction of
the full DeiT III training recipe or a guarantee of small-dataset accuracy.

PatchRNN embeds nonoverlapping patches with a convolution, visits them left to
right and top to bottom, and runs one stacked `nn.RNN`, `nn.GRU` (default), or
`nn.LSTM` sequence module. There is no Python loop over patches. Final sequence
outputs are normalized and mean-pooled for classification. Recurrence is
unidirectional; this is a classifier inspired by PixelRNN, not an autoregressive
pixel generator. Training uses PyTorch's fused sequence implementation where
supported. Gradient-based evaluation disables cuDNN for the recurrent operation
because cuDNN inference does not retain the state needed for backward.

ViP uses residual weighted height, width and channel mixing followed by a channel
MLP. This configurable isotropic variant sets the segment count to the patch
grid side, which must divide the hidden width; it is not a fixed published
multistage preset. Both new families use the existing head, finetuning,
checkpoint and visualization workflows. Existing model IDs remain unchanged;
older checkpoints load with default values for new configuration fields.

The MLP head uses global pooling, configurable hidden layers and a final linear
logit layer. Hidden widths can be one repeated width or one width per hidden
layer. Activations 0–9 are identity, sigmoid, tanh, ReLU, LeakyReLU (custom
slope), PReLU, GELU, SiLU (default), Mish, and SwiGLU. SwiGLU doubles the internal
projection and gates back to the requested hidden width. Zero hidden layers
gives a linear classifier.

Augmentations are opt-in: empty/0 disables them; -1 enables all; comma-separated
IDs select individual operations. The menu lists geometric, color, blur,
erasing/noise, automatic augmentation policies, MixUp and CutMix. All means a
strong composition, including successive MixUp and CutMix; it can be excessive.
Photometric operations act on luminance/RGB while preserving alpha. Geometric
operations transform all channels together. Automatic policies replay the same
geometry and intensity operations on alpha; erasing/noise also affect alpha.
Validation never uses augmentation.

An explicit validation folder uses training class IDs and rejects unknown
classes and overlapping resolved file paths. Otherwise the seeded split is
stratified and keeps at least one training image per class. Fractions satisfy
0 <= fraction < 1; 0 disables validation. A positive fraction requires at least
two images in every class. Validation folders may contain a subset of classes.
**The supplied CIFAR10 folders mix train_* and test_* files.** Percentage splits
use all files; use separately prepared train/validation roots for an official
CIFAR10 benchmark. This program never moves the dataset.

Inverse class-frequency weights drive a replacement sampler with one dataset's
worth of draws per epoch. This balances classes in expectation; it does not
guarantee seeing every image each epoch. Cross entropy is unweighted to avoid
applying inverse-frequency correction twice. The counts and sampler weights
are saved. Validation uses the natural class distribution and reports both
accuracy and mean per-class accuracy. With MixUp/CutMix, training accuracy is the
soft target probability assigned to the predicted class, averaged over samples.
Seed 0 generates and prints a random seed.

Optimizer choices are SGD with momentum 0.9 (lr 0.01), Adam (0.001), Muon
(0.00042), CLion (0.0001), RAdamScheduleFree (0.0025). RAdamScheduleFree trains
at its interpolated weights and swaps to its averaged weights for validation and
every checkpoint, so saved `model_state` is the averaged model. These rates are defaults: pretraining and finetuning
prompt for a finite positive optimizer learning rate. The Python APIs accept
`make_optimizer(model, choice, lr=...)` and `train_model(..., lr=...)`.
The chosen rate applies to all parameter groups, including Muon's fallback,
and is saved in training options and optimizer state. This sets the run's rate;
there is no live mid-epoch control or new scheduler. Muon uses its documented AdamW fallback for the
head, positional embeddings, biases and normalization parameters. Training is
FP32. Each epoch logs loss/accuracy and writes last.pt; best.pt follows validation
loss, or training loss without validation. Checkpoints contain architecture,
preprocessing, classes, model/optimizer states, training options and seed.
Each invocation creates a unique run directory under ImClass (or the save root),
so existing checkpoints are preserved. A checkpoint prompt accepts a .pt file,
a run directory, or a save root (latest run); directories prefer interrupt.pt,
then best.pt, then last.pt.

Ctrl+C during pretraining or finetuning finishes the active training batch,
then saves interrupt.pt and exits. During validation it finishes the active batch;
during an epoch checkpoint write it finishes that write first. This avoids
interrupting optimizer updates or checkpoint writes halfway through. The
interrupt checkpoint includes the current model/optimizer state, configuration,
classes, training options, epoch and number of completed batches. Existing
best.pt/last.pt are preserved. It can be loaded for sampling, DeepDream or
finetuning; finetuning still creates a fresh head/optimizer, rather than resuming
the interrupted training schedule. Interrupting setup before training starts
does not create an interrupt checkpoint.

Finetuning loads that encoder and its preprocessing, asks for a new data root,
and always creates a fresh head and new class mapping. Freeze defaults to yes:
encoder weights and running state stay fixed; selecting no trains everything.

Sampling asks about plots before opening input images, then accepts one file
or recursively processes a folder. It writes probabilities/predictions to CSV
under samples/<unique-run>. Plotting saves the actual resized/cropped input
beside a Grad-CAM overlay for the predicted class. Each attention layer/head
also gets an image of mean received attention across queries. gMLP/aMLP get
spatial gate-magnitude maps; MLP-Mixer gets input-dependent token-mixer response
maps; EfficientNet gets squeeze/excitation-weighted feature energy maps.
These latter maps are not attention probabilities or causal explanations.
Each map is independently scaled. Corrupt images are reported in errors.csv
and processing continues. Output names include a hash of the input path.

Plotting also writes input saliency (mean absolute class-logit gradient across
input channels). ViT options 3/4 add attention rollout: average heads, add an
identity residual, normalize rows, then compose matrices across layers. The
output averages all final token rows to match these models' mean pooling.
Rollout is class-independent and approximate: it does not model MLP mixing,
LayerScale magnitudes, or signed/value projections. It is not offered for aMLP's
attention-in-gating path. Maps remain independently scaled for display.

The sampling menu offers a top-class comparison count (default up to three,
zero disables). Comparison sheets show the input, class IDs/names, probabilities
and class-specific CAM overlays with a shared intensity scale. Full class names
and probabilities remain in predictions.csv. This adds one explanation pass per
compared class. Python callers opt in with `sample(..., compare_classes=3)`;
CAM, saliency, attention/gating/rollout and class comparisons require `plot=True`.

Sampling separately offers feature plots: off (default), learned kernels,
per-unit activations, or both. These work even when CAM plots are disabled.
The maximum units per layer defaults to 0 (all); a positive number selects the
first N output channels/units, not the strongest responses. This limit applies
to kernel output filters and activation units. Each selected convolution filter
still includes every connected input-channel slice. All-unit exports for wide
models and large image folders can create many files and take substantial I/O time.

`kernels/` is written once per sampling run. Every Conv2d, including patch
embeddings, depthwise and 1x1 convolutions, gets paginated weight sheets. Tiles
identify the output filter and the actual input channel, respecting convolution
groups. These are signed weight slices, not reconstructed objects or RGB
photographs; deeper input channels represent features rather than color. Biases
are excluded. A 1x1 kernel therefore appears as a constant-color tile.

`features/<image-stem-and-hash>/` contains the actual preprocessed input and
paginated response sheets from one evaluation forward pass. Coverage includes:

- Every convolution's output channel, before subsequent normalization/activation.
- Explicit activation-module outputs in hierarchical ConvNets.
- Every isotropic encoder block's output and the final normalized encoder features.
  PatchRNN exposes its final recurrent layer's outputs over the patch sequence.
- Every head hidden-layer output after activation and every final class logit.

Spatial outputs are shown as individual channel maps. Token outputs are laid
back onto the patch grid; these describe contextual features at each position,
not isolated patch computations. Head units have no spatial axes and appear as
1x1 response tiles. Fused recurrent gates and intermediate recurrent layers,
and internal token-mixing/attention projections, are not individually exported.
These responses show where features activate; they are not class attribution
maps or reconstructions of what an individual unit "sees".

Sheets contain up to 64 labeled tiles, enlarged with nearest-neighbor sampling
while preserving map aspect ratio. Negative values are blue, zero white, positive
values red. All sheets for one layer share a symmetric scale set by the largest
absolute value among its exported maps/slices; scales differ between layers and
images. Constant maps retain their value relative to that scale. Each folder's
manifest.csv records filenames, zero-based tile indices (row-major), layer/unit
IDs, input channels for kernels, native map dimensions, raw minimum/maximum/mean,
scale, and the total available output-unit count. The manifest makes limits and
display normalization explicit. PNGs are visualizations, not lossless tensors.

Python usage: `sample(..., kernel_plots=True, feature_maps=True, max_units=0)`.
Standalone exporters are `save_kernels(model, new_folder, max_units=0)` and
`save_feature_maps(model, x, new_folder, max_units=0)`, where x is one preprocessed
image `[1,C,H,W]` on the model's device. Export folders must be new to avoid
overwriting earlier results. Feature export uses no gradients, streams layer
outputs to disk, removes hooks, and restores the model's training flags.

DeepDream saves under DeepDream/<unique-run> and offers encoder + head + classes,
head + classes, or classes only. Encoder targets include every Conv2d output
channel and Linear output unit, including internal projections and token-mixer
cells. A filter's objective averages over its spatial locations, not one image
per location of the same shared filter. PatchRNN additionally exposes the final
recurrent layer's output units, averaged over patch positions; fused internal
gates and intermediate recurrent layers are not separately exposed.
Head targets are post-activation hidden
neurons; classes maximize logits. By default every target starts from fresh seeded noise.
DeepDream can also suppress the mean activation of every non-target output in the
same selected layer. The optimized objective is then `target - mean(other units)`;
for a one-unit layer it remains the target activation. The run config and manifest
record whether this option was enabled.
Adam optimizes pixels with total-variation/L2 regularization and [0,1] bounds.
Steps and learning rate are prompted. The manifest records target IDs and
initial/final raw activations. These are local activation-maximizing images,
not unique or guaranteed globally "perfect" images. All-unit mode can create
many thousands of files; its exact count is printed before optimization.

An optional starter image uses the checkpoint's evaluation preprocessing,
including channel conversion and center crop. Set initializations per target
above one to explore multiple starts. Without a starter every run uses fresh
noise. With a starter the first run uses its exact preprocessed pixels; further
runs add uniform noise of configurable amplitude in [0,1] pixel units and clamp
to valid bounds. Zero amplitude deliberately repeats the same starter. Each
initialization is saved separately with its index, starter path, noise amplitude
and initial/final activation in manifest.csv; config.json records run settings
and the seed. The seed prompt controls reproducibility. Python callers use
`deepdream(..., starter='image.png', inits=3, noise=0.1)`; `dream_one` accepts an
already preprocessed [1,C,H,W] starter tensor in [-1,1]. Output preserves alpha.

Architecture references: [EfficientNet](https://arxiv.org/abs/1905.11946),
[DeiT III](https://github.com/facebookresearch/deit/blob/main/models_v2.py),
[MLP-Mixer](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py),
[gMLP/aMLP](https://arxiv.org/abs/2105.08050),
[ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py).
New feature references: [Vision Permutator](https://github.com/houqb/VisionPermutator),
[attention rollout](https://arxiv.org/abs/2005.00928).

Verification: `conda run --no-capture-output -n pytorch2 python -m unittest -v test_imclass`.
Tests use temporary directories and remove their generated images/checkpoints.

## Browser GUI and dataset JSON files

Run `python imclass.py` and choose mode `4/g/gui` (or run `python imclass_gui.py`); it serves
http://127.0.0.1:8766/ in the current folder. Three pages:

- **Dataset**: open a folder of images (recursively) or an existing dataset JSON, create classes, and
  label images one at a time or in batches (click, shift+click, ctrl+click, select all/page/invert;
  keys 1–9 assign, 0/Delete clears). "Use subfolder names" pre-labels images from their folders, and
  "Find duplicates" marks byte-identical files. Save writes a dataset JSON.
- **Train**: pretrain, finetune a checkpoint (new head, optional frozen encoder) or continue a run.
  Duplicate images are removed first and never end up on both sides of the split; continuing a run
  reuses its recorded `split.csv` validation images. A live monitor shows batch loss, per-epoch
  metrics and a fixed held-out probe image's prediction and Grad-CAM after every epoch. Stop saves
  `interrupt.pt` through the same safe boundary as Ctrl+C.
- **Explore**: load a run, then pick an image (browser, upload, drag and drop, random validation
  image, live camera, or a sketch pad) to see class probabilities, Grad-CAM for any class and the
  architecture's other maps, top-class comparison, live activation maps for any layer, convolution
  kernels, occlusion sensitivity, live DeepDream, and an evaluation page with a confusion matrix and
  the most confidently misclassified images.

A dataset JSON is accepted anywhere a class-subfolder directory is (training, finetuning and the
validation prompt, in the CLI too):

```json
{"format": "imclass-dataset", "version": 1, "classes": ["cat", "dog"], "root": "",
 "items": [{"path": "photos/a.jpg", "class": "cat"}, {"path": "photos/b.jpg", "class": null}]}
```

Relative paths resolve against `root`, or the JSON file's folder when `root` is empty. Items with a
null class are unlabeled and skipped; missing files are skipped with a warning.
