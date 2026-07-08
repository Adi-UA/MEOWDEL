# MEOWDEL

[![Tests](https://github.com/Adi-UA/MEOWDEL/actions/workflows/test.yml/badge.svg)](https://github.com/Adi-UA/MEOWDEL/actions/workflows/test.yml)

A GAN, built from scratch in PyTorch + PyTorch Lightning, learning to generate cat faces over 50 epochs:

![Training progress: baseline vs spectral norm vs R1 penalty](assets/training_progress.gif)

Trained on the [Kaggle "Cats faces 64x64" dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models). See [Results](#results) for the full writeup, including a first attempt that mode-collapsed and how it was fixed.

### Layout

```
config.yaml     - all hyperparameters and flags, single source of truth
data.py         - downloads/loads the Kaggle cat faces dataset
model.py        - Generator and Discriminator (DCGAN-style, 64x64 RGB)
gan_module.py   - LitGAN: the Lightning training loop (manual optimization)
train.py        - entrypoint: python train.py [--config path/to/config.yaml]
util.py         - seeding, run-folder + plotting helpers
make_progress_gif.py - stitches one or more runs' per-epoch samples into a side-by-side progress GIF
make_comparison.py    - one or more runs, epoch 0 vs. final epoch, as a labeled grid
tests/                - shape/smoke tests for the model, data pipeline, and save/load (see Testing below)
```

### Setup

Clone the repo and install dependencies. The steps are the same on macOS and Windows except for how you activate the virtual environment:

```
git clone https://github.com/Adi-UA/MEOWDEL.git
cd MEOWDEL
python3 -m venv .venv
```

macOS/Linux:
```
source .venv/bin/activate
```

Windows (PowerShell):
```
.venv\Scripts\Activate.ps1
```

Then, on either machine:
```
pip install -r requirements.txt
```

`pip install torch` alone pulls the CPU-only build on Windows. `requirements.txt` will satisfy itself with that CPU wheel silently, no error, so check for CUDA after installing:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False` on a machine with an NVIDIA GPU, force-reinstall from the CUDA-specific index (match the `cuXXX` tag to your driver's supported CUDA version, shown by `nvidia-smi`):

```
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

`--force-reinstall` is required: pip sees a `torch` package already satisfying the version constraint and skips it otherwise, even though it's the wrong build.

You'll need Kaggle API credentials so `kagglehub` can download the dataset. Either:
- place a `kaggle.json` (with `username` and `key`) at `~/.kaggle/kaggle.json` (`C:\Users\<you>\.kaggle\kaggle.json` on Windows), or
- set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

This is a one-time setup needed on each machine you train on.

### Testing

Shape/smoke tests for the model, data pipeline, and save/load, no dataset download or GPU required:

```
pip install -r requirements-dev.txt
pytest tests/
```

Runs on every push and PR via GitHub Actions (`.github/workflows/test.yml`), CPU-only, ~1-2 minutes. These catch broken shapes and NaN gradients, not training quality, that's what the runs in Results are for.

### Training

```
python train.py
```

No code changes are needed between machines: `train.py` uses `accelerator="auto"`, so PyTorch Lightning detects the best available hardware at runtime — CUDA on the Windows/4060ti box, Apple's MPS on a Mac, or CPU as a fallback. Same command, same config, whichever machine you're on.

To run a different configuration without editing `config.yaml`, copy it and pass `--config`:

```
python train.py --config config.my_experiment.yaml
```

Each run creates a timestamped folder under `outputs/`:

```
outputs/<timestamp>/
  checkpoints/last.ckpt   - Lightning checkpoint (resumable)
  samples/epoch_XXX.png   - a 3x3 grid from a fixed noise vector, saved every epoch
  loss_curve.png          - generator/discriminator/R1 loss curve, saved at the end
  metrics.csv             - raw per-epoch metrics
  events.out.tfevents.*   - TensorBoard logs (`tensorboard --logdir outputs`)
```

Final generator/discriminator weights are also saved to `save_path` from the config (e.g. `models/baseline/generator.pth`), one subfolder per experiment so re-running a different config doesn't overwrite an earlier one's weights.

### Research flags (`config.yaml`)

Two optional techniques from GAN research papers, both off by default. Short technical description first, plain-language version underneath so you can relearn what they actually do without re-reading the papers.

**`spectral_norm`** (bool) — [Miyato et al. 2018](https://arxiv.org/abs/1802.05957). Wraps every discriminator conv layer in spectral normalization, which constrains how much the layer's output can change relative to its input (its "Lipschitz constant"). BatchNorm is skipped on the discriminator when this is on, since the two are known to fight each other.

> In plain terms: without this, the discriminator can become so confident, so fast, that it stops giving the generator anything useful to learn from (its gradient goes flat). Spectral norm puts a speed limit on how sharply the discriminator is allowed to react to small changes in its input, which keeps it from steamrolling the generator early in training.

**`r1_gamma`** (float, `0` disables) + **`r1_interval`** — [Mescheder et al. 2018](https://arxiv.org/abs/1801.04406). Adds a penalty term equal to the squared gradient norm of the discriminator's output with respect to real images, scaled by `r1_gamma`. Applied lazily, only every `r1_interval` steps (the StyleGAN2 approach), to cut the cost of the extra backward pass it requires. A typical value is `r1_gamma: 10`.

> In plain terms: imagine the discriminator's judgment as a landscape, and real images sit at the bottom of valleys in that landscape. If the valley walls are too steep right around real images, the generator gets yanked around wildly once it starts producing samples close to real ones — training oscillates instead of settling down. This penalty forces the discriminator to keep the ground flat around real images, which is what actually lets training converge instead of chasing its own tail forever.

### Results

50 epochs each, ~33 minutes per run on the 4060ti box. Same fixed noise vector sampled every epoch, so the GIF at the top shows the same latent points evolving across runs.

| | Baseline (`config.yaml`) | Spectral norm (`config.spectral_only.yaml`) | R1 penalty (`config.r1_only.yaml`) |
|---|---|---|---|
| Final `g_loss` / `d_loss` | 8.03 / 0.187 | 0.997 / 1.214 | 0.77 / 1.35 |
| Outcome | discriminator winning, but recognizable, varied cats | recognizable, varied cats, though a bit less sharp/diverse than the other two | balanced losses, varied, recognizable cats |

All three are broadly comparable on results, none collapsed or clearly "won" by a wide margin, spectral norm just trails the other two a bit.

Epoch 0 vs. final, side by side:

![Baseline vs Spectral norm vs R1-only, epoch 0 vs final](assets/comparison.png)

Loss curves:

| Baseline | Spectral norm | R1-only |
|---|---|---|
| ![Baseline loss curve](assets/baseline_loss_curve.png) | ![Spectral norm loss curve](assets/spectral_only_loss_curve.png) | ![R1-only loss curve](assets/r1_only_loss_curve.png) |

**Spectral norm's history here**: the first attempt (same architecture, same hyperparameters as the [SN-GAN paper's](https://arxiv.org/abs/1802.05957) own tested setup) mode-collapsed by epoch 0. The working theory was that dropping BatchNorm (which spectral norm requires here, see Research flags above) left the discriminator too unstable at the original learning rate, per [Xiang & Li 2017's](https://ar5iv.labs.arxiv.org/html/1704.03971) finding that BatchNorm-equipped discriminators tolerate higher learning rates than BatchNorm-free ones. Halving `lr_d` to `0.0001` avoided the collapse this run. That's one data point, not proof the theory's right, but it's consistent with it.

Regenerate these visuals from any set of runs:

```
python make_progress_gif.py outputs/<run1> outputs/<run2> outputs/<run3> --labels "Name 1" "Name 2" "Name 3" --output assets/training_progress.gif
python make_comparison.py outputs/<run1> outputs/<run2> outputs/<run3> --labels "Name 1" "Name 2" "Name 3" --output assets/comparison.png
```

### Learnings

A few classic GAN training pitfalls came up while building this, worth remembering for next time:

- **Match the generator's output range to how you normalize the data.** If real images are normalized to `[-1, 1]`, the generator's last layer needs a `Tanh` to match. Otherwise the discriminator can win by checking whether pixel values fall in the expected range, without ever learning real structure.
- **Never inject noise directly into a hard 0/1 label.** It can push the label outside `[0, 1]`, which corrupts `BCEWithLogitsLoss` (a giveaway is the loss going negative, which is mathematically impossible for a valid target). If you want to fight discriminator overconfidence, use a fixed offset (e.g. real target `0.9`) or a principled method like the R1 gradient penalty below, not random noise on the target.
- **BatchNorm isn't optional to just drop without a second thought.** The first spectral norm run (which requires dropping BatchNorm here, see Research flags below) mode-collapsed by epoch 0 at the same learning rate the baseline used fine. Halving the discriminator's learning rate fixed it, see Results below, small differences in how fast the generator and discriminator learn can compound until one saturates the other.
- **Downsample with strided convolutions, not pooling.** Pooling throws away spatial gradient information the discriminator needs.
- **`.detach()` matters.** Feeding the generator's output into the discriminator's own training step without detaching it wastes memory and compute building gradients into the generator that get discarded immediately after.
