# U-Net segmentation assignment

TensorFlow-Keras U-Net for the 936-pair, three-class segmentation assignment.
The workflow includes a fixed-seed overfit sanity check, encoder/decoder depth
analysis, categorical cross-entropy versus CCE+Dice, and augmentation ablation.

## Included files

| File | Role |
| --- | --- |
| `data.py` | Validate pairs, make sklearn splits, load RGB/one-hot masks, apply synchronized augmentation, save and plot predictions. Original membrane helpers remain available. |
| `model.py` | Configurable U-Net depth/width and serializable `CategoricalDiceLoss`. Four pooling/upsampling levels is the standard topology. |
| `main.py` | Shared preparation, training, sklearn F1/Dice evaluation, experiment selection, reporting, and CLI. |
| `dataPrepare.ipynb` | Inspect masks and image alignment; preview data and save a preparation manifest. |
| `trainUnet.ipynb` | Colab/local assignment workflow using the shared modules; sanity run, depth study, loss/augmentation comparisons, measured tables and plots. |
| `requirements.txt` | Local Python dependencies, including scikit-learn. |
| `requirements-gpu.txt` | WSL/Linux TensorFlow CUDA dependencies and the same project libraries. |
| `.gitignore` | Excludes dataset images, results, weights, logs, environments, and caches. |
| `.gitattributes`, `LICENSE` | Repository text settings and license. |
| `img/u-net-architecture.png` | Architecture illustration. |

## Dataset and class names

Keep matching PNG files locally at `data/936/data/<id>.png` and
`data/936/mask/<id>.png`. There must be 936 image/mask pairs. Images are RGB;
masks are same-size single-channel arrays with IDs 0, 1, and 2. Unknown values
or mismatched filenames/dimensions raise errors. RGB is normalized; masks are
resized with nearest-neighbor interpolation and then one-hot encoded for CCE.

The local archive contains no class-name metadata. A closely matching
[Tobacco Aerial Dataset](https://data.mendeley.com/datasets/5dpc5gbgpz/2)
documents a 936-image campaign, 480x352 patches, the same folder layout, and
0=background, 1=crop, 2=weed. This is a **provisional identification**, not proof
that this archive is that dataset. That source describes tobacco, not wheat.
The notebook marks crop/weed names as provisional. Confirm the dataset origin
or instructor mapping before labeling the crop as wheat. IDs are never swapped
or relabeled by the implementation; change `CLASS_NAMES` for verified names.

## Run on Google Colab

1. Copy `data.py`, `model.py`, `main.py`, both notebooks, and `data/936` into
   `MyDrive/unet` in Google Drive. Dataset/results stay local or in your Drive,
   not GitHub. You do not need to upload `.venv` or previous `runs` folders.
2. Open `dataPrepare.ipynb` in Colab. Choose **Runtime -> Change runtime type -> GPU**.
3. Run its setup cell, mount Drive, and set `PROJECT_DIR` if your folder differs
   from `/content/drive/MyDrive/unet`. The setup uses Colab's installed TensorFlow
   and installs scikit-learn, Pillow, and matplotlib as needed.
4. Run all preparation cells. Inspect the mask IDs, class colors, and alignment.
5. Open `trainUnet.ipynb` in Colab, use a GPU runtime, and run its setup/configuration.
6. Run the sanity check and inspect the predictions before the full experiments.
7. Run the depth study and the four loss/augmentation comparisons. Read the
   generated report, curves, per-class Dice, and saved predictions.

Use the same dataset path and seed in both notebooks. If a preparation manifest
matches, training consumes it; otherwise the same shared validation is run again.

## Run locally in VS Code (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Skip creation if `.venv` already exists. Open the folder in VS Code with Python
and Jupyter extensions, select the `.venv` notebook kernel, and run preparation
then training. Restart the kernel after changing imported Python modules.
If needed, register the kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name unet-venv --display-name "Python (.venv U-Net)"
```

Native Windows runs use CPU in the verified environment; Colab GPU is preferred
for all full-dataset experiments. To print the architecture without training:

```powershell
.\.venv\Scripts\python.exe model.py
```

## Experiment design

- Seed 42 is used for sklearn splitting, Python/NumPy/TensorFlow randomness,
  model initialization, shuffling, and augmentation. Deterministic TensorFlow
  operations are enabled; exact values may still differ across software/hardware.
- `train_test_split(..., test_size=0.2, random_state=42)` produces 748 development
  and 188 test pairs. A further 10% of development is validation: 673 train,
  75 validation. The 188 test pairs are never used for training or depth selection.
- Sanity check: five images selected only from the training split; four-level
  U-Net, no dropout/augmentation, 300-epoch budget. Same-sample scores are not
  held-out results. Inspect whether it overfits; increase the budget if necessary.
- Depth study: 2, 3, 4, 5 pooling/upsampling levels, identical seed, split, width,
  resolution, optimizer, and 20-epoch budget. Choose the smallest depth within
  0.005 macro Dice of the best validation score. Report validation Dice, parameter
  count, and training time, rather than deciding the answer in advance.
- At the selected depth, train all four combinations: CCE without/with augmentation
  and CCE+Dice without/with augmentation. These report held-out test scores. All
  configurations and comparisons are fixed before looking at test results.
- Loss: mean categorical CE plus `DICE_WEIGHT * (1 - mean soft class Dice)`.
  The default Dice weight is 1.0; loss is serializable in `.keras` checkpoints.
- Augmentation: joint horizontal/vertical flips and 90-degree rotations, plus
  image-only brightness changes. Validation/test are not augmented.
- Scores: `sklearn.metrics.f1_score` on flattened argmax predictions and true IDs.
  Pixel class F1 equals Dice. Report each ID's Dice, macro Dice over all three,
  and foreground macro Dice over IDs 1 and 2, with `zero_division=0`.

The default resolution is 128x128 and base filter width is 16, retaining the
standard U-Net topology with a smaller width for feasible training. Set width
64 for the canonical reference width. Keep width/resolution unchanged across
comparisons. Deeper networks use more parameters; reduce batch size if GPU
memory is insufficient, then rerun all comparisons with that same batch size.

## Optional CLI

```powershell
# Five-image overfit sanity run
.\.venv\Scripts\python.exe main.py --limit 5 --epochs 300

# Full assignment experiments, using the same seed and 20 epochs per run
.\.venv\Scripts\python.exe main.py --suite --all-images --epochs 20

# Single full-dataset CCE+Dice run with augmentation
.\.venv\Scripts\python.exe main.py --all-images --epochs 20 --loss-mode cce_dice --augmentation
```

`--help` lists depth, width, seed, learning-rate, Dice-weight, class-name, and
output options. Use new output folders to avoid overwriting existing runs.

## Results and report

Each run saves `best.keras`, `split.json`, `config.json`, `history.json`,
`training.csv`, `metrics.json`, up to six prediction mask/color pairs, and a
comparison figure under ignored `runs/`. Each experiment suite also writes:

- `experiment_results.csv` and `.json`: actual measured configuration scores,
  parameter counts, and timings (updated after every completed experiment).
- `depth_selection.json`: validation-only depth-selection criterion and result.
- `performance_differences.json`: signed macro Dice differences for Dice loss
  and augmentation, both with and without the other factor.
- `experiment_report.md`: measured depth/ablation tables and differences.

The notebook adds depth/time/parameter curves and an ablation bar chart.
Do not fabricate metrics or claim unexecuted Colab experiments were measured.
A tiny two-epoch smoke suite verifies code paths only; a lower-resolution
five-epoch full-dataset pilot is preliminary, not the default 20-epoch study.
Use the measured outputs to write your own explanation of why depth, loss,
and augmentation helped or hurt. One fixed seed does not establish statistical
significance; document resolution, width, epoch budget, and mapping uncertainty.

## GitHub exclusions

Dataset images, `runs/`, models, logs, `.venv`, caches, and archives are ignored.
Notebook source files are kept without embedded outputs. Local executed copies
can be saved as `runs/dataPrepare_executed_local.ipynb` and
`runs/trainUnet_executed_local.ipynb`; clear source-notebook outputs before
committing. Removing tracked data/results from the index does not erase old
GitHub history. The architecture illustration is retained.

Based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


## Local RTX GPU training through Ubuntu WSL2

Modern TensorFlow does not use CUDA from the native Windows `.venv`. Use the
separate `.venv-wsl` Linux environment or select a Colab GPU runtime. The training
notebook now has `REQUIRE_GPU=True` and stops early if no GPU is detected.
GPU memory growth prevents TensorFlow from reserving all available GPU memory.

Open this repository with **VS Code: WSL: Reopen Folder in WSL**, then select
**Python (WSL GPU U-Net)** (`unet-gpu`) as the notebook kernel. Run preparation and training
notebooks normally. The Windows `.venv` can still run preparation on CPU.

From an Ubuntu WSL terminal in the repository folder:

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -r requirements-gpu.txt
.venv-wsl/bin/python -m ipykernel install --prefix .venv-wsl --name unet-gpu --display-name "Python (WSL GPU U-Net)"
.venv-wsl/bin/python gpu_env_setup.py
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(.venv-wsl/bin/python -c 'import site; from pathlib import Path; print(":".join(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib")))'):${LD_LIBRARY_PATH:-}"
.venv-wsl/bin/python -m pip check
.venv-wsl/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
.venv-wsl/bin/python main.py --suite --all-images --epochs 20 --require-gpu
```

`requirements-gpu.txt` installs the same project libraries plus the CUDA
dependencies via `tensorflow[and-cuda]`. `gpu_env_setup.py` links CUDA libraries
and configures the registered kernel's CUDA search paths. Both environments and all
GPU results under `runs/` are excluded from Git. See the
[TensorFlow GPU installation instructions](https://www.tensorflow.org/install/pip).
