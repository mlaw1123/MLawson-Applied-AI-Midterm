# Applied AI Midterm: SRGAN-Assisted Classification

This repository is the foundation for a reproducible PyTorch study of whether
GAN-based super-resolution changes binary image-classification performance. It
will compare a transfer-learning baseline trained on original images resized to
128 × 128 with a second classifier trained on 128 × 128 images produced by an
SRGAN from 32 × 32 inputs. No experimental results are claimed at this stage.

## Experimental flow

1. Inventory the original binary image dataset and make one stratified 70/30
   train/test split using seed 42.
2. Save paths and labels in a split manifest shared by every later step.
3. Train Model A on original training images resized to 128 × 128.
4. Train the SRGAN using only training records, pairing 32 × 32 inputs with
   real 128 × 128 targets.
5. Generate 128 × 128 training images with the SRGAN and use them to train
   Model B.
6. Evaluate both classifiers on the same untouched test records and compare
   their metrics, confusion matrices, classification reports, and ROC curves.

Model training is intentionally outside the scope of the current foundation.

## Repository layout

```text
configs/                 Versioned experiment configuration
data/
  raw/                   User-supplied source images (ignored by Git)
  processed/             Split manifests and derived metadata
  generated/             SRGAN-generated image datasets (ignored by Git)
notebooks/               Lightweight notebooks that call the source package
src/applied_ai_midterm/  Reusable, tested Python implementation
tests/                   Automated tests
artifacts/
  checkpoints/           Resumable training state (ignored by Git)
  models/                Exported model binaries (ignored by Git)
reports/figures/         Generated evaluation figures
```

## Planned notebooks

The notebook sequence is designed to keep processing and training logic inside
`src/applied_ai_midterm`:

1. `01_data_audit_and_split.ipynb` — validate the dataset, inspect classes, and
   create the single stratified split manifest.
2. `02_baseline_classifier.ipynb` — train and validate Model A.
3. `03_srgan_training.ipynb` — train or resume the super-resolution GAN.
4. `04_generate_training_images.ipynb` — create Model B's training images.
5. `05_srgan_classifier.ipynb` — train and validate Model B.
6. `06_comparison_and_evaluation.ipynb` — evaluate both models on the reserved
   test set and produce the required visual comparisons.

These files are planned rather than populated during the foundation phase.

## Dataset placement

Place the original data below `data/raw/`, with one directory per binary class:

```text
data/raw/
  class_0/
    image_001.jpg
  class_1/
    image_002.jpg
```

Class names may differ, but exactly two non-empty class directories will be
required. Dataset contents are excluded from version control. Do not place test
images in a separately handcrafted directory; the project will generate and
persist its own stratified split.

## Local setup

Python 3.12 is required. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebooks,dev]'
pytest
ruff check .
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Google Colab workflow

1. Store the repository and dataset in Google Drive, keeping credentials out of
   the repository.
2. Open the appropriate planned notebook in Colab and select a GPU runtime when
   training requires it.
3. Mount Drive, change into the cloned repository, and install dependencies:

   ```python
   from google.colab import drive

   drive.mount("/content/drive")
   %cd /content/drive/MyDrive/MLawson-Applied-AI-Midterm
   !pip install -r requirements.txt
   !pip install -e .
   ```

4. Confirm `configs/config.yaml` and `data/raw/` are available before running a
   notebook.
5. Save SRGAN checkpoints to persistent Drive storage every five epochs so a
   disconnected Colab runtime can resume safely.

The codebase will select CUDA, Apple MPS, or CPU as available; Colab normally
uses CUDA when a GPU runtime is active.

