# Working Agreement: Applied AI SRGAN Midterm

## What this repository must demonstrate

Produce a repeatable PyTorch experiment that measures the effect of GAN-based
super-resolution on binary image classification. The comparison has two arms:

- **Baseline classifier:** fine-tune a pretrained vision network using source
  images scaled directly to 128 × 128 pixels.
- **Super-resolved classifier:** fine-tune the same kind of classifier using
  128 × 128 training images synthesized by an SRGAN from 32 × 32 inputs.

Both classifiers must ultimately be judged against the identical held-out
examples.

## Data protocol and leakage prevention

- Partition the original data exactly once, assigning 70% to training and 30%
  to testing.
- Make this a stratified split and fix its random seed at `42`.
- Persist the path and class label for every split member. All notebooks and
  training jobs must consume that saved manifest instead of producing their
  own partitions.
- The test partition is evaluation-only. Neither classifier nor the SRGAN may
  learn from any test example.
- Baseline inputs are originals resized to 128 × 128.
- Each SRGAN training pair consists of a 32 × 32 low-resolution input and its
  128 × 128 target.
- Train the second classifier only with 128 × 128 images emitted by the trained
  generator for members of the training partition.

## Implementation conventions

- PyTorch and torchvision are the required ML stack.
- Put reusable Python components in `src/applied_ai_midterm`; notebooks should
  remain thin orchestration and exploration layers over those components.
- Use `pathlib` for filesystem work so no operating-system-specific separators
  are embedded in code.
- Add useful type annotations and short, focused docstrings.
- Hardware selection must work across NVIDIA CUDA, Apple Metal (MPS), and CPU
  environments.
- Missing datasets and malformed directory layouts must fail early with clear,
  actionable messages.

## Binary classifier contract

Start from an ImageNet-pretrained torchvision architecture, for example
MobileNetV2 or ResNet18, and use the corresponding ImageNet normalization.
Replace its prediction head with a single-output layer representing one binary
logit. Optimize with `BCEWithLogitsLoss`; do not apply sigmoid during loss
calculation, and apply it only when logits need to be converted to
probabilities.

## Super-resolution model contract

Define the generator and discriminator as distinct classes.

The generator architecture must provide all of the following:

- an entry convolution;
- a stack of residual blocks;
- a skip connection;
- two successive 2× upsampling stages, yielding 4× spatial enlargement in
  total;
- a final layer that returns an RGB image; and
- an explicit output-value convention, such as `tanh`-scaled output, documented
  alongside preprocessing and postprocessing.

The discriminator must deepen progressively through convolutional blocks and
return logits for the real-versus-generated decision.

Document the generator objective as a weighted combination of reconstruction
(pixel/content) loss and adversarial loss. A perceptual term based on pretrained
VGG features may also be included. Unit tests must never initiate downloads of
pretrained parameters.

## SRGAN training durability

- Run SRGAN optimization for no fewer than 150 epochs.
- Write a restartable checkpoint after every fifth epoch.
- Each checkpoint must preserve the current epoch, generator and discriminator
  parameters, both optimizer states, scheduler states for every scheduler in
  use, accumulated training history, and the random-seed/configuration data
  needed to reproduce or resume the run.

## Evaluation and required evidence

Evaluate both classifier variants on the same reserved test records. For each
model, calculate and report:

- accuracy, precision, recall, F1 score, and ROC AUC;
- a confusion matrix;
- a full classification report; and
- an ROC curve.

The visual analysis must make the full transformation path inspectable. Include
examples of original images, normalized images, augmented images, 32 × 32
low-resolution inputs, bicubic enlargements at 128 × 128, generator outputs at
128 × 128, and the corresponding real 128 × 128 targets.

## Files that must remain outside version control

Never commit raw source data, generated-image collections, binary model
checkpoints, virtual/environment directories, or credentials for Colab,
Kaggle, Google Drive, or any other connected data service.

## Change-management expectations

Before making a change that spans multiple files, describe the planned edits.
At the end of every implementation phase:

1. Execute the tests relevant to that phase.
2. Run Ruff or an equivalent static-analysis tool.
3. Summarize which files changed and what changed in them.
4. Leave the work uncommitted unless the user explicitly asks for a Git commit.
