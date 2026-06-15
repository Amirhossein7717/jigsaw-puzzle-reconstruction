# Jigsaw Puzzle Image Reconstruction

Reconstruct a full **96×96 RGB image** from **9 scrambled 28×28 patches** whose order is unknown and whose borders have been partly removed by erosion. The model has to figure out where each patch belongs **and** fill in the missing seam pixels, end to end, with neural networks only.

Built in **Keras / TensorFlow** and trained on the **STL-10** unlabeled set (100,000 images) on Google Colab.

## Results

| Metric | Value |
|---|---|
| **Test MAE** | **0.0458** (std 0.0073) |
| Mean-patch baseline MAE | 0.1826 |
| Improvement over baseline | ~4× lower error |
| Trainable parameters | 1.67M (constraint: < 6M) |

## Approach: "place the real pixels, then inpaint"

A naive encoder–decoder that regenerates every pixel from a small bottleneck comes out blurry (an earlier version of mine reached only 0.109). The key insight here is that each patch is the **center 28×28 of a 32×32 cell**, so about **77% of the pixels are already known exactly** — only the thin eroded seams are missing. So instead of regenerating, the model **carries the real pixels through to the output**:

1. **Shared patch encoder** — a small CNN (shared weights, via `TimeDistributed`) maps each 28×28×3 patch to a 128-d vector.
2. **Self-attention over patches** — two light transformer blocks let the 9 patches reason about each other.
3. **Position-to-patch assignment** — 9 learnable position queries score the patches (scaled dot-product attention) to produce a soft 9×9 assignment, **supervised with the true permutation** for accurate matching.
4. **Assembly** — the assignment places the actual patch pixels into their predicted cells on a 96×96 canvas, leaving the seams blank and flagged by a mask.
5. **U-Net inpainting** — copies the known pixels and fills the seams, ending in `Conv2D(3, sigmoid)`.

Training optimizes `MAE + 0.3 · cross_entropy` (image + assignment). Two models share weights: a two-output training model, and a single-output inference model (patches → image) that is evaluated and saved.

## How to run

1. Open `jigsaw_reconstruction.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. Training is ~30–40 epochs with early stopping (about 1.5–2.5 hours).

The notebook downloads STL-10 automatically, trains the model, reports the test MAE and standard deviation, shows sample reconstructions, and verifies that the saved weights can be re-downloaded and loaded.

## Constraints satisfied

- Fully neural pipeline — no classical jigsaw search or matching algorithm.
- No pretrained models.
- Under 6M trainable parameters.
- Single self-contained Keras notebook, runs on Colab.

## Tech stack

Python · TensorFlow / Keras 3 · NumPy · Matplotlib · Google Colab
