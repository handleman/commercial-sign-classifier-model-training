# Image Classifier Training (TensorFlow.js)

Learning project: how to train an image classifier with TensorFlow.js transfer learning — MobileNetV1 (ImageNet) plus a small custom head, trained in two phases (frozen base, then fine-tune). Commercial sign types are just the example dataset.

## Commands

```bash
npm install       # canvas needs a native build (~1 min)
npm run train     # two-phase training, saves to output/model/
npm run evaluate  # loads output/model/best, prints accuracy + confusion matrix
```

## Data

Training images live under `data/`, one folder per class:

```
data/cabinet/
data/channel_letter/
data/flat_cut/
data/post_panel/
```

- `data/` is **gitignored** — images are never committed. A fresh clone starts with no training data.
- This checkout already contains an example set (~200 images per class, ~800 total) so you can train right away.
- To (re)fetch example images from open sources, use the download scripts (Bing via `icrawler`, DuckDuckGo top-up):
  ```bash
  pip install icrawler duckduckgo-search requests
  python scripts/download_cabinet_signs.py  # or download_channel_letter / download_flat_cut / download_post_panel
  ```
  The scripts save straight into the matching `data/<class>/` folder (creating it if needed). Supported formats: jpg/png (plus single-page PDFs, rendered at 2×).

## Pipeline

1. **Preprocess** (`src/dataLoader.ts`): letterbox to 224×224, train-only flip + brightness jitter, random 80/20 train/val split.
2. **Phase 1** (`src/train.ts`): base frozen, train head only (LR 0.001, ≤20 epochs).
3. **Phase 2**: unfreeze top 30 base layers, fine-tune (LR 0.0001, ≤10 epochs). Early stopping (patience 5 on `val_acc`) in both phases.
4. **Artifacts**: best-by-`val_acc` → `output/model/best/`; final → `output/model/final/`; history → `output/logs/history.json`.

All hyperparameters live in `src/config.ts`. Training is CPU-only; MobileNet weights download from Google Storage on first run.

## Keeping a trained model

`output/` is **gitignored** — checkpoints (~26 MB of binaries, rewritten wholesale on every run) don't belong in git. When you have a checkpoint worth keeping, publish `output/model/best/` as a versioned attachment on a GitHub Release instead:

```bash
npm run train
gh release create v1.0.0 output/model/best/model.json output/model/best/weights.bin \
  --title "v1.0.0" --notes "val_acc 0.93, see output/logs/history.json"
```

Upload both files (not a zip): `model.json` resolves `weights.bin` via a relative path, so keeping both assets side by side under the same release lets TensorFlow.js load the model straight from its URL:

```ts
const model = await tf.loadLayersModel(
  'https://github.com/<user>/<repo>/releases/download/v1.0.0/model.json'
);
```
