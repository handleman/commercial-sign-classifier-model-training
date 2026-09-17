# Releasing a Model

`output/` is gitignored, so trained checkpoints never live in the repo. To keep a checkpoint worth sharing (e.g. for `npm run evaluate`, a demo, or a browser app), publish it as downloadable files on a GitHub Release.

## Prerequisites

- A trained model: `npm run train` → `output/model/best/` (contains `model.json` + `weights.bin`).
- GitHub CLI (`gh`). It's not installed in this checkout — install it (`brew install gh` on macOS) and sign in once with `gh auth login`.

## Steps

1. **Check the metrics** in `output/logs/history.json` (best `val_acc`, no overfitting) and confirm `output/model/best/` holds both `model.json` and `weights.bin`.

2. **Pick a version tag**, e.g. `model-v1.0.0` (the `model-` prefix keeps model releases distinct from code releases).

3. **Create the release with both files as side-by-side assets** (do not zip them — `model.json` locates `weights.bin` via a relative path, so they must sit next to each other):
   ```bash
   gh release create model-v1.0.0 \
     output/model/best/model.json \
     output/model/best/weights.bin \
     output/logs/history.json \
     --title "Model v1.0.0" \
     --notes "val_acc 0.93 on 160-image val split. See history.json for full training log."
   ```
   `history.json` rides along so the release records which metrics this checkpoint hit.

4. **Verify** at https://github.com/handleman/commercial-sign-classifier-model-training/releases — the release must list `model.json` and `weights.bin` as separate assets.

## Using a released model

Load it directly from its URL (relative weight resolution works because both files share the release's download path):

```ts
const model = await tf.loadLayersModel(
  'https://github.com/handleman/commercial-sign-classifier-model-training/releases/download/model-v1.0.0/model.json'
);
```

To run the local `npm run evaluate` against a released checkpoint instead of retraining, download the assets into the expected local layout:

```bash
gh release download model-v1.0.0 -p 'model.json' -p 'weights.bin' -D output/model/best
npm run evaluate
```
