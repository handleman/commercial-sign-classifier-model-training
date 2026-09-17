# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm install       # install deps (canvas requires native build — takes ~1 min)
npm run train     # Phase 1 (frozen base) + Phase 2 (fine-tune), saves to output/model/
npm run evaluate  # load best checkpoint, print accuracy + confusion matrix
```

## Architecture

Transfer learning pipeline: MobileNetV1 (ImageNet) → custom 4-class head.

**Classes:** `cabinet` | `channel_letter` | `flat_cut` | `post_panel`

**Training data:** Place images (jpg/png) or single-page PDFs under:
```
data/cabinet/
data/channel_letter/
data/flat_cut/
data/post_panel/
```

`data/` is gitignored — it is not committed. This checkout already includes an example set (200 images per class, ~800 total). On a fresh clone, either supply your own images or fetch examples from open sources with the download scripts:
```bash
pip install icrawler duckduckgo-search requests
python scripts/download_cabinet_signs.py  # or download_channel_letter / download_flat_cut / download_post_panel
```
The scripts save into the matching `data/<class>/` dir, ready for training.

**Two-phase training:**
1. Base frozen → train head only at LR 0.001 for up to 20 epochs
2. Unfreeze top 30 base layers → fine-tune at LR 0.0001 for up to 10 epochs

EarlyStopping (patience=5) applies to both phases. Best checkpoint (by `val_acc`) saved to `output/model/best/`; final model saved to `output/model/final/`. Training history written to `output/logs/history.json`.

## Key files

| File | Purpose |
|------|---------|
| `src/config.ts` | All hyperparameters and paths — edit here to tune |
| `src/dataLoader.ts` | Image + PDF loading, aspect-ratio-preserving resize (letterbox), augmentation |
| `src/model.ts` | Build MobileNetV1 + head, `unfreezeTopLayers()`, compile |
| `src/train.ts` | Two-phase training loop, checkpointing, history JSON |
| `src/evaluate.ts` | Per-class precision/recall + confusion matrix on val set |

## Notes

- MobileNetV1 weights are downloaded from Google Storage on first run and cached by tfjs. The comment in `model.ts` incorrectly says "MobileNetV2" — the actual URL in `config.ts` is `mobilenet_v1_1.0_224`.
- PDFs are rendered at 2× scale via `pdfjs-dist` + `canvas`, then letterboxed to 224×224. Only the first page is used.
- Augmentation is minimal: random horizontal flip + ±0.1 brightness jitter, applied only to training set.
- `train.ts` patches deprecated `util.isNullOrUndefined`, `util.isFunction`, and `util.isArray` at startup — required because tfjs-node internally uses these Node.js v12-removed helpers.
- Training is CPU-only (`@tensorflow/tfjs-node` native backend).
