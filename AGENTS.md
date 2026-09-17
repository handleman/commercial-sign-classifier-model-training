# AGENTS.md

Learning project: how to train image classifiers with TensorFlow.js transfer learning. The commercial-sign domain (classes, download scripts, `data/` dirs) is only the example dataset — keep the pipeline generic, don't add sign business logic.

## Commands

```bash
npm install       # canvas needs native build (~1 min)
npm run train     # two-phase training, saves to output/model/
npm run evaluate  # loads output/model/best, prints accuracy + confusion matrix
```

No test, lint, or typecheck scripts. No CI. `tsc` exists via `typescript` dep but has no npm shortcut — run `npx tsc --noEmit` for a typecheck.

## Architecture

Transfer learning: MobileNetV1 (ImageNet, URL in `src/config.ts`) → custom 4-class head (`head_dense` → `head_dropout` → `head_output`).

- Classes: `cabinet` | `channel_letter` | `flat_cut` | `post_panel` (`CLASS_NAMES` in `src/config.ts`)
- Tune only via `src/config.ts` (LRs, epochs, `UNFREEZE_TOP_LAYERS=30`, dropout, dense units, batch size, split).
- Training is two-phase in `src/train.ts`: frozen base (LR 0.001, ≤20 epochs) → unfreeze top 30 base layers + recompile (LR 0.0001, ≤10 epochs). EarlyStopping (patience 5 on `val_acc`) in both phases.
- Checkpoints: best-by-`val_acc` → `output/model/best/`; final → `output/model/final/`; history → `output/logs/history.json`.
- Preprocessing (`src/dataLoader.ts`): sharp letterbox to 224×224 (black padding, alpha stripped, /255), PDFs rendered first-page-only at 2× via `pdfjs-dist` + `canvas`. Augmentation (train only): always-on horizontal flip + ±0.1 brightness jitter.
- `src/evaluate.ts` evaluates the `best` checkpoint on the val split.

## Gotchas

- `src/train.ts` and `src/evaluate.ts` must keep the `util.isNullOrUndefined/isFunction/isArray` polyfill at the top, before the `@tensorflow/tfjs-node` import — tfjs-node crashes on modern Node without it. Do not remove or reorder.
- `unfreezeTopLayers()` + `compileModel()` must stay paired in that order — recompiling after changing `trainable` flags is required.
- `src/model.ts` comments were fixed to say MobileNetV1 — if new stale comments appear, trust `src/config.ts` (`mobilenet_v1_1.0_224`, 4 classes).
- Loader glob is only `{jpg,jpeg,png,JPG,JPEG,PNG,pdf,PDF}` — `.webp`/`.gif`/`.bmp` files are silently skipped (there is already a `.webp` in `data/`; download scripts can save `.gif`/`.webp`). Convert or extend the glob.
- `scripts/download_*.py` fetch ~200 images per class from open sources (Bing via `icrawler`, DuckDuckGo top-up) directly into `data/<class>/`. They need `pip install icrawler duckduckgo-search requests` (not in `package.json`).
- `data/` is gitignored, so a fresh clone has no training images. This checkout already contains the example set (200 images × 4 classes, ~800 total) — don't commit it.
- Val split is a fresh random 80/20 shuffle on every run (`buildDatasets`), so `evaluate` does not necessarily score the same val set that training validated on — run-to-run metrics are not directly comparable.
- `.gitignore` ignores `data/`, `node_modules/`, and `output/`. Don't commit training artifacts — publish keepers via `gh release create` with `output/model/best/model.json` + `weights.bin` as side-by-side assets (see README).
- CPU-only (`@tensorflow/tfjs-node`); first run downloads MobileNet weights from Google Storage and caches them via tfjs.

## Workflow

- `AGENTS.md` and `CLAUDE.md` must stay in sync — mirror any repo-wide guidance in both.
- Work on a feature branch (`feat/<name>`), never directly on `main`. When finished and verified, stop and explicitly ask the user before merging — merges go into `main` via PR only.
