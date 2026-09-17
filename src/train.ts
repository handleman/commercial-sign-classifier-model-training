// Polyfill Node.js util functions removed in Node v12+ but still used by tfjs-node
// eslint-disable-next-line @typescript-eslint/no-var-requires
const _util = require('util');
if (!_util.isNullOrUndefined) _util.isNullOrUndefined = (v: unknown) => v == null;
if (!_util.isFunction) _util.isFunction = (v: unknown) => typeof v === 'function';
if (!_util.isArray) _util.isArray = Array.isArray;

import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { buildDatasets } from './dataLoader';
import { buildModel, compileModel, unfreezeTopLayers, printSummary } from './model';
import {
  BATCH_SIZE,
  EPOCHS_FROZEN,
  EPOCHS_FINETUNE,
  LR_FROZEN,
  LR_FINETUNE,
  UNFREEZE_TOP_LAYERS,
  MODEL_DIR,
  LOGS_DIR,
} from './config';

interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  val_loss: number;
  val_acc: number;
}

async function train(): Promise<void> {
  fs.mkdirSync(MODEL_DIR, { recursive: true });
  fs.mkdirSync(LOGS_DIR, { recursive: true });

  // ── Data ──────────────────────────────────────────────────────────────────
  const { trainDataset, valDataset, trainSize, valSize } = await buildDatasets();
  const stepsPerEpoch = Math.floor(trainSize / BATCH_SIZE);
  const valSteps = Math.floor(valSize / BATCH_SIZE);

  // ── Model ─────────────────────────────────────────────────────────────────
  const model = await buildModel();
  compileModel(model, LR_FROZEN);
  printSummary(model);

  const history: EpochLog[] = [];

  // Callback: save best checkpoint by val_accuracy
  let bestValAcc = 0;
  const checkpointPath = path.join(MODEL_DIR, 'best');

  function makeCallbacks(phase: string): tf.CustomCallback {
    return new tf.CustomCallback({
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        const l = logs ?? {};
        const entry: EpochLog = {
          epoch: history.length + 1,
          loss: Number((l['loss'] ?? 0).toFixed(4)),
          acc: Number((l['acc'] ?? 0).toFixed(4)),
          val_loss: Number((l['val_loss'] ?? 0).toFixed(4)),
          val_acc: Number((l['val_acc'] ?? 0).toFixed(4)),
        };
        history.push(entry);

        console.log(
          `[${phase}] Epoch ${entry.epoch} — loss: ${entry.loss}  acc: ${entry.acc}  val_loss: ${entry.val_loss}  val_acc: ${entry.val_acc}`
        );

        if (entry.val_acc > bestValAcc) {
          bestValAcc = entry.val_acc;
          await model.save(`file://${checkpointPath}`);
          console.log(`  ✓ New best val_acc ${bestValAcc} — checkpoint saved`);
        }
      },
    });
  }

  // ── Phase 1: Train head only (base frozen) ────────────────────────────────
  console.log('\n=== Phase 1: Training classification head (base frozen) ===');
  await model.fitDataset(trainDataset, {
    epochs: EPOCHS_FROZEN,
    validationData: valDataset,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 5 }),
      makeCallbacks('frozen'),
    ],
  });

  // ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
  console.log('\n=== Phase 2: Fine-tuning top layers ===');
  unfreezeTopLayers(model, UNFREEZE_TOP_LAYERS);
  compileModel(model, LR_FINETUNE); // must recompile after changing trainable flags

  await model.fitDataset(trainDataset, {
    epochs: EPOCHS_FINETUNE,
    validationData: valDataset,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 5 }),
      makeCallbacks('finetune'),
    ],
  });

  // ── Save final model & history ────────────────────────────────────────────
  const finalPath = path.join(MODEL_DIR, 'final');
  await model.save(`file://${finalPath}`);
  console.log(`\nFinal model saved to ${finalPath}`);

  const historyPath = path.join(LOGS_DIR, 'history.json');
  fs.writeFileSync(historyPath, JSON.stringify(history, null, 2));
  console.log(`Training history saved to ${historyPath}`);

  console.log(`\nBest val_acc: ${bestValAcc}`);
}

train().catch((err) => {
  console.error(err);
  process.exit(1);
});
