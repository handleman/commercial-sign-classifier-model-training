// Polyfill Node.js util functions removed in Node v12+ but still used by tfjs-node
// eslint-disable-next-line @typescript-eslint/no-var-requires
const _util = require('util');
if (!_util.isNullOrUndefined) _util.isNullOrUndefined = (v: unknown) => v == null;
if (!_util.isFunction) _util.isFunction = (v: unknown) => typeof v === 'function';
if (!_util.isArray) _util.isArray = Array.isArray;

import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import { buildDatasets } from './dataLoader';
import { CLASS_NAMES, NUM_CLASSES, MODEL_DIR } from './config';

async function evaluate(): Promise<void> {
  const modelPath = path.join(MODEL_DIR, 'best');
  console.log(`Loading model from ${modelPath}...`);
  const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);

  const { valDataset, valSize } = await buildDatasets();

  // Confusion matrix: rows = actual, cols = predicted
  const matrix: number[][] = Array.from({ length: NUM_CLASSES }, () =>
    new Array(NUM_CLASSES).fill(0)
  );
  let total = 0;
  let correct = 0;

  await valDataset.forEachAsync((batch: { xs: tf.Tensor; ys: tf.Tensor }) => {
    const predictions = model.predict(batch.xs) as tf.Tensor2D;
    const predIndices = predictions.argMax(-1).arraySync() as number[];
    const trueIndices = batch.ys.argMax(-1).arraySync() as number[];

    for (let i = 0; i < trueIndices.length; i++) {
      const actual = trueIndices[i];
      const predicted = predIndices[i];
      matrix[actual][predicted]++;
      total++;
      if (actual === predicted) correct++;
    }

    tf.dispose([batch.xs, batch.ys, predictions]);
  });

  // ── Print results ──────────────────────────────────────────────────────────
  const overallAcc = total > 0 ? ((correct / total) * 100).toFixed(2) : '0.00';
  console.log(`\nOverall accuracy: ${correct}/${total} (${overallAcc}%)\n`);

  // Per-class metrics
  console.log('Per-class metrics:');
  const colWidth = 18;
  const pad = (s: string, w: number) => s.padEnd(w);

  for (let i = 0; i < NUM_CLASSES; i++) {
    const tp = matrix[i][i];
    const rowSum = matrix[i].reduce((a, b) => a + b, 0);
    const colSum = matrix.reduce((a, row) => a + row[i], 0);
    const recall = rowSum > 0 ? ((tp / rowSum) * 100).toFixed(1) : '-';
    const precision = colSum > 0 ? ((tp / colSum) * 100).toFixed(1) : '-';
    console.log(
      `  ${pad(CLASS_NAMES[i], colWidth)} precision: ${String(precision).padStart(5)}%  recall: ${String(recall).padStart(5)}%  (${tp}/${rowSum} correct)`
    );
  }

  // Confusion matrix table
  console.log('\nConfusion matrix (rows=actual, cols=predicted):');
  const header = ''.padEnd(colWidth) + CLASS_NAMES.map((n) => n.padEnd(colWidth)).join('');
  console.log(header);
  for (let i = 0; i < NUM_CLASSES; i++) {
    const row =
      CLASS_NAMES[i].padEnd(colWidth) +
      matrix[i].map((v) => String(v).padEnd(colWidth)).join('');
    console.log(row);
  }

  console.log(`\nValidation set size: ${valSize}`);
}

evaluate().catch((err) => {
  console.error(err);
  process.exit(1);
});
