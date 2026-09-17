import * as tf from '@tensorflow/tfjs-node';
import {
  IMAGE_SIZE,
  NUM_CLASSES,
  DENSE_UNITS,
  DROPOUT_RATE,
  MOBILENET_URL,
} from './config';

/**
 * Load MobileNetV1 pretrained on ImageNet, strip the top classifier,
 * and attach a new example head (number of classes comes from config).
 *
 * Architecture:
 *   MobileNetV1 base (frozen) → GlobalAveragePooling2D → Dense(relu)
 *   → Dropout → Dense(softmax)
 */
export async function buildModel(): Promise<tf.LayersModel> {
  console.log('Loading MobileNetV1 pretrained weights...');
  const mobilenet = await tf.loadLayersModel(MOBILENET_URL);

  // Take the output of the base's global-average-pooling layer as features,
  // discarding the original 1000-class ImageNet head.
  const baseOutput = mobilenet.getLayer('global_average_pooling2d_1').output as tf.SymbolicTensor;

  // Freeze entire base
  for (const layer of mobilenet.layers) {
    layer.trainable = false;
  }

  // Classification head
  const x = tf.layers.dense({ units: DENSE_UNITS, activation: 'relu', name: 'head_dense' })
    .apply(baseOutput) as tf.SymbolicTensor;
  const dropped = tf.layers.dropout({ rate: DROPOUT_RATE, name: 'head_dropout' })
    .apply(x) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax', name: 'head_output' })
    .apply(dropped) as tf.SymbolicTensor;

  const model = tf.model({
    inputs: mobilenet.inputs,
    outputs: output,
    name: 'sign_classifier',
  });

  return model;
}

/**
 * Unfreeze the top N layers of the base for fine-tuning.
 * Call this after the head has converged, then recompile with a lower LR.
 */
export function unfreezeTopLayers(model: tf.LayersModel, n: number): void {
  // Layers belonging to the base are all except the 3 head layers we added
  const baseLayers = model.layers.filter(
    (l) => !['head_dense', 'head_dropout', 'head_output'].includes(l.name)
  );
  const unfreezeFrom = Math.max(0, baseLayers.length - n);
  for (let i = unfreezeFrom; i < baseLayers.length; i++) {
    baseLayers[i].trainable = true;
  }
  console.log(`Unfroze ${baseLayers.length - unfreezeFrom} base layers for fine-tuning.`);
}

export function compileModel(model: tf.LayersModel, lr: number): void {
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}

export function printSummary(model: tf.LayersModel): void {
  const trainable = model.layers.filter((l) => l.trainable).length;
  console.log(`Layers: ${model.layers.length} total, ${trainable} trainable`);
  model.summary();
}
