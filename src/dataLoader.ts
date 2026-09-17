import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import { glob } from 'glob';
import { createCanvas } from 'canvas';
import {
  CLASS_NAMES,
  NUM_CLASSES,
  IMAGE_SIZE,
  BATCH_SIZE,
  VALIDATION_SPLIT,
  DATA_DIR,
} from './config';

// pdfjs-dist legacy build works in Node.js without a DOM
// eslint-disable-next-line @typescript-eslint/no-var-requires
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');

interface Sample {
  imagePath: string;
  labelIndex: number;
}

async function loadSamples(): Promise<Sample[]> {
  const samples: Sample[] = [];

  for (let i = 0; i < CLASS_NAMES.length; i++) {
    const classDir = path.join(DATA_DIR, CLASS_NAMES[i]);
    if (!fs.existsSync(classDir)) {
      console.warn(`Warning: data directory not found: ${classDir}`);
      continue;
    }
    const files = await glob(
      `${classDir}/**/*.{jpg,jpeg,png,JPG,JPEG,PNG,pdf,PDF}`
    );
    for (const file of files) {
      samples.push({ imagePath: file, labelIndex: i });
    }
    console.log(`Class "${CLASS_NAMES[i]}": ${files.length} images`);
  }

  // Shuffle
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }

  return samples;
}

/**
 * Render first page of a PDF to a raw PNG buffer using pdfjs-dist + canvas.
 * Scale 2.0 gives ~1200×1600px from a typical A4 — plenty for downscaling.
 */
async function pdfToBuffer(pdfPath: string): Promise<Buffer> {
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const doc = await pdfjsLib.getDocument({ data }).promise;
  const page = await doc.getPage(1);
  const viewport = page.getViewport({ scale: 2.0 });

  const canvas = createCanvas(viewport.width, viewport.height);
  const ctx = canvas.getContext('2d');

  await page.render({
    canvasContext: ctx as unknown as any,
    viewport,
  }).promise;

  return canvas.toBuffer('image/png');
}

/**
 * Load any supported image (jpg/png/pdf) and return a 224×224×3 float32 tensor
 * with aspect ratio preserved via letterboxing (black padding).
 */
async function imageToTensor(imagePath: string): Promise<tf.Tensor3D> {
  const ext = path.extname(imagePath).toLowerCase();

  let sharpInput: sharp.Sharp;
  if (ext === '.pdf') {
    const pngBuffer = await pdfToBuffer(imagePath);
    sharpInput = sharp(pngBuffer);
  } else {
    sharpInput = sharp(imagePath);
  }

  const buffer = await sharpInput
    .resize(IMAGE_SIZE, IMAGE_SIZE, {
      fit: 'contain',           // preserve aspect ratio
      background: { r: 0, g: 0, b: 0 },  // black letterbox padding
    })
    .removeAlpha()
    .raw()
    .toBuffer();

  const tensor = tf.tensor3d(new Uint8Array(buffer), [IMAGE_SIZE, IMAGE_SIZE, 3], 'float32');
  return tensor.div(255.0) as tf.Tensor3D;
}

function augment(tensor: tf.Tensor3D): tf.Tensor3D {
  let t = tf.image.flipLeftRight(tensor.expandDims(0) as tf.Tensor4D);
  t = t.add((Math.random() - 0.5) * 0.2) as tf.Tensor4D;
  t = tf.clipByValue(t, 0, 1) as tf.Tensor4D;
  return t.squeeze([0]) as tf.Tensor3D;
}

export async function buildDatasets(): Promise<{
  trainDataset: tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }>;
  valDataset: tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }>;
  trainSize: number;
  valSize: number;
}> {
  const samples = await loadSamples();
  if (samples.length === 0) {
    throw new Error(
      'No training images found. Add images/PDFs to data/cabinet, data/channel_letter, data/flat_cut, data/post_panel.'
    );
  }

  const splitIdx = Math.floor(samples.length * (1 - VALIDATION_SPLIT));
  const trainSamples = samples.slice(0, splitIdx);
  const valSamples = samples.slice(splitIdx);

  console.log(
    `Total: ${samples.length} | Train: ${trainSamples.length} | Val: ${valSamples.length}`
  );

  function samplesToDataset(
    items: Sample[],
    doAugment: boolean
  ): tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }> {
    return tf.data
      .generator(async function* () {
        for (const sample of items) {
          let xs = await imageToTensor(sample.imagePath);
          if (doAugment) xs = augment(xs);
          const ys = tf.oneHot(sample.labelIndex, NUM_CLASSES);
          yield { xs, ys };
        }
      })
      .shuffle(Math.min(200, items.length))
      .batch(BATCH_SIZE) as tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }>;
  }

  return {
    trainDataset: samplesToDataset(trainSamples, true),
    valDataset: samplesToDataset(valSamples, false),
    trainSize: trainSamples.length,
    valSize: valSamples.length,
  };
}
