import * as path from 'path';

export const IMAGE_SIZE = 224;
export const BATCH_SIZE = 16;
export const VALIDATION_SPLIT = 0.2;

export const EPOCHS_FROZEN = 20;
export const EPOCHS_FINETUNE = 10;
export const LR_FROZEN = 0.001;
export const LR_FINETUNE = 0.0001;

export const UNFREEZE_TOP_LAYERS = 30;
export const DROPOUT_RATE = 0.3;
export const DENSE_UNITS = 128;

export const CLASS_NAMES = ['cabinet', 'channel_letter', 'flat_cut', 'post_panel'] as const;
export type ClassName = typeof CLASS_NAMES[number];
export const NUM_CLASSES = CLASS_NAMES.length;

export const DATA_DIR = path.resolve(__dirname, '..', 'data');
export const OUTPUT_DIR = path.resolve(__dirname, '..', 'output');
export const MODEL_DIR = path.join(OUTPUT_DIR, 'model');
export const LOGS_DIR = path.join(OUTPUT_DIR, 'logs');

// MobileNetV1 pretrained weights — downloaded once and cached by tfjs
export const MOBILENET_URL =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
