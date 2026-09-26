# ATLAS drop-zone model training

This folder trains the model that finds the drop-zone marker from the drone's downward camera.
It produces a ~1 MB INT8 TensorFlow Lite model that runs on the Raspberry Pi 4 CPU
(15 fps target), so the drone decides where to drop without any offboard link.

## What the model does

For every camera frame it answers two questions:

| Output | Shape | Meaning |
|---|---|---|
| `presence` | 1 value, 0 to 1 | Probability that the marker is in the frame |
| `box` | 4 values, 0 to 1 | Marker centre and size `(cx, cy, w, h)` as a fraction of the image |

There is exactly one marker per delivery, so a single-object localiser is enough. It is much
lighter than a multi-box detector such as SSD or YOLO.

## Architecture

```
uint8 RGB 160x160 (raw camera pixels)
  -> Rescaling to [-1, 1]            normalisation is inside the model
  -> MobileNetV2, width 0.5          ImageNet weights, no top
  -> Global average pooling -> Dropout 0.2
  -> Dense(1, sigmoid)  "presence"
  -> Dense(4, sigmoid)  "box"
```

- **Loss:** binary cross-entropy for `presence`; Huber for `box`, weighted by the presence label
  so frames without a marker never train the box head.
- **Two-stage training:** heads only on a frozen backbone (lr 1e-3), then the top 40 % of the
  backbone is unfrozen with BatchNorm kept frozen (lr 1e-4, cosine decay).
- **Augmentation for a nadir camera:** flips and 90-degree rotations (the box is transformed
  with the image), brightness, contrast and saturation.

## Files

| File | Purpose |
|---|---|
| `common.py` | Label format, CSV reading and writing, image loading |
| `synth_data.py` | Generates a synthetic dataset: the ring-and-H marker composited onto aerial-style backgrounds with shadows, blur, tilt and look-alike distractors |
| `yolo_to_csv.py` | Converts real frames labelled in Roboflow, CVAT or Label Studio (YOLO export) into `labels.csv` |
| `train.py` | Builds and trains the model; saves `dropzone_float.keras` plus the validation and calibration splits |
| `quantize.py` | Full-integer INT8 conversion, then compares INT8 with float and fails if accuracy drops more than 2 points |
| `benchmark.py` | Measures latency and frame rate for 1 to 4 threads. Run it on the Pi |

## Dataset format

```
data/<set>/
  images/   *.jpg
  labels.csv
```

```csv
filename,present,cx,cy,w,h
img_0001.jpg,1,0.512,0.430,0.18,0.17
img_0002.jpg,0,0,0,0,0
```

Box values are normalised to the image. Keep about 40 % negative frames (no marker): grass,
roofs, shadows, the drone's own legs. They are what teach the model not to fire on clutter.

## Workflow

```bash
pip install -r requirements-train.txt          # GPU machine or Google Colab

# 1. data
python training/synth_data.py --out data/synth --n 6000
python training/yolo_to_csv.py --root data/field

# 2. train (synthetic + real frames)
python training/train.py --data data/synth data/field --out models

# 3. quantise and verify
python training/quantize.py --model models/dropzone_float.keras \
                            --out models/dropzone_mnv2_int8.tflite

# 4. on the Raspberry Pi
python training/benchmark.py --model models/dropzone_mnv2_int8.tflite
```

Set `perception.model_path` in `config/atlas.yaml` to the new model. If the Pi falls short of
15 fps, retrain with `--alpha 0.35` or `--size 128` and update `perception.input_size`.

## Collecting real data

Synthetic data alone is not enough for flight. Before flying:

- Print the marker on a 1 m x 1 m banner. At 12 m the Camera Module 3 sees about 15 m x 9 m of ground.
- Lock the camera focus at infinity and use a short exposure (around 2 ms) to limit motion blur.
- Record frames from the drone's own camera at 3, 6, 9, 12 and 15 m, at different times of day,
  over different ground. Aim for 1,500 to 2,000 labelled images.
- Keep one whole flight out as the test set so near-identical frames do not leak between train and test.
- Targets on that held-out flight: at least 95 % presence accuracy and a centre error under 0.05 of the frame.

## Bundled smoke-test model

`models/dropzone_mnv2_int8_synthetic.tflite` was trained from scratch for 12 epochs on 2,400
synthetic images, because ImageNet weights were not available in the build environment.

| | Float | INT8 |
|---|---|---|
| Presence accuracy (360 validation images) | 99.2 % | 98.9 % |
| Centre error (fraction of frame) | 0.093 | 0.092 |

It proves that the train, quantise and deploy chain works end to end. It says nothing about
accuracy on real flight imagery, so do not fly with it.
