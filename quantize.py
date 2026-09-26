"""Post-training full-integer (INT8) quantisation for the Raspberry Pi 4.

* weights and activations int8, calibrated on ~300 real training frames
  (the representative dataset must look like flight imagery, otherwise the
  activation ranges are wrong and accuracy collapses)
* uint8 input (raw camera pixels, scale 1 / zero-point 0), float32 outputs
  so the onboard code does not need to know the output quantisation
* ``TFLITE_BUILTINS_INT8`` only: conversion fails loudly if any op would
  fall back to float, instead of silently producing a slow hybrid model

After converting, the INT8 model is compared with the float model on the
validation split; the build fails if the accuracy drop exceeds ``--max-drop``.

    python training/quantize.py --model models/dropzone_float.keras --out models/dropzone_mnv2_int8.tflite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import IMG_SIZE, load_image  # noqa: E402


def convert(model, calib_rows, size=IMG_SIZE):
    import tensorflow as tf

    def rep():
        for r in calib_rows:
            yield [load_image(r["path"], size).astype(np.float32)[None]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    return conv.convert()


def tflite_predict(model_bytes, images):
    from atlas.perception.detector import TFLiteDropZoneDetector, _load_interpreter  # noqa: F401
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    it.allocate_tensors()
    det = TFLiteDropZoneDetector("", images[0].shape[0], interpreter=it, conf_threshold=0.5)
    out = []
    for img in images:
        d = det.detect(img, 0.0)
        out.append((d.confidence, d.cx, d.cy, d.w, d.h))
    return np.array(out, np.float32)


def score(pred, rows, thr=0.5):
    pres = np.array([r["present"] for r in rows]) > 0.5
    boxes = np.array([r["box"] for r in rows], np.float32)
    p = pred[:, 0] >= thr
    acc = float((p == pres).mean())
    centre_err = float(np.mean(np.hypot(pred[pres, 1] - boxes[pres, 0], pred[pres, 2] - boxes[pres, 1]))) if pres.any() else 0.0
    return {"accuracy": acc, "centre_err": centre_err}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dropzone_float.keras")
    ap.add_argument("--calib", default=None, help="json rows for calibration (default: next to model)")
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="models/dropzone_mnv2_int8.tflite")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--max-drop", type=float, default=0.02, help="max allowed accuracy drop vs float")
    a = ap.parse_args(argv)

    import tensorflow as tf

    mdir = Path(a.model).parent
    calib = json.loads(Path(a.calib or mdir / "calib_split.json").read_text())
    val = json.loads(Path(a.val or mdir / "val_split.json").read_text())
    model = tf.keras.models.load_model(a.model)
    tfl = convert(model, calib, a.size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(tfl)
    print(f"INT8 model: {a.out} ({len(tfl) / 1024:.0f} KiB)")

    imgs = [load_image(r["path"], a.size) for r in val]
    fp = model.predict(np.stack(imgs).astype(np.float32), verbose=0)
    fpred = np.concatenate([fp["presence"], fp["box"]], axis=1)
    qpred = tflite_predict(tfl, imgs)
    fs, qs = score(fpred, val), score(qpred, val)
    print(f"float : acc {fs['accuracy']:.3f}  centre err {fs['centre_err']:.4f}")
    print(f"int8  : acc {qs['accuracy']:.3f}  centre err {qs['centre_err']:.4f}")
    report = {"float": fs, "int8": qs, "size_kib": len(tfl) / 1024}
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    if fs["accuracy"] - qs["accuracy"] > a.max_drop:
        print("FAIL: quantisation accuracy drop too large")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
