"""Train the MobileNetV2 drop-zone localiser.

Architecture
    uint8 RGB 160x160
      -> Rescaling(1/127.5, -1)                  (normalisation inside the model)
      -> MobileNetV2 (alpha 0.5, ImageNet weights, no top)
      -> GlobalAveragePooling -> Dropout(0.2)
      -> Dense(1, sigmoid)  "presence"
      -> Dense(4, sigmoid)  "box" = (cx, cy, w, h)

Loss
    presence: binary cross-entropy
    box:      Huber, weighted by the presence label, so negatives never
              pull the box head toward an arbitrary (0,0,0,0)

Schedule
    stage 1: backbone frozen, head only           (lr 1e-3)
    stage 2: top ~40% of backbone unfrozen, BN frozen  (lr 1e-4, cosine decay)
    BatchNorm layers stay in inference mode during fine-tuning; updating
    their statistics on a small dataset is the most common way to wreck a
    pretrained MobileNet.

    python training/train.py --data data/synth --out models/ --epochs1 8 --epochs2 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_SIZE, load_image, read_labels  # noqa: E402


def build_model(size=IMG_SIZE, alpha=0.5, weights="imagenet"):
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = layers.Input((size, size, 3), dtype=tf.float32, name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="normalise")(inp)
    backbone = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=alpha,
                                                 include_top=False, weights=weights)
    backbone._name = "backbone"
    # Pretrained: keep BatchNorm in inference mode (training=False) so its
    # ImageNet statistics survive fine-tuning. From scratch there are no
    # statistics yet, so BN must run in normal training mode.
    if not weights:
        # MobileNetV2's BN momentum (0.999) needs thousands of steps before the
        # moving statistics used at validation/inference time are meaningful;
        # on small from-scratch runs that makes val accuracy look like chance.
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = 0.9
    x = backbone(x, training=False) if weights else backbone(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    presence = layers.Dense(1, activation="sigmoid", name="presence")(x)
    box = layers.Dense(4, activation="sigmoid", name="box")(x)
    return tf.keras.Model(inp, {"presence": presence, "box": box}, name="atlas_dropzone"), backbone


def make_dataset(rows, size, batch, augment, shuffle):
    import tensorflow as tf

    paths = [r["path"] for r in rows]
    pres = np.array([[r["present"]] for r in rows], np.float32)
    boxes = np.array([r["box"] for r in rows], np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, pres, boxes))
    if shuffle:
        ds = ds.shuffle(len(rows), reshuffle_each_iteration=True)

    def load(path, p, b):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        return img, p, b

    def aug(img, p, b):
        cx, cy, w, h = b[0], b[1], b[2], b[3]
        # horizontal / vertical flips (box follows)
        if tf.random.uniform(()) < 0.5:
            img, cx = tf.image.flip_left_right(img), tf.where(p[0] > 0, 1.0 - cx, cx)
        if tf.random.uniform(()) < 0.5:
            img, cy = tf.image.flip_up_down(img), tf.where(p[0] > 0, 1.0 - cy, cy)
        # 90-degree rotation (nadir view has no preferred "up")
        if tf.random.uniform(()) < 0.5:
            img = tf.image.rot90(img, k=1)          # counter-clockwise
            cx, cy, w, h = (tf.where(p[0] > 0, cy, cx), tf.where(p[0] > 0, 1.0 - cx, cy),
                            tf.where(p[0] > 0, h, w), tf.where(p[0] > 0, w, h))
        img = tf.image.random_brightness(img, 40.0)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, p, tf.stack([cx, cy, w, h])

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda img, p, b: (img, {"presence": p, "box": b}, {"presence": tf.ones_like(p[0]), "box": p[0]}))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def split(rows, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    nv = max(1, int(len(rows) * val_frac))
    return [rows[i] for i in idx[nv:]], [rows[i] for i in idx[:nv]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more dataset roots")
    ap.add_argument("--out", default="models")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (offline)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs1", type=int, default=8)
    ap.add_argument("--epochs2", type=int, default=20)
    ap.add_argument("--unfreeze-frac", type=float, default=0.4)
    a = ap.parse_args(argv)

    import tensorflow as tf

    rows = [r for root in a.data for r in read_labels(root)]
    train_rows, val_rows = split(rows)
    print(f"train {len(train_rows)}  val {len(val_rows)}")
    train_ds = make_dataset(train_rows, a.size, a.batch, augment=True, shuffle=True)
    val_ds = make_dataset(val_rows, a.size, a.batch, augment=False, shuffle=False)

    model, backbone = build_model(a.size, a.alpha, None if a.weights == "none" else a.weights)
    losses = {"presence": tf.keras.losses.BinaryCrossentropy(),
              "box": tf.keras.losses.Huber(delta=0.1)}
    metrics = {"presence": [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
               "box": [tf.keras.metrics.MeanAbsoluteError(name="mae")]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(out / "dropzone_best.keras"), monitor="val_loss",
                                              save_best_only=True)

    # stage 1: head only
    backbone.trainable = a.weights == "none"   # from scratch: train everything
    model.compile(tf.keras.optimizers.Adam(1e-3), loss=losses, loss_weights={"presence": 1.0, "box": 5.0},
                  metrics=metrics)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs1, callbacks=[ckpt], verbose=2)

    # stage 2: fine-tune the top of the backbone, keep BatchNorm frozen
    h2 = None
    if a.epochs2 > 0:
        backbone.trainable = True
        n_frozen = int(len(backbone.layers) * (1 - a.unfreeze_frac))
        for i, layer in enumerate(backbone.layers):
            layer.trainable = i >= n_frozen and not isinstance(layer, tf.keras.layers.BatchNormalization)
        steps = a.epochs2 * max(1, len(train_rows) // a.batch)
        lr = tf.keras.optimizers.schedules.CosineDecay(1e-4, steps)
        model.compile(tf.keras.optimizers.Adam(lr), loss=losses, loss_weights={"presence": 1.0, "box": 5.0},
                      metrics=metrics)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs2, callbacks=[ckpt], verbose=2)

    model = tf.keras.models.load_model(out / "dropzone_best.keras")
    model.save(out / "dropzone_float.keras")
    hist = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
    if h2:
        for k, vals in h2.history.items():
            hist.setdefault(k, []).extend(float(v) for v in vals)
    (out / "train_history.json").write_text(json.dumps(hist, indent=1))
    # save the validation split for quantisation calibration and evaluation
    (out / "val_split.json").write_text(json.dumps(val_rows))
    (out / "calib_split.json").write_text(json.dumps(train_rows[:300]))
    print(f"saved {out / 'dropzone_float.keras'}")


if __name__ == "__main__":
    main()
