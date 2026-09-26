"""Drop-zone detector: INT8-quantised MobileNetV2 running under TFLite.

Model contract (produced by ``training/train.py`` + ``training/quantize.py``):
  input   uint8 [1, S, S, 3] raw RGB pixels, S = 160 by default
          (normalisation to [-1, 1] happens inside the model)
  output  "presence" [1, 1]  sigmoid probability a drop-zone marker is visible
          "box"      [1, 4]  sigmoid (cx, cy, w, h) normalised to the image

A single-object localiser (not a multi-box SSD) is enough because there is
exactly one marker per delivery; it keeps the head tiny so the MobileNetV2
backbone dominates latency (budget <= 45 ms/frame on a Pi 4 with 4 threads,
leaving headroom for the 15 fps / 66 ms frame budget).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from atlas.core.state import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame_rgb: np.ndarray | None, now: float) -> Detection: ...


def _load_interpreter(path: str, threads: int):
    try:
        from tflite_runtime.interpreter import Interpreter  # Pi: pip install tflite-runtime
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # dev machines
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:  # nearest-neighbour fallback without OpenCV
        h, w = frame.shape[:2]
        ys = (np.arange(size) * h / size).astype(int)
        xs = (np.arange(size) * w / size).astype(int)
        return frame[ys][:, xs]


def quantize_input(img_uint8: np.ndarray, detail: dict) -> np.ndarray:
    """Map an RGB uint8 image to the model's input tensor.

    The MobileNetV2 normalisation (x/127.5 - 1) is baked into the model as a
    Rescaling layer, so the model's float input domain is raw pixels 0..255.
    Exported with ``inference_input_type=tf.uint8`` the input quantisation is
    scale=1, zero_point=0 and the camera frame is fed with no arithmetic at
    all; other quantisations are handled generically as q = x/scale + zp.
    """
    dtype = detail["dtype"]
    scale, zp = detail["quantization"]
    if dtype == np.float32:
        return img_uint8.astype(np.float32)[None]
    if dtype == np.uint8 and abs(scale - 1.0) < 1e-6 and zp == 0:
        return np.ascontiguousarray(img_uint8, dtype=np.uint8)[None]
    q = np.round(img_uint8.astype(np.float32) / scale + zp)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)[None]


def dequantize(arr: np.ndarray, detail: dict) -> np.ndarray:
    scale, zp = detail["quantization"]
    if arr.dtype == np.float32 or scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zp) * scale


class TFLiteDropZoneDetector(Detector):
    def __init__(self, model_path: str, input_size: int = 160, threads: int = 4,
                 conf_threshold: float = 0.6, interpreter=None):
        self.it = interpreter or _load_interpreter(model_path, threads)
        self.size = input_size
        self.conf = conf_threshold
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        # identify outputs by shape: presence has 1 value, box has 4
        self.out_presence = next(o for o in outs if int(np.prod(o["shape"])) == 1)
        self.out_box = next(o for o in outs if int(np.prod(o["shape"])) == 4)

    def detect(self, frame_rgb, now: float) -> Detection:
        t0 = time.perf_counter()
        x = quantize_input(resize_rgb(frame_rgb, self.size), self.inp)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        p = float(dequantize(self.it.get_tensor(self.out_presence["index"]), self.out_presence).ravel()[0])
        cx, cy, w, h = dequantize(self.it.get_tensor(self.out_box["index"]), self.out_box).ravel()[:4]
        lat = time.perf_counter() - t0
        present = p >= self.conf
        return Detection(now, present, p, float(cx), float(cy), float(w), float(h), lat)
