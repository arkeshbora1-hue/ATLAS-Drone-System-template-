"""Camera capture: Picamera2 on the Pi, OpenCV VideoCapture elsewhere.

A capture thread keeps only the newest frame so inference always runs on
the freshest image; frames that arrive while the model is busy are dropped
rather than queued (queuing would add latency, not throughput).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class Camera:
    def __init__(self, width: int = 640, height: int = 480, index: int = 0):
        self.size = (width, height)
        self.index = index
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                controls={"FrameRate": 30}))
            cam.start()
            self._backend = ("picamera2", cam)
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(self.index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            if not cap.isOpened():
                raise RuntimeError("no camera available")
            self._backend = ("opencv", cap)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _grab(self) -> Optional[np.ndarray]:
        kind, dev = self._backend
        if kind == "picamera2":
            # Picamera2 "RGB888" is BGR byte order in memory; flip to RGB
            return dev.capture_array("main")[:, :, ::-1]
        ok, frame = dev.read()
        return frame[:, :, ::-1] if ok else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            f = self._grab()
            if f is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._stamp = f, time.monotonic()

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame, self._stamp

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(1.0)
        if self._backend:
            kind, dev = self._backend
            dev.stop() if kind == "picamera2" else dev.release()
