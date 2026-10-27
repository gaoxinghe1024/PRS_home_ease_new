"""One physical capture per device; lossless shared frames for isolated workers."""
import logging
import os
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np

log = logging.getLogger(__name__)
HEADER = 16


class FrameReader:
    def __init__(self, descriptor):
        self.memory = shared_memory.SharedMemory(name=descriptor["name"])
        self.shape = tuple(descriptor["shape"])
        self.size = int(np.prod(self.shape))

    def read(self):
        # A sequence check prevents reading pixels from two different captures.
        for _ in range(100):
            before, stamp = struct.unpack_from("<Qd", self.memory.buf)
            if before and not before % 2:
                pixels = bytes(self.memory.buf[HEADER:HEADER + self.size])
                after = struct.unpack_from("<Q", self.memory.buf)[0]
                if before == after:
                    if time.perf_counter() - stamp > 5:
                        raise RuntimeError("The camera image was interrupted. Check the connection and restart")
                    return np.frombuffer(pixels, np.uint8).reshape(self.shape).copy(), before // 2, stamp
            time.sleep(.001)
        raise RuntimeError("Camera image is temporarily unavailable")

    def close(self):
        self.memory.close()


class CameraCapture:
    def __init__(self, index, factory=None):
        import cv2
        self.index = index
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.jpeg = None
        self.sequence = 0
        self.frames = 0
        self.started = time.perf_counter()
        self.users = set()
        self.memory = None
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = (factory or cv2.VideoCapture)(index, backend)
        try:
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = (factory or cv2.VideoCapture)(index)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open camera {index}. Close other camera or video-call apps and check the camera privacy permission")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise ValueError(f"Camera {index} is connected but returns no frames. Check whether it is in use by another app or disconnected")
            self.shape = frame.shape
            self.memory = shared_memory.SharedMemory(create=True, size=HEADER + frame.nbytes)
            self.memory.buf[:HEADER] = bytes(HEADER)
            self._publish(frame)
        except Exception:
            self.cap.release()
            if self.memory:
                self.memory.close()
                self.memory.unlink()
            raise
        self.thread = threading.Thread(target=self._run, name=f"camera-{index}", daemon=True)
        self.thread.start()
        log.info("Camera %s opened, shape=%s", index, self.shape)

    def _publish(self, frame):
        import cv2
        if frame.shape != self.shape:
            raise RuntimeError("The camera resolution changed during capture. Please restart")
        self.sequence += 2
        struct.pack_into("<Q", self.memory.buf, 0, self.sequence - 1)
        self.memory.buf[HEADER:] = frame.tobytes()
        struct.pack_into("<d", self.memory.buf, 8, time.perf_counter())
        struct.pack_into("<Q", self.memory.buf, 0, self.sequence)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            if ok:
                self.jpeg = jpeg.tobytes()
            self.frames += 1

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Camera connection lost")
                self._publish(frame)
        except Exception as exc:
            self.error = str(exc)
            struct.pack_into("<d", self.memory.buf, 8, 0)
            log.exception("Camera %s capture failed", self.index)
        finally:
            self.cap.release()

    def descriptor(self):
        return {"name": self.memory.name, "shape": list(self.shape), "index": self.index}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.cap.release()
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("The camera capture thread has not exited yet, so the device cannot be reopened")
        self.memory.close()
        self.memory.unlink()
        log.info("Camera %s released", self.index)


class CameraHub:
    def __init__(self, factory=None):
        self.lock = threading.RLock()
        self.devices = {}
        self.factory = factory

    def acquire(self, index, owner):
        with self.lock:
            capture = self.devices.get(index)
            if capture and capture.error:
                raise ValueError(capture.error + ", please stop the tasks using this camera and try again")
            if capture is None:
                capture = CameraCapture(index, self.factory)
                self.devices[index] = capture
            capture.users.add(owner)
            log.info("Camera %s attached to %s", index, owner)
            return capture.descriptor()

    def descriptor(self, owner):
        """The frame description ``owner`` is currently using (reused directly on a handover)."""
        with self.lock:
            capture = next((c for c in self.devices.values() if owner in c.users), None)
            return capture.descriptor() if capture else None

    def users(self, owner):
        with self.lock:
            capture = next((c for c in self.devices.values() if owner in c.users), None)
            return set(capture.users) if capture else set()

    def busy(self, owner):
        """Other users of this stream (usable as a boolean, and shows who is using it)."""
        return self.users(owner) - {owner, "exercise-opening"}

    def release(self, owner, accept=()):
        """Release one user; an id in ``accept`` continues the same task and takes over this stream.

        Always-on mode hands the camera from video detection to exercise detection: when both use
        the same physical camera only the owner id is rebound, so the device is not reopened and the image never stops.
        """
        with self.lock:
            keep = set(accept)
            for index, capture in list(self.devices.items()):
                if owner not in capture.users:
                    continue
                capture.users.discard(owner)
                if not keep:
                    if not capture.users:
                        capture.close()
                        del self.devices[index]
                    continue
                capture.users |= keep
                if not capture.users:
                    capture.close()
                    del self.devices[index]

    def release_except(self, owners):
        """Release every user except ``owners``; the device and the pixel pipeline keep running.

        Only the owner ids are removed. Closing the device during a handover would give the new user
        a pipeline that has stopped sampling, and its first frame would report an interruption. A stream
        nobody uses any more is closed by ``release``.
        """
        keep = set(owners)
        with self.lock:
            for index, capture in list(self.devices.items()):
                capture.users &= keep
                if not capture.users:
                    capture.close()
                    del self.devices[index]

    def preview(self, owner):
        with self.lock:
            capture = next((c for c in self.devices.values() if owner in c.users), None)
            if capture is None or capture.error:
                return None
            with capture.lock:
                return capture.jpeg

    def status(self):
        with self.lock:
            return [{"index": c.index, "users": sorted(c.users), "error": c.error,
                     "capture_fps": round(c.frames / max(.1, time.perf_counter() - c.started), 1)}
                    for c in self.devices.values()]

    def close(self):
        with self.lock:
            for index, capture in list(self.devices.items()):
                capture.close()
                del self.devices[index]
