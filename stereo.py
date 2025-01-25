import inspect
import sys
import time

sys.path.append("/usr/lib/python3/dist-packages")

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Generic, TypeAlias, cast

# must add "/usr/lib/python3/dist-packages" to your PYTHONPATH
from picamera2.picamera2 import Picamera2
from picamera2.job import Job
from picamera2.metadata import Metadata
from picamera2.controls import Controls
from picamera2.sensor_format import SensorFormat
from picamera2.request import CompletedRequest
import threading

if TYPE_CHECKING:
    RequestJob = Job[CompletedRequest]
else:
    RequestJob = Job


class Camera:
    cam: Picamera2
    format: str
    size: tuple[int, int]

    request_job: RequestJob | None = None
    image: NDArray | None = None
    metadata: Metadata | None = None
    timestamp: int = 0
    event: threading.Event = threading.Event()
    thread: threading.Thread | None = None

    def __init__(self, cam_idx: int) -> None:
        self.cam = Picamera2(camera_num=cam_idx)

    def start(self) -> None:
        self.cam.start()

    def capture_metadata(self) -> Metadata:
        return self.cam.capture_metadata()

    def configure(self, ctrls: Controls | dict[str, Any] | None = None) -> None:
        config = self.cam.create_still_configuration(
            buffer_count=1,
            queue=False,
            main={"format": "BGR888", "size": (4608, 2592)},
        )
        self.cam.configure(config)
        if ctrls is not None:
            self.cam.set_controls(ctrls)

    def set_controls(self, ctrls: Controls | dict[str, Any]) -> None:
        self.cam.set_controls(ctrls)

    def _signal_capture_request(self) -> Callable[[Job], None]:
        def fn(job: RequestJob) -> None:
            request: CompletedRequest | None = None
            try:
                request = job.get_result()
                self.image = request.make_buffer("main").reshape(2592, 4608, 3)
                self.metadata = cast(Metadata, request.get_metadata())
            finally:
                if request is not None:
                    request.release()
            self.timestamp = self.metadata["SensorTimestamp"]
            self.event.set()

        return fn

    def capture_request(self) -> None:
        self.request_job = self.cam.capture_request(
            flush=False, signal_function=self._signal_capture_request()
        )

    def capture_request_sync(self) -> None:
        request: CompletedRequest = self.cam.capture_request(flush=True)
        try:
            self.image = request.make_buffer("main").reshape(2592, 4608, 3)
            self.metadata = cast(Metadata, request.get_metadata())
        finally:
            if request is not None:
                request.release()
        self.timestamp = self.metadata["SensorTimestamp"]

    def wait(self):
        self.event.wait()
        self.event.clear()

ctrl: Controls = cast(Controls, {})
ctrl["AfMode"] = 0
ctrl["AeEnable"] = False
ctrl["ExposureTime"] = 10000

cam1 = Camera(0)
cam1.configure(ctrl)
cam1.start()

metadata: dict[str, Any] = cam1.capture_metadata()
ctrl["ExposureTime"] = metadata["ExposureTime"]
ctrl["AnalogueGain"] = metadata["AnalogueGain"]
ctrl["ColourGains"] = metadata["ColourGains"]
ctrl["AwbEnable"] = False

cam2 = Camera(1)
cam2.configure(ctrl)
cam2.start()

time.sleep(1)

for _ in range(10):
    cam1.capture_request()
    cam2.capture_request()

    cam1.wait()
    cam2.wait()

    diff_ns = abs(cam1.timestamp - cam2.timestamp)
    print(f"Timestamp difference: {diff_ns} ns")

cv2.imwrite("cam1.png", cam1.image)
cv2.imwrite("cam2.png", cam2.image)

cam1.cam.stop()
cam1.cam.close()

cam2.cam.stop()
cam2.cam.close()