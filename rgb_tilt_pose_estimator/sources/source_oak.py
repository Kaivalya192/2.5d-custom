from typing import Generator

import numpy as np

try:
    import depthai as dai
except Exception:
    dai = None


def _oak_enum_or_default(container, key: str, default_name: str):
    if hasattr(container, key):
        return getattr(container, key)
    return getattr(container, default_name)


def _oak_socket(name: str):
    s = str(name).upper()
    if s in ("RGB", "CAM_A", "A"):
        return _oak_enum_or_default(dai.CameraBoardSocket, "CAM_A", "RGB")
    if s in ("CAM_B", "B"):
        return _oak_enum_or_default(dai.CameraBoardSocket, "CAM_B", "LEFT")
    if s in ("CAM_C", "C"):
        return _oak_enum_or_default(dai.CameraBoardSocket, "CAM_C", "RIGHT")
    return _oak_enum_or_default(dai.CameraBoardSocket, "CAM_A", "RGB")


def _oak_sensor_resolution(name: str):
    n = str(name).lower()
    sr = dai.ColorCameraProperties.SensorResolution
    if n == "720p" and hasattr(sr, "THE_720_P"):
        return sr.THE_720_P
    if n == "4k" and hasattr(sr, "THE_4_K"):
        return sr.THE_4_K
    if n == "12mp" and hasattr(sr, "THE_12_MP"):
        return sr.THE_12_MP
    return sr.THE_1080_P


def _oak_xlink_nodes():
    if hasattr(dai.node, "XLinkOut") and hasattr(dai.node, "XLinkIn"):
        return dai.node.XLinkOut, dai.node.XLinkIn
    if hasattr(dai.node, "internal") and hasattr(dai.node.internal, "XLinkOut") and hasattr(dai.node.internal, "XLinkIn"):
        return dai.node.internal.XLinkOut, dai.node.internal.XLinkIn
    raise RuntimeError("DepthAI build does not expose XLinkOut/XLinkIn nodes.")


def source_oak(
    width: int,
    height: int,
    fps: float,
    socket: str,
    sensor_resolution: str,
    warmup_frames: int,
    auto_exposure: bool,
    exposure_us: float,
    iso: float,
    auto_white_balance: bool,
    white_balance: float,
) -> Generator[np.ndarray, None, None]:
    if dai is None:
        raise RuntimeError("depthai is not available. Install depthai in the current environment.")

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    xout_node, xin_node = _oak_xlink_nodes()
    xout = pipeline.create(xout_node)
    xin = pipeline.create(xin_node)
    xout.setStreamName("rgb")
    xin.setStreamName("control")

    cam.setBoardSocket(_oak_socket(socket))
    cam.setResolution(_oak_sensor_resolution(sensor_resolution))
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setInterleaved(False)
    cam.setFps(float(fps))
    cam.setVideoSize(int(width), int(height))
    cam.video.link(xout.input)
    xin.out.link(cam.inputControl)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_ctrl = device.getInputQueue(name="control")

        ctrl = dai.CameraControl()
        if auto_exposure:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(int(max(1.0, exposure_us)), int(max(50.0, iso)))
        if auto_white_balance:
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        else:
            ctrl.setManualWhiteBalance(int(max(1000.0, white_balance)))
        q_ctrl.send(ctrl)

        for _ in range(int(max(0, warmup_frames))):
            _ = q_rgb.get()

        while True:
            pkt = q_rgb.get()
            frame = pkt.getCvFrame()
            if frame is not None and frame.size > 0:
                yield frame
