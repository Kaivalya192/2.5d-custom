from typing import Generator

import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def _configure_realsense_color_sensor(
    profile,
    exposure: float,
    gain: float,
    white_balance: float,
    auto_exposure: bool,
    auto_white_balance: bool,
) -> None:
    dev = profile.get_device()
    sensors = dev.query_sensors()
    color_sensor = None
    for s in sensors:
        name = s.get_info(rs.camera_info.name).lower()
        if "rgb" in name or "color" in name:
            color_sensor = s
            break
    if color_sensor is None:
        return
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
    if color_sensor.supports(rs.option.exposure):
        color_sensor.set_option(rs.option.exposure, float(exposure))
    if color_sensor.supports(rs.option.gain):
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if auto_white_balance else 0.0)
    if color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def source_realsense(
    width: int,
    height: int,
    fps: int,
    exposure: float,
    gain: float,
    white_balance: float,
    auto_exposure: bool,
    auto_white_balance: bool,
) -> Generator[np.ndarray, None, None]:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available.")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    _configure_realsense_color_sensor(
        profile,
        exposure=exposure,
        gain=gain,
        white_balance=white_balance,
        auto_exposure=auto_exposure,
        auto_white_balance=auto_white_balance,
    )
    try:
        for _ in range(15):
            _ = pipe.wait_for_frames(5000)
        while True:
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            if color:
                yield np.asanyarray(color.get_data())
    finally:
        pipe.stop()

