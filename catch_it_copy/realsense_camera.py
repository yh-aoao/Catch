#!/usr/bin/env python3
"""Small RealSense D435i interface for the robot computer.

The module talks to librealsense directly and does not require a ROS node.  It
can be imported from another program or run as a command-line preview tool.

Example (from this directory)::

    python3 realsense_camera.py --preview

Import example::

    from realsense_camera import RealSenseCamera

    with RealSenseCamera(width=640, height=480, fps=30) as camera:
        frame = camera.read()
        print(frame.depth_m.shape, camera.intrinsics)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CameraFrame:
    """One synchronized camera sample.

    ``color`` is BGR (OpenCV convention), ``depth`` is the native uint16
    depth image, and ``depth_m`` is depth converted to metres.  A field is
    ``None`` when that stream was disabled or did not arrive in the sample.
    """

    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    depth_m: Optional[np.ndarray]
    timestamp_ms: float
    accel: Optional[Tuple[float, float, float]] = None
    gyro: Optional[Tuple[float, float, float]] = None


class RealSenseCamera:
    """Synchronous RealSense D435/D435i camera wrapper."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        enable_color: bool = True,
        enable_depth: bool = True,
        enable_imu: bool = False,
        serial: Optional[str] = None,
        align_to_color: bool = True,
        warmup_frames: int = 5,
    ) -> None:
        if not enable_color and not enable_depth and not enable_imu:
            raise ValueError("At least one RealSense stream must be enabled")
        if align_to_color and not enable_color:
            raise ValueError("align_to_color=True requires enable_color=True")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height and fps must be positive")

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_imu = enable_imu
        self.serial = serial
        self.align_to_color = align_to_color
        self.warmup_frames = max(0, warmup_frames)

        self._rs: Any = None
        self._pipeline: Any = None
        self._profile: Any = None
        self._align: Any = None
        self._depth_scale = 0.001
        self._intrinsics: Optional[
            Dict[str, Union[float, int, str, List[float]]]
        ] = None
        self._started = False

    @staticmethod
    def _import_realsense() -> Any:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install librealsense and its "
                "Python bindings on the robot computer first."
            ) from exc
        return rs

    @classmethod
    def list_devices(cls) -> List[Dict[str, str]]:
        """Return connected RealSense devices and their basic information."""
        rs = cls._import_realsense()
        devices: List[Dict[str, str]] = []
        for device in rs.context().query_devices():
            info = {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
            devices.append(info)
        return devices

    def start(self) -> "RealSenseCamera":
        """Configure and start the selected device."""
        if self._started:
            return self

        rs = self._import_realsense()
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "No RealSense device found. Check the USB cable, power, and "
                "that another process is not exclusively using the camera."
            )

        if self.serial is not None:
            serials = [
                device.get_info(rs.camera_info.serial_number) for device in devices
            ]
            if self.serial not in serials:
                raise RuntimeError(
                    f"RealSense serial {self.serial!r} was not found. "
                    f"Connected serials: {serials}"
                )

        config = rs.config()
        if self.serial is not None:
            config.enable_device(self.serial)
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
        if self.enable_color:
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
        if self.enable_imu:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        self._pipeline = rs.pipeline()
        try:
            self._profile = self._pipeline.start(config)
        except Exception:
            self._pipeline = None
            raise

        self._rs = rs
        self._started = True
        try:
            self._align = (
                rs.align(rs.stream.color)
                if self.align_to_color and self.enable_depth
                else None
            )

            if self.enable_depth:
                depth_sensor = self._profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                intrinsics_stream = (
                    rs.stream.color if self.align_to_color else rs.stream.depth
                )
                video_profile = self._profile.get_stream(
                    intrinsics_stream
                ).as_video_stream_profile()
                intrinsics = video_profile.get_intrinsics()
                self._intrinsics = {
                    "width": int(intrinsics.width),
                    "height": int(intrinsics.height),
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "ppx": float(intrinsics.ppx),
                    "ppy": float(intrinsics.ppy),
                    "model": str(intrinsics.model),
                    "coeffs": [float(value) for value in intrinsics.coeffs],
                }

            for _ in range(self.warmup_frames):
                self._pipeline.wait_for_frames()
        except Exception:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        """Stop streaming. Safe to call more than once."""
        if self._pipeline is not None and self._started:
            self._pipeline.stop()
        self._started = False
        self._pipeline = None
        self._profile = None
        self._align = None

    close = stop

    def __enter__(self) -> "RealSenseCamera":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    @property
    def depth_scale(self) -> float:
        """Native depth unit in metres (usually 0.001 for D435i)."""
        return self._depth_scale

    @property
    def intrinsics(self) -> Optional[Dict[str, Union[float, int, str, List[float]]]]:
        """Intrinsics for the returned depth image, available after start."""
        return self._intrinsics

    def read(self, timeout_ms: int = 5000) -> CameraFrame:
        """Wait for and return one synchronized sample."""
        if not self._started or self._pipeline is None:
            raise RuntimeError("Camera is not started; call start() first")

        raw_frames = self._pipeline.wait_for_frames(timeout_ms)
        frames = raw_frames
        if self._align is not None:
            frames = self._align.process(raw_frames)

        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color = (
            np.asanyarray(color_frame.get_data()).copy() if color_frame else None
        )
        depth = (
            np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
        )
        depth_m = depth.astype(np.float32) * self._depth_scale if depth is not None else None

        accel = (
            self._motion_value(raw_frames, self._rs.stream.accel)
            if self.enable_imu
            else None
        )
        gyro = (
            self._motion_value(raw_frames, self._rs.stream.gyro)
            if self.enable_imu
            else None
        )
        timestamps = [frame.get_timestamp() for frame in (color_frame, depth_frame) if frame]
        timestamp_ms = float(timestamps[0] if timestamps else time.time() * 1000.0)
        return CameraFrame(color, depth, depth_m, timestamp_ms, accel, gyro)

    @staticmethod
    def _motion_value(frames: Any, stream_type: Any) -> Optional[Tuple[float, float, float]]:
        frame = frames.first_or_default(stream_type)
        if not frame:
            return None
        motion = frame.as_motion_frame()
        value = motion.get_motion_data()
        return (float(value.x), float(value.y), float(value.z))

    def depth_at(self, x: int, y: int, frame: Optional[CameraFrame] = None) -> float:
        """Return depth at pixel ``(x, y)`` in metres, or ``nan`` if invalid."""
        if frame is None:
            frame = self.read()
        if frame.depth_m is None:
            raise RuntimeError("Depth stream is disabled")
        height, width = frame.depth_m.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise IndexError(f"Pixel ({x}, {y}) is outside {width}x{height}")
        value = float(frame.depth_m[y, x])
        return value if value > 0 else float("nan")


def _run(args: argparse.Namespace) -> int:
    cv2 = None
    if args.preview or args.save_dir:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for previewing or saving PNG files"
            ) from exc

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_color=not args.no_color,
        enable_depth=not args.no_depth,
        enable_imu=args.imu,
        serial=args.serial,
        align_to_color=not args.no_color,
    )
    print("Connected devices:", RealSenseCamera.list_devices())
    with camera:
        print(
            f"Streaming at {args.width}x{args.height}@{args.fps}; "
            f"depth scale={camera.depth_scale}"
        )
        print("Intrinsics:", camera.intrinsics)
        count = 0
        started_at = time.monotonic()
        try:
            while args.frames <= 0 or count < args.frames:
                frame = camera.read()
                if args.preview and frame.color is not None:
                    cv2.imshow("RealSense color", frame.color)
                if args.preview and frame.depth_m is not None:
                    depth_vis = cv2.convertScaleAbs(
                        frame.depth_m, alpha=255.0 / args.depth_max_m
                    )
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imshow("RealSense depth", depth_vis)
                if save_dir and frame.color is not None:
                    cv2.imwrite(str(save_dir / f"color_{count:06d}.png"), frame.color)
                if save_dir and frame.depth is not None:
                    np.save(save_dir / f"depth_{count:06d}.npy", frame.depth)

                count += 1
                if count == 1 or time.monotonic() - started_at >= 1.0:
                    center_depth = float("nan")
                    if frame.depth_m is not None:
                        center_depth = camera.depth_at(
                            frame.depth_m.shape[1] // 2,
                            frame.depth_m.shape[0] // 2,
                            frame,
                        )
                    print(
                        f"frames={count}, center_depth={center_depth:.3f} m, "
                        f"accel={frame.accel}, gyro={frame.gyro}"
                    )
                    started_at = time.monotonic()
                if args.preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        except KeyboardInterrupt:
            print("Stopping camera")
    if args.preview:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start and preview an Intel RealSense D435i"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--serial", help="RealSense serial number when multiple cameras are connected"
    )
    parser.add_argument(
        "--frames", type=int, default=0, help="Frames to capture; 0 means until q/Esc"
    )
    parser.add_argument("--save-dir", help="Save color PNG and native depth NPY frames")
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=4.0,
        help="Depth range shown in preview",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Also enable accelerometer and gyroscope",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live OpenCV windows (omit this over a headless SSH session)",
    )
    args = parser.parse_args()

    if args.no_color and args.no_depth and not args.imu:
        parser.error("at least one of color, depth, or IMU must be enabled")
    if args.depth_max_m <= 0:
        parser.error("--depth-max-m must be positive")

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
