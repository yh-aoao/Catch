#!/usr/bin/env python3
"""Publish Intel RealSense D435i data as standard ROS2 sensor messages."""

from __future__ import annotations

import argparse
from typing import Any, Optional

import numpy as np

from realsense_camera import CameraFrame, RealSenseCamera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--imu", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def _image_message(image: np.ndarray, encoding: str, frame_id: str, stamp: Any) -> Any:
    from sensor_msgs.msg import Image

    contiguous = np.ascontiguousarray(image)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = contiguous.shape[0]
    message.width = contiguous.shape[1]
    message.encoding = encoding
    message.is_bigendian = False
    message.step = contiguous.strides[0]
    message.data = contiguous.tobytes()
    return message


def _camera_info_message(camera: RealSenseCamera, frame_id: str, stamp: Any) -> Any:
    from sensor_msgs.msg import CameraInfo

    intrinsics = camera.intrinsics
    if intrinsics is None:
        raise RuntimeError("Camera intrinsics are unavailable")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.width = int(intrinsics["width"])
    message.height = int(intrinsics["height"])
    message.distortion_model = "plumb_bob"
    message.d = list(intrinsics["coeffs"])
    message.k = [fx, 0.0, ppx, 0.0, fy, ppy, 0.0, 0.0, 1.0]
    message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message.p = [fx, 0.0, ppx, 0.0, 0.0, fy, ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return message


def main() -> int:
    args = _parse_args()
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, Imu
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are unavailable. Run "
            "'source /opt/ros/foxy/setup.bash' first."
        ) from exc

    class RealSenseNode(Node):
        def __init__(self) -> None:
            super().__init__("realsense_python")
            self.camera = RealSenseCamera(
                width=args.width,
                height=args.height,
                fps=args.fps,
                enable_imu=args.imu,
                serial=args.serial,
            )
            self.camera.start()
            self.color_pub = self.create_publisher(
                Image, "/camera/color/image_raw", qos_profile_sensor_data
            )
            self.depth_pub = self.create_publisher(
                Image,
                "/camera/aligned_depth_to_color/image_raw",
                qos_profile_sensor_data,
            )
            self.info_pub = self.create_publisher(
                CameraInfo, "/camera/color/camera_info", qos_profile_sensor_data
            )
            self.imu_pub: Optional[Any] = None
            if args.imu:
                self.imu_pub = self.create_publisher(
                    Imu, "/camera/imu", qos_profile_sensor_data
                )
            self.timer = self.create_timer(1.0 / args.fps, self.publish_frame)
            self.get_logger().info(
                f"D435i started at {args.width}x{args.height}@{args.fps}"
            )

        def publish_frame(self) -> None:
            try:
                frame = self.camera.read(timeout_ms=2000)
            except Exception as exc:
                self.get_logger().error(f"Failed to read RealSense frame: {exc}")
                return

            stamp = self.get_clock().now().to_msg()
            frame_id = "camera_color_optical_frame"
            if frame.color is not None:
                self.color_pub.publish(
                    _image_message(frame.color, "bgr8", frame_id, stamp)
                )
            if frame.depth_m is not None:
                self.depth_pub.publish(
                    _image_message(frame.depth_m, "32FC1", frame_id, stamp)
                )
            self.info_pub.publish(
                _camera_info_message(self.camera, frame_id, stamp)
            )
            self._publish_imu(frame, stamp)

        def _publish_imu(self, frame: CameraFrame, stamp: Any) -> None:
            if self.imu_pub is None or (frame.accel is None and frame.gyro is None):
                return
            message = Imu()
            message.header.stamp = stamp
            message.header.frame_id = "camera_imu_optical_frame"
            message.orientation_covariance[0] = -1.0
            if frame.accel is not None:
                (
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ) = frame.accel
            if frame.gyro is not None:
                (
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ) = frame.gyro
            self.imu_pub.publish(message)

        def destroy_node(self) -> bool:
            self.camera.stop()
            return super().destroy_node()

    rclpy.init()
    node = None
    try:
        node = RealSenseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
