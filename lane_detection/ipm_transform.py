"""
Apply a saved inverse-perspective-mapping (IPM) calibration.

Examples:
    python ipm_transform.py --source pictures/camera_view.jpg
    python ipm_transform.py --source 0
    python ipm_transform.py --source videos/track.mp4 --save outputs/track_bev.mp4

Controls for camera/video:
    q or Esc: quit
    s: save the current BEV frame as a JPG

The calibration is tied to the source image dimensions and camera pose.
If either changes, recalibrate rather than silently reusing the old matrix.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


class IPMTransformer:
    """Load and apply a previously saved four-point homography."""

    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        with self.calibration_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.source_width = int(self.config["source_image_width"])
        self.source_height = int(self.config["source_image_height"])
        self.output_width = int(self.config["destination_width_px"])
        self.output_height = int(self.config["destination_height_px"])
        self.source_points = np.asarray(
            self.config["source_points_px"], dtype=np.float32
        )
        self.homography = np.asarray(self.config["homography"], dtype=np.float64)

        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Calibration source dimensions must be positive.")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("BEV output dimensions must be positive.")
        if self.source_points.shape != (4, 2):
            raise ValueError("Calibration must contain exactly four 2D source points.")
        if self.homography.shape != (3, 3):
            raise ValueError("Homography must be a 3x3 matrix.")
        if not np.isfinite(self.homography).all():
            raise ValueError("Homography contains non-finite values.")
        if abs(np.linalg.det(self.homography)) < 1e-12:
            raise ValueError("Homography is singular and cannot be inverted.")

    def validate_frame(self, frame):
        """Reject mismatched image dimensions instead of applying a bad warp."""
        height, width = frame.shape[:2]
        expected = (self.source_width, self.source_height)
        actual = (width, height)
        if actual != expected:
            raise ValueError(
                f"Frame size is {width}x{height}, but calibration expects "
                f"{self.source_width}x{self.source_height}. "
                "Use the calibrated camera resolution or recalibrate."
            )

    def transform(self, frame):
        self.validate_frame(frame)
        return cv2.warpPerspective(
            frame,
            self.homography,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a saved IPM calibration to an image, video, or camera."
    )
    parser.add_argument(
        "--source", required=True,
        help="Image path, video path, or camera index (e.g. 0)."
    )
    parser.add_argument(
        "--calibration", default="calibration/ipm_calibration.json",
        help="Path to calibration JSON."
    )
    parser.add_argument(
        "--save", default=None,
        help="Optional output video path. For images, saves a BEV JPG."
    )
    return parser.parse_args()


def open_source(source):
    path = Path(source)
    if path.is_file():
        image = cv2.imread(str(path))
        if image is not None:
            return "image", image, None

    source_value = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return "stream", None, capture


def make_comparison(frame, bev):
    """Resize only for display; transformation uses the original frame."""
    display_original = cv2.resize(
        frame, (bev.shape[1], bev.shape[0]), interpolation=cv2.INTER_AREA
    )
    return np.hstack((display_original, bev))


def main():
    args = parse_args()
    transformer = IPMTransformer(args.calibration)
    source_type, image, capture = open_source(args.source)

    writer = None
    output_path = Path(args.save) if args.save else None

    try:
        if source_type == "image":
            frame = image
            bev = transformer.transform(frame)
            comparison = make_comparison(frame, bev)
            cv2.imshow("Original (display resized) | BEV", comparison)

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output_path), bev):
                    raise RuntimeError(f"Could not save BEV image: {output_path}")
                print(f"Saved BEV image: {output_path}")

            print("Image transformed. Press any key in the image window to close.")
            cv2.waitKey(0)
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 20.0

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps,
                (transformer.output_width * 2, transformer.output_height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

        frame_count = 0
        start_time = time.perf_counter()
        last_bev = None

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            bev = transformer.transform(frame)
            comparison = make_comparison(frame, bev)
            cv2.imshow("Original (display resized) | BEV", comparison)

            if writer:
                writer.write(comparison)

            frame_count += 1
            last_bev = bev

            elapsed = time.perf_counter() - start_time
            if elapsed > 0 and frame_count % 30 == 0:
                print(f"Processing FPS: {frame_count / elapsed:.2f}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s") and last_bev is not None:
                snapshot = Path("outputs") / "bev_snapshot.jpg"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot), last_bev)
                print(f"Saved snapshot: {snapshot}")

    finally:
        if capture:
            capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
