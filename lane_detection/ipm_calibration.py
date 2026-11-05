"""
Interactive inverse perspective mapping (IPM) calibration.

Usage:
    python ipm_calibration.py --source camera_view.jpg
    python ipm_calibration.py --source 0
    python ipm_calibration.py --source camera_view.jpg --width 900 --height 600

Controls:
    Left click: select four source points in this order:
        1. top-left, 2. top-right, 3. bottom-right, 4. bottom-left
    u: undo last point
    r: reset all points
    Enter: generate/update bird's-eye view
    s: save calibration JSON and BEV image (after 4 points are selected)
    q or Esc: quit

Notes:
    - Select points on the road surface, not on raised objects.
    - The camera must remain fixed after calibration.
    - Destination dimensions are output pixels, not physical metres.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "IPM calibration - click TL, TR, BR, BL"


def load_frame(source: str):
    """Load one image frame or one frame from a camera/video source."""
    path = Path(source)
    if path.is_file():
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        return frame, "image"

    camera_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from: {source}")
    return frame, "capture"


def validate_points(points, frame_shape):
    if len(points) != 4:
        raise ValueError("Exactly four source points are required.")

    height, width = frame_shape[:2]
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point {(x, y)} is outside the image.")

    contour = np.array(points, dtype=np.float32)
    area = abs(cv2.contourArea(contour))
    if area < 1000:
        raise ValueError("Selected ROI is too small. Choose a larger road region.")

    return contour


def create_bev(frame, source_points, output_width, output_height):
    """Warp the selected road quadrilateral into a rectangular BEV image."""
    src = validate_points(source_points, frame.shape)

    dst = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(
        frame,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return bev, homography


class CalibrationUI:
    def __init__(self, frame, output_width, output_height, output_prefix):
        self.frame = frame
        self.display = frame.copy()
        # Keep large camera frames usable on ordinary laptop displays.
        self.display_scale = min(1.0, 1280 / frame.shape[1], 800 / frame.shape[0])
        self.points = []
        self.output_width = output_width
        self.output_height = output_height
        self.output_prefix = Path(output_prefix)
        self.bev = None
        self.homography = None

    def redraw(self):
        self.display = self.frame.copy()

        for index, point in enumerate(self.points):
            cv2.circle(self.display, point, 8, (0, 0, 255), -1)
            cv2.putText(
                self.display,
                f"{index + 1}",
                (point[0] + 12, point[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, self.points[i], self.points[i + 1],
                         (0, 220, 255), 2)

        if len(self.points) == 4:
            cv2.line(self.display, self.points[3], self.points[0],
                     (0, 220, 255), 2)
            overlay = self.display.copy()
            polygon = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (255, 180, 0))
            self.display = cv2.addWeighted(overlay, 0.18, self.display, 0.82, 0)

        instructions = [
            "Click: TL -> TR -> BR -> BL",
            "u: undo | r: reset | Enter: warp | s: save | q: quit",
            f"Selected points: {len(self.points)}/4",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(self.display, line, (15, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 4)
            cv2.putText(self.display, line, (15, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        shown = self.display
        if self.display_scale < 1.0:
            shown = cv2.resize(
                self.display,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(WINDOW_NAME, shown)

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            # Convert displayed-window coordinates back to original image pixels.
            source_x = int(round(x / self.display_scale))
            source_y = int(round(y / self.display_scale))
            self.points.append((source_x, source_y))
            self.redraw()

    def generate(self):
        if len(self.points) != 4:
            print("Select all four points first.")
            return
        try:
            self.bev, self.homography = create_bev(
                self.frame, self.points, self.output_width, self.output_height
            )
            cv2.imshow("Bird's-eye view (BEV)", self.bev)
            print("\nHomography matrix:")
            print(self.homography)
            print("BEV generated. Inspect it before saving.")
        except ValueError as exc:
            print(f"Calibration error: {exc}")

    def save(self):
        if self.bev is None or self.homography is None:
            print("Generate a BEV first by pressing Enter.")
            return

        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        config_path = self.output_prefix.with_suffix(".json")
        bev_path = self.output_prefix.with_name(
            self.output_prefix.name + "_bev.jpg"
        )

        config = {
            "source_image_width": int(self.frame.shape[1]),
            "source_image_height": int(self.frame.shape[0]),
            "source_points_px": [list(p) for p in self.points],
            "destination_width_px": self.output_width,
            "destination_height_px": self.output_height,
            "homography": self.homography.tolist(),
            "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "physical_scale": None,
            "status": "pixel-calibrated; physical scale not established",
        }

        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

        if not cv2.imwrite(str(bev_path), self.bev):
            raise RuntimeError(f"Could not save BEV image to {bev_path}")

        print(f"Saved calibration: {config_path}")
        print(f"Saved BEV image:   {bev_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive IPM calibration")
    parser.add_argument("--source", required=True,
                        help="Image path, video path, or camera index (e.g. 0)")
    parser.add_argument("--width", type=int, default=800,
                        help="BEV output width in pixels (default: 800)")
    parser.add_argument("--height", type=int, default=600,
                        help="BEV output height in pixels (default: 600)")
    parser.add_argument("--output", default="calibration/ipm_calibration",
                        help="Output prefix for JSON and BEV image")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.width < 2 or args.height < 2:
        raise ValueError("Output width and height must both be at least 2.")

    frame, source_type = load_frame(args.source)
    print(f"Loaded {source_type}: {frame.shape[1]}x{frame.shape[0]}")
    print("Select the road quadrilateral in this order:")
    print("1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left")
    print("Choose points on the ground plane, along the boundaries of the lane.")

    ui = CalibrationUI(frame, args.width, args.height, args.output)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, ui.on_mouse)
    ui.redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("u"):
            if ui.points:
                ui.points.pop()
                ui.bev = None
                ui.homography = None
                ui.redraw()
        elif key == ord("r"):
            ui.points.clear()
            ui.bev = None
            ui.homography = None
            ui.redraw()
        elif key in (10, 13):
            ui.generate()
        elif key == ord("s"):
            ui.save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
