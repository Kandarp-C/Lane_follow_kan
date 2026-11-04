#!/usr/bin/env python3
"""
Classical lane detection for white lane markings on a dark floor.

Supported sources:
  Webcam:          --source 0
  Video file:      --source "videos/track.mp4"
  Still image:     --source "pictures/track.jpg"
  Network stream:  --source "rtsp://192.168.1.100:554/stream"

Pipeline:
  source -> grayscale -> Gaussian blur -> Canny -> road ROI
         -> Hough segments -> left/right line fit -> lane midpoint/error

This program visualizes lane geometry only. It DOES NOT control motors.
"""

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def parse_args():
    """Parse command-line options. --source is deliberately a string."""
    parser = argparse.ArgumentParser(
        description="OpenCV lane detector for white lines on a dark floor."
    )
    parser.add_argument(
        "--source", required=True,
        help='Camera index (0), image/video path, or stream URL.'
    )
    parser.add_argument("--width", type=int, default=640,
                        help="Requested capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Requested capture height (default: 480)")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=25)
    parser.add_argument("--min-line-length", type=int, default=25)
    parser.add_argument("--max-line-gap", type=int, default=60)
    parser.add_argument("--roi-top", type=float, default=0.55,
                        help="Top of trapezoid ROI as fraction of image height.")
    parser.add_argument("--lookahead", type=float, default=0.75,
                        help="Lane measurement row as fraction of image height.")
    parser.add_argument("--realtime", action="store_true",
                        help="Pace video-file playback using its FPS.")
    parser.add_argument("--loop", action="store_true",
                        help="Restart a video file when it reaches its end.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Do not show the Canny ROI window.")
    return parser.parse_args()


def classify_source(source):
    """
    Decide how to open --source.

    Numeric strings mean camera indices. Known image extensions mean still
    images. Everything else is passed to VideoCapture (video or stream URL).
    """
    if source.isdigit():
        return "camera", int(source)

    suffix = Path(source).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image", source

    parsed = urlparse(source)
    if parsed.scheme.lower() in {"rtsp", "rtmp", "http", "https", "udp", "tcp"}:
        return "stream", source

    return "video", source


def make_roi_mask(image_shape, roi_top):
    """Create a trapezoid mask covering the lower road area."""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array([[
        (int(0.05 * width), height - 1),
        (int(0.38 * width), int(roi_top * height)),
        (int(0.62 * width), int(roi_top * height)),
        (int(0.95 * width), height - 1),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return mask, polygon


def preprocess(frame, args):
    """Grayscale -> blur -> Canny -> mask to the road ROI."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)

    mask, polygon = make_roi_mask(edges.shape, args.roi_top)
    return cv2.bitwise_and(edges, mask), polygon


def extract_lane_segments(roi_edges, args):
    """Detect Hough segments and group them by image half."""
    height, width = roi_edges.shape
    center_x = width / 2.0

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=args.hough_threshold,
        minLineLength=args.min_line_length,
        maxLineGap=args.max_line_gap,
    )

    left, right = [], []
    if lines is None:
        return left, right

    for line in lines:
    
        # Convert each detected line into a flat NumPy array.
        # This handles both (N, 1, 4) and (N, 4) formats.
        coordinates = np.asarray(line).reshape(-1)
    
        if coordinates.size < 4:
            continue
    
        x1, y1, x2, y2 = map(int, coordinates[:4])
    
        dx = x2 - x1
        dy = y2 - y1
    
        # Avoid division by zero for vertical lines.
        if dx == 0:
            continue
    
        length = float(np.hypot(dx, dy))
        slope = dy / float(dx)
    
        # Reject short and nearly horizontal segments.
        if length < args.min_line_length or abs(slope) < 0.25:
            continue
    
        midpoint_x = (x1 + x2) / 2.0
    
        segment = (x1, y1, x2, y2, length)
    
        if midpoint_x < center_x:
            left.append(segment)
        else:
            right.append(segment)

    return left, right


def fit_boundary(segments):
    """
    Fit x = a*y + b to all endpoints on one side.
    Longer Hough segments receive more weight.
    """
    if not segments:
        return None

    xs, ys, weights = [], [], []
    for x1, y1, x2, y2, length in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    try:
        a, b = np.polyfit(
            np.asarray(ys, dtype=np.float64),
            np.asarray(xs, dtype=np.float64),
            1,
            w=np.sqrt(np.asarray(weights, dtype=np.float64)),
        )
        return float(a), float(b)
    except (ValueError, np.linalg.LinAlgError):
        return None


def x_at_y(line, y):
    """Calculate horizontal x coordinate on fitted line at row y."""
    if line is None:
        return None
    a, b = line
    return int(round(a * y + b))


def draw_boundary(image, line, color, y_top, y_bottom):
    """Draw a fitted boundary between the ROI top and bottom of frame."""
    if line is None:
        return
    cv2.line(
        image,
        (x_at_y(line, y_top), y_top),
        (x_at_y(line, y_bottom), y_bottom),
        color,
        3,
    )


def process_frame(frame, args, smoothed_error):
    """
    Process one frame and return:
      annotated frame, Canny ROI, updated smoothed error, valid error or None.
    This is shared by webcams, videos, images, and network streams.
    """
    height, width = frame.shape[:2]
    roi_edges, roi_polygon = preprocess(frame, args)

    left_segments, right_segments = extract_lane_segments(roi_edges, args)
    left_line = fit_boundary(left_segments)
    right_line = fit_boundary(right_segments)

    output = frame.copy()
    cv2.polylines(output, roi_polygon, True, (255, 255, 0), 2)

    lookahead_y = int(args.lookahead * height)
    roi_top_y = int(args.roi_top * height)

    draw_boundary(output, left_line, (255, 80, 0), roi_top_y, height - 1)
    draw_boundary(output, right_line, (0, 80, 255), roi_top_y, height - 1)

    valid_error = None

    # A centerline is valid only when both boundaries are currently visible.
    if left_line is not None and right_line is not None:
        x_left = x_at_y(left_line, lookahead_y)
        x_right = x_at_y(right_line, lookahead_y)

        if 0 <= x_left < width and 0 <= x_right < width and x_left < x_right:
            midpoint = (x_left + x_right) // 2
            image_center = width // 2
            raw_error = midpoint - image_center

            alpha = 0.35
            if smoothed_error is None:
                smoothed_error = float(raw_error)
            else:
                smoothed_error = (
                    alpha * raw_error + (1.0 - alpha) * smoothed_error
                )

            valid_error = smoothed_error
            normalized_error = smoothed_error / max(image_center, 1)

            cv2.circle(output, (x_left, lookahead_y), 7, (255, 80, 0), -1)
            cv2.circle(output, (x_right, lookahead_y), 7, (0, 80, 255), -1)
            cv2.circle(output, (midpoint, lookahead_y), 8, (0, 255, 0), -1)
            cv2.line(output, (image_center, height - 1),
                     (midpoint, lookahead_y), (0, 255, 255), 2)

            cv2.putText(
                output,
                f"Error: {smoothed_error:+.1f}px  Normalized: {normalized_error:+.3f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            cv2.putText(output, "BOTH LANES DETECTED", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            smoothed_error = None
            cv2.putText(output, "INVALID LANE GEOMETRY", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        smoothed_error = None
        missing = []
        if left_line is None:
            missing.append("LEFT")
        if right_line is None:
            missing.append("RIGHT")
        cv2.putText(output, "MISSING: " + " + ".join(missing) + " LANE",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output, "NO VALID CENTERLINE", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return output, roi_edges, smoothed_error, valid_error


def show_frame(output, roi_edges, args):
    """Display the annotated result and optionally the edge image."""
    cv2.imshow("Lane Detection", output)
    if not args.no_debug:
        cv2.imshow("Canny ROI", roi_edges)


def open_capture(source, args, source_kind):
    """Open a webcam, video file, or network stream using VideoCapture."""
    # CAP_DSHOW can improve webcam startup reliability on Windows.
    if source_kind == "camera" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open source: {source!r}. Check the camera index, "
            "file path, stream URL, network connection, and codec support."
        )

    if source_kind == "camera":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    return capture


def run_image(image_path, args):
    """Load and process exactly one still image."""
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(
            f"OpenCV could not decode image: {image_path}. "
            "Check its format and file integrity."
        )

    output, roi_edges, _, _ = process_frame(frame, args, None)
    show_frame(output, roi_edges, args)
    print("Image processed. Press any key in the image window to close.")
    cv2.waitKey(0)


def run_capture(source, source_kind, args):
    """Read and process frames from camera, video file, or network stream."""
    if source_kind == "video" and not Path(source).is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")

    capture = open_capture(source, args, source_kind)
    is_file_video = source_kind == "video"

    # Video files have a stored FPS. Use it only when --realtime is requested.
    video_fps = capture.get(cv2.CAP_PROP_FPS) if is_file_video else 0.0
    frame_period = 1.0 / video_fps if video_fps and video_fps > 0 else 0.0

    smoothed_error = None
    print(f"Opened {source_kind}: {source}")
    if is_file_video:
        print(f"Video FPS reported by file: {video_fps:.2f}")
        if args.realtime and frame_period == 0.0:
            print("Warning: video FPS unavailable; playback will run as fast as frames arrive.")

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = capture.read()

            if not ok:
                # End of a local video: optionally seek back to the first frame.
                if is_file_video and args.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    smoothed_error = None
                    continue

                if is_file_video:
                    print("Video finished.")
                else:
                    print("Frame read failed or stream ended.")
                break

            output, roi_edges, smoothed_error, _ = process_frame(
                frame, args, smoothed_error
            )
            show_frame(output, roi_edges, args)

            # For prerecorded files, wait so display playback follows source FPS.
            if is_file_video and args.realtime and frame_period > 0:
                remaining = frame_period - (time.perf_counter() - loop_start)
                wait_ms = max(1, int(remaining * 1000))
            else:
                wait_ms = 1

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
    finally:
        capture.release()


def main():
    args = parse_args()

    if not 0 < args.roi_top < 1:
        raise ValueError("--roi-top must be between 0 and 1.")
    if not 0 < args.lookahead < 1:
        raise ValueError("--lookahead must be between 0 and 1.")
    if args.canny_low >= args.canny_high:
        raise ValueError("--canny-low must be smaller than --canny-high.")
    if args.realtime and args.source.isdigit():
        print("Note: --realtime is for prerecorded video files; ignored for cameras.")

    source_kind, source = classify_source(args.source)
    print(f"Source type: {source_kind}")

    try:
        if source_kind == "image":
            run_image(source, args)
        else:
            run_capture(source, source_kind, args)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
