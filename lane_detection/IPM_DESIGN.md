# IPM Design and Calibration

## Status

- Project milestone: `v0.2.0 — Inverse Perspective Mapping`
- Current sub-phase: interactive pixel-space calibration prototype
- Implementation status: first standalone calibration tool created; not yet validated on the target vehicle camera
- Current source image: 1920 × 1080 image captured with the webcam connected directly to a PC
- Camera mounting: temporary / not yet fixed in its final vehicle position
- Track: white lane markings on a dark floor; target lane is the left-side lane in the supplied test image
- Physical lane width: not yet measured; competition lane dimensions may differ

## Objective

Transform a selected quadrilateral on the road plane from the camera image into a rectangular bird's-eye-view (BEV) image using a planar homography.

IPM is intended to make road-plane geometry easier to reason about. It does not itself detect lanes, identify the correct route, or straighten curved roads. Curves remain curves in BEV.

## Scope of the first prototype

The standalone `ipm_calibration.py` tool:

1. Loads one image, or captures one frame from a camera/video source.
2. Lets the operator click four source-image points.
3. Uses the point order: top-left, top-right, bottom-right, bottom-left.
4. Computes a homography with `cv2.getPerspectiveTransform`.
5. Warps the image with `cv2.warpPerspective`.
6. Displays the BEV result.
7. Saves a JSON calibration and a BEV preview when requested.

The tool is intentionally independent of ROS 2, IMU, odometry, and the steering PID.

## Calibration assumptions and limitations

- The selected points must lie on the same approximately planar road/floor surface.
- The camera must remain rigidly fixed after calibration. A camera position or angle change invalidates the calibration.
- The current webcam image was captured with the camera connected directly to a PC and is not yet confirmed as the final mounted camera pose.
- The first calibration is pixel-space only. The destination rectangle width and height are output pixels, not metres.
- A metric bird's-eye view requires a known physical measurement on the ground plane, such as measured lane width or a calibration marker.
- A homography models a plane. Raised objects, people, cones, and other objects above the floor will not be geometrically rectified correctly.
- Lens distortion is not corrected in this first prototype. If distortion is significant, camera intrinsic calibration and undistortion should be added before IPM.
- The target is the left-side lane in the supplied test image. The ROI must exclude the adjacent lane unless the task explicitly changes.

## Initial implementation

Run the prototype from the repository root:

```powershell
python ipm_calibration.py --source "path\to\camera_view.jpg"
```

Optional output dimensions:

```powershell
python ipm_calibration.py --source "path\to\camera_view.jpg" --width 800 --height 600
```

For a single frame from webcam index 0:

```powershell
python ipm_calibration.py --source 0
```

### Interactive controls

- Left click four points in order: top-left, top-right, bottom-right, bottom-left.
- `u`: undo the last point.
- `r`: reset all points.
- `Enter`: generate or update the BEV.
- `s`: save the calibration JSON and BEV preview.
- `q` or `Esc`: quit.

Default outputs:

- `calibration/ipm_calibration.json`
- `calibration/ipm_calibration_bev.jpg`

## Validation checklist

A calibration is not accepted merely because the warp executes.

- [ ] The selected quadrilateral follows the intended lane boundaries.
- [ ] The ROI does not accidentally include the adjacent lane.
- [ ] The four points lie on the road/floor plane.
- [ ] The BEV contains the full useful near-to-far lane region.
- [ ] The lane boundaries appear geometrically plausible and do not cross due to incorrect point order.
- [ ] Curved lane geometry remains continuous in BEV.
- [ ] Repeating the calibration with the same camera pose produces a consistent result.
- [ ] Calibration is repeated after the webcam is rigidly mounted in its final position.
- [ ] Physical scale is measured before interpreting BEV distances as metres.

## Known open questions

1. What is the final webcam mounting position, height, pitch, and yaw on the chassis?
2. What image resolution and camera index will be used on the Raspberry Pi 5?
3. What are the physical lane width and useful look-ahead distance on the competition track?
4. Is lens distortion significant enough to require undistortion?
5. What BEV output dimensions provide sufficient detail without excessive CPU cost?

## Next tasks

1. Run the calibration tool on the supplied test image.
2. Select and inspect the four points for the left-side lane.
3. Adjust the quadrilateral until the BEV is useful and stable.
4. Save the calibration JSON and preview.
5. Repeat calibration after the camera is mounted in its final position.
6. Only after validation, extract the reusable transform into the lane-perception pipeline.
7. Add automated tests for point validation, homography generation, output dimensions, and saved configuration loading.

## Handoff instructions for a future contributor or AI

Before continuing:

1. Read the root `README.md` for the overall project roadmap and system architecture.
2. Read `CHANGELOG.md` for implemented changes.
3. Read this file for IPM assumptions, calibration state, limitations, and next tasks.
4. Inspect the current source code and saved calibration JSON; do not assume planned work is implemented.
5. Preserve the separation between perception/waypoint generation and the vehicle PID/actuation layer.
6. Do not require ROS 2, IMU, odometry, or lab data for this local IPM prototype.
7. Do not treat the temporary webcam image calibration as final vehicle calibration.


## Calibration attempt 4 — saved baseline

The user supplied `ipm_calibration.json` for calibration attempt 4. The JSON was inspected and its required fields and matrix dimensions were validated.

- Source frame: 1920 × 1080 pixels
- BEV output: 800 × 600 pixels
- Point order: top-left, top-right, bottom-right, bottom-left
- Top-left: (912, 526)
- Top-right: (1160, 525)
- Bottom-right: (1498, 940)
- Bottom-left: (548, 924)
- Physical scale: unknown (`null`)

This is an experimental pixel-space calibration. It is not the final vehicle calibration because the webcam mounting pose is temporary. Preserve the JSON as baseline evidence; do not treat its homography as portable to another camera pose or image resolution.

## Reusable transform implementation

`ipm_transform.py` loads the saved calibration and applies its homography without requiring point selection on every frame.

Features:
- Supports a still image, video file, or camera index.
- Validates input frame dimensions against the calibration.
- Produces the configured BEV output dimensions.
- Displays the original frame (resized for side-by-side display only) next to the BEV.
- Can save a BEV image or side-by-side output video.
- Reports processing FPS for video/camera sources.

Run from the repository root:

```powershell
python ipm_transform.py --source "pictures/camera_view.jpg" --calibration "calibration/ipm_calibration.json"
```

For webcam index 0:

```powershell
python ipm_transform.py --source 0 --calibration "calibration/ipm_calibration.json"
```

Save a transformed image:

```powershell
python ipm_transform.py --source "pictures/camera_view.jpg" --calibration "calibration/ipm_calibration.json" --save "outputs/camera_bev.jpg"
```

Save a transformed video:

```powershell
python ipm_transform.py --source "videos/track.mp4" --calibration "calibration/ipm_calibration.json" --save "outputs/track_bev.mp4"
```

The side-by-side original panel is resized for display. The actual homography is always applied to the original-resolution frame.

## Automated tests

`test_ipm_transform.py` contains tests for:
- BEV output dimensions.
- Rejection of input frames whose dimensions do not match the calibration.
- Rejection of malformed homography matrices.

Run:

```powershell
python -m unittest test_ipm_transform.py -v
```

## Current status and next tasks

- [x] Interactive four-point calibration prototype.
- [x] Generate and inspect multiple experimental BEV calibrations.
- [x] Save attempt 4 calibration JSON.
- [x] Implement reusable calibration loading and frame transformation.
- [x] Add basic automated tests.
- [ ] Run automated tests in the user's Windows project environment.
- [ ] Run the saved calibration against the exact 1920 × 1080 source image.
- [ ] Verify camera stream resolution and camera pose after final mounting.
- [ ] Recalibrate after final mounting.
- [ ] Validate on multiple track scenes and measure CPU latency.
- [ ] Integrate BEV output into lane-marking extraction only after validation.

## Handoff notes

A future contributor or AI should:
1. Read the root README and changelog.
2. Read this IPM design document.
3. Inspect `ipm_calibration.json` and note its temporary camera-pose limitation.
4. Run the unit tests.
5. Run `ipm_transform.py` against an image with matching dimensions.
6. Do not change lane detection or controller/PID interfaces as part of this IPM-only milestone.
