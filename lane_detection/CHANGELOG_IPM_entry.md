## [Unreleased]

### Added
- Added `ipm_transform.py` to load a saved calibration and transform image, video, or camera frames.
- Added input-dimension and homography validation.
- Added side-by-side original/BEV visualization and optional output saving.
- Added `test_ipm_transform.py` with basic transformation and validation tests.
- Preserved calibration attempt 4 as an experimental baseline configuration.

### Documentation
- Recorded calibration attempt 4 coordinates and its temporary camera-pose limitation.
- Documented reusable transform usage, test commands, validation status, and handoff steps.

### Notes
- Physical scale is not established; calibration remains pixel-space only.
- The attempt 4 calibration is not final until the webcam is fixed in its competition mounting position.
