# sc-pose-utils Conventions

This document captures the attitude conventions used throughout the repo.

## 1. Quaternion Convention

- Default quaternion format in this repo is right-handed scalar-first (RSF): `[w, x, y, z]`.
- Quaternion math utilities in `sc_pose.math.quaternion` assume RSF inputs.
- Use conversion helpers at package level when integrating with scalar-last systems (for example ROS-style `[x, y, z, w]`):
  - `xyzw_to_wxyz`
  - `wxyz_to_xyzw`
- Normalize quaternions before sensitive operations when data source is uncertain (`q_norm`).

## 2. Rotation Convention

- `q2rotm(q)` returns an active rotation matrix.
- `q2trfm(q)` returns a passive transform matrix (transpose of active form under this implementation).
- Active/passive language should be explicit in docstrings/comments whenever frame semantics matter.

## 3. Frame and Pose Naming

- Use explicit frame-direction names in variables and function arguments.
- Current standard pattern:
  - `q_TARGET_2_CAM`
  - `r_Co2To_CAM`
  - `points_xyz_TARGET`
- Projection equation used throughout examples/comments:
  - `p^C = R_{TARGET->CAM} p^TARGET + r_{Co->To}^C`
- Translation vectors should state:
  - from which origin,
  - to which origin,
  - and the frame they are expressed in.

## 4. Camera Model and Distortion

- Pinhole model uses Brown-Conrady distortion terms.
- Distortion coefficient order follows OpenCV convention:
  - `[k1, k2, p1, p2, k3]`
- Skew is typically zero for pinhole unless explicitly set.
- Principal point default is image center:
  - `cx = (img_w - 1) / 2`
  - `cy = (img_h - 1) / 2`

## 5. Units and Numeric Assumptions

- Sensor dimensions and focal length are in millimeters.
- Image dimensions and projected coordinates are in pixels.
- Angles are in radians unless explicitly stated otherwise.
- In projection, points with invalid depth (`Z <= eps`) are treated as non-projectable (NaN output with warning in classless path).

## 6. Shape and Validation Expectations

- Quaternion shape: `(4,)`
- Translation shape: `(3,)`
- 3D points: `(..., 3)` (or `(N, 3)` in classless projection API; `(3, N)` is accepted then transposed)
- Intrinsics matrix `K`: `(3, 3)`
- Public APIs should validate shape and finite values and raise clear `ValueError` messages on mismatch.

## 7. Interop Guidance (ROS / External Libraries)

- External systems often use quaternion `[x, y, z, w]`.
- Convert at boundaries:
  - ingress: `xyzw_to_wxyz`
  - egress: `wxyz_to_xyzw`
- Keep core math and internal filters in RSF to avoid silent convention drift.
