# src/sc_pose/visualization/cad_geometry.py
"""load metric meshes and project calibrated camera points"""
# SPDX-License-Identifier: GPL-3.0-only
# adapted from Will-Lipscomb24/CamCal notation a82d9573163da0a71c6c1d1cda02477c8292f46f

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

_DEFAULT_RENDER_CONFIG: dict[str, Any]  = {
    "backend": "opencv",
    "mesh_units": "m",
    "mesh_scale": 1.0,
    "near_clip_m": 1.0e-6,
    "far_clip_m": None,
    "backface_culling": False,
    "supersample": 1,
    "coverage_threshold": 0.25,
    # values >1 batch triangle fills through OpenCV's C++ loop.  The default
    # remains the exact historical per-face union; offline H150 feature
    # extraction may opt into a small, audited batch (for example 32)
    "opencv_batch_faces": 1,
    # OpenCV order (k1, k2, p1, p2[, k3, ...]).  ``None`` preserves the
    # historical ideal-pinhole projection exactly.  Raw-camera masks can pass
    # the coefficients recorded in their trajectory-pack metadata
    "distortion_coefficients": None,
}


class RenderMaskUnavailableError(RuntimeError):
    """Raised when a requested silhouette cannot be rendered safely."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    """
    copy an optional configuration mapping

    Inputs:
    value (Any): mapping or None; name (str): option label

    Outputs:
    config (dict): independent mapping
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None, got {type(value).__name__}")
    return dict(value)


def _section(config: Mapping[str, Any] | None, section: str) -> dict[str, Any]:
    """
    resolve direct or nested rendering options

    Inputs:
    config (Mapping | None): configuration; section (str): requested section

    Outputs:
    options (dict): selected rendering configuration
    """
    current = _mapping(config, name = "config")
    if isinstance(current.get("candidate_selector"), Mapping):
        current = dict(current["candidate_selector"])
    if isinstance(current.get("render_mask_agreement"), Mapping):
        current = dict(current["render_mask_agreement"])
    if section and isinstance(current.get(section), Mapping):
        current = dict(current[section])
    return current


def _merged(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """
    merge supplied renderer options over the defaults

    Inputs:
    defaults, overrides (Mapping): default and supplied options

    Outputs:
    config (dict): merged options
    """
    result  = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            nested  = dict(result[key])
            nested.update(value)
            result[key] = nested
        else:
            result[key] = value
    return result


def _number(config: Mapping[str, Any], key: str, default: float) -> float:
    """
    read a finite scalar rendering option

    Inputs:
    config (Mapping): options; key (str): option name; default (float): fallback

    Outputs:
    value (float): finite scalar
    """
    value   = config.get(key, default)
    if value is None:
        value   = default
    value   = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return value


def _positive_number(config: Mapping[str, Any], key: str, default: float) -> float:
    """
    read a positive scalar rendering option

    Inputs:
    config (Mapping): options; key (str): option name; default (float): fallback

    Outputs:
    value (float): positive scalar
    """
    value   = _number(config, key, default)
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value!r}")
    return value


def _image_shape(image_size: Sequence[int] | Mapping[str, Any]) -> tuple[int, int]:
    """
    resolve the positive image height and width

    Inputs:
    image_size (Sequence | Mapping): image dimensions

    Outputs:
    shape (tuple): integer height and width
    """
    if isinstance(image_size, Mapping):
        height  = image_size.get("height", image_size.get("height_px"))
        width   = image_size.get("width", image_size.get("width_px"))
        if height is None or width is None:
            raise ValueError("image_size mapping must contain height and width")
    else:
        values  = tuple(image_size)
        if len(values) < 2:
            raise ValueError("image_size must contain (height, width)")
        height, width   = values[:2]
    height, width   = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {(height, width)}")
    return height, width


def _unit_scale_to_m(units: str) -> float:
    """
    convert source mesh units to meters

    Inputs:
    units (str): configured source units

    Outputs:
    scale (float): meters per source unit
    """
    normalized  = str(units).strip().lower()
    scales      = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "cm": 1.0e-2,
        "centimeter": 1.0e-2,
        "centimeters": 1.0e-2,
        "mm": 1.0e-3,
        "millimeter": 1.0e-3,
        "millimeters": 1.0e-3,
    }
    if normalized not in scales:
        raise ValueError(f"unsupported mesh units: {units!r}")
    return scales[normalized]


def _scene_to_mesh(scene: Any, trimesh_module: Any) -> Any:
    """
    combine a loaded trimesh scene into a triangle mesh

    Inputs:
    scene, trimesh_module (Any): loaded scene and trimesh module

    Outputs:
    mesh (Any): combined triangle mesh
    """
    try:
        mesh    = scene.to_mesh()
    except (AttributeError, TypeError):
        mesh    = scene.dump(concatenate = True)
    if not isinstance(mesh, trimesh_module.Trimesh):
        raise RenderMaskUnavailableError("trimesh scene could not be combined into one mesh")
    return mesh


@lru_cache(maxsize = 8)
def _load_mesh_path_cached(
    path_text: str,
    modification_time_ns: int,
    file_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    cache mesh arrays until local file identity changes

    Inputs:
    path_text (str): mesh path; modification_time_ns, file_size (int): cache identity

    Outputs:
    mesh (tuple): immutable vertices and triangle indices
    """
    # the stat fields are deliberately part of the key, so replacing a mesh at
    # the same path cannot silently reuse stale geometry
    del modification_time_ns, file_size
    path    = Path(path_text)
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - navy has trimesh
        raise RenderMaskUnavailableError(
            "loading a mesh path requires trimesh; pass vertices/faces directly or install trimesh"
        ) from exc
    try:
        loaded  = trimesh.load_mesh(path, process = False)
    except Exception as exc:
        raise RenderMaskUnavailableError(f"failed to load CAD mesh {path}: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        loaded  = _scene_to_mesh(loaded, trimesh)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RenderMaskUnavailableError(
            f"unsupported loaded mesh type for {path}: {type(loaded).__name__}"
        )
    vertices    = np.ascontiguousarray(np.asarray(loaded.vertices, dtype = np.float64))
    faces       = np.ascontiguousarray(np.asarray(loaded.faces, dtype = np.int64))
    vertices.setflags(write = False)
    faces.setflags(write = False)
    return vertices, faces


def _load_mesh_arrays(mesh: Any, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    load metric arrays without requiring trimesh for array-backed input

    Inputs:
    mesh (Any): arrays, mesh object or path; config (Mapping): units and scale

    Outputs:
    mesh (tuple): contiguous vertices and triangular faces
    """
    mesh_units      = config.get("mesh_units") or _DEFAULT_RENDER_CONFIG["mesh_units"]
    mesh_scale      = _number(config, "mesh_scale", 1.0)
    ypr_offset_deg  = config.get("ypr_offset_deg")

    source  = mesh
    if hasattr(source, "mesh_path") and not hasattr(source, "vertices"):
        mesh_units  = config.get("mesh_units") or getattr(source, "mesh_units", mesh_units)
        if ypr_offset_deg is None:
            ypr_offset_deg  = getattr(source, "ypr_offset_deg", None)
        # MegaPose RigidObject.scale already includes its unit conversion.  Use
        # it only when the caller did not explicitly provide mesh_scale
        if "mesh_scale" not in config and hasattr(source, "scale"):
            mesh_scale  = float(source.scale)
            mesh_units  = "m"
        source  = source.mesh_path

    if isinstance(source, Mapping):
        if "vertices" not in source or "faces" not in source:
            raise ValueError("mesh mapping must contain vertices and faces")
        vertices    = np.asarray(source["vertices"], dtype = np.float64)
        faces       = np.asarray(source["faces"], dtype = np.int64)
    elif isinstance(source, (tuple, list)) and len(source) == 2:
        vertices    = np.asarray(source[0], dtype = np.float64)
        faces       = np.asarray(source[1], dtype = np.int64)
    elif hasattr(source, "vertices") and hasattr(source, "faces"):
        vertices    = np.asarray(source.vertices, dtype = np.float64)
        faces       = np.asarray(source.faces, dtype = np.int64)
    elif isinstance(source, (str, Path)):
        path    = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"CAD mesh not found: {path}")
        stat            = path.stat()
        vertices, faces = _load_mesh_path_cached(
            str(path.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
    else:
        raise TypeError(
            "mesh must be a path, a (vertices, faces) pair, a mapping, or an object with vertices/faces"
        )

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"mesh vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"mesh faces must be triangular with shape (M, 3), got {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices contain non-finite values")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("mesh faces contain out-of-range vertex indices")
    scale   = _unit_scale_to_m(str(mesh_units)) * mesh_scale
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"combined mesh scale must be finite and positive, got {scale!r}")
    vertices    = vertices * scale
    if ypr_offset_deg is not None:
        ypr = np.asarray(ypr_offset_deg, dtype = np.float64)
        if ypr.shape != (3,) or not np.isfinite(ypr).all():
            raise ValueError("ypr_offset_deg must contain three finite values")
        heading, pitch, roll    = np.deg2rad(ypr)
        ch, sh                  = math.cos(heading), math.sin(heading)
        cp, sp                  = math.cos(pitch), math.sin(pitch)
        cr, sr                  = math.cos(roll), math.sin(roll)
        rotation_z              = np.asarray([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
        rotation_x              = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        rotation_y              = np.asarray([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
        vertices                = ((rotation_z @ rotation_x @ rotation_y) @ vertices.T).T
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _validate_camera(K: Any, T_CO: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    validate camera intrinsics and object-to-camera pose

    Inputs:
    K, T_CO (Any): calibrated camera matrix and rigid pose

    Outputs:
    camera (tuple): float64 intrinsics and pose
    """
    intrinsics  = np.asarray(K, dtype = np.float64)
    pose        = np.asarray(T_CO, dtype = np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"K must be a finite 3x3 matrix, got {intrinsics.shape}")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("K must have positive fx and fy")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"T_CO must be a finite 4x4 matrix, got {pose.shape}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol = 1.0e-9, rtol = 0.0):
        raise ValueError("T_CO must have homogeneous bottom row [0, 0, 0, 1]")
    rotation    = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol = 1.0e-6, rtol = 0.0):
        raise ValueError("T_CO rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol = 1.0e-6, rtol = 0.0):
        raise ValueError("T_CO rotation determinant must be +1")
    return intrinsics, pose


def _clip_polygon_depth(
    polygon: np.ndarray,
    *,
    plane_z: float,
    keep_greater: bool,
) -> np.ndarray:
    """
    clip polygon depth

    Inputs:
    polygon, plane_z, keep_greater: geometry or renderer options

    Outputs:
    result: converted geometry or rendering value
    """
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        """
        inside

        Inputs:
        point: geometry or renderer options

        Outputs:
        result: converted geometry or rendering value
        """
        return bool(point[2] >= plane_z) if keep_greater else bool(point[2] <= plane_z)

    output: list[np.ndarray]    = []
    previous                    = polygon[-1]
    previous_inside             = inside(previous)
    for current in polygon:
        current_inside  = inside(current)
        if current_inside != previous_inside:
            dz  = current[2] - previous[2]
            if abs(float(dz)) > np.finfo(np.float64).eps:
                alpha   = (plane_z - previous[2]) / dz
                output.append(previous + alpha * (current - previous))
        if current_inside:
            output.append(current)
        previous        = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 3), dtype = np.float64)
    return np.asarray(output, dtype = np.float64)


def _validated_distortion_coefficients(value: Any) -> np.ndarray | None:
    """
    resolve a supported OpenCV distortion vector

    Inputs:
    value (Any): optional distortion coefficients

    Outputs:
    coefficients (array | None): finite nonzero coefficients, or ideal pinhole
    """
    if value is None:
        return None
    coefficients    = np.asarray(value, dtype = np.float64).reshape(-1)
    if coefficients.size == 0:
        return None
    if coefficients.size not in {4, 5, 8, 12, 14}:
        raise ValueError(
            "distortion_coefficients must contain 4, 5, 8, 12, or 14 values"
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("distortion_coefficients contain non-finite values")
    if not np.any(coefficients != 0.0):
        return None
    return np.ascontiguousarray(coefficients)


def project_points(
    K: np.ndarray,
    points_c: np.ndarray,
    distortion_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """
    project calibrated camera-space points

    Inputs:
    K, points_c (arrays): intrinsics and points; distortion_coefficients (array | None): OpenCV model

    Outputs:
    pixels (array): projected image coordinates
    """
    if distortion_coefficients is None:
        homogeneous = (K @ points_c.T).T
        return homogeneous[:, :2] / homogeneous[:, 2:3]
    # the common OpenCV Brown-Conrady/rational models are inexpensive to apply
    # directly to points already expressed in the camera frame.  Calling
    # cv2.projectPoints would redundantly build Rodrigues/Jacobian machinery
    # for every one of the H candidate poses (and was ~50 ms for this mesh)
    # the uncommon 14-coefficient tilted-sensor model retains OpenCV's exact
    # implementation
    if distortion_coefficients.size == 14:
        projected, _jacobian    = cv2.projectPoints(
            np.ascontiguousarray(points_c, dtype = np.float64),
            np.zeros(3, dtype = np.float64),
            np.zeros(3, dtype = np.float64),
            K,
            distortion_coefficients,
        )
        return np.asarray(projected[:, 0, :], dtype = np.float64)

    z                                               = points_c[:, 2]
    x                                               = points_c[:, 0] / z
    y                                               = points_c[:, 1] / z
    r2                                              = x * x + y * y
    r4                                              = r2 * r2
    r6                                              = r4 * r2
    coefficients                                    = np.zeros(12, dtype = np.float64)
    coefficients[: distortion_coefficients.size]    = distortion_coefficients
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4  = coefficients
    radial_numerator                                = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_denominator                              = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial                                          = radial_numerator / radial_denominator
    xy                                              = x * y
    x_distorted                                     = (
        x * radial
        + 2.0 * p1 * xy
        + p2 * (r2 + 2.0 * x * x)
        + s1 * r2
        + s2 * r4
    )
    y_distorted                                     = (
        y * radial
        + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * xy
        + s3 * r2
        + s4 * r4
    )
    projected                                       = np.empty((len(points_c), 2), dtype = np.float64)
    projected[:, 0]                                 = K[0, 0] * x_distorted + K[0, 1] * y_distorted + K[0, 2]
    projected[:, 1]                                 = K[1, 0] * x_distorted + K[1, 1] * y_distorted + K[1, 2]
    return projected


def _validate_rendered_mask(mask: Any, image_size: tuple[int, int]) -> np.ndarray:
    """
    validate a custom renderer silhouette

    Inputs:
    mask (Any): rendered mask; image_size (tuple): expected height and width

    Outputs:
    mask (array): boolean silhouette
    """
    rendered    = np.asarray(mask)
    if rendered.shape != image_size:
        raise RenderMaskUnavailableError(
            f"renderer returned shape {rendered.shape}, expected {image_size}"
        )
    if rendered.dtype.kind == "f" and not np.isfinite(rendered).all():
        raise RenderMaskUnavailableError("renderer returned non-finite mask values")
    return np.asarray(rendered > 0, dtype = bool)


def default_render_config():
    """
    copy the deterministic silhouette-rendering defaults

    Inputs:
    none

    Outputs:
    config (dict): independent rendering options
    """
    return dict(_DEFAULT_RENDER_CONFIG)
