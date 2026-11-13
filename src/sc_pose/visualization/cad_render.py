# src/sc_pose/visualization/cad_render.py
"""render binary CAD silhouettes and measure true-contour distances"""
# SPDX-License-Identifier: GPL-3.0-only
# adapted from Will-Lipscomb24/CamCal notation a82d9573163da0a71c6c1d1cda02477c8292f46f

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .cad_geometry import (
    RenderMaskUnavailableError, _DEFAULT_RENDER_CONFIG, _clip_polygon_depth, _image_shape,
    _load_mesh_arrays, _merged, _number, _positive_number, project_points, _section,
    _validate_camera, _validate_rendered_mask, _validated_distortion_coefficients,
)
from .cad_rasterization import (
    _clip_projected_polygon_to_viewport, _rasterize_opencv_fast, _rasterize_zbuffer_triangle,
)

def render_cad_silhouette(
    mesh: Any,
    K: Any,
    T_CO: Any,
    image_size: Sequence[int] | Mapping[str, Any],
    renderer_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """
    render a metric CAD silhouette at an object-to-camera pose

    Inputs:
    mesh (Any): vertices/faces or mesh path; K, T_CO (arrays): intrinsics and pose
    image_size (Sequence | Mapping): height and width; renderer_config (Mapping | None): rendering options

    Outputs:
    mask (array): boolean silhouette in image coordinates
    """
    raw_config      = _section(renderer_config, "renderer_config")
    config          = _merged(_DEFAULT_RENDER_CONFIG, raw_config)
    height, width   = _image_shape(image_size)
    K_array, pose   = _validate_camera(K, T_CO)

    custom_renderer = config.get("renderer")
    backend         = str(config.get("backend", "opencv")).strip().lower()
    if custom_renderer is not None or backend in {"callable", "custom"}:
        if custom_renderer is None:
            raise RenderMaskUnavailableError(
                "renderer_config.backend is callable/custom but renderer is missing"
            )
        try:
            if hasattr(custom_renderer, "render_silhouette"):
                result  = custom_renderer.render_silhouette(
                    mesh = mesh,
                    K = K_array,
                    T_CO = pose,
                    image_size = (height, width),
                    renderer_config = config,
                )
            else:
                result  = custom_renderer(
                    mesh = mesh,
                    K = K_array,
                    T_CO = pose,
                    image_size = (height, width),
                    renderer_config = config,
                )
        except Exception as exc:
            raise RenderMaskUnavailableError(f"custom silhouette renderer failed: {exc}") from exc
        return _validate_rendered_mask(result, (height, width))

    if backend not in {"opencv", "cpu", "cpu_opencv", "cpu_zbuffer", "zbuffer"}:
        raise RenderMaskUnavailableError(f"unsupported silhouette renderer backend: {backend!r}")

    vertices_o, faces   = _load_mesh_arrays(mesh, raw_config)
    vertices_c          = (pose[:3, :3] @ vertices_o.T).T + pose[:3, 3]
    near_clip           = _positive_number(config, "near_clip_m", 1.0e-6)
    far_value           = config.get("far_clip_m")
    far_clip            = None if far_value is None else float(far_value)
    if far_clip is not None and (not np.isfinite(far_clip) or far_clip <= near_clip):
        raise ValueError("far_clip_m must be finite and greater than near_clip_m")
    supersample = int(config.get("supersample", 1) or 1)
    if supersample < 1 or supersample > 8:
        raise ValueError("supersample must be an integer in [1, 8]")

    render_height, render_width = height * supersample, width * supersample
    K_render                    = K_array.copy()
    K_render[0, :]              *= supersample
    K_render[1, :]              *= supersample
    mask_u8                     = np.zeros((render_height, render_width), dtype = np.uint8)
    use_zbuffer                 = backend in {"cpu_zbuffer", "zbuffer"}
    z_buffer                    = (
        np.full((render_height, render_width), np.inf, dtype = np.float64)
        if use_zbuffer
        else None
    )
    cull_backfaces              = bool(config.get("backface_culling", False))
    distortion_coefficients     = _validated_distortion_coefficients(
        config.get("distortion_coefficients")
    )
    batch_faces                 = int(config.get("opencv_batch_faces", 1) or 1)
    if batch_faces < 1 or batch_faces > 1024:
        raise ValueError("opencv_batch_faces must be an integer in [1, 1024]")

    if use_zbuffer:
        slow_faces  = faces
    else:
        slow_face_indices   = _rasterize_opencv_fast(
            vertices_c,
            faces,
            K_render,
            mask_u8,
            near_clip = near_clip,
            far_clip = far_clip,
            cull_backfaces = cull_backfaces,
            distortion_coefficients = distortion_coefficients,
            batch_faces = batch_faces,
        )
        slow_faces          = faces[slow_face_indices]

    for face in slow_faces:
        triangle    = vertices_c[face]
        if not np.isfinite(triangle).all():
            continue
        if cull_backfaces:
            normal  = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            # outward front faces point toward the camera at the origin
            if float(np.dot(normal, np.mean(triangle, axis = 0))) >= 0.0:
                continue
        polygon = _clip_polygon_depth(triangle, plane_z = near_clip, keep_greater = True)
        if far_clip is not None:
            polygon = _clip_polygon_depth(polygon, plane_z = far_clip, keep_greater = False)
        if len(polygon) < 3:
            continue
        # clipping a triangle against near/far planes produces a convex polygon;
        # fan triangulation is therefore safe
        for index in range(1, len(polygon) - 1):
            clipped_triangle    = np.stack((polygon[0], polygon[index], polygon[index + 1]))
            projected           = project_points(
                K_render,
                clipped_triangle,
                distortion_coefficients,
            )
            if not np.isfinite(projected).all():
                continue
            if (
                np.max(projected[:, 0]) < 0.0
                or np.min(projected[:, 0]) >= render_width
                or np.max(projected[:, 1]) < 0.0
                or np.min(projected[:, 1]) >= render_height
            ):
                continue
            if use_zbuffer:
                assert z_buffer is not None
                _rasterize_zbuffer_triangle(
                    projected,
                    clipped_triangle[:, 2],
                    z_buffer,
                    mask_u8,
                )
            else:
                # subpixel coordinates prevent premature rounding on small
                # objects while staying within OpenCV's integer API
                projected_visible   = _clip_projected_polygon_to_viewport(
                    projected,
                    width = render_width,
                    height = render_height,
                )
                if len(projected_visible) < 3:
                    continue
                shift   = 4
                points  = np.rint(projected_visible * (1 << shift)).astype(np.int32)
                cv2.fillConvexPoly(mask_u8, points, color = 1, lineType = cv2.LINE_8, shift = shift)

    if supersample > 1:
        coverage    = cv2.resize(
            mask_u8.astype(np.float32),
            (width, height),
            interpolation = cv2.INTER_AREA,
        )
        threshold   = _number(config, "coverage_threshold", 0.25)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0, 1]")
        return np.asarray(coverage >= threshold, dtype = bool)
    return np.asarray(mask_u8 > 0, dtype = bool)


def inner_contour(mask: np.ndarray) -> np.ndarray:
    """
    extract a binary inner boundary using the original erosion rule

    Inputs:
    mask (array): binary foreground

    Outputs:
    contour (array): boolean inner boundary
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype = bool)
    kernel  = np.ones((3, 3), dtype = np.uint8)
    eroded  = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations = 1,
        borderType = cv2.BORDER_CONSTANT,
        borderValue = 0,
    )
    return mask & ~(eroded > 0)


def mean_contour_distance(source_contour: np.ndarray, target_contour: np.ndarray) -> float:
    """
    measure the mean source-to-target contour distance

    Inputs:
    source_contour, target_contour (arrays): binary boundary rasters

    Outputs:
    distance (float): mean pixels, or NaN for empty boundaries
    """
    if not np.any(source_contour) or not np.any(target_contour):
        return float("nan")
    distance    = cv2.distanceTransform(
        (~target_contour).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return float(np.mean(distance[source_contour]))
