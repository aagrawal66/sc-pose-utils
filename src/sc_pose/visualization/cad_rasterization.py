# src/sc_pose/visualization/cad_rasterization.py
"""clip and rasterize camera-space triangles without changing silhouette rules"""
# SPDX-License-Identifier: GPL-3.0-only
# adapted from Will-Lipscomb24/CamCal notation a82d9573163da0a71c6c1d1cda02477c8292f46f

from __future__ import annotations

import math

import numpy as np
import cv2

from .cad_geometry import project_points

def _clip_projected_polygon_to_viewport(
    polygon: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """
    clip an image polygon before fixed-point rasterization

    Inputs:
    polygon (array): image vertices; width, height (int): viewport dimensions

    Outputs:
    polygon (array): clipped image vertices
    """
    output      = np.asarray(polygon, dtype = np.float64)
    boundaries  = (
        (0, 0.0, True),
        (0, float(width - 1), False),
        (1, 0.0, True),
        (1, float(height - 1), False),
    )
    for axis, boundary, keep_greater in boundaries:
        if len(output) == 0:
            break
        clipped: list[np.ndarray]   = []
        previous                    = output[-1]
        previous_inside             = (
            previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        )
        for current in output:
            current_inside  = (
                current[axis] >= boundary if keep_greater else current[axis] <= boundary
            )
            if bool(current_inside) != bool(previous_inside):
                delta   = current[axis] - previous[axis]
                if abs(float(delta)) > np.finfo(np.float64).eps:
                    alpha   = (boundary - previous[axis]) / delta
                    clipped.append(previous + alpha * (current - previous))
            if current_inside:
                clipped.append(current)
            previous        = current
            previous_inside = current_inside
        output  = (
            np.asarray(clipped, dtype = np.float64)
            if clipped
            else np.empty((0, 2), dtype = np.float64)
        )
    return output


def _rasterize_zbuffer_triangle(
    projected: np.ndarray,
    depth: np.ndarray,
    z_buffer: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
    rasterize one triangle with perspective-correct depth testing

    Inputs:
    projected, depth (arrays): triangle pixels and depths; z_buffer, mask (arrays): output buffers

    Outputs:
    none, updates depth and binary silhouette buffers
    """
    height, width   = mask.shape
    min_x           = max(0, int(math.floor(float(np.min(projected[:, 0])))))
    max_x           = min(width - 1, int(math.ceil(float(np.max(projected[:, 0])))))
    min_y           = max(0, int(math.floor(float(np.min(projected[:, 1])))))
    max_y           = min(height - 1, int(math.ceil(float(np.max(projected[:, 1])))))
    if min_x > max_x or min_y > max_y:
        return

    x0, y0      = projected[0]
    x1, y1      = projected[1]
    x2, y2      = projected[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) <= 1.0e-12:
        return
    ys, xs  = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    px      = xs.astype(np.float64) + 0.5
    py      = ys.astype(np.float64) + 0.5
    w0      = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
    w1      = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
    w2      = 1.0 - w0 - w1
    epsilon = 1.0e-9
    covered = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
    if not np.any(covered):
        return
    inverse_depth   = w0 / depth[0] + w1 / depth[1] + w2 / depth[2]
    valid           = covered & np.isfinite(inverse_depth) & (inverse_depth > 0.0)
    if not np.any(valid):
        return
    z           = np.full_like(inverse_depth, np.inf, dtype = np.float64)
    z[valid]    = 1.0 / inverse_depth[valid]
    view        = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
    update      = valid & (z < view)
    if np.any(update):
        view[update]        = z[update]
        mask_view           = mask[min_y : max_y + 1, min_x : max_x + 1]
        mask_view[update]   = 1


def _rasterize_opencv_fast(
    vertices_c: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    *,
    near_clip: float,
    far_clip: float | None,
    cull_backfaces: bool,
    distortion_coefficients: np.ndarray | None,
    batch_faces: int,
) -> np.ndarray:
    """
    rasterize ordinary faces with vectorized projection and exact triangle union

    Inputs:
    vertices_c, faces, K (arrays): camera geometry; mask (array): output silhouette
    near_clip, far_clip (floats): depth bounds; cull_backfaces (bool): visibility; distortion_coefficients (array | None): camera model; batch_faces (int): fill budget

    Outputs:
    indices (array): faces requiring the clipped rasterization path
    """
    triangles_c         = vertices_c[faces]
    finite              = np.isfinite(triangles_c).all(axis = (1, 2))
    minimum_z           = np.min(triangles_c[:, :, 2], axis = 1)
    maximum_z           = np.max(triangles_c[:, :, 2], axis = 1)
    intersects_depth    = finite & (maximum_z >= near_clip)
    if far_clip is not None:
        intersects_depth    &= minimum_z <= far_clip
    all_inside  = intersects_depth & (minimum_z >= near_clip)
    if far_clip is not None:
        all_inside  &= maximum_z <= far_clip

    front_facing    = np.ones(len(faces), dtype = bool)
    if cull_backfaces:
        normals             = np.cross(
            triangles_c[:, 1] - triangles_c[:, 0],
            triangles_c[:, 2] - triangles_c[:, 0],
        )
        front_facing        = np.einsum("ij,ij->i", normals, np.mean(triangles_c, axis = 1)) < 0.0
        intersects_depth    &= front_facing
        all_inside          &= front_facing

    simple_indices  = np.flatnonzero(all_inside)
    unsafe_indices  = np.empty((0,), dtype = np.int64)
    if simple_indices.size:
        simple_faces                            = faces[simple_indices]
        used_vertex_indices                     = np.unique(simple_faces.reshape(-1))
        projected_vertices                      = np.full((len(vertices_c), 2), np.nan, dtype = np.float64)
        projected_vertices[used_vertex_indices] = project_points(
            K,
            vertices_c[used_vertex_indices],
            distortion_coefficients,
        )
        projected                               = projected_vertices[simple_faces]
        finite_projection                       = np.isfinite(projected).all(axis = (1, 2))
        min_x                                   = np.min(projected[:, :, 0], axis = 1)
        max_x                                   = np.max(projected[:, :, 0], axis = 1)
        min_y                                   = np.min(projected[:, :, 1], axis = 1)
        max_y                                   = np.max(projected[:, :, 1], axis = 1)
        height, width                           = mask.shape
        intersects_view                         = (
            finite_projection
            & (max_x >= 0.0)
            & (min_x < width)
            & (max_y >= 0.0)
            & (min_y < height)
        )
        signed_twice_area                       = (
            (projected[:, 1, 0] - projected[:, 0, 0])
            * (projected[:, 2, 1] - projected[:, 0, 1])
            - (projected[:, 1, 1] - projected[:, 0, 1])
            * (projected[:, 2, 0] - projected[:, 0, 0])
        )
        nondegenerate                           = np.abs(signed_twice_area) > 1.0e-12
        # with four fixed-point fractional bits, this bound remains safely
        # inside signed int32.  Rarer extreme near-plane projections take the
        # robust viewport-clipped slow path
        coordinate_safe = np.max(np.abs(projected), axis = (1, 2)) < 1.0e8
        fast            = intersects_view & nondegenerate & coordinate_safe
        points          = np.rint(projected[fast] * 16.0).astype(np.int32)
        if batch_faces <= 1:
            for triangle_points in points:
                cv2.fillConvexPoly(
                    mask,
                    triangle_points,
                    color = 1,
                    lineType = cv2.LINE_8,
                    shift = 4,
                )
        else:
            # fillPoly uses an even/odd rule within one call, so very large
            # batches can cancel overlapping front/back triangles.  Small
            # batches are ORed into the exact union and offer a controlled
            # speed/edge approximation tradeoff.  The chosen value is part of
            # selector configuration/provenance and is never implicit
            scratch = np.zeros_like(mask)
            for start in range(0, len(points), batch_faces):
                scratch.fill(0)
                cv2.fillPoly(
                    scratch,
                    points[start : start + batch_faces],
                    color = 1,
                    lineType = cv2.LINE_8,
                    shift = 4,
                )
                cv2.bitwise_or(mask, scratch, dst = mask)
        unsafe_indices  = simple_indices[
            intersects_view & nondegenerate & ~coordinate_safe
        ]

    clipping_indices    = np.flatnonzero(intersects_depth & ~all_inside)
    if unsafe_indices.size == 0:
        return clipping_indices
    return np.concatenate((clipping_indices, unsafe_indices))
