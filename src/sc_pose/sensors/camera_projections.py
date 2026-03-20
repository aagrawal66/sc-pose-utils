""" Code to project pose information on to a camera image, uses a Camera Object """

import numpy as np 
from numpy.typing import NDArray 
from pathlib import Path
from typing import Tuple, Optional
import cv2
import warnings

# local imports
from sc_pose.sensors.camera import CameraBase
from sc_pose.mathtils.quaternion import q2trfm, q2rotm

r"""
Note on passive and active rotations:

A passive rotation is a change of coordinate frame, while an active rotation is a rotation of vectors in space.

Example 1
\vspace{-2.5mm}
\begin{itemize}[noitemsep]
    \item Imagine you're standing still facing north, and you're holding an arrow straight out in front of you, facing north as well.
    \item You are the coordinate frame, the arrow at its tip is a point in space.
    \item Also, imagine the vector from you (your origin) to the arrow tip as a position vector.
    \item If you stay facing north, but rotate the arrow 90 degrees to point east, that's an active rotation of the position vector.
    \item On the other hand, if you instead rotate yourself to face east, while keeping the arrow pointing straight out in front of you, that's a passive rotation of the coordinate frame.
\end{itemize}

Example 2
\vspace{-2.5mm}
\begin{itemize}[noitemsep]
    \item Imagine a camera looking at an object in space. The camera has its own body-fixed coordinate frame (C), and the object has its own body-fixed coordinate frame (T).
    \item Imagine a set of points in the target object's frame. 
    \item We need to express that point in the camera's frame in order to project it onto the camera's image plane.
    \item To project a point onto the camera image plane, we must express that point in the camera frame. In effect, we align the axes of t with those of C (rotation), then account for their different origins (translation).
    \item In short, we're aligning the axes of the target frame with the axes of the camera frame. We will then use 
    translation to express the position of the target frame origin in the camera frame.
    \item There are two ways to think of the projections here.
    \begin{itemize}[noitemsep]
        \item Employing active rotation: $\pp^C = \mathbf{R}_{T\to C} \cdot \pp^T + \rr_{C_o\to T_o}^C$ 
        \item Employing passive rotation: $\pp^C = \Tr^C_T \cdot ( \pp^T - \rr_{T_o\to C_o}^T) = \Tr^C_T \pp^T + \rr_{C_o\to T_o}^C$
        \item Note that $ \Tr^C_T \pp^T$ are the coordinates of the same physical point, expressed in the camera axes, but still referenced to the target origin. We need to add the translation between the target and camera frame origins to get into the camera frame.
    \end{itemize}
\end{itemize}"""

class PoseProjector:

    def __init__(self, camera: CameraBase | None):
        """ Initialize PoseProjector with a Camera object """
        if not isinstance(camera, CameraBase):
            if camera is None:
                warning_msg = f"PoseProjector expects camera init to take in a CameraBase, got {type(camera).__name__}, assuming that classless_pinhole_project_to_image is intended"
                warnings.warn(warning_msg, RuntimeWarning)
            else:
                raise TypeError(f"camera must be CameraBase or None, got {type(camera).__name__}")

        self.camera = camera
    
    def project_to_image(
                            self, 
                            q_CAM_2_TARGET: NDArray[np.floating],
                            r_Co2To_CAM: NDArray[np.floating], 
                            points_xyz_TARGET: NDArray[np.floating]
                        ) -> NDArray[np.floating]:
        """ 
        Project 3D points in TARGET frame to 2D image points in CAMERA frame
        TARGET frame is often called the WORLD, OBJECT, or BODY frame depending on context

        Args:
            q_CAM_2_TARGET: right-handed scalar first (RSF) quaternion representing rotation from CAMERA to TARGETframe (4,)
            r_Co2To_CAM: translation vector from CAMERA body-fixed frame origin to TARGET frame origin, expressed in CAMERA frame (3,)
            points_xyz_TARGET: 3D points (with shape (..., 3) ) in TARGET frame to be projected

        Returns:
            points_uv_CAM: 2D points (with shape (..., 2) ) in image plane of CAMERA frame 
        """
        # sanity checks 
        # quaternion size check
        q_CAM_2_TARGET      = np.asarray(q_CAM_2_TARGET)
        if q_CAM_2_TARGET.shape != (4,):
            raise ValueError(f"q_CAM_2_TARGET must have shape (4,), got {q_CAM_2_TARGET.shape}")
        # translation vector size check
        r_Co2To_CAM      = np.asarray(r_Co2To_CAM)
        if r_Co2To_CAM.shape != (3,):
            raise ValueError(f"r_Co2To_CAM must have shape (3,), got {r_Co2To_CAM.shape}")
        # points size check
        points_xyz_TARGET      = np.asarray(points_xyz_TARGET)
        if points_xyz_TARGET.ndim < 2 or points_xyz_TARGET.shape[-1] != 3:
            raise ValueError(f"points_xyz_TARGET must have shape (..., 3), got {points_xyz_TARGET.shape}")
        # check for NaN or Inf values
        if not np.isfinite(points_xyz_TARGET).all():
            raise ValueError("points_xyz_TARGET contains NaN or Inf values")
        
        # with active rotation, we rotate vectors in space, keep the coordinate frame fixed
        # q2rotm returns a 3x3 an active rotation matrix, it describes rotation of vectors from one frame to another
        # in this case, we are rotating the position vectors from the camera origin to each point, expressing those vectors in the CAMERA frame
        # this means
        R_TARGET_2_CAM      = q2rotm(q_CAM_2_TARGET) # R_TARGET_2_CAM rotates vectors from TARGET frame to CAMERA frame = Trfm_TARGET_2_CAM which is the passive rotation matrix that transforms coordinates of points from TARGET frame to CAMERA frame
        points_shape        = points_xyz_TARGET.shape # (..., 3) original shape of points
        points_flat         = points_xyz_TARGET.reshape(-1, 3) # (N, 3) points flattened for matrix multiplication
        # first transpose makes it (3, N) for broadcasting multiplication, then apply rotation, then transpose back to (N, 3)
        points_flat_xyz_CAM = (R_TARGET_2_CAM @ points_flat.T).T + r_Co2To_CAM # (N, 3) points in CAMERA frame
        points_xyz_CAM      = points_flat_xyz_CAM.reshape(points_shape) # (..., 3) points in CAMERA frame
        points_uv_CAM       = self.camera.project_camera3Dxyz_to_imageUV(points_xyz_CAM) # (..., 2) points in image plane 
        return points_uv_CAM
    

    def T4x4_2_uv(self, T4x4_A_B: NDArray, pts_A: NDArray) -> NDArray[np.floating]:
        """ 
        Project 3D points in A frame to 2D pixel coordinates in the camera frame, given the 4x4 
        homogeneous transformation from A to B and the camera object with calibration data 
        """
        # Trfm_A_B    = T4x4_A_B[:3,:3]
        # tr_B_A_in_B = T4x4_A_B[:3,3]
        # # pts_B       = Trfm_A_B @ pts_A + tr_B_A_in_B
        pts_B           = np.empty_like(pts_A, dtype = float)
        for i, p in enumerate(pts_A):
            p_h_A       = np.array([p[0], p[1], p[2], 1.0]) # homogeneous point
            p_h_B       = T4x4_A_B @ p_h_A
            pts_B[i]    = p_h_B[:3]
        points_uv_CAM   = self.camera.project_camera3Dxyz_to_imageUV(pts_B)
        return points_uv_CAM

    # classess version of T4x4_2_uv, which takes in a Camera object and a 4x4 transformation, and projects points from A frame to the camera image plane
    @staticmethod
    def classless_pinhole_project_T4x4_2_uv(
                                            T_TARGET_CAM: NDArray[np.floating],
                                            Kmat: NDArray[np.floating],
                                            points_xyz_TARGET: NDArray[np.floating],
                                            BC_dist_coeffs: NDArray[np.floating] | None = None
                                        ) -> NDArray[np.floating]:

        """
        Projecting 3D points to 2D image coordinates that without using a Camera object
        Uses pinhole camera model with radial and tangential distortion (Brown-Conrady model)
        
        Args:
            T_TARGET_CAM: 4x4 transformation matrix from target frame to camera frame (NDArray)
            T_TARGET_CAM is a homogeneous transformation matrix that transforms points from the TARGET frame to the CAMERA frame
                T_TARGET_CAM = [ Trfm_TARGET_CAM | r_Co2To_CAM ] where Trfm is a passive rotation matrix used for coordinate transformation 
            Kmat: camera intrinsic matrix (3,3) (NDArray)
            points_xyz_TARGET: 3D points (N, 3) (NDArray)
            BC_dist_coeffs: distortion coefficients (5,) (NDArray), defaults to np.zeros(5) if None
        
        BC_dist_coeffs layout follows OpenCV convention::
        BC_dist_coeffs[0]: k1 (radial)
        BC_dist_coeffs[1]: k2 (radial)
        BC_dist_coeffs[2]: p1 (tangential)
        BC_dist_coeffs[3]: p2 (tangential)
        BC_dist_coeffs[4]: k3 (radial)

        Returns:
            points2D: 2D points (N, 2) (NDArray)
        """
        # ensure inputs are numpy arrays
        T_TARGET_CAM        = np.asarray(T_TARGET_CAM)
        Kmat                = np.asarray(Kmat)
        points_xyz_TARGET   = np.asarray(points_xyz_TARGET)

        # input validation
        if T_TARGET_CAM.shape != (4, 4):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: T_TARGET_CAM must have shape (4, 4), got {T_TARGET_CAM.shape}")
        if Kmat.shape != (3, 3):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: Kmat must have shape (3, 3), got {Kmat.shape}")

        if points_xyz_TARGET.ndim == 2 and points_xyz_TARGET.shape[1] == 3:
            pass  # correct shape, (N, 3)
        elif points_xyz_TARGET.ndim == 2 and points_xyz_TARGET.shape[0] == 3:
            points_xyz_TARGET   = points_xyz_TARGET.T # (3, N) -> (N, 3)
        else:
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: points_xyz_TARGET must have shape (N, 3) or (3, N), got {points_xyz_TARGET.shape}")
        
        if BC_dist_coeffs is None:
            BC_dist_coeffs  = np.zeros(5, dtype = T_TARGET_CAM.dtype)
        BC_dist_coeffs      = np.asarray(BC_dist_coeffs)
        if BC_dist_coeffs.shape != (5,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: BC_dist_coeffs must have shape (5,), got {BC_dist_coeffs.shape}")
        
        # check values
        for name, arr in [("T_TARGET_CAM", T_TARGET_CAM), ("Kmat", Kmat), ("points_xyz_TARGET", points_xyz_TARGET)]:
            if not np.isfinite(arr).all():
                raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: {name} contains NaN/Inf")
        
        # need to make points_xyz_TARGET homoegenous
        num_pts             = points_xyz_TARGET.shape[0] 
        points_xyzh_TARGET  = np.column_stack([points_xyz_TARGET, np.ones(num_pts, dtype = points_xyz_TARGET.dtype)])
        points_xyzh_CAM     = (T_TARGET_CAM @ points_xyzh_TARGET.T).T # inner transpose for batched matrix multiplication, outer transpose to get back to (N, 4)
        points_xyz_CAM      = points_xyzh_CAM[:, :3] # convert back to non-homogeneous coordinates, shape (N, 3) 
        # normalize points to image plane
        XX                  = points_xyz_CAM[:, 0] 
        YY                  = points_xyz_CAM[:, 1]
        ZZ                  = points_xyz_CAM[:, 2]
        eps                 = 1e-8
        valid               = ZZ > eps  # only project points in front of the camera and avoid division by zero
        x0                  = np.full_like(ZZ, np.nan, dtype = points_xyz_CAM.dtype)
        y0                  = np.full_like(ZZ, np.nan, dtype = points_xyz_CAM.dtype)
        x0[valid]           = XX[valid] / ZZ[valid]
        y0[valid]           = YY[valid] / ZZ[valid]
        num_invalid         = np.count_nonzero(~valid) 
        if num_invalid > 0:
            warning_msg     = f"PoseProjector.classless_pinhole_project_to_image: {num_invalid} points are behind the camera or too close to the camera plane and will be assigned NaN in the output"
            warnings.warn(warning_msg, RuntimeWarning)
        # apply Brown-Conrady distortion model
        k1          = BC_dist_coeffs[0] 
        k2          = BC_dist_coeffs[1]
        p1          = BC_dist_coeffs[2]
        p2          = BC_dist_coeffs[3]
        k3          = BC_dist_coeffs[4]
        # radial distortion
        r2          = x0*x0 + y0*y0
        r4          = r2 * r2
        r6          = r2 * r4
        cdist       = 1 + k1*r2 + k2*r4 + k3*r6
        # tangential distortion
        xdist       = x0*cdist + 2*p1*x0*y0 + p2*(r2 + 2*x0*x0)
        ydist       = y0*cdist + p1*(r2 + 2*y0*y0) + 2*p2*x0*y0

        # apply camera matrix
        fx          = Kmat[0, 0]
        fy          = Kmat[1, 1]
        c_x         = Kmat[0, 2]
        c_y         = Kmat[1, 2]
        skew        = Kmat[0, 1]
        # final pixel coordinates
        u           = fx*xdist + skew*ydist + c_x
        v           = fy*ydist + c_y
        points2D    = np.column_stack([u, v])  # N x 2

        return points2D

    @staticmethod
    def classless_pinhole_project_to_image(
                                            q_CAM_2_TARGET: NDArray[np.floating], 
                                            r_Co2To_CAM: NDArray[np.floating], 
                                            Kmat: NDArray[np.floating],
                                            points_xyz_TARGET: NDArray[np.floating],
                                            BC_dist_coeffs: NDArray[np.floating] | None = None
                                        ) -> NDArray[np.floating]:

        """
        Projecting 3D points to 2D image coordinates that without using a Camera object
        Uses pinhole camera model with radial and tangential distortion (Brown-Conrady model)
        
        Args:
            q_CAM_2_TARGET: quaternion (4,) (NDArray)
            r_Co2To_CAM: position (3,) (NDArray)
            Kmat: camera intrinsic matrix (3,3) (NDArray)
            points_xyz_TARGET: 3D points (N, 3) (NDArray)
            BC_dist_coeffs: distortion coefficients (5,) (NDArray), defaults to np.zeros(5) if None
        
        BC_dist_coeffs layout follows OpenCV convention::
        BC_dist_coeffs[0]: k1 (radial)
        BC_dist_coeffs[1]: k2 (radial)
        BC_dist_coeffs[2]: p1 (tangential)
        BC_dist_coeffs[3]: p2 (tangential)
        BC_dist_coeffs[4]: k3 (radial)

        Returns:
            points2D: 2D points (N, 2) (NDArray)
        """
        # ensure inputs are numpy arrays
        q_CAM_2_TARGET      = np.asarray(q_CAM_2_TARGET)
        r_Co2To_CAM         = np.asarray(r_Co2To_CAM)
        Kmat                = np.asarray(Kmat)
        points_xyz_TARGET   = np.asarray(points_xyz_TARGET)

        # input validation
        if q_CAM_2_TARGET.shape != (4,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: q_CAM_2_TARGET must have shape (4,), got {q_CAM_2_TARGET.shape}")
        if r_Co2To_CAM.shape != (3,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: r_Co2To_CAM must have shape (3,), got {r_Co2To_CAM.shape}")
        if Kmat.shape != (3, 3):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: Kmat must have shape (3, 3), got {Kmat.shape}")

        if points_xyz_TARGET.ndim == 2 and points_xyz_TARGET.shape[1] == 3:
            pass  # correct shape, (N, 3)
        elif points_xyz_TARGET.ndim == 2 and points_xyz_TARGET.shape[0] == 3:
            points_xyz_TARGET   = points_xyz_TARGET.T # (3, N) -> (N, 3)
        else:
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: points_xyz_TARGET must have shape (N, 3) or (3, N), got {points_xyz_TARGET.shape}")
        
        if BC_dist_coeffs is None:
            BC_dist_coeffs  = np.zeros(5, dtype = q_CAM_2_TARGET.dtype)
        BC_dist_coeffs      = np.asarray(BC_dist_coeffs)
        if BC_dist_coeffs.shape != (5,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: BC_dist_coeffs must have shape (5,), got {BC_dist_coeffs.shape}")
        
        # check values
        for name, arr in [("q_CAM_2_TARGET", q_CAM_2_TARGET), ("r_Co2To_CAM", r_Co2To_CAM),
                  ("Kmat", Kmat), ("points_xyz_TARGET", points_xyz_TARGET)]:
            if not np.isfinite(arr).all():
                raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: {name} contains NaN/Inf")
        

        R_TARGET_to_CAM     = q2rotm(q_CAM_2_TARGET) # active rotation matrix from CAMERA to TARGETframe
        # first transpose makes it (3, N) for broadcasting multiplication, then apply rotation, then transpose back to (N, 3)
        points_xyz_CAM      = (R_TARGET_to_CAM @ points_xyz_TARGET.T).T + r_Co2To_CAM # N x 3 points in CAMERA frame  
        # normalize points to image plane
        XX                  = points_xyz_CAM[:, 0] 
        YY                  = points_xyz_CAM[:, 1]
        ZZ                  = points_xyz_CAM[:, 2]
        eps                 = 1e-8
        valid               = ZZ > eps  # only project points in front of the camera and avoid division by zero
        x0                  = np.full_like(ZZ, np.nan, dtype = points_xyz_CAM.dtype)
        y0                  = np.full_like(ZZ, np.nan, dtype = points_xyz_CAM.dtype)
        x0[valid]           = XX[valid] / ZZ[valid]
        y0[valid]           = YY[valid] / ZZ[valid]
        num_invalid         = np.count_nonzero(~valid) 
        if num_invalid > 0:
            warning_msg     = f"PoseProjector.classless_pinhole_project_to_image: {num_invalid} points are behind the camera or too close to the camera plane and will be assigned NaN in the output"
            warnings.warn(warning_msg, RuntimeWarning)
        # apply Brown-Conrady distortion model
        k1          = BC_dist_coeffs[0] 
        k2          = BC_dist_coeffs[1]
        p1          = BC_dist_coeffs[2]
        p2          = BC_dist_coeffs[3]
        k3          = BC_dist_coeffs[4]
        # radial distortion
        r2          = x0*x0 + y0*y0
        r4          = r2 * r2
        r6          = r2 * r4
        cdist       = 1 + k1*r2 + k2*r4 + k3*r6
        # tangential distortion
        xdist       = x0*cdist + 2*p1*x0*y0 + p2*(r2 + 2*x0*x0)
        ydist       = y0*cdist + p1*(r2 + 2*y0*y0) + 2*p2*x0*y0

        # apply camera matrix
        fx          = Kmat[0, 0]
        fy          = Kmat[1, 1]
        c_x         = Kmat[0, 2]
        c_y         = Kmat[1, 2]
        skew        = Kmat[0, 1]
        # final pixel coordinates
        u           = fx*xdist + skew*ydist + c_x
        v           = fy*ydist + c_y
        points2D    = np.column_stack([u, v])  # N x 2

        return points2D


def draw_uv_points_on_image(
                                img_or_path: NDArray[np.uint8] | Path | str,
                                points_uv: NDArray[np.floating],
                                *,
                                point_color: Tuple[int, int, int] = (0, 255, 0),
                                point_radius: int = 5,
                                point_thickness: int = -1,
                                **kwargs # additional keyword arguments for cv2.circle
                        ) -> NDArray:
    """ 
    Draw (u, v) points on image and return the image with drawn points
    
    Args:
        img_or_path: image as numpy array (H, W, 3) or path to image file
        points_uv: 2D points (N, 2) to be drawn on image
        point_color: color of points in BGR format, default is green (0, 255, 0)
        point_radius: radius of points to be drawn, default is 5
        point_thickness: thickness of points to be drawn, default is -1 (filled circle)
        **kwargs: additional keyword arguments for cv2.circle function
    
    Returns:
        img_with_points: image with drawn points as numpy array (H, W, 3)
    """
    # load / normalize image 
    if isinstance(img_or_path, (str, Path)):
        img     = cv2.imread(str(img_or_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_or_path}")
        # ensure 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_bgr = img
    elif isinstance(img_or_path, np.ndarray):
        img         = img_or_path
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img_bgr = img.copy()
        elif img.ndim == 3 and img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"unsupported image shape: {img.shape}")
    else:
        raise TypeError("img_or_path must be a numpy array or a path-like")

    H, W    = img_bgr.shape[:2]
    uv      = np.asarray(points_uv)
    num_pts = uv.shape[0]
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (N,2), got {uv.shape}")
    total_skipped   = 0
    # draw outputs
    for i, (u, v) in enumerate(uv):
        if (not np.isfinite(u) or not np.isfinite(v)):
            total_skipped += 1
            continue
        ui  = int(np.round(u))
        vi  = int(np.round(v))
        if ui < 0 or ui >= W or vi < 0 or vi >= H:
            total_skipped += 1
            continue

        cv2.circle(
                    img_bgr, 
                    (ui, vi), 
                    point_radius, 
                    point_color, 
                    point_thickness,
                    **kwargs
                )
    print(f"total points skipped (out of bounds or invalid) out of {num_pts}: ", total_skipped)
    
    return img_bgr
