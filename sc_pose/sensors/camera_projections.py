""" Code to project pose information on to a camera image, uses a Camera Object """

import numpy as np 
from numpy.typing import NDArray 
import warnings

# local imports
from sc_pose.sensors.camera import CameraBase
from sc_pose.math.quaternion import q2trfm

class PoseProjector:

    def __init__(self, camera: CameraBase):
        """ Initialize PoseProjector with a Camera object """
        if not isinstance(camera, CameraBase):
            raise TypeError(f"camera must be CameraBase, got {type(camera).__name__}")
        self.camera     = camera
    
    def project_to_image(
                            self, 
                            q_TARGET_2_CAM: NDArray[np.floating],
                            r_Co2To_CAM: NDArray[np.floating], 
                            points_xyz_TARGET: NDArray[np.floating]
                        ) -> NDArray[np.floating]:
        """ 
        Project 3D points in TARGET frame to 2D image points in CAMERA frame
        TARGET frame is often called the WORLD, OBJECT, or BODY frame depending on context

        Args:
            q_TARGET_2_CAM: right-handed scalar first (RSF) quaternion representing rotation from TARGET to CAMERA frame (4,)
            r_Co2To_CAM: translation vector from CAMERA body-fixed frame origin to TARGET frame origin, expressed in CAMERA frame (3,)
            points_xyz_TARGET: 3D points (with shape (..., 3) ) in TARGET frame to be projected

        Returns:
            points_uv_CAM: 2D points (with shape (..., 2) ) in image plane of CAMERA frame 
        """
        # sanity checks 
        # quaternion size check
        q_TARGET_2_CAM      = np.asarray(q_TARGET_2_CAM)
        if q_TARGET_2_CAM.shape != (4,):
            raise ValueError(f"q_TARGET_2_CAM must have shape (4,), got {q_TARGET_2_CAM.shape}")
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
        # q2trfm returns a 3x3 transformation (passive rotation, also known as Direction Cosine Matrix) matrix from TARGET to CAMERA frame
        T_TARGET_2_CAM      = q2trfm(q_TARGET_2_CAM)
        points_shape        = points_xyz_TARGET.shape # (..., 3) original shape of points
        points_flat         = points_xyz_TARGET.reshape(-1, 3) # (N, 3) points flattened for matrix multiplication
        points_flat_xyz_CAM = (T_TARGET_2_CAM @ points_flat.T).T + r_Co2To_CAM # (N, 3) points in CAMERA frame
        points_xyz_CAM      = points_flat_xyz_CAM.reshape(points_shape) # (..., 3) points in CAMERA frame
        points_uv_CAM       = self.camera.project_camera3Dxyz_to_imageUV(points_xyz_CAM) # (..., 2) points in image plane 
        return points_uv_CAM
    

    @staticmethod
    def classless_pinhole_project_to_image(
                                            q_TARGET_2_CAM: NDArray[np.floating], 
                                            r_Co2To_CAM: NDArray[np.floating], 
                                            Kmat: NDArray[np.floating],
                                            points_xyz_TARGET: NDArray[np.floating],
                                            BC_dist_coeffs: NDArray[np.floating] | None = None
                                        ) -> NDArray[np.floating]:

        """
        Projecting 3D points to 2D image coordinates that without using a Camera object
        Uses pinhole camera model with radial and tangential distortion (Brown-Conrady model)
        
        Args:
            q_TARGET_2_CAM: quaternion (4,) (NDArray)
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
        q_TARGET_2_CAM      = np.asarray(q_TARGET_2_CAM)
        r_Co2To_CAM         = np.asarray(r_Co2To_CAM)
        Kmat                = np.asarray(Kmat)
        points_xyz_TARGET   = np.asarray(points_xyz_TARGET)

        # input validation
        if q_TARGET_2_CAM.shape != (4,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: q_TARGET_2_CAM must have shape (4,), got {q_TARGET_2_CAM.shape}")
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
            BC_dist_coeffs  = np.zeros(5, dtype = q_TARGET_2_CAM.dtype)
        BC_dist_coeffs      = np.asarray(BC_dist_coeffs)
        if BC_dist_coeffs.shape != (5,):
            raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: BC_dist_coeffs must have shape (5,), got {BC_dist_coeffs.shape}")
        
        # check values
        for name, arr in [("q_TARGET_2_CAM", q_TARGET_2_CAM), ("r_Co2To_CAM", r_Co2To_CAM),
                  ("Kmat", Kmat), ("points_xyz_TARGET", points_xyz_TARGET)]:
            if not np.isfinite(arr).all():
                raise ValueError(f"PoseProjector.classless_pinhole_project_to_image: {name} contains NaN/Inf")
        
        # num_pts             = points_xyz_TARGET.shape[0]
        T_TARGET_2_CAM      = q2trfm(q_TARGET_2_CAM) # 3 x 3 transformation matrix from TARGET to CAMERA frame
        # first transpose makes it (3, N) for broadcasting multiplication, then apply rotation, then transpose back to (N, 3)
        points_xyz_CAM      = (T_TARGET_2_CAM @ points_xyz_TARGET.T).T + r_Co2To_CAM # N x 3 points in CAMERA frame  
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
