""" Camera Model Objects and associated functions """

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np 
from typing import Any, Mapping, Tuple
from numpy.typing import NDArray
import yaml
import warnings 

class CameraBase(ABC):
    """ 
    Abstract Base Class for Camera Models 
    
    Attributes:
        sw_mm: Sensor width in mm
        sh_mm: Sensor height in mm
        img_w: Image width in pixels
        img_h: Image height in pixels
        fl_mm: Focal length in mm
        sq_px: Flag for square pixels (fx = fy) in focal length calculation

    primary initialization path:
        Explicit parameters for sensor size, image size, and focal length
    secondary initialization path:
        from configuration with fixed keys
    """
    def __init__(
                    self,
                    sensor_width_mm: float,
                    sensor_height_mm: float,
                    image_width_px: int,
                    image_height_px: int,
                    focal_length_mm: float,
                    square_pixels: bool = True,
                    *, 
                    dtype: np.dtype = np.float32
                  ) -> None:
        # initialize camera parameters and ensure correct types
        self.sw_mm  = float(sensor_width_mm) # sensor width in mm
        self.sh_mm  = float(sensor_height_mm) # sensor height in mm
        self.img_w  = int(image_width_px) # image width in pixels
        self.img_h  = int(image_height_px) # image height in pixels
        self.fl_mm  = float(focal_length_mm) # focal length in 
        self.sq_px  = square_pixels # whether pixels are square (fx = fy)
        self._validate_params() # validate intrinsic parameters

        # intrinsics override (None means "compute from physical")
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._skew: float = 0.0  # pinhole convention: keep at 0 unless explicitly set
        self._K_cache: NDArray[np.float32] | None = None # cache for intrinsic matrix K
        # TODO: use this everywhere dtype is needed
        self.dtype  = np.dtype(dtype) # data type for computations


    @classmethod # a class method is used to create an instance from a config
    # classmethod is used to create an instance from a config
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CameraBase":
        """ Initialize CameraBase from config dict with fixed keys, each concrete camera model must implement this """
        raise NotImplementedError("CameraBase.from_config() must be implemented in subclass") 

    def _validate_params(self) -> None:
        # validation checks
        if not (self.sw_mm > 0 and self.sh_mm > 0):
            raise ValueError("CameraBase: sensor dimensions must be positive values")
        if not (self.img_w > 0 and self.img_h > 0):
            raise ValueError("CameraBase: image dimensions must be positive values")
        if not (self.fl_mm > 0):
            raise ValueError("CameraBase: focal length must be a positive value")
            

    def focal_length_px(self) -> Tuple[float, float]:
        """ Calculate and return the focal length in pixels (fx, fy), if square fx = fy """
        x_px_per_mm = self.img_w / self.sw_mm
        y_px_per_mm = self.img_h / self.sh_mm
        fx          = self.fl_mm * x_px_per_mm
        if self.sq_px:
            fy      = fx
        else:
            fy      = self.fl_mm * y_px_per_mm
        return float(fx), float(fy)


    def center_principal_point(self) -> Tuple[float, float]:
        """ Return principal point (c_x, c_y) in pixels at image center"""
        c_x     = (self.img_w  - 1) / 2.0
        c_y     = (self.img_h  - 1) / 2.0
        return float(c_x), float(c_y)
     

    @staticmethod
    def build_Kmat(fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> NDArray[np.float32]:
        """ 
        build the camera intrinsic matrix K 
        LaTeX: K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
        """
        Kmat    = np.array([
                            [fx, skew, cx],
                            [0.0, fy, cy],
                            [0.0,  0.0,  1.0]
                            ], dtype = np.float32)
        return Kmat    


    @abstractmethod
    def calc_Kmat(self) -> NDArray[np.float32]:
        """ Return the 3x3 camera intrinsic matrix K """
        raise NotImplementedError("CameraBase.calc_Kmat() must be implemented in subclass")
    

    @abstractmethod
    def project_camera3Dxyz_to_imageUV(self, xyz: NDArray[np.float32]) -> NDArray[np.float32]:
        """ Project 3D camera coordinates (X, Y, Z) to image pixel coordinates (u, v) """
        raise NotImplementedError("CameraBase.project_camera3Dxyz_to_imageUV() must be implemented in subclass")
    
    @abstractmethod
    def project_imageUV_to_cameraRay(self, uv: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """ Project image pixel coordinates (u, v) to normalized 3D camera coordinates (X, Y, Z = 1) """
        raise NotImplementedError("CameraBase.project_imageUV_to_cameraRay() must be implemented in subclass")
    
    @abstractmethod
    def set_calibration(self, **kwargs: Any) -> "CameraBase":
        """ Set camera calibration parameters specific to the camera model """
        raise NotImplementedError("CameraBase.set_calibration() must be implemented in subclass")
    
class PinholeCamera(CameraBase):
    """ This object models a pinhole camera with Brown-Conrady distortion  """
    def __init__(
                self,
                sensor_width_mm: float,
                sensor_height_mm: float,
                image_width_px: int,
                image_height_px: int,
                focal_length_mm: float,
                square_pixels: bool = True,
                *, # after this, all args are keyword-only, not positional
                k1: float = 0.0, # first radial distortion coefficient
                k2: float = 0.0, # second radial distortion coefficient
                k3: float = 0.0, # third radial distortion coefficient
                p1: float = 0.0, # first tangential distortion coefficient
                p2: float = 0.0, # second tangential distortion coefficient
                ) -> None:
        super().__init__(sensor_width_mm, sensor_height_mm, image_width_px, image_height_px, focal_length_mm, square_pixels)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self._validate_basic_CB_distortion()
        
        # Brown–Conrady Distortion Model (OpenCV-compatible) Explained
        # -------------------------------------------------
        # let (x, y) be normalized image coordinates after perspective division:
        #     x = X / Z
        #     y = Y / Z

        # radial distance:
        #     r² = x² + y²

        # radial distortion:
        #     x_r = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
        #     y_r = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)

        # tangential distortion:
        #     x_d = x_r + 2*p1*x*y + p2*(r² + 2*x²)
        #     y_d = y_r + p1*(r² + 2*y²) + 2*p2*x*y

        # final pixel coordinates (intrinsics K):
        #     u = fx*x_d + s*y_d + cx = fx* [ x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²) ] + s* [ y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y ] + cx
        #     v = fy*y_d + cy = fy* [ y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y ] + cy

        # where:
        #     k1, k2, k3  : radial distortion coefficients
        #     p1, p2      : tangential distortion coefficients
        #     fx, fy      : focal lengths in pixels
        #     cx, cy      : principal point
        #     s           : skew (usually 0, 0 for pinhole model)

        # Notes:
        # - distortion is applied in normalized camera coordinates.
        # - OpenCV distortion vector order:
        #       [k1, k2, p1, p2, k3]
        # - higher-order terms (k4–k6) are omitted here
        # - assumes +Z points forward out of the camera

        

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PinholeCamera":
        """ Initialize PinholeCamera from config dict with fixed keys """
        try:
            return cls(
                        sensor_width_mm     = cfg['sensor_width_mm'],
                        sensor_height_mm    = cfg['sensor_height_mm'],
                        image_width_px      = cfg['image_width_px'],
                        image_height_px     = cfg['image_height_px'],
                        focal_length_mm     = cfg['focal_length_mm'],
                        square_pixels       = cfg.get('square_pixels', True),
                        k1                  = cfg.get('k1', 0.0),
                        k2                  = cfg.get('k2', 0.0),
                        k3                  = cfg.get('k3', 0.0),
                        p1                  = cfg.get('p1', 0.0),
                        p2                  = cfg.get('p2', 0.0),
                        )
        except KeyError as e:
            missing     = str(e).strip("'")
            raise KeyError(f"PinholeCamera.from_config: missing key '{missing}' in config") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"PinholeCamera.from_config: invalid value in config: {e}") from e
    
    def _validate_basic_CB_distortion(self) -> None:
        """ Validate Brown-Conrady distortion coefficients """
        # no specific validation rules for distortion coefficients, checking type and finiteness
        for name, val in [("k1", self.k1), ("k2", self.k2), ("k3", self.k3), ("p1", self.p1), ("p2", self.p2)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(val).__name__}")
            if not np.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val}")


    def calc_Kmat(self) -> NDArray[np.float32]:
        """ Calculate and return the intrinsic matrix K (3x3), skew is always set to 0.0 in pinhole model """
        if self._K_cache is not None:
            # use cached K matrix
            return self._K_cache
        if all(v is not None for v in (self._fx, self._fy, self._cx, self._cy)):
            fx      = float(self._fx)
            fy      = float(self._fy)
            cx      = float(self._cx)
            cy      = float(self._cy)
        else:
            if any(v is not None for v in (self._fx, self._fy, self._cx, self._cy)):
                raise ValueError(
                                    "PinholeCamera.calc_Kmat: partial calibration override detected, " \
                                    "set fx, fy, cx, cy together (or clear all)"
                            )
            fx, fy  = self.focal_length_px()
            cx, cy  = self.center_principal_point()
        Kmat        = self.build_Kmat(fx = fx, fy = fy, cx = cx, cy = cy, skew = 0.0) # pinhole convention: zero skew
        self._K_cache   = Kmat # cache K matrix
        return Kmat
    
    def set_calibration(
                            self,
                            * ,
                            fx: float,
                            fy: float,
                            cx: float,
                            cy: float,
                            k1: float,
                            k2: float,
                            k3: float,
                            p1: float,
                            p2: float
                        ) -> "PinholeCamera": # enables method chaining
        """ Set the camera intrinsics and distortion coefficients for the camera based on calibration results """
        # set intrinsic parameters: fx, fy, cx, cy
        self._fx    = float(fx)
        self._fy    = float(fy)
        if (self._fx != self._fy):
            self.sq_px  = False
            warnings.warn("PinholeCamera.set_calibration: setting fx != fy on a non-square pixel camera", RuntimeWarning)
        self._cx    = float(cx)
        self._cy    = float(cy)
        # set distortion coefficients
        self.k1     = float(k1)
        self.k2     = float(k2)
        self.k3     = float(k3)
        self.p1     = float(p1)
        self.p2     = float(p2)
        self._validate_basic_CB_distortion()
        self._K_cache   = None # clear K matrix cache

        # returning self enables method chaining:
        # when methods return an object (often self) so the next method can be called
        # # e.g., 
        # text = "  Hello World  ".strip().lower().replace(" ", "_")
        # print(text)

        return self
    
    def set_calibration_yaml(self, yaml_path: str | Path) -> "PinholeCamera":
        """ 
        Return a PinholeCamera with a YAML file containing calibrated instrincs and distortion coefficients 
        
        Required keys:
            - fx
            - fy
            - cx
            - cy
            - k1
            - k2
            - k3
            - p1
            - p2    
        """

        # load YAML file
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # set calibration parameters from YAML
        try:
            return self.set_calibration(
                                        fx  = cfg['fx'],
                                        fy  = cfg['fy'],
                                        cx  = cfg['cx'],
                                        cy  = cfg['cy'],
                                        k1  = cfg['k1'],
                                        k2  = cfg['k2'],
                                        k3  = cfg['k3'],
                                        p1  = cfg['p1'],
                                        p2  = cfg['p2']
                                      )
        except KeyError as e:
            missing     = str(e).strip("'")
            raise KeyError(f"PinholeCamera.set_calibration_yaml: missing key '{missing}' in YAML file") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"PinholeCamera.set_calibration_yaml: invalid value in YAML file: {e}") from e
    
    def project_camera3Dxyz_to_imageUV(self, xyz: NDArray[np.float32]) -> NDArray[np.float32]:
        """ 
        Project 3D camera coordinates (X, Y, Z) to image pixel coordinates (u, v) using Brown-Conrady distortion 

        Args:
            xyz: (..., 3) array of 3D points in camera coordinates
        
        Returns:
            uv: (..., 2) array of 2D points in image pixel coordinates
        
        Brown–Conrady Distortion Model applied with skew = 0.0 (see class init for added details)
        """

        # sanity check input

        xyz     = np.asarray(xyz, dtype = np.float32)

        if xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (..., 3)")
        
        XX      = xyz[..., 0]
        YY      = xyz[..., 1]
        ZZ      = xyz[..., 2]
        eps     = 1e-8
        behind  = ZZ < 0
        close   = (ZZ >= 0) & (ZZ <= eps)
        invalid = behind | close
        valid   = ZZ > eps
        num_behind  = int(np.count_nonzero(behind))
        num_close   = int(np.count_nonzero(close))
        num_invalid = int(np.count_nonzero(invalid))
        num_valid   = int(np.count_nonzero(valid))
        if num_behind > 0:
            warnings.warn(f"PinholeCamera.project_camera3Dxyz_to_imageUV: {num_behind} points are behind the camera (Z < 0), they will be set to NaN in output", RuntimeWarning)
        if num_close > 0:
            warnings.warn(f"PinholeCamera.project_camera3Dxyz_to_imageUV: {num_close} points are very close to the camera (0 <= Z <= {eps}), they will be set to NaN in output", RuntimeWarning)
        if num_invalid > 0:
            warnings.warn(f"PinholeCamera.project_camera3Dxyz_to_imageUV: there are a total of {num_invalid} invalid points (behind or very close to the camera)", RuntimeWarning)
        if num_valid == 0:
            raise ValueError("PinholeCamera.project_camera3Dxyz_to_imageUV: no valid points with Z > 0 found")
        x       = np.full_like(ZZ, np.nan, dtype = np.float32)
        y       = np.full_like(ZZ, np.nan, dtype = np.float32)
        x[valid] = XX[valid] / ZZ[valid]
        y[valid] = YY[valid] / ZZ[valid]   
        r2      = x * x  + y* y # r^2
        r4      = r2 * r2 # r^4
        r6      = r2 * r4 # r^6
        Kmat    = self.calc_Kmat()
        fx      = Kmat[0, 0]
        fy      = Kmat[1, 1]
        cx      = Kmat[0, 2]
        cy      = Kmat[1, 2]
        skew    = Kmat[0, 1] 
        x_r     = x * (1 + self.k1*r2 + self.k2*r4 + self.k3*r6)
        y_r     = y * (1 + self.k1*r2 + self.k2*r4 + self.k3*r6)
        x_d     = x_r + 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
        y_d     = y_r + self.p1*(r2 + 2*y*y) + 2*self.p2*x*y
        uu      = fx * x_d + skew * y_d + cx # skew will be zero in pinhole model, included for completeness
        vv      = fy * y_d + cy
        UV      = np.stack([uu, vv], axis = -1) # this will have shape (..., 2)
        return UV

    def undistort_normalized_points(
                                        self, 
                                        xd: NDArray[np.float32], 
                                        yd: NDArray[np.float32],
                                        iters: int = 25,
                                        tol: float = 1e-8
                                    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """ 
        Run iterative undistortion on normalized image coordinates (xd, yd) to get (x, y)

        Solving x and y from distorted coordinates (xd, yd) using iterative method: 
        x_d = x * (1 + k1 r² + k2 r⁴ + k3 r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
        y_d = y * (1 + k1 r² + k2 r⁴ + k3 r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
        """
        x   = xd.astype(np.float32, copy = True)
        y   = yd.astype(np.float32, copy = True)
        for _ in range(iters):
            x_prev  = x.copy()
            y_prev  = y.copy()
            r2      = x*x + y*y
            r4      = r2*r2
            r6      = r2*r4
            radial  = 1.0 + self.k1*r2 + self.k2*r4 + self.k3*r6
            # tangential terms (Brown–Conrady)
            dx      = 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
            dy      = self.p1*(r2 + 2*y*y) + 2*self.p2*x*y
            # forward model: xd = x*radial + dx  =>  solve for x,y
            x       = (xd - dx) / radial
            y       = (yd - dy) / radial
            if (np.max(np.abs(x - x_prev)) < tol) and (np.max(np.abs(y - y_prev)) < tol):
                break
        return x, y

    def project_imageUV_to_cameraRay(self, uv: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """ 
        Project image pixel coordinates (u, v) to normalized 3D camera coordinates (X, Y, Z = 1) 

        Args:
            uv: (..., 2) array of 2D points in image pixel coordinates
        
        Returns:
            rays: (..., 3) array of undistored normalized 3D points in camera coordinates (X, Y, Z = 1)
            unit_rays: (..., 3) array of undistored unit-length 3D points in camera coordinates (X, Y, Z)
        """
        # sanity check input
        uv      = np.asarray(uv, dtype = np.float32)

        if uv.shape[-1] != 2:
            raise ValueError("uv must have shape (..., 2)")
        
        u       = uv[..., 0]
        v       = uv[..., 1]
        Kmat    = self.calc_Kmat()
        fx      = Kmat[0, 0]
        fy      = Kmat[1, 1]
        cx      = Kmat[0, 2]
        cy      = Kmat[1, 2]
        skew    = Kmat[0, 1]
        # do the inversion 
        y_d     = ( v - cy ) / fy
        x_d     = ( u - cx - skew * y_d ) / fx
        if any(abs(c) > 0 for c in (self.k1, self.k2, self.k3, self.p1, self.p2)):
            x,y = self.undistort_normalized_points(x_d, y_d)
        else:
            x   = x_d
            y   = y_d
        ones    = np.ones_like(x)
        # perspective back to normalized camera coordinates
        XYZ     = np.stack([x, y, ones], axis = -1) # shape (..., 3)
        XYZU    = XYZ / np.linalg.norm(XYZ, axis = -1, keepdims = True) # normalize to unit length
        return XYZ, XYZU