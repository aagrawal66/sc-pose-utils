""" Public package interface for sc_pose """

from sc_pose.mathtils.quaternion import q_norm, q2rotm, rotm2q, xyzw_to_wxyz, wxyz_to_xyzw
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


# the syntax below allows users to do when installing this repo as a package:
# from sc_pose import q_norm
# instead of:
# from sc_pose.mathtils.quaternion import q_norm
# and from sc_pose import * will import q_norm and other functions

__all__ = [
            "q_norm",
            "q2rotm",
            "rotm2q",
            "xyzw_to_wxyz",
            "wxyz_to_xyzw",
            "PinholeCamera",
            "PoseProjector",
            "draw_uv_points_on_image"
        ]
