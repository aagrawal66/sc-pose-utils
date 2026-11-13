""" Public package interface for sc_pose """

from sc_pose.mathtils.quaternion import q_norm, q2rotm, rotm2q, xyzw_to_wxyz, wxyz_to_xyzw
from sc_pose.sensors.camera import PinholeCamera


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


def __getattr__(name):
    """
    load optional OpenCV projection helpers when requested

    Inputs:
    name (str): requested package export

    Outputs:
    value: public projection helper
    """
    if name in ('PoseProjector', 'draw_uv_points_on_image'):
        from sc_pose.sensors import camera_projections

        return getattr(camera_projections, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
