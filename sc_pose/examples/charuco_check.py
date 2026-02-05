""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
import numpy as np
from pathlib import Path
import pandas as pd
import json 
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import rotm2q
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
test_name       = "2025_11_14_003"
calib_yaml      = HERE / "artifacts" / test_name / "calibration.yaml" 
poses_json      = HERE / "artifacts" / test_name / "poses.json"
charuco_json    = HERE / "artifacts" / test_name / "calibration.json"
res_path        = HERE / 'results'
calib_flag      = True
##################################### Inputs #####################################

# setup keys
cj_key_charuco      = "charuco_board"
cj_key_3d_points    = "corners_3d"
pj_key_poses        = "poses"
pj_key_Rotm         = "R_B_to_C" # active rotation matrix from board to camera
pj_key_tr           = "t_C_Co_to_Bo"  # translation from camera origin to board origin in camera frame
pj_key_imgs         = "image_path"

Rmats               = [] 
trs                 = []
img_paths           = []

with open(poses_json, 'r') as f:
    pose_data       = json.load(f)
    Rmats           = [np.array(p[pj_key_Rotm]) for p in pose_data[pj_key_poses]]
    trs             = [np.array(p[pj_key_tr]) for p in pose_data[pj_key_poses]]
    img_paths       = [Path(p[pj_key_imgs]) for p in pose_data[pj_key_poses]]
trs_array           = np.array(trs)  # (N, 3)
Rmats_array         = np.array(Rmats)  # (N, 3, 3)

with open(charuco_json, 'r') as f:
    charuco_data    = json.load(f)
    charuco_X_B     = {
                        int(k): np.array(v)
                        for k, v in charuco_data[cj_key_charuco][cj_key_3d_points].items()
                    }
    target_BFF_pts  = np.stack([v for _, v in sorted(charuco_X_B.items())])

pdb.set_trace()


# setup camera 
cam     = PinholeCamera.from_config(str(calib_yaml))
cam.print_state()
Kmat    = cam.calc_Kmat()
print("Camera K-matrix:\n", Kmat)
proj    = PoseProjector(camera = cam)


for i, img_path in enumerate(img_paths):
    
    img_base        = img_path.stem
    translation     = trs_array[i]
    R_B_to_C        = Rmats_array[i]
    pdb.set_trace()
    quaternion      = rotm2q(R_B_to_C)
    r_Co2To_CAM     = translation
    q_TARGET_2_CAM  = quaternion

    uv_cam  = proj.project_to_image(
                                        q_TARGET_2_CAM = q_TARGET_2_CAM,
                                        r_Co2To_CAM = r_Co2To_CAM,
                                        points_xyz_TARGET = target_BFF_pts 
                                    )
    
    # fill this in with image info
    img     = draw_uv_points_on_image(
                                    img_or_path = img_path,
                                    points_uv = uv_cam,
                                    point_color = (0, 0, 255),
                                    point_radius = 3, 
                                    point_thickness = -1
                                 
                                )
    cv2.imwrite( res_path / f"charuco_check_{img_base}.png", img)
    

pdb.set_trace()
