""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
import numpy as np
from pathlib import Path
import json 
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import rotm2q
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
data_file       = HERE / "artifacts" / "soho_reframed_mesh_pose_pack"
meta_data       = data_file / "meta.json"
kps_file        = data_file / "mesh_points_1000.json"
# setup keys
pose_key        = "poses"
index_key       = "index"
img_name_key    = "image_filename"
cam_mat_key     = "K"
trf_mat_key     = "T_C_O"
tr_vec_key      = "t_C_O_m"
res_path        = HERE / 'results'
##################################### Inputs #####################################

# create projection object
proj    = PoseProjector(None)

Rmats               = [] 
trs                 = []
img_paths           = []
with open(kps_file, 'r') as f:
    kps_mm          = np.array( json.load(f) ) # obj in mm
    kps_m           = kps_mm / 1e3
    target_BFF_pts  = kps_m  
with open(meta_data, 'r') as f:
    data        = json.load(f)
    Kmat        = np.array( data[cam_mat_key] )
    pose_data   = data[pose_key]
    img_paths   = [data_file / p[img_name_key] for p in pose_data]
    trs         = [np.array(p[tr_vec_key]) for p in pose_data]
    Rmats       = [np.array(p[trf_mat_key])[:3,:3] for p in pose_data]
trs_array       = np.array(trs)  # (N, 3)
Rmats_array     = np.array(Rmats)  # (N, 3, 3)

for i, img_path in enumerate(img_paths):
    
    img_base        = img_path.stem
    translation     = trs_array[i]
    R_B_to_C        = Rmats_array[i]
    quaternion      = rotm2q(R_B_to_C)
    r_Co2To_CAM     = translation
    q_TARGET_2_CAM  = quaternion

    uv_cam          = proj.classless_pinhole_project_to_image(
                                        q_TARGET_2_CAM = q_TARGET_2_CAM,
                                        r_Co2To_CAM = r_Co2To_CAM,
                                        Kmat = Kmat,
                                        points_xyz_TARGET = target_BFF_pts 
                                    )
    
    # fill this in with image info
    img             = draw_uv_points_on_image(
                                    img_or_path = img_path,
                                    points_uv = uv_cam,
                                    point_color = (0, 0, 255),
                                    point_radius = 5, 
                                    point_thickness = 2
                                 
                                )
    cv2.imwrite( res_path / f"pandas_check_{img_base}.png", img)
    

pdb.set_trace()
