""" An example script of how to use the Pinhole Camera model with synthetic image data """
import numpy as np
from pathlib import Path
import pandas as pd
import json 
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import q2trfm, q_mult_shu, q_conj
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
target_name     = "soho"
target_kps      = HERE / "artifacts" / "target_kps" / "soho_128.npy"
target_kps      = HERE / "artifacts" / "target_kps" / "soho_1000.npy"  
img_list        = [
                    HERE / "artifacts" / "synthetic_images" / "image_09973.png",
                    HERE / "artifacts" / "synthetic_images" / "image_09975.png"
                ]
meta_list      = [
                    HERE / "artifacts" / "synthetic_meta" / "meta_09973.json",
                    HERE / "artifacts" / "synthetic_meta" / "meta_09975.json"
                ]
img_width       = 1024  
img_height      = 1024
focal_length    = 50 # in mm 
sensor_width    = 36 # in mm
sensor_height   = 24 # in mm


meta_keys        = 'pose'
res_path        = HERE / 'results'
calib_flag      = True
calib_flag      = False
##################################### Inputs #####################################


# create camera object
cam     = PinholeCamera(
                            sensor_width_mm = sensor_width,
                            sensor_height_mm = sensor_height,
                            image_width_px  = img_width,
                            image_height_px = img_height,
                            focal_length_mm = focal_length
                        )
cam.print_state()
Kmat    = cam.calc_Kmat()
print("Camera K-matrix:\n", Kmat)

target_BFF_pts  = np.load(target_kps)  # (N, 3)
proj        = PoseProjector(camera = cam)
for img_path, meta_path in zip(img_list, meta_list):
    img_base        = img_path.stem
    # load metadata
    with open(meta_path, 'r') as f:
        meta_dict   = json.load(f)
    pose            = meta_dict[meta_keys]
    translation     = np.array( pose[:3] )
    quaternion      = np.array( pose[3:] ) 
    r_Co2To_CAM     = translation
    q_TARGET_2_CAM  = quaternion
    # # we want q_TARGET_2_CAM and r_Co2To_CAM
    # q_TARGET_2_CAM  = q_mult_shu(q2 = q_VICON_2_CAM, q1 = q_conj(q_VICON_2_TARGET))
    # r_Co2To_CAM    = q2trfm(q_VICON_2_CAM) @ ( r_Vo2To_VICON - r_Vo2Co_VICON)

    # fill this in with vicon info
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
                                    point_radius = 10, 
                                    point_thickness = -1
                                 
                                )
    cv2.imwrite( res_path / f"synthetic_check_{img_base}.png", img)
    

pdb.set_trace()
