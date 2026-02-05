""" An example script of how to use the Pinhole Camera model with synthetic image data that have all the same camera parameters """
import numpy as np
from pathlib import Path
import pandas as pd
import json 
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import q_norm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
# create a list of synthetic images, their metadata, and target keypoints

target_name     = ["soho", "soho", "cygnus", "soho", "soho"]
target_kps1     = HERE / "artifacts" / "target_kps" / "soho_1000.npy"
target_kps2     = HERE / "artifacts" / "target_kps" / "soho_128.npy"
target_kps3     = HERE / "artifacts" / "target_kps" / "cygnus_keypoints.npy"
img_list        = [
                    HERE / "artifacts" / "synthetic_images" / "image_09973.png",
                    HERE / "artifacts" / "synthetic_images" / "image_09975.png",
                    HERE / "artifacts" / "synthetic_images" / "image_001080.png",
                    HERE / "artifacts" / "synthetic_images" / "image_00056.png",
                    HERE / "artifacts" / "synthetic_images" / "image_00000.png"
                ]
meta_list      = [
                    HERE / "artifacts" / "synthetic_meta" / "meta_09973.json",
                    HERE / "artifacts" / "synthetic_meta" / "meta_09975.json",
                    HERE / "artifacts" / "synthetic_meta" / "meta_001080.json",
                    HERE / "artifacts" / "synthetic_meta" / "meta_00056.json",
                    HERE / "artifacts" / "synthetic_meta" / "meta_00000.json"
                ]
target_kps_list = [
                    target_kps1,
                    target_kps1, 
                    target_kps3,
                    target_kps1,
                    target_kps1
                ]
   
img_widths      = [1024, 1024, 1024, 512, 512]  
img_heights     = [1024, 1024, 1024, 512, 512]
focal_length    = 50 # in mm 
sensor_width    = 36 # in mm
sensor_height   = 24 # in mm


meta_quat_key   = ['pose', 'pose', 'pose', 'pose', 'pose']
meta_tr_key     = ['pose', 'pose', 'translation', 'pose', 'pose']
res_path        = HERE / 'results'
calib_flag      = True
calib_flag      = False
num_kps         = 200
##################################### Inputs #####################################



for target_name, img_path, meta_path, target_kps_path, img_width, img_height, meta_quat_key, meta_tr_key in zip(target_name, img_list, meta_list, target_kps_list, img_widths, img_heights,  meta_quat_key, meta_tr_key):
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
    proj        = PoseProjector(camera = cam)

    
    target_BFF_all  = np.load(target_kps_path)  # (N, 3)
    target_BFF_pts  = target_BFF_all[:num_kps, :]
    img_base        = img_path.stem
    # load metadata
    with open(meta_path, 'r') as f:
        meta_dict   = json.load(f)
    
    if meta_quat_key == 'pose' and meta_tr_key == 'pose':
        pose            = meta_dict[meta_quat_key]
        translation     = np.array( pose[:3] )
        quaternion      = q_norm(np.array( pose[3:] ))
    else:
        # pdb.set_trace()
        translation     = np.array( meta_dict[meta_tr_key] )
        quaternion      = q_norm(np.array( meta_dict[meta_quat_key] ))
    
    r_Co2To_CAM     = translation
    q_TARGET_2_CAM  = quaternion
    # # we want q_TARGET_2_CAM and r_Co2To_CAM
    # q_TARGET_2_CAM  = q_mult_shu(q2 = q_VICON_2_CAM, q1 = q_conj(q_VICON_2_TARGET))
    # r_Co2To_CAM    = q2trfm(q_VICON_2_CAM) @ ( r_Vo2To_VICON - r_Vo2Co_VICON)

    uv_cam  = proj.project_to_image(
                                        q_TARGET_2_CAM = q_TARGET_2_CAM,
                                        r_Co2To_CAM = r_Co2To_CAM,
                                        points_xyz_TARGET = target_BFF_pts 
                                    )
    
    uv_cam2 = proj.classless_pinhole_project_to_image(
                                                        q_TARGET_2_CAM, 
                                                        r_Co2To_CAM, 
                                                        Kmat,
                                                        target_BFF_pts
                                                 ) 
    # fill this in with image info
    img     = draw_uv_points_on_image(
                                    img_or_path = img_path,
                                    points_uv = uv_cam,
                                    point_color = (0, 0, 255),
                                    point_radius = 3, 
                                    point_thickness = -1
                                 
                                )
    cv2.imwrite( res_path / f"synthetic_check_{img_base}.png", img)
    

pdb.set_trace()
