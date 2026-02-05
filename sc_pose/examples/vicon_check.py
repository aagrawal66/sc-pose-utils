""" An example script of how to use Pinhole Camera model with Vicon data """
import numpy as np
from pathlib import Path
import pandas as pd
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import q2trfm, q_mult_shu, q_conj
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
# Basler ace acA4112-30uc camera specs: https://www.baslerweb.com/en-us/shop/aca4112-30uc/
img_width       = 4096 
img_height      = 3000
focal_length    = 25.0  # in mm 
sensor_width    = 14.13  # in mm
sensor_height   = 10.35 # in mm

cali_yaml       = HERE / "artifacts" / "calibration.yaml"
calib_img_dir   = HERE / "artifacts" / "calibration_images"
vicon_csv       = HERE / "artifacts" / "vicon_data.csv"
csv_keys        = {
                'x_target': 'soho_x',
                'y_target': 'soho_y',
                'z_target': 'soho_z',
                'qw_target': 'soho_qw',
                'qx_target': 'soho_qx',
                'qy_target': 'soho_qy',
                'qz_target': 'soho_qz',
                'x_cam': 'basler_x',
                'y_cam': 'basler_y',
                'z_cam': 'basler_z',
                'qw_cam': 'basler_qw',
                'qx_cam': 'basler_qx',
                'qy_cam': 'basler_qy',
                'qz_cam': 'basler_qz'
                }

img_filepaths   = list(calib_img_dir.glob("*.png"))
res_path        = HERE / 'results'
target_BFF_pts  = np.array([[0.0, 0.0, 0.0]]) # (1, 3)
calib_flag      = True
calib_flag      = False
##################################### Inputs #####################################


calib_data          = Path(cali_yaml).resolve(strict = True)
# create camera object
cam     = PinholeCamera(
                            sensor_width_mm = sensor_width,
                            sensor_height_mm = sensor_height,
                            image_width_px  = img_width,
                            image_height_px = img_height,
                            focal_length_mm = focal_length
                        )
cam.print_state()

# set calibration
if calib_flag:
    cam.set_calibration_yaml(calib_data)
    cam.print_state()

proj    = PoseProjector(camera = cam)
data                = pd.read_csv(vicon_csv)
for index, row in data.iterrows():
    # extract image path
    img_path        = img_filepaths[index]
    # extract vicon poses
    r_Vo2Co_VICON   = np.array([row[csv_keys['x_cam']], row[csv_keys['y_cam']], row[csv_keys['z_cam']]])
    q_VICON_2_CAM   = np.array([row[csv_keys['qw_cam']], row[csv_keys['qx_cam']], row[csv_keys['qy_cam']], row[csv_keys['qz_cam']]])
    r_Vo2To_VICON   = np.array([row[csv_keys['x_target']], row[csv_keys['y_target']], row[csv_keys['z_target']]])
    q_VICON_2_TARGET= np.array([row[csv_keys['qw_target']], row[csv_keys['qx_target']], row[csv_keys['qy_target']], row[csv_keys['qz_target']]])
    # we want q_TARGET_2_CAM and r_Co2To_CAM
    q_TARGET_2_CAM  = q_mult_shu(q2 = q_VICON_2_CAM, q1 = q_conj(q_VICON_2_TARGET))
    r_Co2To_CAM     = q2trfm(q_VICON_2_CAM) @ ( r_Vo2To_VICON - r_Vo2Co_VICON)

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
    cv2.imwrite( res_path / f"vicon_check_{index+1:02d}.png", img)
    

# pdb.set_trace()
