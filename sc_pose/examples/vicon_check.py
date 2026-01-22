""" An example script of how to use Pinhole Camera model with Vicon data """
import numpy as np
from pathlib import Path
import pdb

# local imports
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
# Basler ace acA4112-30uc camera specs: https://www.baslerweb.com/en-us/shop/aca4112-30uc/
img_width       = 4096 
img_height      = 3000
focal_length    = 24.0  # in mm, or 25, need to check 
sensor_width    = 14.13  # in mm
sensor_height   = 10.35 # in mm

cali_yaml       = HERE / "artifacts" / "calibration.yaml"

# # vicon data 
# img_path            = HERE / "artifacts" / "vicon_testimg_001.png"
# r_Vo2Co_VICON       = 
# r_Vo2To_VICON       =
# q_VICON_2_CAM       = 
# q_VICON_2_TARGET    =  

# target_BFF_pts  = np.array([0, 0, 0])
##################################### Inputs #####################################

# resolve paths
calib_data  = Path(cali_yaml).resolve(strict = True)

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
cam.set_calibration_yaml(calib_data)
cam.print_state()

proj    = PoseProjector(camera = cam)
# fill this in with vicon info
# uv_cam  = proj.project_to_image(
#                                     q_TARGET_2_CAM = ,
#                                     r_Co2To_CAM = ,
#                                     points_xyz_TARGET = target_BFF_pts 
#                                 )

# # fill this in with image info
# img     = draw_uv_points_on_image(
#                                     img_or_path = ,
#                                     points_uv = uv_cam, 
#                                 )
pdb.set_trace()