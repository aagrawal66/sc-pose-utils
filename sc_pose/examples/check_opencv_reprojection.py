""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
from pathlib import Path
import json 
import cv2
from scipy.spatial.transform import Rotation as R
import os 
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pdb

# local imports
from sc_pose.math_utils.quaternion import rotm2q, q2rotm, q2trfm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE                = Path(__file__).parent.resolve()
##################################### Inputs #####################################
data_folder         = HERE / "artifacts" / "offset" / "expm_001"
data_name           = data_folder.name
image_folder        = data_folder / "images"
skip_csv_header     = True
kps_file            = HERE / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_1000.json" # origin shifted to edge
# kps_centered_file   = 
opencv_pose_est     = data_folder / "camera_poses.csv"
vicon_pose_est      = data_folder / "vicon_data.csv"
# calib_data          = data_folder / "calibration.yaml" # wrong, ANAND EDIT, what?
calib_data          = data_folder / "calibration_2025_11_14.yaml" # a different calibration file
offset_data         = data_folder / "offset_results.json"
# setup keys
res_path            = HERE / "results" / data_name

# camera parameters 
img_width       = 4096 
img_height      = 3000
focal_length    = 25.0  # in mm 
sensor_width    = 14.13  # in mm
sensor_height   = 10.35 # in mm

# opencv_keys
opencv_keys     = {
                    'frame': 'frame',
                    'rvec_x': 'rvec_x',
                    'rvec_y': 'rvec_y',
                    'rvec_z': 'rvec_z',
                    'tvec_x': 'tvec_x',
                    'tvec_y': 'tvec_y',
                    'tvec_z': 'tvec_z'
                }
# vicon keys
vicon_keys      = {
                    'frame': 'image_number',
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
##################################### Inputs #####################################


################################ Helper Functions ################################
def Trfm_4x4_inverse(T: NDArray) -> NDArray:
    """ Invert a homogeneous transformation matrix """
    R               = T[:3,:3]
    t               = T[:3,3]
    R_inv           = R.T
    t_inv           = -R_inv @ t
    T_inv           = np.eye(4)
    T_inv[:3,:3]    = R_inv
    T_inv[:3,3]     = t_inv
    return T_inv
################################ Helper Functions ################################






# make results path
os.makedirs(res_path, exist_ok = True)


# create projection object
cam     = PinholeCamera(
                            sensor_width_mm = sensor_width,
                            sensor_height_mm = sensor_height,
                            image_width_px  = img_width,
                            image_height_px = img_height,
                            focal_length_mm = focal_length
                        )
# cam.print_state()

# set calibration
cam.set_calibration_yaml(calib_data)
Kmat_cal    = cam.calc_Kmat()
dist_coeffs = cam._dist_coeffs_as_array()
proj        = PoseProjector(camera = cam)




# kps from drew/anand
with open(kps_file, 'r') as f:
    kps_mm          = np.array( json.load(f) ) # obj in mm
    kps_m           = kps_mm / 1e3
    target_BFF_pts  = kps_m

# add origin to start of kps
target_BFF_pts_with_origin  = np.vstack( (np.zeros((1,3)), target_BFF_pts) ) 

opencv_df   = pd.read_csv(opencv_pose_est)
vicon_df    = pd.read_csv(vicon_pose_est)

# extract opencv pose estimates

# extract vicon pose estimates


# load both opencv and vicon data into dataframes
# iterate through rows
Rmats           = [] 
trs             = []
img_paths       = []
img_nums        = []


for i, row in opencv_df.iterrows():
    img_name    = row['frame']
    img_base    = Path(img_name).stem 
    img_num     = int(img_base.split("_")[-1])
    print(f"Processing row {i}: {img_name}")
    img_path    = image_folder / img_name 
    img_outpath = res_path / f"opencv_reproj_{img_base}.png" 
    
    # extract opencv outputted rotation and translation
    rvec    = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
    tvec    = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])

    # convert to left scalar first quaternion
    # covert rotation vector to rotation matrix, active rotation from target to camera
    rot     = R.from_rotvec(rvec).as_matrix() 
    Rt      = R.from_matrix(rot)
    q_xyzw  = Rt.as_quat()
    q_TC    = np.roll(q_xyzw, 1)
    # Translation (Camera to Target in Camera frame)
    r_CT    = tvec

    R_T_to_C, _     = cv2.Rodrigues(rvec)
    q_T_to_C        = rotm2q(R_T_to_C)
    T_Co2To_C       = tvec

    Rmats.append(R_T_to_C)
    trs.append(T_Co2To_C)
    img_paths.append(img_path)
    img_nums.append(img_num)

    # project 3D points to 2D image coordinates
    uv_cam  = proj.classless_pinhole_project_to_image(
                                                        q_TARGET_2_CAM    = q_TC,
                                                        r_Co2To_CAM       = r_CT,
                                                        Kmat              = Kmat_cal,
                                                        BC_dist_coeffs    = dist_coeffs,
                                                        points_xyz_TARGET = target_BFF_pts_with_origin 
                                                    )
    
    # draw the points and save the verification image
    # Using str(img_path) to ensure compatibility with cv2/drawing functions
    img_out = draw_uv_points_on_image(
                                        img_or_path     = str(img_path),
                                        points_uv       = uv_cam,
                                        point_color     = (0, 0, 255),
                                        point_radius    = 15, 
                                        point_thickness = 2
                                    )
    # highlight origin point in a different color
    uv_origin   = uv_cam[0]  # assuming the first point is the origin
    img_out     = draw_uv_points_on_image(
                                            img_or_path     = img_out,
                                            points_uv       = uv_origin.reshape(1, 2),
                                            point_color     = (0, 255, 0),  # Green for origin
                                            point_radius    = 20, 
                                            point_thickness = 3
                                        )
    cv2.imwrite(img_outpath, img_out)

trs_array       = np.array(trs)  # (N, 3)
Rmats_array     = np.array(Rmats)  # (N, 3, 3)
img_files       = sorted(os.listdir(image_folder))

print(f'Results located at: {res_path}')


