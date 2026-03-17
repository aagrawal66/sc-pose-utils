""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
from pathlib import Path
import json 
import sys
import cv2
from pyparsing import line
from pyparsing import line
from scipy.spatial.transform import Rotation as R
import os 
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pdb

# local imports
from ..mathtils.quaternion import rotm2q, q2rotm, q2trfm, q_mult_shu, q_conj
from ..sensors.camera import PinholeCamera
from ..sensors.camera_projections import PoseProjector, draw_uv_points_on_image


################################ Helper Functions ################################
def Trfm_4x4_inverse(T4x4: NDArray) -> NDArray:
    """ 
    Invert a 4x4 homogeneous transformation matrix
    of the form:
    [ R | t ]
    [ 0 | 1 ]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    The inverse is given by:
    [ R^T | -R^T t ]
    [ 0   | 1       ] 
    """
    Rmat            = T4x4[:3,:3]
    tvec            = T4x4[:3,3]
    Rmat_inv        = Rmat.T
    t_inv           = -Rmat_inv @ tvec
    T4x4_inv        = np.eye(4)
    T4x4_inv[:3,:3] = Rmat_inv
    T4x4_inv[:3,3]  = t_inv
    return T4x4_inv

def _process_vicon_offset_v01(row, T_CvC, T_TvT):
    """ 
    Process vicon data using 4x4 Homogeneous transformation matrices
    
    T_CvC: transformation from Vicon camera frame to true camera frame
    T_TvT: transformation from Vicon target frame to true target frame
    """
    soho_x          = float( row[1] ) * 1E-3
    soho_y          = float( row[2] ) * 1E-3
    soho_z          = float( row[3] ) * 1E-3
    soho_qw         = float( row[4] )
    soho_qx         = float( row[5] )
    soho_qy         = float( row[6] )
    soho_qz         = float( row[7] )
    soho_VTv        = np.array( [ soho_x, soho_y, soho_z ] )
    soho_quatVTv    = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] )

    cam_x           = float( row[8] )  * 1E-3
    cam_y           = float( row[9] )  * 1E-3
    cam_z           = float( row[10] ) * 1E-3
    cam_qw          = float( row[11] )
    cam_qx          = float( row[12] )
    cam_qy          = float( row[13] )
    cam_qz          = float( row[14] )
    cam_VCv         = np.array( [ cam_x, cam_y, cam_z ] )
    cam_quatVCv     = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] )

    
    # convert Vicon information into a transformation matrix from Vicon frame to the Vicon target frame
    R_VTv           = R.from_quat( soho_quatVTv, scalar_first = True ).as_matrix()
    T_VTv           = np.eye(4)
    T_VTv[:3,:3]    = R_VTv
    T_VTv[:3,3]     = soho_VTv

    # convert Vicon information into a transformation matrix from Vicon frame to the Vicon camera frame
    R_VCv           = R.from_quat( cam_quatVCv, scalar_first = True ).as_matrix()
    T_VCv           = np.eye(4)
    T_VCv[:3,:3]    = R_VCv
    T_VCv[:3,3]     = cam_VCv

    # homogenous transformation matrix from vicon camera frame to vicon target frame
    T_CvTv          = Trfm_4x4_inverse(T_VCv) @ T_VTv
    # homogenous transformation matrix from true camera frame to true target frame
    T_CT            = Trfm_4x4_inverse(T_CvC) @ T_CvTv @ Trfm_4x4_inverse(T_TvT)

    # extracting quaternion and translation from homogeneous transformation matrix
    q_CT            = R.from_matrix( T_CT[:3,:3] ).as_quat()  # [x, y, z, w]
    q_wxyz          = np.roll(q_CT, 1)  
    r_CT            = T_CT[:3,3]

    # corrected frames
    # attitude transformation from camera to target, translation from camera to target in camera frame
    q_CAMERA_2_TARGET   = q_wxyz
    r_Co2To_CAMERA     = r_CT

    return q_CAMERA_2_TARGET, r_Co2To_CAMERA

def _process_vicon_offset_v02(row, T_CvC, T_TvT):
    """
    Process vicon data using passive rotation matricies and translation vectors
    
    T_CvC: transformation from Vicon camera frame to true camera frame
    T_TvT: transformation from Vicon target frame to true target frame
    """  
    # assuming error in Vicon frame definitions
    # CT: Camera Tilde frame
    # TT: Target Tilde frame
    # VICON: Vicon frame
    # we have transformations from VICON to both CT and TT
    # we want to find the transformation from CT to TT
    r_Vo2CTo_VICON  = np.array([row[vicon_keys['x_cam']], row[vicon_keys['y_cam']], row[vicon_keys['z_cam']]])
    q_VICON_2_CT    = np.array([row[vicon_keys['qw_cam']], row[vicon_keys['qx_cam']], row[vicon_keys['qy_cam']], row[vicon_keys['qz_cam']]])
    r_Vo2TTo_VICON  = np.array([row[vicon_keys['x_target']], row[vicon_keys['y_target']], row[vicon_keys['z_target']]])
    q_VICON_2_TT    = np.array([row[vicon_keys['qw_target']], row[vicon_keys['qx_target']], row[vicon_keys['qy_target']], row[vicon_keys['qz_target']]])
    
    # # we want q_TARGET_TILDE_2_CAM_TILDE and r_CTo2To_CAM_TILDE
    # q_TT_2_CT       = q_mult_shu(q2 = q_VICON_2_CT, q1 = q_conj(q_VICON_2_TT))
    # r_CTo2TTo_CT    = q2trfm(q_VICON_2_CT) @ ( r_Vo2TTo_VICON - r_Vo2CTo_VICON)
    
    # extract the rotation and translation from the transformations from vicon to camera/target, then apply the correction from vicon to true frames
    # camera
    r_CTo2Co_C      = T_CvC[:3, -1]
    T_CT_2_C        = T_CvC[:3, :3]
    R_CT_2_C        = T_CT_2_C.T 
    q_CT_2_C         = rotm2q(R_CT_2_C)
    # target
    r_TTo2To_T      = T_TvT[:3, -1]
    T_TT_2_T        = T_TvT[:3, :3]
    R_TT_2_T        = T_TT_2_T.T
    q_TT_2_T        = rotm2q(R_TT_2_T)

    # calculate the quaterion relating true camera frame to true target frame    
    # go from Vicon to Camera Tilde then Camera Tilde to true Camera to get Vicon to true Camera
    q_VICON_2_C     = q_mult_shu(q2 = q_CT_2_C, q1 = q_VICON_2_CT)
    T_VICON_2_C       = q2trfm(q_VICON_2_C)
    # go from Vicon to Target Tilde then Target Tilde to true Target to get Vicon to true Target
    q_VICON_2_T     = q_mult_shu(q2 = q_TT_2_T, q1 = q_VICON_2_TT)
    # go from Camera To Vicon then Vicon to Target to get Camera to Target
    T_VICON_2_T     = q2trfm(q_VICON_2_T)
    q_C_2_T         = q_mult_shu(q2 = q_VICON_2_T, q1 = q_conj(q_VICON_2_C))

    # calculate the translation from true camera to true target frame in the true camera frame
    r_Vo2Co_C       = T_VICON_2_C @ (r_Vo2CTo_VICON +  T_VICON_2_C.T @ r_CTo2Co_C )
    r_Vo2To_C       = T_VICON_2_C @ (r_Vo2TTo_VICON +  T_VICON_2_T.T @ r_TTo2To_T )
    r_Co2To_C       = r_Vo2Co_C - r_Vo2To_C
    
    q_CAMERA_2_TARGET   = q_C_2_T
    r_Co2To_CAMERA      = r_Co2To_C
    return q_CAMERA_2_TARGET, r_Co2To_CAMERA

def _load_offset_estimates(offset_data_path, offset_keys):
    # extract offset estimates from json
    with open(offset_data_path, 'r') as f:
        offset_json                         = json.load(f)
    offset_key_list                     = list(offset_keys)
    CV_C_key                            = offset_key_list[0] # "Trf_4x4_CamViconDef_Cam"
    TV_T_key                            = offset_key_list[1] # "Trf_4x4_TargetViconDef_Target"
    Trf4x4_CAMVICON_2_CAM_TRUE          = np.array( offset_json[CV_C_key] )
    Trf4x4_TARGETVICON_2_TARGET_TRUE    = np.array( offset_json[TV_T_key] )
    
    return Trf4x4_CAMVICON_2_CAM_TRUE, Trf4x4_TARGETVICON_2_TARGET_TRUE

def _process_opencv_pose_v01(rvec, tvec):
    """" processing opencv pose estimates using scipy """
    rot                 = R.from_rotvec(rvec).as_matrix()
    Rt                  = R.from_matrix(rot)
    q_xyzw              = Rt.as_quat()
    q_TC                = np.roll(q_xyzw, 1)
    r_CT                = tvec
    q_TARGET_2_CAMERA   = q_TC
    r_Co2To_CAMERA      = r_CT
    return q_TARGET_2_CAMERA, r_Co2To_CAMERA

def _process_opencv_pose_v02(rvec, tvec):
    """ processing opencv pose estimates using cv2 """
    R_T_to_C, _         = cv2.Rodrigues(rvec)
    q_T_to_C            = rotm2q(R_T_to_C)
    r_Co2To_C           = tvec
    q_TARGET_2_CAMERA   = q_T_to_C
    r_Co2To_CAMERA      = r_Co2To_C
    return q_TARGET_2_CAMERA, r_Co2To_CAMERA


################################ Helper Functions ################################


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

# offset keys
offset_keys     = {
                    'Trf_4x4_CamViconDef_Cam': 'T_CvC',
                    'Trf_4x4_TargetViconDef_Target': 'T_TvT'
                }
##################################### Inputs #####################################

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

# matched loop
for i, row in opencv_df.iterrows():
    img_name    = row['frame']
    img_base    = Path(img_name).stem 
    img_num     = int(img_base.split("_")[-1])
    print(f"Processing row {i}: {img_name}")
    img_path    = image_folder / img_name 
    img_outpath = res_path / f"opencv_reproj_{img_base}.png" 
    pdb.set_trace()
    # extract opencv outputted rotation and translation
    rvec    = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
    tvec    = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])

    q_T_2_C, r_Co2To_C  = _process_opencv_pose_v01(rvec, tvec)
    R_T_to_C            = q2rotm(q_T_2_C)
    pdb.set_trace()

    Rmats.append(R_T_to_C)
    trs.append(r_Co2To_C)
    img_paths.append(img_path)
    img_nums.append(img_num)

    # project 3D points to 2D image coordinates
    uv_cam  = proj.classless_pinhole_project_to_image(
                                                        q_TARGET_2_CAM    = q_T_2_C,
                                                        r_Co2To_CAM       = r_Co2To_C,
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

# reproject all loop based on the pose estimates from vicon

print(f'Results located at: {res_path}')

