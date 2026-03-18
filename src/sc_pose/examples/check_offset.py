""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
from pathlib import Path
import json 
import cv2
from scipy.spatial.transform import Rotation as R
import os, shutil
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pdb

# local imports
from sc_pose.mathtils.quaternion import rotm2q, q2rotm, q2trfm, q_mult_shu, q_conj
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


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

def _process_vicon_offset_v01(row, T_CvC, T_TvT, vicon_keys):
    """ 
    Process vicon data using 4x4 Homogeneous transformation matrices
    
    T_CvC: transformation from Vicon camera frame to true camera frame
    T_TvT: transformation from Vicon target frame to true target frame

    Return: 
    q_TARGET_2_CAMERA: quaternion representing rotation from true camera frame to true target frame
    r_Co2To_CAMERA: translation from true camera frame to true target frame in the true camera frame

    Note: pose filtering may want q_CAMERA_2_TARGET instead, which is the conjugate of q_TARGET_2_CAMERA

    Notation:
    ^ A T_B is a 4x4 homogeneous transformation matrix that transforms points from frame B to frame A
    [ A^R_B | A^t_{Ao -> B}]: 
    A^R_B is a passive rotation matrix from frame B to frame A, meaning it rotates the coordinate axes of frame B to align with frame A
    A^t_{Ao -> B} is a translation vector from the origin of frame A to the origin of frame B, expressed in frame A's coordinates 
    """
    soho_x          = float(row[vicon_keys['x_target']]) * 1E-3
    soho_y          = float(row[vicon_keys['y_target']]) * 1E-3
    soho_z          = float(row[vicon_keys['z_target']]) * 1E-3
    soho_qw         = float(row[vicon_keys['qw_target']])
    soho_qx         = float(row[vicon_keys['qx_target']])
    soho_qy         = float(row[vicon_keys['qy_target']])
    soho_qz         = float(row[vicon_keys['qz_target']])
    soho_VTv        = np.array( [ soho_x, soho_y, soho_z ] )
    soho_quatVTv    = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] )

    cam_x           = float(row[vicon_keys['x_cam']]) * 1E-3
    cam_y           = float(row[vicon_keys['y_cam']]) * 1E-3
    cam_z           = float(row[vicon_keys['z_cam']]) * 1E-3
    cam_qw          = float(row[vicon_keys['qw_cam']])
    cam_qx          = float(row[vicon_keys['qx_cam']])
    cam_qy          = float(row[vicon_keys['qy_cam']])
    cam_qz          = float(row[vicon_keys['qz_cam']])
    cam_VCv         = np.array( [ cam_x, cam_y, cam_z ] )
    cam_quatVCv     = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] )

    # R in this fcn are passive rotation matrices
    # _ABv means passive rotation from A to B Vicon frame
    
    # general notation:
    # ^A T_B = from B to A
    # ^B T_C = from C to B
    # therefore ^A T_B * ^B T_C = from C to A = ^A T_C
    # so ^{Tv} T_{Cv} = ^{Tv} T_{V} * ^{V} T_{Cv} = ^{Tv} T_{V} * ( ^{Cv} T_{V} )^-1 


    # we have soho_VTv, which is in the V frame 
    # so to build a 4x4 homogenous, we need R_TvV, we have soho_quatVTv, so we will need to take its conjugate and get the resulting passive rotation matrix
    # we will build T_TvV
    T_TvV           = np.eye(4)
    T_TvV[:3, 3]    = soho_VTv
    T_TvV[:3, :3]   = q2trfm(q_conj(soho_quatVTv)) # R_TvV

    # we have cam_VCv, which is in the V frame
    # to build a 4x4 homogenous, we need R_CvV, we have cam_quatVCv, so we will need to take its conjugate and get the resulting passive rotation matrix
    # we will build T_CvV
    T_CvV           = np.eye(4)
    T_CvV[:3, 3]    = cam_VCv
    T_CvV[:3, :3]   = q2trfm(q_conj(cam_quatVCv)) # R_CvV
    
    # we want T_TC: from true target frame to true camera frame
    T_TvCv          = Trfm_4x4_inverse(T_CvV) @ T_TvV
    # full sequence: True Target -> Vicon Target -> Vicon Target -> Vicon Camera -> Vicon Camera -> True Camera 
    T_TC            = T_CvC @ T_TvCv @ Trfm_4x4_inverse(T_TvT)
    
    R_TC            = T_TC[:3, :3]
    q_TC            = rotm2q(R_TC.T) # need to transpose b/c rotm2q assumes active rotation, but we are using passive rotation matrices
    r_Co2To_C       = T_TC[:3, 3]
    
    q_TARGET_2_CAMERA   = q_TC
    r_Co2To_CAMERA      = r_Co2To_C
    
    
    # # # convert Vicon information into a transformation matrix from Vicon frame to the Vicon target frame
    # # # R_VTv           = R.from_quat( soho_quatVTv, scalar_first = True ).as_matrix() #, this is most likley active rotation, need transpose
    # # T_VTv[:3,3]     = soho_VTv # in V frame 

    # #     R_VTv           = q2trfm(soho_quatVTv).T
    # # T_VTv           = np.eye(4)
    # # T_VTv[:3,:3]    = R_VTv


    # # convert Vicon information into a transformation matrix from Vicon frame to the Vicon camera frame
    # # R_VCv           = R.from_quat( cam_quatVCv, scalar_first = True ).as_matrix() #, this is most likley active rotation, need transpose
    # R_VCv           = q2trfm(cam_quatVCv).T
    # T_VCv           = np.eye(4)
    # T_VCv[:3,:3]    = R_VCv
    # T_VCv[:3,3]     = cam_VCv


    # # homogenous transformation matrix from vicon camera frame to vicon target frame
    # # full sequence: Camera Vicon -> Vicon -> Vicon -> Target Vicon = Camera Vicon -> Target Vicon
    # T_CvTv      = T_VTv @ Trfm_4x4_inverse(T_VCv)
    #  # homogenous transformation matrix from true camera frame to true target frame
    #  # full sequence: Camera True -> Camera Vicon -> Camera Vicon -> Target Vicon -> Target Vicon -> Target True = Camera True -> Target True
    # T_CT        = Trfm_4x4_inverse(T_TvT) @ T_CvTv @ Trfm_4x4_inverse(T_CvC)
    # T_TC        = Trfm_4x4_inverse(T_CT)
    # r_Co2To_C   = T_TC[:3, 3]
    # R_TC        = T_TC[:3, :3]
    # q_TC        = q2trfm(R_TC)

    # # # extracting quaternion and translation from homogeneous transformation matrix
    # # q_CT            = R.from_matrix( T_CT[:3,:3] ).as_quat()  # [x, y, z, w]
    # # q_wxyz          = np.roll(q_CT, 1)  
    # # r_CT            = T_CT[:3,3]

    # # corrected frames
    # # attitude transformation from camera to target, translation from camera to target in camera frame
    # q_TARGET_2_CAMERA   = q_wxyz
    # r_Co2To_CAMERA      = r_CT

    return q_TARGET_2_CAMERA, r_Co2To_CAMERA

def _process_vicon_offset_v02(row, T_CvC, T_TvT, vicon_keys):
    """
    Process vicon data using passive rotation matricies and translation vectors
    
    T_CvC: transformation from Vicon camera frame to true camera frame
    T_TvT: transformation from Vicon target frame to true target frame
    """  
    # assuming error in Vicon frame definitions
    # CT: Camera Tilde frame
    # TT: Target Tilde frame
    # V: Vicon frame
    # we have transformations from VICON to both CT and TT
    # we want to find the transformation from CT to TT
    r_Vo2CTo_V  = np.array([row[vicon_keys['x_cam']], row[vicon_keys['y_cam']], row[vicon_keys['z_cam']]]) * 1E-3 # convert from mm to m
    q_V_2_CT    = np.array([row[vicon_keys['qw_cam']], row[vicon_keys['qx_cam']], row[vicon_keys['qy_cam']], row[vicon_keys['qz_cam']]])
    r_Vo2TTo_V  = np.array([row[vicon_keys['x_target']], row[vicon_keys['y_target']], row[vicon_keys['z_target']]]) * 1E-3 # convert from mm to m
    q_V_2_TT    = np.array([row[vicon_keys['qw_target']], row[vicon_keys['qx_target']], row[vicon_keys['qy_target']], row[vicon_keys['qz_target']]])
    Trfm_V_2_CT = q2trfm(q_V_2_CT)
    Trfm_V_2_TT = q2trfm(q_V_2_TT)

    
    # extract the passive rotation and translation from the transformations from Camera Vicon to true Camera and from Target Vicon to true Target
    # camera
    r_Co2CTo_C  = T_CvC[:3, -1]
    Trfm_CT_2_C = T_CvC[:3, :3]
    
    # target
    r_To2TTo_T  = T_TvT[:3, -1]
    Trfm_TT_2_T = T_TvT[:3, :3]

    # we want Trfm_T_2_C
    # this is the transformation from the true target frame to the true camera frame
    # full sequence: true target -> target vicon -> target vicon -> vicon -> vicon -> camera vicon -> camera vicon -> true camera = true target -> camera vicon
    Trfm_TT_CT  = Trfm_V_2_CT.T @  Trfm_V_2_TT.T # from target vicon -> vicon -> vicon -> camera vicon = from target vicon to camera vicon
    Trfm_T_2_C  = Trfm_CT_2_C @ Trfm_TT_CT @ Trfm_TT_2_T.T
    Rotm_T_2_C  = Trfm_T_2_C.T

    # we want r_Co2To_C, the translation from true camera to true target in the true camera frame
    # we have r_Co2CTo_C, the translation from true camera to camera vicon in the true camera frame
    # we need r_TTo2To_C, we have r_To2TTo_T, the translation from true target to target vicon in the true target frame 
    r_To2TTo_C      = Trfm_T_2_C @ (-r_To2TTo_T) 
    # we need r_CTo2TTo_C, the translation from the vicon camera frame to the vicon target frame in the true camera frame
    r_CTo2TTo_V     = r_Vo2TTo_V - r_Vo2CTo_V
    Trfm_V_2_C      = Trfm_CT_2_C @ Trfm_V_2_CT.T
    r_CTo2TTo_C     = Trfm_V_2_C @ r_CTo2TTo_V
    r_Co2To_C       = r_Co2CTo_C + r_CTo2TTo_C + r_To2TTo_C
    # so full sequence in the camera frame: true camera origin -> camera vicon origin -> camera vicon origin-> target vicon origin -> target vicon origin -> true target origin
    # = true camera origin -> true target origin

    q_TARGET_2_CAMERA   = rotm2q(Rotm_T_2_C)
    r_Co2To_CAMERA      = r_Co2To_C
    return q_TARGET_2_CAMERA, r_Co2To_CAMERA


def _process_vicon_offset_v03(row, T_CvC, T_TvT, vicon_keys):
        """
        Process vicon data using passive rotation matrices and translation vectors
        
        T_CvC: transformation from Vicon camera frame to true camera frame
        T_TvT: transformation from Vicon target frame to true target frame
        """
    
        soho_x          = float(row[vicon_keys['x_target']]) * 1E-3
        soho_y          = float(row[vicon_keys['y_target']]) * 1E-3
        soho_z          = float(row[vicon_keys['z_target']]) * 1E-3
        soho_qw         = float(row[vicon_keys['qw_target']])
        soho_qx         = float(row[vicon_keys['qx_target']])
        soho_qy         = float(row[vicon_keys['qy_target']])
        soho_qz         = float(row[vicon_keys['qz_target']])
        soho_TvV        = np.array( [ soho_x, soho_y, soho_z ] ).T
        soho_quatTvV    = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] ).T

        cam_x           = float(row[vicon_keys['x_cam']]) * 1E-3
        cam_y           = float(row[vicon_keys['y_cam']]) * 1E-3
        cam_z           = float(row[vicon_keys['z_cam']]) * 1E-3
        cam_qw          = float(row[vicon_keys['qw_cam']])
        cam_qx          = float(row[vicon_keys['qx_cam']])
        cam_qy          = float(row[vicon_keys['qy_cam']])
        cam_qz          = float(row[vicon_keys['qz_cam']])
        cam_CvV         = np.array( [ cam_x, cam_y, cam_z ] ).T
        cam_quatCvV     = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] ).T

        R_CCv           = T_CvC[:3,:3]
        R_TTv           = T_TvT[:3,:3]
        t_CCv           = T_CvC[:3,3]
        t_TTv           = T_TvT[:3,3]
        Trfm_TvV        = q2trfm( soho_quatTvV )
        Trfm_CvV        = q2trfm( cam_quatCvV )
        Trfm_VTv        = Trfm_TvV.T
        Trfm_VCv        = Trfm_CvV.T
        Trfm_CCv        = R_CCv.T
        Trfm_TTv        = R_TTv.T
        Trfm_VC         = ( Trfm_CCv @ Trfm_CvV ).T
        Trfm_VT         = ( Trfm_TTv @ Trfm_TvV ).T
        
        # post-offset
        translation_posto   = Trfm_VC.T \
                            @ ( \
                                ( soho_TvV + Trfm_VTv @ t_TTv ) \
                              - ( cam_CvV  + Trfm_VCv @ t_CCv ) \
                            )
        Trfm_TC_posto       = ( Trfm_VT.T @ Trfm_VC )
        quaternion_posto    = rotm2q( Trfm_TC_posto.T )

        r_Co2To_CAM         = translation_posto
        q_TARGET_2_CAM      = quaternion_posto

        # # pre-offset
        # translation_preo    = Trfm_CvV \
        #                     @ ( \
        #                         soho_TvV  \
        #                       - cam_CvV   \
        #                     )
        # Trfm_TC_preo        =  ( Trfm_TvV @ Trfm_CvV.T )
        # quaternion_preo     = rotm2q( Trfm_TC_preo.T )

        return q_TARGET_2_CAM, r_Co2To_CAM


def _load_offset_estimates(offset_data_path, offset_keys):
    # extract offset estimates from json
    with open(offset_data_path, 'r') as f:
        offset_json                         = json.load(f)
    offset_key_list                     = list(offset_keys)
    CV_C_key                            = offset_keys[offset_key_list[0]] # "Trf_4x4_CamViconDef_Cam"
    TV_T_key                            = offset_keys[offset_key_list[1]] # "Trf_4x4_TargetViconDef_Target"
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


def _select_processor(processor_name, processor_map, processor_kind):
    """ resolve a version string from the Inputs section into a callable function """
    try:
        return processor_map[processor_name]
    except KeyError as exc:
        available   = ", ".join(sorted(processor_map))
        raise ValueError(
            f"Unknown {processor_kind} '{processor_name}'. Expected one of: {available}"
        ) from exc


################################ Helper Functions ################################
def main():
    HERE                = Path(__file__).parent.resolve()
    ##################################### Inputs #####################################
    data_folder         = HERE / "artifacts" / "offset" / "expm_001"
    data_name           = data_folder.name
    image_folder        = data_folder / "images"
    # kps_file is in mm
    kps_file            = HERE / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_50000.json" # origin shifted to edge
    # kps_centered_file   = 
    opencv_pose_est     = data_folder / "camera_poses.csv" # attitude and positon in meters
    vicon_pose_est      = data_folder / "vicon_data.csv" # position in mm
    calib_data          = data_folder / "calibration.yaml" # wrong, ANAND EDIT, what?
    # calib_data          = data_folder / "calibration_2025_11_14.yaml" # a different calibration file
    offset_data         = data_folder / "offset_results.json"
    # setup keys
    res_path            = HERE / "results" / f'{data_name}_v03'

    # choose type of vicon and opencv pose processing
    selected_vicon_offset_processor = "v03"  # options: v01, v02, v03
    selected_opencv_pose_processor  = "v01"  # options: v01, v02


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
                        'x_cam': 'cam_x',
                        'y_cam': 'cam_y',
                        'z_cam': 'cam_z',
                        'qw_cam': 'cam_qw',
                        'qx_cam': 'cam_qx',
                        'qy_cam': 'cam_qy',
                        'qz_cam': 'cam_qz'
                    }

    # offset keys
    offset_keys     = {
                        'Trf_4x4_CamViconDef_Cam': 'T_CvC',
                        'Trf_4x4_TargetViconDef_Target': 'T_TvT'
                    }
    ##################################### Inputs #####################################

    ############################## Secondary Input Setup #############################


    vicon_offset_processors = {
                                "v01": _process_vicon_offset_v01,
                                "v02": _process_vicon_offset_v02,
                                "v03": _process_vicon_offset_v03
                            }

    opencv_pose_processors  = {
                                "v01": _process_opencv_pose_v01,
                                "v02": _process_opencv_pose_v02
                            }

    process_vicon_offset    = _select_processor(
                                                    selected_vicon_offset_processor,
                                                    vicon_offset_processors,
                                                    "vicon offset pose processor"
                                                )
    process_opencv_pose     = _select_processor(
                                                    selected_opencv_pose_processor,
                                                    opencv_pose_processors,
                                                    "opencv pose processor"
                                                )
    ############################## Secondary Input Setup #############################
    # make results path
    if res_path.exists():
        print(f"Warning: {res_path} already exists, deleting and recreating...")
        shutil.rmtree(res_path)
    os.makedirs(res_path, exist_ok = True)
    print(f"Results will be saved to: {res_path}")


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

    opencv_df       = pd.read_csv(opencv_pose_est)
    vicon_df        = pd.read_csv(vicon_pose_est)


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
        # extract opencv outputted rotation and translation
        rvec        = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
        tvec        = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])

        q_T_2_C, r_Co2To_C  = process_opencv_pose(rvec, tvec)
        R_T_to_C            = q2rotm(q_T_2_C)

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

        vicon_match = vicon_df[vicon_df[vicon_keys['frame']] == img_num]
        if vicon_match.empty:
            print(f"Image number {img_num} for {img_name} not found in vicon data, skipping...")
            continue
        vicon_row   = vicon_match.iloc[0, :]
        
        vicon_img_outpath       = res_path / f"vicon_reproj_{img_base}.png" 
        combined_img_out_path   = res_path / f"combined_reproj_{img_base}.png" 
        # uv_cam_vicon = proj.classless_pinhole_project_to_image(
        Trf4x4_CAMVICON_2_CAM_TRUE, Trf4x4_TARGETVICON_2_TARGET_TRUE    = _load_offset_estimates(
                                                                                                    offset_data_path = offset_data, 
                                                                                                    offset_keys = offset_keys
                                                                                                )
        q_proc_T_2_C, r_proc_Co2To_C                                    = process_vicon_offset(
                                                                                                        row = vicon_row,
                                                                                                        T_CvC = Trf4x4_CAMVICON_2_CAM_TRUE,
                                                                                                        T_TvT = Trf4x4_TARGETVICON_2_TARGET_TRUE,
                                                                                                        vicon_keys = vicon_keys
                                                                                                    )
        uv_cam_vicon    = proj.classless_pinhole_project_to_image(
                                                                    q_TARGET_2_CAM    = q_proc_T_2_C,
                                                                    r_Co2To_CAM       = r_proc_Co2To_C,
                                                                    Kmat              = Kmat_cal,
                                                                    BC_dist_coeffs    = dist_coeffs,
                                                                    points_xyz_TARGET = target_BFF_pts_with_origin
                                                                )
        img_vicon_out   = draw_uv_points_on_image(
                                                    img_or_path     = str(img_path),
                                                    points_uv       = uv_cam_vicon,
                                                    point_color     = (255, 0, 0),
                                                    point_radius    = 15, 
                                                    point_thickness = 2
                                                )
        # highlight origin point in a different color
        uv_origin_vicon = uv_cam_vicon[0]  # assuming the first point is the origin
        img_vicon_out   = draw_uv_points_on_image(
                                                    img_or_path     = img_vicon_out,
                                                    points_uv       = uv_origin_vicon.reshape(1, 2),
                                                    point_color     = (0, 255, 0),  # Green for origin
                                                    point_radius    = 20, 
                                                    point_thickness = 3
                                                )
        cv2.imwrite(vicon_img_outpath, img_vicon_out)
        # combined
        img_overlay_out = draw_uv_points_on_image(
                                                    img_or_path     = img_out,
                                                    points_uv       = uv_cam_vicon,
                                                    point_color     = (128, 0, 128),
                                                    point_radius    = 17, 
                                                    point_thickness = 3
                                                )
        cv2.imwrite(combined_img_out_path, img_overlay_out)


    trs_array       = np.array(trs)  # (N, 3)
    Rmats_array     = np.array(Rmats)  # (N, 3, 3)

    # reproject all loop based on the pose estimates from vicon

    print(f'Results located at: {res_path}')

if __name__ == "__main__":
    main()
