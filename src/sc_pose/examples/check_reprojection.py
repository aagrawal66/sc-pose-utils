""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
import numpy as np
from pathlib import Path
import json 
from numpy.typing import NDArray
import yaml
import csv
import cv2
import pdb
from scipy.spatial.transform import Rotation as R
import os 

# local imports
from sc_pose.math_utils.quaternion import rotm2q, q2rotm, q2trfm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
data_file       = HERE / "artifacts" / "reprojection_images"
offset_meta_data= data_file / "metadata" / "offset_results.json"
camera_meta_data= data_file / "metadata" / "calibration.yaml" # wrong
vicon_meta_data = data_file / "metadata" / "vicon_data.csv" # wrong
skip_csv_header = True
kps_file        = HERE / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_1000.json"
# setup keys
res_path        = HERE / 'reproj_results'
os.makedirs(res_path, exist_ok=True)
##################################### Inputs #####################################
def T_inverse(T: NDArray) -> NDArray:
    """ Invert a homogeneous transformation matrix """
    R = T[:3,:3]
    t = T[:3,3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3,:3] = R_inv
    T_inv[:3,3] = t_inv
    return T_inv



# create projection object
proj    = PoseProjector(None)

Rmats               = [] 
trs                 = []
img_paths           = []

# kps from drew/anand
with open(kps_file, 'r') as f:
    kps_mm          = np.array( json.load(f) ) # obj in mm
    kps_m           = kps_mm / 1e3
    target_BFF_pts  = kps_m  

# offset metadata
with open(offset_meta_data, 'r') as f:
    camera_json = json.load(f)
    T_CvC       = np.array( camera_json["T_CvC"] )
    T_TvT       = np.array( camera_json["T_TvT"] )

# R_CvC = T_CvC[:3,:3]
# R_TvT = T_TvT[:3,:3]
# t_TvT = T_TvT[:3,3]
# t_CvC = T_CvC[:3,3]


# camera metadata
with open(camera_meta_data, 'r') as f:
    data = yaml.safe_load(f)
    Kmat = np.array(
        [
            [ data["fx"], 0. , data["cx"] ],
            [ 0., data["fy"] , data["cy"] ],
            [ 0., 0., 1. ]
        ]
    )
    dist_coeffs = np.array( [
        data["k1"],
        data["k2"],
        data["p1"],
        data["p2"],
        data["k3"]
    ] ) 

trs_array       = np.array(trs)  # (N, 3)
Rmats_array     = np.array(Rmats)  # (N, 3, 3)

# for i, img_path in enumerate(img_paths):
with open(vicon_meta_data,'r',newline='') as csv_f:

    reader =csv.reader( csv_f )
    if skip_csv_header:
        next(reader, None) 

    for i, line in enumerate( reader ):
        img_n           = line[0]   
        img_path = data_file / f"cal_image_{img_n}.png"
        img_base        = img_path.stem
        


        # loading vicon data and converting to meters
        soho_x  = float( line[1] ) * 1E-3
        soho_y  = float( line[2] ) * 1E-3
        soho_z  = float( line[3] ) * 1E-3
        soho_qw = float( line[4] )
        soho_qx = float( line[5] )
        soho_qy = float( line[6] )
        soho_qz = float( line[7] )
        soho_VTv     = np.array( [ soho_x, soho_y, soho_z ] )
        soho_quatVTv = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] )

        cam_x   = float( line[8] )  * 1E-3
        cam_y   = float( line[9] )  * 1E-3
        cam_z   = float( line[10] ) * 1E-3
        cam_qw  = float( line[11] )
        cam_qx  = float( line[12] )
        cam_qy  = float( line[13] )
        cam_qz  = float( line[14] )
        cam_VCv     = np.array( [ cam_x, cam_y, cam_z ] )
        cam_quatVCv = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] )

        
        # Convert Vicon information into a transformation matrix from Vicon frame to the Vicon target frame
        R_VTv      = R.from_quat( soho_quatVTv, scalar_first=True ).as_matrix()
        T_VTv      = np.eye(4)
        T_VTv[:3,:3] = R_VTv
        T_VTv[:3,3] = soho_VTv

        # Convert Vicon information into a transformation matrix from Vicon frame to the Vicon camera frame
        R_VCv      = R.from_quat( cam_quatVCv, scalar_first=True ).as_matrix()
        T_VCv      = np.eye(4)
        T_VCv[:3,:3] = R_VCv
        T_VCv[:3,3] = cam_VCv

        # Transformation from vicon camera frame to vicon target frame
        T_CvTv = T_inverse(T_VCv) @ T_VTv
        # Transformation from true camera frame to true target frame
        T_CT = T_inverse(T_CvC) @ T_CvTv @ T_TvT

        # Extracting quaternion and translation from transformation matrix
        q_CT = R.from_matrix( T_CT[:3,:3] ).as_quat()  # [x, y, z, w]
        q_wxyz = np.roll(q_CT, 1)  
        r_CT = T_CT[:3,3]

        uv_cam          = proj.classless_pinhole_project_to_image(
                                            q_TARGET_2_CAM = q_wxyz,
                                            r_Co2To_CAM = r_CT,
                                            Kmat = Kmat,
                                            BC_dist_coeffs = dist_coeffs,
                                            points_xyz_TARGET = target_BFF_pts 
                                        )
        
        # fill this in with image info
        img             = draw_uv_points_on_image(
                                        img_or_path = img_path,
                                        points_uv = uv_cam,
                                        point_color = (0, 0, 255),
                                        point_radius = 15, 
                                        point_thickness = 2
                                    
                                    )
        cv2.imwrite( res_path / f"offset_check_{img_base}.png", img)
    

# pdb.set_trace()
