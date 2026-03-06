""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
import numpy as np
from pathlib import Path
import json 
import yaml
import csv
import cv2
import pdb

# local imports
from sc_pose.math.quaternion import rotm2q, q2rotm, q2trfm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
data_file       = HERE / "artifacts" / "reprojection_images"
offset_meta_data= data_file / "metadata" / "offset_results.json"
camera_meta_data= data_file / "metadata" / "calibration.yaml" # wrong
vicon_meta_data = data_file / "metadata" / "vicon_data.csv" # wrong
skip_csv_header = True
kps_file        = HERE / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_50000.json"
# setup keys
res_path        = HERE / 'reproj_results'
##################################### Inputs #####################################

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

R_CCv = T_CvC[:3,:3]
R_TTv = T_TvT[:3,:3]

t_CCv = T_CvC[:3,3]
t_TTv = T_TvT[:3,3]


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
        # if( i != 18 and i != 0 and i != 34 ): continue

        img_n   = int( line[0] ) 

        soho_x  = float( line[1] ) * 1E-3
        soho_y  = float( line[2] ) * 1E-3
        soho_z  = float( line[3] ) * 1E-3
        soho_qw = float( line[4] )
        soho_qx = float( line[5] )
        soho_qy = float( line[6] )
        soho_qz = float( line[7] )
        soho_TvV     = np.array( [ soho_x, soho_y, soho_z ] ).T
        soho_quatTvV = np.array( [ soho_qw, soho_qx, soho_qy, soho_qz ] ).T

        
        cam_x   = float( line[8] )  * 1E-3
        cam_y   = float( line[9] )  * 1E-3
        cam_z   = float( line[10] ) * 1E-3
        cam_qw  = float( line[11] )
        cam_qx  = float( line[12] )
        cam_qy  = float( line[13] )
        cam_qz  = float( line[14] )
        cam_CvV     = np.array( [ cam_x, cam_y, cam_z ] ).T
        cam_quatCvV = np.array( [ cam_qw, cam_qx, cam_qy, cam_qz ] ).T


        img_path = data_file / f"cal_image_{img_n}.png"
    
        img_base        = img_path.stem


        Trfm_TvV        = q2trfm( soho_quatTvV )
        Trfm_CvV        = q2trfm( cam_quatCvV )
        Trfm_VTv        = Trfm_TvV.T
        Trfm_VCv        = Trfm_CvV.T
        Trfm_CCv        = R_CCv.T
        Trfm_TTv        = R_TTv.T
        Trfm_VC         = ( Trfm_CCv @ Trfm_CvV ).T
        Trfm_VT         = ( Trfm_TTv @ Trfm_TvV ).T

        
        
        # post-offset
        translation_posto     = Trfm_VC.T \
                            @ ( \
                                ( soho_TvV + Trfm_VTv @ t_TTv ) \
                              - ( cam_CvV  + Trfm_VCv @ t_CCv ) \
                            )
        Trfm_TC_posto         =  ( Trfm_VT.T @ Trfm_VC )
        quaternion_posto      = rotm2q( Trfm_TC_posto.T )

        r_Co2To_CAM     = translation_posto
        q_TARGET_2_CAM  = quaternion_posto

        # pre-offset
        translation_preo     = Trfm_CvV \
                            @ ( \
                                soho_TvV  \
                              - cam_CvV   \
                            )
        Trfm_TC_preo         =  ( Trfm_TvV @ Trfm_CvV.T )
        quaternion_preo      = rotm2q( Trfm_TC_preo.T )




        uv_cam_posto          = proj.classless_pinhole_project_to_image(
                                            q_TARGET_2_CAM = quaternion_posto,
                                            r_Co2To_CAM = translation_posto,
                                            Kmat = Kmat,
                                            BC_dist_coeffs = dist_coeffs,
                                            points_xyz_TARGET = target_BFF_pts
                                        )
        
        uv_cam_preo          = proj.classless_pinhole_project_to_image(
                                            q_TARGET_2_CAM = quaternion_preo,
                                            r_Co2To_CAM = translation_preo,
                                            Kmat = Kmat,
                                            BC_dist_coeffs = dist_coeffs,
                                            points_xyz_TARGET = target_BFF_pts 
                                        )
        
        uv_cam_origin          = proj.classless_pinhole_project_to_image(
                                            q_TARGET_2_CAM = quaternion_posto,
                                            r_Co2To_CAM = translation_posto,
                                            Kmat = Kmat,
                                            BC_dist_coeffs = dist_coeffs,
                                            points_xyz_TARGET = np.zeros( (3,1) )
                                        )
        
        img             = draw_uv_points_on_image(
                                        img_or_path = img_path,
                                        points_uv = uv_cam_posto,
                                        point_color = (0, 255, 0),
                                        point_radius = 15, 
                                        point_thickness = -1
                                    )
        
        img             = draw_uv_points_on_image(
                                        img_or_path = img,
                                        points_uv = uv_cam_preo,
                                        point_color = (0, 0, 255),
                                        point_radius = 15, 
                                        point_thickness = 2
                                    )
        
        img             = draw_uv_points_on_image(
                                        img_or_path = img,
                                        points_uv = uv_cam_origin,
                                        point_color = (255, 0, 0),
                                        point_radius = 30, 
                                        point_thickness = -1
                                    )


        cv2.imwrite( res_path / f"pandas_check_{img_base}.png", img)
    

# pdb.set_trace()
