""" An example script of how to use the Pinhole Camera model with calibration imagery data, see CamCal repo, that have all the same camera parameters """
from pathlib import Path
import json 
import yaml
import csv
import cv2
import pdb
from scipy.spatial.transform import Rotation as R
import os 
import pandas as pd
import numpy as np

# local imports
from sc_pose.math_utils.quaternion import rotm2q, q2rotm, q2trfm
from sc_pose.sensors.camera import PinholeCamera
from sc_pose.sensors.camera_projections import PoseProjector, draw_uv_points_on_image


HERE            = Path(__file__).parent.resolve()
##################################### Inputs #####################################
data_file       = HERE / "artifacts" / "reprojection_images"
skip_csv_header = True
kps_file        = HERE / "artifacts" / "soho_reframed_mesh_pose_pack" / "mesh_points_1000.json"
opencv_meta_data   = HERE / "artifacts" / "camera_poses.csv"
camera_meta_data= data_file / "metadata" / "calibration.yaml" # wrong
# setup keys
res_path        = HERE / 'reproj_results'
os.makedirs(res_path, exist_ok=True)
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

img_files = sorted(os.listdir(data_file))


# 1. Open the CSV metadata
with open(opencv_meta_data, mode='r') as f:
    reader = csv.reader(f)
    header = next(reader)  

df = pd.read_csv(opencv_meta_data)

# 2. Iterate through rows using itertuples for speed and clean access
for i, row in df.iterrows():
    # Optional filter (pandas index starts at 0)
    # if i not in [0, 18, 34]: continue
    
    img_name = row['frame'] 
    print(f"Processing row {i}: {img_name}")
    
    img_base = os.path.splitext(img_name)[0]
    img_path = data_file / img_name 

    #Extract opencv outputted rotation and translation
    rvec = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
    tvec = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])

    # Convert to left scalar first quaternion
    rot = R.from_rotvec(rvec).as_matrix()
    Rt = R.from_matrix(rot)
    q_xyzw = Rt.as_quat()
    q_TC = np.roll(q_xyzw, 1)

    # 5. Translation (Camera to Target in Camera frame)
    r_CT = tvec

    # 6. Project 3D points to 2D image coordinates
    uv_cam = proj.classless_pinhole_project_to_image(
        q_TARGET_2_CAM    = q_TC,
        r_Co2To_CAM       = r_CT,
        Kmat              = Kmat,
        BC_dist_coeffs    = dist_coeffs,
        points_xyz_TARGET = target_BFF_pts 
    )
    
    # 7. Draw the points and save the verification image
    # Using str(img_path) to ensure compatibility with cv2/drawing functions
    img_out = draw_uv_points_on_image(
        img_or_path     = str(img_path),
        points_uv       = uv_cam,
        point_color     = (0, 0, 255),
        point_radius    = 15, 
        point_thickness = 2
    )
    
    # Save output to the results path
    output_filename = f"pandas_check_{img_base}.png"
    cv2.imwrite(str(res_path / output_filename), img_out)
    
    print(f"Processed frame {i}: {img_name}")



