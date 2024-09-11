# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
from adherent.data_processing import utils
import jaxsim.api as js
import pathlib

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Visualize the latest retargeted motion (i.e. the one stored in the "retargeted_motion.txt" file)
parser.add_argument("--latest", help="Visualize the latest retargeted motion (i.e. the one stored in the "
                                     "retargeted_motion.txt file)", action="store_true")

# Our custom dataset is divided in two datasets: D2 and D3
parser.add_argument("--dataset", help="Select a dataset between D2 and D3.", type=str, default="D2")

# Each dataset is divided into portions. D2 includes portions [1,5]. D3 includes portions [6,11].
parser.add_argument("--portion", help="Select a portion of the chosen dataset. Available choices: from 1 to 5 for D2,"
                                      "from 6 to 11 for D3.", type=int, default=1)

# Each portion of each dataset has been retargeted as it is or mirrored. Select if you want to visualize the mirrored version
parser.add_argument("--mirrored", help="Visualize the mirrored version of the selected dataset portion.", action="store_true")

args = parser.parse_args()

latest = args.latest
dataset = args.dataset
retargeted_mocap_index = args.portion
mirrored = args.mirrored

# ====================
# LOAD RETARGETED DATA
# ====================

# Retrieve script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

if latest:

    # Mocap path for the latest retargeted motion
    retargeted_mocap_path = "retargeted_motion.txt"
    retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

    # Load the retargeted mocap data
    timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path)

else:

    # Define the selected subsection of the dataset to be loaded and the correspondent interesting frame interval
    if dataset == "D2":
        retargeted_mocaps = {1:"1_forward_normal_step",2:"2_backward_normal_step",3:"3_left_and_right_normal_step",
                             4:"4_diagonal_normal_step",5:"5_mixed_normal_step"}
        limits = {1: [3750, 35750], 2: [1850, 34500], 3: [2400, 36850], 4: [1550, 16000], 5: [2550, 82250]}
    elif dataset == "D3":
        retargeted_mocaps = {6:"6_forward_small_step",7:"7_backward_small_step",8:"8_left_and_right_small_step",
                             9:"9_diagonal_small_step",10:"10_mixed_small_step",11:"11_mixed_normal_and_small_step"}
        limits = {6: [1500, 28500], 7: [1750, 34000], 8: [2900, 36450], 9: [1250, 17050], 10: [1450, 78420], 11: [1600, 61350]}
    initial_frame = limits[retargeted_mocap_index][0]
    final_frame = limits[retargeted_mocap_index][1]

    # Define the retargeted mocap path
    if not mirrored:
        retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_RETARGETED.txt"
    else:
        retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "_mirrored/" + retargeted_mocaps[retargeted_mocap_index]  + "_RETARGETED_MIRRORED.txt"
    retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

    # Load the retargeted mocap data
    timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path,
                                                                     initial_frame=initial_frame,
                                                                     final_frame=final_frame)

# ===============
# MODEL INSERTION
# ===============

# Retrieve the robot urdf model
urdf_path = pathlib.Path("../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Init jaxsim model for visualization and joint names/positions
js_model = js.model.JaxSimModel.build_from_model_description(
    model_description=urdf_path, is_urdf=True
)

# Get the joint name list
joint_names = [str(joint_name) for joint_name in js_model.joint_names()]

# Define the joints of interest for the features computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm
controlled_joints_indexes = [joint_names.index(elem) for elem in controlled_joints]

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

input("Press Enter to start the visualization of the retargeted motion")
utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, js_model=js_model,
                                  controlled_joints=controlled_joints, controlled_joints_indexes=controlled_joints_indexes)
