# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import numpy as np
from pi_trajectory_generation.data_processing import utils
from pi_trajectory_generation.data_processing import motion_data
from pi_trajectory_generation.data_processing import xsens_data_converter
from pi_trajectory_generation.data_processing import motion_data_retargeter
import pathlib
import idyntree.swig as idyn
import bipedal_locomotion_framework.bindings as blf
import jaxsim.api as js

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--filename", help="Mocap file to be retargeted. Relative path from script folder.",
                    type=str, default="../datasets/mocap/treadmill_walking.txt")
parser.add_argument("--mirroring", help="Mirror the mocap data.", action="store_true")
parser.add_argument("--KFWBGR", help="Kinematically feasible Whole-Body Geometric Retargeting.", action="store_true")
parser.add_argument("--save", help="Store the retargeted motion in json format.", action="store_true")
parser.add_argument("--deactivate_horizontal_feet", help="Deactivate horizontal feet enforcing.", action="store_true")
parser.add_argument("--deactivate_straight_head", help="Deactivate straight head enforcing.", action="store_true")
parser.add_argument("--deactivate_wider_legs", help="Deactivate wider legs enforcing.", action="store_true")
parser.add_argument("--deactivate_visualization", help="Do not visualize the retargeted motion.", action="store_true")
parser.add_argument("--plot_ik_solutions", help="Show plots of the target task values and the IK solutions.", action="store_true")

args = parser.parse_args()

mocap_filename = args.filename
mirroring = args.mirroring
kinematically_feasible_base_retargeting = args.KFWBGR
store_as_json = args.save
horizontal_feet = not args.deactivate_horizontal_feet
straight_head = not args.deactivate_straight_head
wider_legs = not args.deactivate_wider_legs
visualize_retargeted_motion = not args.deactivate_visualization
plot_ik_solutions = args.plot_ik_solutions

# =====================
# XSENS DATA CONVERSION
# =====================

# Original mocap data
script_directory = os.path.dirname(os.path.abspath(__file__))
mocap_filename = os.path.join(script_directory, mocap_filename)

# Define the relevant data for retargeting purposes: timestamps and link orientations
metadata = motion_data.MocapMetadata.build()
metadata.add_timestamp()
metadata.add_link("Pelvis", is_root=True)
metadata.add_link("T8", position=False)
metadata.add_link("Head", position=False)
metadata.add_link("RightUpperLeg", position=False)
metadata.add_link("RightLowerLeg", position=False)
metadata.add_link("RightFoot", position=False)
metadata.add_link("RightUpperArm", position=False)
metadata.add_link("RightForeArm", position=False)
metadata.add_link("LeftUpperLeg", position=False)
metadata.add_link("LeftLowerLeg", position=False)
metadata.add_link("LeftFoot", position=False)
metadata.add_link("LeftUpperArm", position=False)
metadata.add_link("LeftForeArm", position=False)

# Instantiate the data converter
converter = xsens_data_converter.XSensDataConverter.build(mocap_filename=mocap_filename,
                                                          mocap_metadata=metadata)
# Convert the mocap data
motiondata = converter.convert()

# ===============
# MODEL INSERTION
# ===============

# Retrieve the robot urdf model
urdf_path = pathlib.Path("../src/pi_trajectory_generation/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Init jaxsim model for visualization and joint names/positions
js_model = js.model.JaxSimModel.build_from_model_description(
    model_description=urdf_path, is_urdf=True
)

# Get the joint name list
joint_names = [str(joint_name) for joint_name in js_model.joint_names()]

# Load the idyn model
model_loader = idyn.ModelLoader()
assert model_loader.loadReducedModelFromFile(str(urdf_path), joint_names)

# create KinDynComputationsDescriptor
kindyn = idyn.KinDynComputations()
assert kindyn.loadRobotModel(model_loader.model())

# ==========================
# INVERSE KINEMATICS SETTING
# ==========================

# Set the parameters from toml file
qp_ik_params = blf.parameters_handler.TomlParametersHandler()
toml = pathlib.Path("../src/pi_trajectory_generation/data_processing/qpik.toml").expanduser()
assert toml.is_file()
ok = qp_ik_params.set_from_file(str(toml))
qp_ik_params.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
assert ok

# Build the QPIK object
(variables_handler, tasks, ik_solver) = blf.ik.QPInverseKinematics.build(
    param_handler=qp_ik_params, kin_dyn=kindyn
)

ik_solver: blf.ik.QPInverseKinematics

# ===========
# RETARGETING
# ===========

# Define robot-specific feet frames
feet_frames, feet_links = utils.define_feet_frames_and_links(robot="ergoCubV1")

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = utils.define_foot_vertices(robot="ergoCubV1")

# Define robot-specific quaternions from the robot base frame to the target base frame
robot_to_target_base_quat = utils.define_robot_to_target_base_quat(robot="ergoCubV1")

# Instantiate the retargeter
if kinematically_feasible_base_retargeting:
    retargeter = motion_data_retargeter.KFWBGR.build(motiondata=motiondata,
                                                     metadata=metadata,
                                                     ik_solver=ik_solver,
                                                     joint_names=joint_names,
                                                     mirroring=mirroring,
                                                     horizontal_feet=horizontal_feet,
                                                     straight_head=straight_head,
                                                     wider_legs=wider_legs,
                                                     robot_to_target_base_quat=robot_to_target_base_quat,
                                                     kindyn=kindyn,
                                                     local_foot_vertices_pos=local_foot_vertices_pos,
                                                     feet_frames=feet_frames)
else:
    retargeter = motion_data_retargeter.WBGR.build(motiondata=motiondata,
                                                   metadata=metadata,
                                                   ik_solver=ik_solver,
                                                   joint_names=joint_names,
                                                   kindyn=kindyn,
                                                   mirroring=mirroring,
                                                   horizontal_feet=horizontal_feet,
                                                   straight_head=straight_head,
                                                   wider_legs=wider_legs,
                                                   robot_to_target_base_quat=robot_to_target_base_quat)

# Retrieve ik solutions
if kinematically_feasible_base_retargeting:
    timestamps, ik_solutions = retargeter.KF_retarget(plot_ik_solutions=plot_ik_solutions)
else:
    timestamps, ik_solutions = retargeter.retarget(plot_ik_solutions=plot_ik_solutions)

# =============
# STORE AS JSON
# =============

if store_as_json:

    outfile_name = os.path.join(script_directory, "retargeted_motion.txt")

    input("Press Enter to store the retargeted mocap into a json file")
    utils.store_retargeted_mocap_as_json(timestamps=timestamps, ik_solutions=ik_solutions, outfile_name=outfile_name)
    print("\nThe retargeted mocap data have been saved in", outfile_name, "\n")

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

if visualize_retargeted_motion:

    input("Press Enter to start the visualization of the retargeted motion")
    utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions,
                                      js_model=js_model, controlled_joints=joint_names)

input("Close")
