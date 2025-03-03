# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import json
import numpy as np
from typing import List, Dict
from scipy.spatial.transform import Rotation
import dataclasses
import idyntree.swig as idyn

import mujoco
import mujoco.viewer
import jaxsim.api as js
from jaxsim.mujoco import ModelToMjcf

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =================
# RETARGETING UTILS
# =================

@dataclasses.dataclass
class IKSolution:

    joint_configuration: np.ndarray
    base_position: np.array = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    base_quaternion: np.array = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

def define_robot_to_target_base_quat(robot: str) -> List:
    """Define the robot-specific quaternions from the robot base frame to the target base frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the robot base frame is rotated of -180 degs on z w.r.t. the target base frame
        robot_to_target_base_quat = [0, 0, 0, -1.0]

    elif robot == "iCubV3":
        # For iCubV3, the robot base frame is the same as the target base frame
        robot_to_target_base_quat = [0, 0, 0, 0.0]

    elif robot == "ergoCubV1":
        # For ergoCubV1, the robot base frame is the same as the target base frame
        robot_to_target_base_quat = [0, 0, 0, 0.0]

    else:
        raise Exception("Quaternions from the robot to the target base frame only defined for iCubV2_5, iCubV3 and ergoCubV1.")

    return robot_to_target_base_quat

def define_feet_frames_and_links(robot: str) -> Dict:
    """Define the robot-specific feet frames and links."""

    if robot == "iCubV2_5":
        right_foot_frame = "r_foot"
        left_foot_frame = "l_foot"
        right_foot_link = "r_ankle_2"
        left_foot_link = "l_ankle_2"

    elif robot == "iCubV3":
        right_foot_frame = "r_sole"
        left_foot_frame = "l_sole"
        right_foot_link = "r_ankle_2"
        left_foot_link = "l_ankle_2"

    elif robot == "ergoCubV1":
        right_foot_frame = "r_sole"
        left_foot_frame = "l_sole"
        right_foot_link = "r_ankle_2"
        left_foot_link = "l_ankle_2"

    else:
        raise Exception("Feet frames and links only defined for iCubV2_5, iCubV3 and ergoCubV1.")

    feet_frames = {"right_foot": right_foot_frame, "left_foot": left_foot_frame}
    feet_links = {feet_frames["right_foot"]: right_foot_link, feet_frames["left_foot"]: left_foot_link}

    return feet_frames, feet_links

def define_foot_vertices(robot: str) -> List:
    """Define the robot-specific positions of the feet vertices in the foot frame."""

    if robot == "iCubV2_5":

        # For iCubV2_5, the feet vertices are not symmetrically placed wrt the foot frame origin.
        # The foot frame has z pointing down, x pointing forward and y pointing right.

        # Origin of the box which represents the foot (in the foot frame)
        box_origin = [0.03, 0.005, 0.014]

        # Size of the box which represents the foot
        box_size = [0.16, 0.072, 0.001]

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
        FR_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]
        BL_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
        BR_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]

    elif robot == "iCubV3":

        # For iCubV3, the feet vertices are symmetrically placed wrt the sole frame origin.
        # The sole frame has z pointing up, x pointing forward and y pointing left.

        # Size of the box which represents the foot rear
        box_size = [0.117, 0.1, 0.006]

        # Distance between the foot rear and the foot front boxes
        boxes_distance = 0.00225

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_size[0] + boxes_distance / 2, box_size[1] / 2, 0]
        FR_vertex_pos = [box_size[0] + boxes_distance / 2, - box_size[1] / 2, 0]
        BL_vertex_pos = [- box_size[0] - boxes_distance / 2, box_size[1] / 2, 0]
        BR_vertex_pos = [- box_size[0] - boxes_distance / 2, - box_size[1] / 2, 0]

    elif robot == "ergoCubV1":

        # For ergoCubV1, the feet vertices are symmetrically placed wrt the sole frame origin.
        # The sole frame has z pointing up, x pointing forward and y pointing left.

        # Size of the box which represents the foot rear
        box_size = [0.117, 0.1, 0.006]

        # Distance between the foot rear and the foot front boxes # TODO: doublecheck
        boxes_distance = 0.00225

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_size[0] + boxes_distance / 2, box_size[1] / 2, 0]
        FR_vertex_pos = [box_size[0] + boxes_distance / 2, - box_size[1] / 2, 0]
        BL_vertex_pos = [- box_size[0] - boxes_distance / 2, box_size[1] / 2, 0]
        BR_vertex_pos = [- box_size[0] - boxes_distance / 2, - box_size[1] / 2, 0]

    else:
        raise Exception("Feet vertices positions only defined for iCubV2_5, iCubV3 and ergoCubV1.")

    # Vertices positions in the foot (F) frame
    F_vertices_pos = [FL_vertex_pos, FR_vertex_pos, BL_vertex_pos, BR_vertex_pos]

    return F_vertices_pos

def quaternion_multiply(quat1: List, quat2: List) -> np.array:
    """Auxiliary function for quaternion multiplication."""

    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    res = np.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                     x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                     -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                     x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])

    return res

def to_xyzw(wxyz: List) -> List:
    """Auxiliary function to convert quaternions from wxyz to xyzw format."""

    return wxyz[[1, 2, 3, 0]]

def to_wxyz(xyzw: List) -> List:
    """Auxiliary function to convert quaternions from xyzw to wxyz format."""

    return xyzw[[3, 0, 1, 2]]

def store_retargeted_mocap_as_json(timestamps: List, ik_solutions: List, outfile_name: str) -> None:
    """Auxiliary function to store the retargeted motion."""

    ik_solutions_json = []

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        ik_solution_json = {"joint_positions": ik_solution.joint_configuration.tolist(),
                            "base_position": ik_solution.base_position.tolist(),
                            "base_quaternion": ik_solution.base_quaternion.tolist(),
                            "timestamp": timestamps[i]}

        ik_solutions_json.append(ik_solution_json)

    with open(outfile_name, "w") as outfile:
        json.dump(ik_solutions_json, outfile)

def load_retargeted_mocap_from_json(input_file_name: str, initial_frame: int = 0, final_frame: int = -1) -> (List, List):
    """Auxiliary function to load the retargeted mocap data."""

    # Load ik solutions
    with open(input_file_name, 'r') as openfile:
        ik_solutions = json.load(openfile)

    # If a final frame has been passed, extract relevant ik solutions
    if initial_frame != -1:
        ik_solutions = ik_solutions[initial_frame:final_frame]

    # Extract timestamps
    timestamps = [ik_solution["timestamp"] for ik_solution in ik_solutions]

    return timestamps, ik_solutions

# =========================
# FEATURES EXTRACTION UTILS
# =========================

def define_frontal_base_direction(robot: str) -> List:
    """Define the robot-specific frontal base direction in the base frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the reversed x axis of the base frame is pointing forward
        frontal_base_direction = [-1, 0, 0]

    elif robot == "iCubV3":
        # For iCubV3, the x axis is pointing forward
        frontal_base_direction = [1, 0, 0]

    elif robot == "ergoCubV1":
        # For ergoCubV1, the x axis is pointing forward
        frontal_base_direction = [1, 0, 0]

    else:
        raise Exception("Frontal base direction only defined for iCubV2_5, iCubV3 and ergoCubV1.")

    return frontal_base_direction

def define_frontal_chest_direction(robot: str) -> List:
    """Define the robot-specific frontal chest direction in the chest frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the z axis of the chest frame is pointing forward
        frontal_chest_direction = [0, 0, 1]

    elif robot == "iCubV3":
        # For iCubV3, the x axis of the chest frame is pointing forward
        frontal_chest_direction = [1, 0, 0]

    elif robot == "ergoCubV1":
        # For ergoCubV1, the x axis of the chest frame is pointing forward
        frontal_chest_direction = [1, 0, 0]

    else:
        raise Exception("Frontal chest direction only defined for iCubV2_5, iCubV3 and ergoCubV1.")

    return frontal_chest_direction

def rotation_2D(angle: float) -> np.array:
    """Auxiliary function for a 2-dimensional rotation matrix."""

    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

def transform_from_pos_quat(position: List, quaternion: List) -> np.ndarray:

    H = np.eye(4)
    H[:3,3] = position
    H[:3,:3] = Rotation.from_quat(to_xyzw(quaternion)).as_matrix()

    return H

def reset_robot_configuration(kindyn: idyn.KinDynComputations, joint_positions: List, base_position: List, base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = idyn.Transform(transform_from_pos_quat(base_position, base_quaternion))
        s = idyn.VectorDynSize.FromPython(joint_positions)
        ds = idyn.VectorDynSize.FromPython(np.zeros(len(joint_positions)))
        base_twist = idyn.Twist.FromPython(np.zeros(6))
        g = idyn.Vector3.FromPython(np.array([0, 0, -9.806]))

        kindyn.setRobotState(world_H_base, s, base_twist, ds, g)

def idyn_transform_to_np(H_idyn: idyn.Transform) -> np.ndarray:

    transform_as_string = H_idyn.toString()
    idyn_style_transform_as_np = np.fromstring(transform_as_string, dtype=float, sep=' ').reshape(4,3)
    H_np = np.eye(4)
    H_np[:3,:3] = idyn_style_transform_as_np[:3,:]
    H_np[:3,-1] = idyn_style_transform_as_np[-1,:]

    return H_np

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_retargeted_motion(timestamps: List,
                                ik_solutions: List,
                                js_model: js.model.JaxSimModel,
                                controlled_joints: List,
                                controlled_joints_indexes: List = []) -> None:
    """Auxiliary function to visualize retargeted motion."""

    timestamp_prev = -1

    # Define the vantage point and convert the env to a mujoco string
    camera = {
     "name":"ergocub_camera",
     "mode":"fixed",
     "pos":"20 20 4",
     "xyaxes":"0 0 4 0 1 0",
     "fovy":"60",
    }
    mjcf_string, assets = ModelToMjcf.convert(model=js_model.built_from, cameras=camera)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    with handle as viewer:
        for i in range(1, len(ik_solutions)):

            ik_solution = ik_solutions[i]

            # Retrieve the base pose and the joint positions, based on the type of ik_solution
            if type(ik_solution) == IKSolution:
                joint_positions = ik_solution.joint_configuration
                base_position = ik_solution.base_position
                base_quaternion = ik_solution.base_quaternion

            elif type(ik_solution) == dict:
                joint_positions = np.asarray(ik_solution["joint_positions"])
                base_position = np.asarray(ik_solution["base_position"])
                base_quaternion = np.asarray(ik_solution["base_quaternion"])

                # Reorder the joint positions to match the controlled joints
                joint_positions = np.array([joint_positions[index] for index in controlled_joints_indexes])

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Visualize the retargeted motion at the time rate of the collected data
            timestamp = timestamps[i]
            if timestamp_prev == -1:
                dt = 1 / 100
            else:
                dt = timestamp - timestamp_prev
            timestamp_prev = timestamp

            mujoco.mj_forward(env, data)
            viewer.sync()
            time.sleep(dt)

            if i == 1:
                input("Start visualization")

    print("Visualization ended")
    time.sleep(1)

def visualize_global_features(global_window_features,
                              global_frame_features,
                              ik_solutions: List,
                              js_model: js.model.JaxSimModel,
                              controlled_joints: List,
                              plot_facing_directions: bool = True,
                              plot_base_velocities: bool = True) -> None:
    """Visualize the retargeted frames along with the associated global features."""

    window_length_frames = global_window_features.window_length_frames
    window_step = global_window_features.window_step
    window_indexes = global_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    # Define the vantage point and convert the env to a mujoco string
    camera = {
     "name":"ergocub_camera",
     "mode":"fixed",
     "pos":"20 20 4",
     "xyaxes":"0 0 4 0 1 0",
     "fovy":"60",
    }
    mjcf_string, assets = ModelToMjcf.convert(model=js_model.built_from, cameras=camera)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    with handle as viewer:
        for i in range(initial_frame, final_frame):

            if not viewer.is_running():
                break
            # Debug
            print(i - initial_frame, "/", final_frame - initial_frame)

            # Retrieve the base pose and the joint positions
            joint_positions = global_frame_features.s[i]
            base_position = global_frame_features.base_positions[i]
            base_quaternion = global_frame_features.base_quaternions[i]

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Retrieve global features
            base_positions = global_window_features.base_positions[i - window_length_frames]
            base_velocities = global_window_features.base_velocities[i - window_length_frames]
            base_quaternions = global_window_features.base_quaternions[i - window_length_frames]

            mujoco.mj_forward(env, data)
            viewer.sync()
            time.sleep(1/50)

            if i == 1:
                input("Start visualization")

            # =================
            # FACING DIRECTIONS
            # =================

            if plot_facing_directions:

                # Figure 1 for the facing directions
                plt.figure(1)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base directions
                    base_euler_angle = Rotation.from_quat(to_xyzw(base_quaternions[j])).as_euler('xyz')[2]
                    base_direction = np.array([np.cos(base_euler_angle), np.sin(base_euler_angle)])
                    base_direction = base_direction / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base direction in blue
                        plt.plot([base_position[0], base_position[0] + 2 * base_direction[0]],
                                 [base_position[1], base_position[1] + 2 * base_direction[1]],
                                 'b', label="Current facing direction")
                    else:
                        # Other base directions in green
                        plt.plot([base_position[0], base_position[0] + base_direction[0]],
                                 [base_position[1], base_position[1] + base_direction[1]], 'g')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Facing directions (global view)")
                plt.legend()

            # ===============
            # BASE VELOCITIES
            # ===============

            if plot_base_velocities:

                # Figure 2 for the base velocities
                plt.figure(2)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base velocities
                    base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base velocity in magenta
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]],
                                  'm', label="Current base velocity")
                    else:
                        # Other base velocities in gray
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]], 'gray')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Base velocities (global view)")
                plt.legend()

            # Plot
            plt.show()
            plt.pause(0.0001)

def visualize_local_features(local_window_features,
                             global_frame_features,
                             ik_solutions: List,
                             js_model: js.model.JaxSimModel,
                             controlled_joints: List,
                             plot_facing_directions: bool = True,
                             plot_base_velocities: bool = True) -> None:
    """Visualize the retargeted frames along with the associated local features."""

    window_length_frames = local_window_features.window_length_frames
    window_step = local_window_features.window_step
    window_indexes = local_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    # Define the vantage point and convert the env to a mujoco string
    camera = {
     "name":"ergocub_camera",
     "mode":"fixed",
     "pos":"20 20 4",
     "xyaxes":"0 0 4 0 1 0",
     "fovy":"60",
    }
    mjcf_string, assets = ModelToMjcf.convert(model=js_model.built_from, cameras=camera)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    with handle as viewer:
        for i in range(initial_frame, final_frame):

            if not viewer.is_running():
                break
            # Debug
            print(i - initial_frame, "/", final_frame - initial_frame)

            # Retrieve the base pose and the joint positions
            joint_positions = global_frame_features.s[i]
            base_position = global_frame_features.base_positions[i]
            base_quaternion = global_frame_features.base_quaternions[i]

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Retrieve global features
            base_positions = local_window_features.base_positions[i - window_length_frames]
            base_velocities = local_window_features.base_velocities[i - window_length_frames]
            base_quaternions = local_window_features.base_quaternions[i - window_length_frames]

            mujoco.mj_forward(env, data)
            viewer.sync()
            time.sleep(1/50)

            if i == 1:
                input("Start visualization")

            # =================
            # FACING DIRECTIONS
            # =================

            if plot_facing_directions:

                # Figure 1 for the facing directions
                plt.figure(1)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot facing directions
                    base_euler_angle = Rotation.from_quat(to_xyzw(base_quaternions[j])).as_euler('xyz')[2]
                    base_direction = np.array([np.cos(base_euler_angle), np.sin(base_euler_angle)])
                    base_direction = base_direction / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current facing direction in blue
                        plt.plot([base_position[0], base_position[0] + 2 * base_direction[0]],
                                [base_position[1], base_position[1] + 2 * base_direction[1]], 'b',
                                label="Current facing direction")
                    else:
                        # Other facing directions in green
                        plt.plot([base_position[0], base_position[0] + base_direction[0]],
                                 [base_position[1], base_position[1] + base_direction[1]], 'g')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Facing directions (local view)")
                plt.legend()

            # ===============
            # BASE VELOCITIES
            # ===============

            if plot_base_velocities:

                # Figure 2 for the base velocities
                plt.figure(2)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base velocities
                    base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base velocity in magenta
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]],
                                  'm', label="Current base velocity")
                    else:
                        # Other base velocities in gray
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]], 'gray')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Base velocities (local view)")
                plt.legend()

            # Plot
            plt.show()
            plt.pause(0.0001)