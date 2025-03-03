# Authors: Evelyn D'Elia, Giulio Romualdi, Paolo Maria Viceconte
from idyntree.visualize import MeshcatVisualizer
import idyntree.bindings as idyn
import numpy as np
import manifpy as manif
import bipedal_locomotion_framework as blf
import resolve_robotics_uri_py
import argparse
import os
from scipy.spatial.transform import Rotation as R

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif'})

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="File name of the trained model to be simulated.", type=str, default="../datasets/onnx/training_subsampled_mirrored_10x_pi_20240514-173315_ep130.onnx")
args = parser.parse_args()
model_name = args.model_name

# Get the configuration files
script_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_directory, "../config/config_mann.toml")
params_network = blf.parameters_handler.TomlParametersHandler()
params_network.set_from_file(str(config_path))

joypad_config_path = os.path.join(script_directory, "../config/config_joypad.toml")
params_joypad = blf.parameters_handler.TomlParametersHandler()
params_joypad.set_from_file(str(joypad_config_path))

# Get the path of the robot model
robot_model_path = str(resolve_robotics_uri_py.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))
ml = idyn.ModelLoader()
ml.loadReducedModelFromFile(robot_model_path, params_network.get_parameter_vector_string("joints_list"))
viz = MeshcatVisualizer()
viz.load_model(ml.model())

# Create the trajectory generator
mann_trajectory_generator = blf.ml.VelMANNAutoregressive()
assert mann_trajectory_generator.set_robot_model(ml.model())
assert mann_trajectory_generator.initialize(params_network)

# Create the input builder
input_builder = blf.ml.VelMANNAutoregressiveInputBuilder()
assert input_builder.initialize(params_joypad)

# Initial joint positions configuration. The serialization is specified in the config file
joint_positions = params_network.get_parameter_vector_float("initial_joints_configuration")

# Initial base pose. This pose makes the robot stand on the ground with the feet flat
base_pose = manif.SE3.Identity()
initial_base_height = params_network.get_parameter_float("initial_base_height")
quat = params_network.get_parameter_vector_float("initial_base_quaternion")
quat = quat / np.linalg.norm(quat) # Normalize the quaternion
base_pose = manif.SE3([0, 0, initial_base_height], quat)

# Create user inputs
# - `left_stick_x`: 1.0
# - `left_stick_y`: 0.0
# - `right_stick_x`: 1.0
# - `right_stick_y`: 0.0
input_builder_input = blf.ml.VelMANNDirectionalInput()
input_builder_input.motion_direction = np.array([1, 0])
input_builder_input.base_direction = np.array([1, 0])

# Network execution
viz.open()
input("Press a key to start the trajectory generation")

# Reset the trajectory generator
mann_trajectory_generator.reset(joint_positions, base_pose)
length_of_time = 500

# Initialize arrays to store values of interest
base_translations = np.zeros(shape=(3,length_of_time))
base_rotations = np.zeros(shape=(3,length_of_time))
foot_contacts = np.zeros(shape=(2,length_of_time))
left_foot_velocities = np.zeros(shape=(6,length_of_time))
right_foot_velocities = np.zeros(shape=(6,length_of_time))
left_foot_positions = np.zeros(shape=(3,length_of_time))
left_foot_rotations = np.zeros(shape=(3,length_of_time))
right_foot_positions = np.zeros(shape=(3,length_of_time))
right_foot_rotations = np.zeros(shape=(3,length_of_time))

for i in range(length_of_time):

    # Set the input to the builder
    input_builder.set_input(input_builder_input)
    assert input_builder.advance()
    assert input_builder.is_output_valid()

    # Set the input to the trajectory generator
    mann_trajectory_generator.set_input(input_builder.get_output())
    assert mann_trajectory_generator.advance()
    assert mann_trajectory_generator.is_output_valid()

    # Get the output of the trajectory generator and update the visualization
    mann_output = mann_trajectory_generator.get_output()
    base_translations[:,i] = np.array(mann_output.base_pose.translation())
    base_rotations[:,i] = R.from_matrix(mann_output.base_pose.rotation()).as_euler('xyz')
    foot_contacts[:,i] = np.array([mann_output.left_foot.is_active, mann_output.right_foot.is_active])
    left_foot_velocities[:,i] = np.array(mann_output.left_foot_velocity)
    right_foot_velocities[:,i] = np.array(mann_output.right_foot_velocity)
    left_foot_positions[:,i] = np.array(mann_output.left_foot_pose.translation())
    right_foot_positions[:,i] = np.array(mann_output.right_foot_pose.translation())
    left_foot_rotations[:,i] = R.from_matrix(mann_output.left_foot_pose.rotation()).as_euler('xyz')
    right_foot_rotations[:,i] = R.from_matrix(mann_output.right_foot_pose.rotation()).as_euler('xyz')

    viz.set_multibody_system_state(mann_output.base_pose.translation(),
                                   mann_output.base_pose.rotation(),
                                   mann_output.joint_positions)

# get the mse of the foot vels only when that foot is in contact
left_foot_vel_error = np.zeros(shape=(length_of_time,1))
right_foot_vel_error = np.zeros(shape=(length_of_time,1))
left_foot_ang_vel_error = np.zeros(shape=(length_of_time,1))
right_foot_ang_vel_error = np.zeros(shape=(length_of_time,1))
for i in range(length_of_time):
    if foot_contacts[0,i] > 0.5:
        # then the left foot is in contact, take the root mean squared
        left_foot_vel_error[i] = np.sqrt(np.mean(left_foot_velocities[:3,i]**2))
        left_foot_ang_vel_error[i] = np.sqrt(np.mean(left_foot_velocities[3:,i]**2))
    if foot_contacts[1,i] > 0.5:
        # then the right foot is in contact, take the root mean squared
        right_foot_vel_error[i] = np.sqrt(np.mean(right_foot_velocities[:3,i]**2))
        right_foot_ang_vel_error[i] = np.sqrt(np.mean(right_foot_velocities[3:,i]**2))

print("Total RMSE linear, angular vel avg: ", np.mean([np.sum(left_foot_vel_error), np.sum(right_foot_vel_error)]), " | ", np.mean([np.sum(left_foot_ang_vel_error), np.sum(right_foot_ang_vel_error)]))

# Plots
timesteps = np.arange(0,len(base_translations[0,:])/50, 1/50)

# Plot base displacement and rotation over time to compare performance with and without correction
figsize=(6, 3)
fig = plt.figure(figsize=figsize)
plt.title("Foot displacement")
plt.xticks([])
plt.yticks([])
ax = plt.subplot(2,1,1)
# set x limits
ax.set_xlim([0, max(timesteps)])
ax.set_xticks([])
ax.set_ylim([-0.5, 1])
ax.plot(timesteps, base_translations[0,:] - base_translations[0,0])
ax.plot(timesteps, base_translations[1,:] - base_translations[1,0])
ax.plot(timesteps, base_translations[2,:] - base_translations[2,0])
plt.ylabel("(m)")
ax = plt.subplot(2,1,2)
# set x limits
ax.set_xlim([0, max(timesteps)])
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_ylim([-0.5, 0.1])
ax.plot(timesteps, base_rotations[0,:] - base_rotations[0,0])
ax.plot(timesteps, base_rotations[1,:] - base_rotations[1,0])
ax.plot(timesteps, base_rotations[2,:] - base_rotations[2,0])

plt.xlabel("Time (s)")
plt.ylabel("(rad)", labelpad=-1)
fig.text(0.01, 0.5, 'Displacement', va='center', rotation='vertical')
plt.legend(['x','y','z'], loc="lower left",
                bbox_transform=fig.transFigure, ncol=3)
plt.tight_layout()

# Plot the foot heights
timesteps = np.arange(0,len(foot_contacts[0,:])/50, 1/50)
fig = plt.figure(figsize=(6,2))
plt.fill_between(timesteps, foot_contacts[0,:], alpha=0.5)
plt.fill_between(timesteps, foot_contacts[1,:], alpha=0.5)
plt.plot(timesteps, left_foot_positions[2,:])
plt.plot(timesteps, right_foot_positions[2,:])
plt.ylim([0, 0.1])
plt.xlim([min(timesteps), max(timesteps)])
plt.xticks([0, 2, 4, 6, 8, 10])
plt.xlabel("Time (s)", labelpad=-7)
plt.ylabel("Position (m)")
plt.legend(['left contact', 'right contact', 'left z', 'right z'], bbox_to_anchor=(0.95, -0.07), loc="lower right",
                bbox_transform=fig.transFigure, ncol=4)
plt.title("Foot heights")
plt.tight_layout()

plt.show()

