# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", help="Relative path that contains data from robot tests.", type=str, default="../datasets/robot_tests/")

args = parser.parse_args()

data_dir = args.data_dir

script_directory = os.path.dirname(os.path.abspath(__file__))
mpc_path = os.path.join(script_directory, data_dir, "robot_logger_device_2024_06_18_11_46_54.mat")
dcm_path = os.path.join(script_directory, data_dir, "robot_logger_device_2024_07_09_15_26_45.mat")

args = parser.parse_args()

# read in the mat files from the data
dcm_file = h5py.File(dcm_path, 'r')
mpc_file = h5py.File(mpc_path, 'r')

controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # head
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

# Get the values from them
joints_considered = ['torso_yaw', 'neck_yaw', 'l_shoulder_pitch', 'l_elbow']

sets = ['desired', 'measured', 'postural']

figsize = (9.5, 9)

fig = plt.figure(figsize=figsize)

n = 2.3
dt = 14.3 - n
change_times = [n, n+dt, n+2*dt, n+3*dt]
start_time = dcm_file['/robot_logger_device/cmw/joints/positions/postural/timestamps'][1250]

dcm_timestamps = dcm_file['/robot_logger_device/cmw/joints/positions/postural/timestamps'][1250:] - start_time

#DCM result
for joint in joints_considered:
    ax = plt.subplot(len(joints_considered), 1, joints_considered.index(joint)+1)

    # set x limits
    ax.set_xlim([min(dcm_timestamps), max(dcm_timestamps)])
    ax.tick_params(axis='both', labelsize=12)

    # remove extra x labels
    if joints_considered.index(joint) < (len(joints_considered) - 1):
        ax.set_xticks([])

    # shade the directions
    ax.axvspan(change_times[0], change_times[1], color='peachpuff') #forward
    ax.axvspan(change_times[1], change_times[2], color='honeydew') #side
    ax.axvspan(change_times[2], change_times[3], color='lightcyan') #backward

    for metric in sets:
        ax.plot(dcm_timestamps,
                np.squeeze(dcm_file['/robot_logger_device/cmw/joints/positions/' + metric + '/data'][1250:,:,controlled_joints.index(joint)]))
        ax.set_title(joint.replace("_", " ").capitalize(), fontsize=20, pad=2.0)

ax.set_xlabel("Time (s)", fontsize=16)
plt.subplots_adjust(left=0.1,
                    right=0.95,
                    hspace=0.2)
fig.text(0.025, 0.5, 'Angle (rad)', va='center', rotation='vertical', fontsize=16)
plt.legend(['forward', 'side right', 'backward'] + sets, bbox_to_anchor=(0.5, -0.00), loc="lower center",
                bbox_transform=fig.transFigure, ncol=6, fontsize=14, columnspacing=0.5)

# plt.savefig('dcm_joint_plots.pdf', bbox_inches='tight')

sets_mpc = ['cmw/joints_state/positions/desired', 'joints_state/positions', 'cmw/joints_state/positions/mann']

#joints_state are in a different order
controlled_joints_joint_state = ['neck_pitch', 'neck_roll', 'neck_yaw', 'camera_tilt', # head
                                 'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                                 'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow',
                                  'l_wrist_pitch', 'l_wrist_roll', 'l_wrist_yaw', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow',
                      'l_wrist_pitch', 'l_wrist_roll', 'l_wrist_yaw', # right arm
                                 'l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll']  # right leg

# in MPC, the measurements from joint_state also need to be shifted in time to account for data offset

fig = plt.figure(figsize=figsize)

# Check where the direction changes occur
joypad_x = mpc_file['/robot_logger_device/cmw/joypad/motion_direction/data'][:,:,0]
change_idxs = np.where(joypad_x[:-1] != joypad_x[1:])[0]

start_time = mpc_file['/robot_logger_device/cmw/joints_state/positions/desired/timestamps'][650]
mpc_timestamps = mpc_file['/robot_logger_device/cmw/joints_state/positions/desired/timestamps'][650:] - start_time

#get the timestamps at which inputs change
change_times = np.squeeze(mpc_file['/robot_logger_device/cmw/joints_state/positions/desired/timestamps'][change_idxs] - start_time)

#MPC result
for joint in joints_considered:
    ax = plt.subplot(len(joints_considered), 1, joints_considered.index(joint)+1)

    # set x limits
    # ax.set_xlim([0, len(joypad_x[650:])])
    ax.set_xlim([min(mpc_timestamps), max(mpc_timestamps)])
    ax.tick_params(axis='both', labelsize=12)

    # remove extra x labels
    if joints_considered.index(joint) < (len(joints_considered) - 1):
        ax.set_xticks([])

    # shade the directions
    ax.axvspan(change_times[0], change_times[1], color='peachpuff') #forward
    ax.axvspan(change_times[1], change_times[2], color='honeydew') #side
    ax.axvspan(change_times[2], change_times[3]-4.5, color='lightcyan') #backward

    for metric in sets_mpc:
        if metric == sets_mpc[1]:
            ax.plot(mpc_timestamps, np.squeeze(mpc_file['/robot_logger_device/' + metric + '/data'][735+650:735+650+len(mpc_timestamps),:,controlled_joints_joint_state.index(joint)]))
        else:
            ax.plot(mpc_timestamps, np.squeeze(mpc_file['/robot_logger_device/' + metric + '/data'][650:,:,controlled_joints.index(joint)]))
        ax.set_title(joint.replace("_", " ").capitalize(), fontsize=20, pad=2.0)

ax.set_xlabel("Time (s)", fontsize=16)
plt.subplots_adjust(left=0.1,
                    right=0.95,
                    hspace=0.2)
fig.text(0.025, 0.5, 'Angle (rad)', va='center', rotation='vertical', fontsize=16)
plt.legend(['forward', 'side right', 'backward'] + sets, bbox_to_anchor=(0.5, -0.00), loc="lower center",
                bbox_transform=fig.transFigure, ncol=6, fontsize=14, columnspacing=0.5)
# plt.savefig('mpc_joint_plots.pdf', bbox_inches='tight')

plt.show()