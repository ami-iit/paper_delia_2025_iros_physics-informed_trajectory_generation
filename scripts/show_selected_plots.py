# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import h5py

plt.rcParams.update({'font.family':'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

script_directory = os.path.dirname(os.path.abspath(__file__))

#create data arrays
vel_rmse_data = scipy.io.loadmat(script_directory + '/../datasets/reproduce_plots/velocity_rmse_values.mat')
lin_data = [[d[0] for d in vel_rmse_data['mse']],
            [d[0] for d in vel_rmse_data['pi_1x']],
            [d[0] for d in vel_rmse_data['pi_10x']],
            [d[0] for d in vel_rmse_data['pi_20x']]]
ang_data = [[d[1] for d in vel_rmse_data['mse']],
            [d[1] for d in vel_rmse_data['pi_1x']],
            [d[1] for d in vel_rmse_data['pi_10x']],
            [d[1] for d in vel_rmse_data['pi_20x']]]

figsize=(4,3)

# Define mean properties
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')

colors = ['gray', 'lightcoral', 'palegreen', 'cornflowerblue']

#Calculate the 5-element statistical repr of this
fig, ax = plt.subplots(figsize=figsize)
ax.axhline(22.880557330235362, color='k', linestyle='--') #value from the default onnx from original adherent
bplot = ax.boxplot(lin_data, showmeans=True, patch_artist=True, meanprops=meanprops, whis=[0, 100])
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticks([1, 2, 3, 4])  # Ensure the number of ticks matches the number of labels
ax.set(xticklabels=["$w=0$", "$w=1$", "$w=10$", "$w=20$"])
plt.ylabel('RMSE (m/s)')
plt.legend(["baseline"])

# Angular velocity boxplot with broken y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [1, 3]})

# Plot the lower part
ax1.axhline(17.73790623177127, color='k', linestyle='--')  # Reference line
bplot1 = ax1.boxplot(ang_data, showmeans=True, patch_artist=True, meanprops=meanprops, whis=[0, 100])
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylim(17.5, 18)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(bottom=False)
ax1.set_yticks([18])

# Plot the upper part
ax2.axhline(17.73790623177127, color='k', linestyle='--')  # Reference line
bplot2 = ax2.boxplot(ang_data, showmeans=True, patch_artist=True, meanprops=meanprops, whis=[0, 100])
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_ylim(0, 7)
ax2.spines['top'].set_visible(False)
ax2.set_xticks([1, 2, 3, 4])  # Ensure the number of ticks matches the number of labels
ax2.set(xticklabels=["$w=0$", "$w=1$", "$w=10$", "$w=20$"])
ax2.set_ylabel('RMSE (rad/s)')

# Add diagonal lines to indicate the break
d = .015  # how big to make the diagonal lines in axes coordinates
h = 3.0
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-h*d, +h*d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-h*d, +h*d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.legend(["baseline"], loc="lower left")

# Plot base x vs y displacement to compare performance with and without correction
figsize=(5, 4)
fig = plt.figure(figsize=figsize)
oldadh = scipy.io.loadmat(script_directory + '/../datasets/reproduce_plots/oldadh.mat')
piadh_10x_no_pid = scipy.io.loadmat(script_directory + '/../datasets/reproduce_plots/piadh_no_pid.mat')
piadh_10x_pid = scipy.io.loadmat(script_directory + '/../datasets/reproduce_plots/piadh_pid.mat')
plt.plot(oldadh['x'][0], oldadh['y'][0])
plt.plot(piadh_10x_no_pid['x'][0], piadh_10x_no_pid['y'][0])
plt.plot(piadh_10x_pid['x'][0], piadh_10x_pid['y'][0])
plt.xlabel('Displacement x (m)')
plt.ylabel('Displacement y (m)')
plt.legend(['Baseline', 'No correction', 'With correction'], loc='lower left')

# Plot the left and right foot pitch to compare PI weights
figsize= (6,3.5)
fig = plt.figure(figsize=figsize)
# Read in the data
foot_pitch_data = scipy.io.loadmat(script_directory + '/../datasets/reproduce_plots/foot_pitch_data.mat')

ax = plt.subplot(2,1,1)
time_range = np.arange(0,len(foot_pitch_data['mse']['lf_pitch'][0][0][0])/50, 1/50)

ax.set_xlim([min(time_range), max(time_range)])
#This plot is the left foot pitch for each weight
ax.plot(time_range, foot_pitch_data['mse']['lf_pitch'][0][0][0])
ax.plot(time_range, foot_pitch_data['pi_1x']['lf_pitch'][0][0][0], '--')
ax.plot(time_range, foot_pitch_data['pi_10x']['lf_pitch'][0][0][0], '-.')
ax.plot(time_range, foot_pitch_data['pi_20x']['lf_pitch'][0][0][0], ':')
ax.plot(time_range, foot_pitch_data['pi_100x']['lf_pitch'][0][0][0])
ax.set_xticks([])
ax.set_title('Left foot')

ax = plt.subplot(2,1,2)
ax.set_xlim([min(time_range), max(time_range)])
#This plot is the right foot pitch for each weight
ax.plot(time_range, foot_pitch_data['mse']['rf_pitch'][0][0][0])
ax.plot(time_range, foot_pitch_data['pi_1x']['rf_pitch'][0][0][0], '--')
ax.plot(time_range, foot_pitch_data['pi_10x']['rf_pitch'][0][0][0], '-.')
ax.plot(time_range, foot_pitch_data['pi_20x']['rf_pitch'][0][0][0], ':')
ax.plot(time_range, foot_pitch_data['pi_100x']['rf_pitch'][0][0][0])
ax.set_title('Right foot')
ax.set_xticks([0, 1, 2, 3, 4, 5])

fig.text(0.03, 0.5, 'Angle (rad)', va='center', rotation='vertical')
plt.xlabel('Time (s)')
plt.legend(['$w=0$', '$w=1$', '$w=10$', '$w=20$', '$w=100$'], bbox_to_anchor=(0.91, -0.06), loc="lower right",
                bbox_transform=fig.transFigure, ncol=5, columnspacing=0.8)

# Plot the data from robot experiments
script_directory = os.path.dirname(os.path.abspath(__file__))
mpc_path = os.path.join(script_directory, "../datasets/reproduce_plots/robot_tests/robot_logger_device_2024_06_18_11_46_54.mat")
dcm_path = os.path.join(script_directory, "../datasets/reproduce_plots/robot_tests/robot_logger_device_2024_07_09_15_26_45.mat")

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

# For the MPC, the measurements from joint_state also need to be shifted in time to account for data offset
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

plt.show()