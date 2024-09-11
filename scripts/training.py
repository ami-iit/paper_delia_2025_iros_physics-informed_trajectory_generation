import os
import torch
import argparse
import numpy as np
from torch import nn
from typing import List, Dict
from pi_trajectory_generation.mann_pytorch.MANN import MANN
from torch.utils.data import DataLoader
from pi_trajectory_generation.mann_pytorch.utils import create_path
from pi_trajectory_generation.mann_pytorch.DataHandler import DataHandler
from torch.utils.tensorboard import SummaryWriter

import adam
from adam.pytorch import KinDynComputationsBatch
import pathlib

# Check whether the gpu or the cpu is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Set cuda as default for all future tensors if the GPU is available
if device == "cuda":
    torch.set_default_device('cuda')

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()
parser.add_argument("--deactivate_mirroring", help="Discard features from mirrored mocap data.", action="store_true")
parser.add_argument("--pi_weight", help="Weight of the pi loss.", type=float, default=10.0)
parser.add_argument("--portions", help="Select which subsets of the data to train on.", type=int, nargs='+', default=[1, 2, 3, 4, 5])
args = parser.parse_args()
mirroring = not args.deactivate_mirroring
pi_weight = args.pi_weight
portions = args.portions

# ===============
# MODEL INSERTION
# ===============

# Retrieve the robot urdf model
urdf_path = pathlib.Path("../src/pi_trajectory_generation/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Define the joints of interest for the features computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

# Create a KinDynComputations object with adam
kinDyn = KinDynComputationsBatch(urdf_path, controlled_joints, 'root_link')
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)

# =====================
# DATASET CONFIGURATION
# =====================

# Auxiliary function to retrieve the portions associated to each dataset
def get_dataset_portions(portions) -> Dict:
    """Retrieve the portions associated to each dataset."""

    portions_dict = {1: "1_forward_normal_step",
                2: "2_backward_normal_step",
                3: "3_left_and_right_normal_step",
                4: "4_diagonal_normal_step",
                5: "5_mixed_normal_step"}

    #get portions from dict depending on args
    filtered_dict = {k: portions_dict[k] for k in portions}

    return filtered_dict

# Auxiliary function to define the input and output filenames
def define_input_and_output_paths(mirroring: bool) -> (List, List):
    """Given the mirroring flag, retrieve the list of input and output filenames."""

    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Initialize input filenames list
    input_paths = []

    # Fill input filenames list

    inputs = get_dataset_portions(portions)

    for index in inputs.keys():
        input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_D2/" + inputs[index] + "_X.txt"
        input_paths.append(input_path)

        if mirroring:
            input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_mirrored_D2/" + inputs[index] + "_X_MIRRORED.txt"
            input_paths.append(input_path)

    # Debug
    print("\nInput files:")
    for input_path in input_paths:
        print(input_path)

    # Initialize output filenames list
    output_paths = []

    # Fill output filenames list

    outputs = get_dataset_portions(portions)

    for index in outputs.keys():
        output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_D2/" + outputs[index] + "_Y.txt"
        output_paths.append(output_path)

        if mirroring:
            output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_mirrored_D2/" + outputs[index] + "_Y_MIRRORED.txt"
            output_paths.append(output_path)

    # Debug
    print("\nOutput files:")
    for output_path in output_paths:
        print(output_path)

    return input_paths, output_paths

# Auxiliary function to define where to store the training results
def define_storage_folder(mirroring: bool) -> str:
    """Given the mirroring flag, retrieve the storage folder."""

    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set storage folder
    if not mirroring:
        storage_folder = script_directory + '/../datasets/training_subsampled_D2'
    else:
        storage_folder = script_directory + '/../datasets/training_subsampled_mirrored_D2'

    storage_folder += "_test_paper_code"

    # Debug
    print("\nStorage folder:", storage_folder, "\n")

    return storage_folder

# Retrieve inputs and outputs global paths
input_paths, output_paths = define_input_and_output_paths(mirroring)

# Retrieve global storage folder
storage_folder = define_storage_folder(mirroring)

# Retrieve the training and testing datasets
data_handler = DataHandler(input_paths=input_paths, output_paths=output_paths, storage_folder=storage_folder,
                           training=True, training_set_percentage=98)
training_data = data_handler.get_training_data()
testing_data = data_handler.get_testing_data()

# ======================
# TRAINING CONFIGURATION
# ======================

# Debug
input("\nPress Enter to start the training")

# Random seed
torch.manual_seed(23456)

# Training hyperparameters
num_experts = 4
batch_size = 32
dropout_probability = 0.3
gn_hidden_size = 32
mpn_hidden_size = 512
epochs = 150
Te = 10
Tmult = 2
learning_rate_ini = 0.0001
weightDecay_ini = 0.0025
Te_cumulative = Te

# Configure the datasets for training and testing
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Normalize weight decay
total_batches = int(len(train_dataloader))
weightDecay_ini = weightDecay_ini / (np.power(total_batches * Te, 0.5))

# Initialize the MANN architecture
mann = MANN(train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_experts=num_experts,
            gn_hidden_size=gn_hidden_size,
            mpn_hidden_size=mpn_hidden_size,
            dropout_probability=dropout_probability,
            savepath=data_handler.get_savepath(),
            kinDyn=kinDyn,
            pi_weight=pi_weight)

# Check the trainable parameters in the model
for name, param in mann.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Define the loss function
loss_fn = nn.MSELoss(reduction="mean")

# Initialize the optimizer
optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini, weight_decay=weightDecay_ini)

# Initialize learning rate and weight decay schedulers
fake_lr_optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini)
fake_wd_optimizer = torch.optim.AdamW(mann.parameters(), lr=weightDecay_ini)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_lr_optimizer, T_max=Te)
wd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_wd_optimizer, T_max=Te)

# Configure tensorboard writer
writer_path = data_handler.get_savepath() + "/logs/"
create_path(writer_path)
writer = SummaryWriter(log_dir=writer_path)

# Create the path to periodically store the learned models
model_path = data_handler.get_savepath() + "/models/"
create_path(model_path)
last_model_path = ""

# =============
# TRAINING LOOP
# =============

for epoch in range(epochs):

    # Debug
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Perform one epoch of training and testing
    mann.train_loop(loss_fn, optimizer, epoch, writer)
    mann.test_loop(loss_fn, epoch, writer)

    # Save the trained model periodically and at the very last iteration
    if epoch % 10 == 0 or epoch == epochs - 1:
        current_model_path = model_path + "/model_" + str(epoch) + ".pth"
        torch.save(mann, current_model_path)
        last_model_path = current_model_path

    # Update current learning rate and weight decay
    lr_scheduler.step()
    wd_scheduler.step()
    optimizer.param_groups[0]['lr'] = lr_scheduler.get_last_lr()[0]
    optimizer.param_groups[0]['weight_decay'] = wd_scheduler.get_last_lr()[0]

    # Reinitialize learning rate and weight decay
    if epoch == Te_cumulative - 1:
        Te = Tmult * Te
        Te_cumulative += Te
        fake_lr_optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini)
        fake_wd_optimizer = torch.optim.AdamW(mann.parameters(), lr=weightDecay_ini)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_lr_optimizer, T_max=Te)
        wd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_wd_optimizer, T_max=Te)

# Close tensorboard writer
writer.close()

# Debug
print("Training over!")
