import os
from typing import List, Dict
import json
import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from pi_trajectory_generation.mann_pytorch.GatingNetwork import GatingNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from pi_trajectory_generation.mann_pytorch.MotionPredictionNetwork import MotionPredictionNetwork

# for vmapping with jax
from jax2torch import jax2torch
import jax.numpy as jnp
from jax import jit, vmap
import functools
from adam.pytorch import KinDynComputationsBatch

@jit
@functools.partial(vmap, in_axes=(0, 0, 0, 0))
def V_b_label_fun(gamma, full_jacobian_LF, full_jacobian_RF, joint_velocity):
    jacobian_LF_inv = jnp.linalg.inv(full_jacobian_LF[:, :6])
    jacobian_RF_inv = jnp.linalg.inv(full_jacobian_RF[:, :6])
    M = (- gamma * jacobian_LF_inv @ full_jacobian_LF[:, 6:] @ joint_velocity
        - (1 - gamma) * jacobian_RF_inv @ full_jacobian_RF[:, 6:] @ joint_velocity)
    print("[INFO] Compiling PI loss function")
    return M
class MANN(nn.Module):
    """Class for the Mode-Adaptive Neural Network."""

    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 num_experts: int, gn_hidden_size: int, mpn_hidden_size: int, dropout_probability: float,
                 savepath: str,
                 kinDyn: KinDynComputationsBatch,
                 pi_weight: float):
        """Mode-Adaptive Neural Network constructor.

        Args:
            train_dataloader (DataLoader): Iterable over the training dataset
            test_dataloader (DataLoader): Iterable over the testing dataset
            num_experts (int): The number of expert weights constituting the Motion Prediction Network
            gn_hidden_size (int): The dimension of the 3 hidden layers of the Gating Network
            mpn_hidden_size (int): The dimension of the 3 hidden layers of the Motion Prediction Network
            dropout_probability (float): The probability of an element to be zeroed in the network training
        """

        # Superclass constructor
        super(MANN, self).__init__()

        # Retrieve input and output dimensions from the training dataset
        train_features, train_labels = next(iter(train_dataloader))
        input_size = train_features.size()[-1]
        output_size = train_labels.size()[-1]

        # Define the two subnetworks composing the MANN architecture
        self.gn = GatingNetwork(input_size=input_size,
                                output_size=num_experts,
                                hidden_size=gn_hidden_size,
                                dropout_probability=dropout_probability)
        self.mpn = MotionPredictionNetwork(num_experts=num_experts,
                                           input_size=input_size,
                                           output_size=output_size,
                                           hidden_size=mpn_hidden_size,
                                           dropout_probability=dropout_probability)

        # Store the dataloaders for training and testing
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.savepath = savepath

        # Store the kindyn object
        self.kinDyn = kinDyn

        self.pi_weight = pi_weight

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['kinDyn']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mode-Adaptive Neural Network architecture.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            y (torch.Tensor): The output of the Motion Prediction Network
        """

        # Retrieve the output of the Gating Network
        blending_coefficients = self.gn(x.T)

        # Retrieve the output of the Motion Prediction Network
        y = self.mpn(x, blending_coefficients=blending_coefficients)

        return y

    def read_from_file(self, filename: str) -> np.array:
        """Read data as json from file."""

        with open(filename, 'r') as openfile:
            data = json.load(openfile)

        return np.array(data)

    def load_input_mean_and_std(self, datapath: str) -> (List, List):
        """Compute input mean and standard deviation."""

        # Full-input mean and std
        Xmean = self.read_from_file(datapath + 'X_mean.txt')
        Xstd = self.read_from_file(datapath + 'X_std.txt')

        # Remove zeroes from Xstd
        for i in range(Xstd.size):
            if Xstd[i] == 0:
                Xstd[i] = 1

        Xmean = torch.from_numpy(Xmean)
        Xstd = torch.from_numpy(Xstd)

        return Xmean, Xstd

    def load_output_mean_and_std(self, datapath: str) -> (List, List):
        """Compute output mean and standard deviation."""

        # Full-output mean and std
        Ymean = self.read_from_file(datapath + 'Y_mean.txt')
        Ystd = self.read_from_file(datapath + 'Y_std.txt')

        # Remove zeroes from Ystd
        for i in range(Ystd.size):
            if Ystd[i] == 0:
                Ystd[i] = 1

        Ymean = torch.from_numpy(Ymean)
        Ystd = torch.from_numpy(Ystd)

        return Ymean, Ystd

    def denormalize(self, X: torch.Tensor, Xmean: torch.Tensor, Xstd: torch.Tensor) -> torch.Tensor:
        """Denormalize X, given its mean and std."""

        if torch.cuda.is_available:
            Xmean = Xmean.to('cuda')
            Xstd = Xstd.to('cuda')

        # Denormalize
        X = X * Xstd + Xmean

        return X

    def euler_to_quaternion(self, angle):
        """
        Convert Euler angles to quaternion representation.

        Args:
            angle: Tensor containing xyz Euler angles

        Returns:
            quaternions: Tensor containing quaternions in xyzw format.
        """

        roll, pitch, yaw = angle.unbind(dim=-1)

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr

        return torch.stack((qx, qy, qz, qw), dim=-1)

    def quaternion_position_to_transform(self, quaternion: torch.Tensor, position: torch.Tensor):

        # Normalize quaternion
        quaternion = quaternion / torch.norm(quaternion)

        # Extract individual components of the quaternion
        x, y, z, w = quaternion.unbind(dim=-1)

        # Compute rotation matrix components
        xx = x * x
        xy = x * y
        xz = x * z
        yy = y * y
        yz = y * z
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotation_matrix = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
            ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))

        # Stack rotation matrix with position
        transformation_matrix = torch.cat([
            torch.cat([rotation_matrix, position.unsqueeze(-1)], dim=-1),
            torch.tensor([0, 0, 0, 1], dtype=quaternion.dtype, device=quaternion.device).expand(position.shape[:-1] + (1, 4))
            ], dim=-2)

        return transformation_matrix

    def compute_Vb_label(self, base_position: torch.Tensor, base_angle: torch.Tensor, joint_position: torch.Tensor, V_b: torch.Tensor,
                         joint_velocity: torch.Tensor) -> torch.Tensor:

        # Get a base transform matrix from the data
        base_quaternion = self.euler_to_quaternion(base_angle)
        H_b = self.quaternion_position_to_transform(base_quaternion, base_position)

        # Compute the foot jacobian matrices
        full_jacobian_LF = self.kinDyn.jacobian("l_sole", H_b, joint_position)
        full_jacobian_RF = self.kinDyn.jacobian("r_sole", H_b, joint_position)

        # Check which foot is lower to determine support foot (gamma=1 for LF support, gamma=0 for RF support)
        H_LF = self.kinDyn.forward_kinematics("l_sole", H_b, joint_position)
        H_RF = self.kinDyn.forward_kinematics("r_sole", H_b, joint_position)
        z_diff = H_LF[:,2,3] - H_RF[:,2,3]
        condition = z_diff > 0
        gamma = torch.where(condition, 0, 1).float()

        # Get the function
        V_b_label_torch = jax2torch(V_b_label_fun)

        # Call the function to get the tensor result
        V_b_label = V_b_label_torch(gamma, full_jacobian_LF, full_jacobian_RF, joint_velocity)

        return V_b_label

    def get_pi_loss_components(self, X: torch.Tensor, pred: torch.Tensor) -> (torch.Tensor, torch.Tensor):

            datapath = os.path.join(self.savepath, "normalization/")
            Xmean, Xstd = self.load_input_mean_and_std(datapath)
            Ymean, Ystd = self.load_output_mean_and_std(datapath)

            # Denormalize for correct calculations
            X = self.denormalize(X, Xmean, Xstd)
            pred = self.denormalize(pred, Ymean, Ystd)

            # Get Vb from network output
            V_b_linear = pred[:,:3].float()
            V_b_angular = pred[:,21:24].float()
            joint_position_batch = pred[:, 42:68].float()
            joint_velocity_batch = pred[:,68:94].float()
            base_position_batch = pred[:,-6:-3].float()
            base_angle_batch = pred[:,-3:].float()
            V_b = torch.cat((V_b_linear, V_b_angular), 1).float()

            # Calculate Vb from data for each elem
            V_b_label_tensor = self.compute_Vb_label(base_position_batch, base_angle_batch, joint_position_batch, V_b, joint_velocity_batch)

            return V_b_label_tensor, V_b

    def train_loop(self, loss_fn: _Loss, optimizer: Optimizer, epoch: int, writer: SummaryWriter) -> None:
        """Run one epoch of training.

        Args:
            loss_fn (_Loss): The loss function used in the training process
            optimizer (Optimizer): The optimizer used in the training process
            epoch (int): The current training epoch
            writer (SummaryWriter): The updater of the event files to be consumed by TensorBoard
        """

        # Total number of batches
        total_batches = int(len(self.train_dataloader))

        # Cumulative loss
        cumulative_loss = 0
        cumulative_mse_loss = 0
        cumulative_pi_loss = 0

        # Print the learning rate and weight decay of the current epoch
        print('Current lr:', optimizer.param_groups[0]['lr'])
        print('Current wd:', optimizer.param_groups[0]['weight_decay'])

        # Iterate over batches
        for batch, (X, y) in enumerate(self.train_dataloader):

            pred = self(X.float()).double()
            mse_loss = loss_fn(pred, y)

            # Add MSE of Vb and Vbpred
            V_b_label_tensor, V_b = self.get_pi_loss_components(X, pred)
            pi_loss = self.pi_weight * loss_fn(V_b_label_tensor, V_b)

            loss = mse_loss + pi_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()
            cumulative_mse_loss += mse_loss.item()
            cumulative_pi_loss += pi_loss.item()

            # Periodically print the current average loss
            if batch % 1000 == 0:
                current_avg_loss = cumulative_loss/(batch+1)
                current_avg_mse_loss = cumulative_mse_loss/(batch+1)
                current_avg_pi_loss = cumulative_pi_loss/(batch+1)
                print(f"avg loss: {current_avg_loss:>7f}  [{batch:>5d}/{total_batches:>5d}]")
                print(f"avg MSE loss: {current_avg_mse_loss:>7f}")
                print(f"avg PI loss: {current_avg_pi_loss:>7f}")

        # Print the average loss of the current epoch
        avg_loss = cumulative_loss/total_batches
        avg_mse_loss = cumulative_mse_loss/total_batches
        avg_pi_loss = cumulative_pi_loss/total_batches
        print("Final avg loss:", avg_loss)
        print("Final avg MSE loss:", avg_mse_loss)
        print("Final avg PI loss:", avg_pi_loss)

        # Store the average loss, loss components, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_loss', avg_loss, epoch)
        writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch)
        writer.add_scalar('avg_pi_loss', avg_pi_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('wd', optimizer.param_groups[0]['weight_decay'], epoch)
        writer.flush()

    def test_loop(self, loss_fn: _Loss, epoch: int, writer: SummaryWriter) -> None:
        """Test the trained model on the test data.

        Args:
            loss_fn (_Loss): The loss function used for testing
        """

        # Dataset dimension
        num_batches = len(self.test_dataloader)

        # Cumulative loss
        cumulative_test_loss = 0
        cumulative_test_mse_loss = 0
        cumulative_test_pi_loss = 0

        with torch.no_grad():

            # Iterate over the testing dataset
            for X, y in self.test_dataloader:

                pred = self(X.float()).double()
                cumulative_test_mse_loss += loss_fn(pred, y).item()

                V_b_label_tensor, V_b = self.get_pi_loss_components(X, pred)
                cumulative_test_pi_loss += self.pi_weight * loss_fn(V_b_label_tensor, V_b).item()

                cumulative_test_loss = cumulative_test_mse_loss + cumulative_test_pi_loss

        # Print the average test loss at the current epoch
        avg_test_loss = cumulative_test_loss/num_batches
        avg_test_mse_loss = cumulative_test_mse_loss/num_batches
        avg_test_pi_loss = cumulative_test_pi_loss/num_batches
        print(f"Avg test loss: {avg_test_loss:>8f} \n")
        print(f"Avg test MSE loss: {avg_test_mse_loss:>8f} \n")
        print(f"Avg test PI loss: {avg_test_pi_loss:>8f} \n")

        # Store the average loss, loss components, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_test_loss', avg_test_loss, epoch)
        writer.add_scalar('avg_test_mse_loss', avg_test_mse_loss, epoch)
        writer.add_scalar('avg_test_pi_loss', avg_test_pi_loss, epoch)
        writer.flush()

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference step on the given input.

        Args:
            x (torch.Tensor): The input vector for both the Gating and Motion Prediction networks

        Returns:
            pred (torch.Tensor): The output of the Motion Prediction Network
        """

        with torch.no_grad():
            pred = self(x.float()).double()

        return pred

