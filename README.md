<h1 align="center">
Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning and Control-Informed Steering
</h1>

<div align="center">
Evelyn D'Elia, Paolo Maria Viceconte, Lorenzo Rapetti, Diego Ferigo, Giulio Romualdi, Giuseppe L'Erario, Raffaello Camoriano, and Daniele Pucci
</div>

<br>


https://github.com/user-attachments/assets/00144d6f-c059-47f7-9e19-f2533a9a40a1


<div align="center">
    Submitted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
</div>

<div align="center">
    <a href="#installation"> <b> Installation </b> | </a> <a href="https://github.com/ami-iit/paper_delia_2025_iros_physics-informed_trajectory_generation/tree/main"> <b> Paper </b>  | </a> <a href="https://huggingface.co/datasets/evelyd/paper_delia_2025_iros_physics-informed_trajectory_generation_dataset"> <b> Dataset </b> </a>
</div>

## Installation

We recommend using conda/mamba, available [here](https://github.com/conda-forge/miniforge), for this installation.

First, create and activate a conda environment with the necessary dependencies.
```bash
mamba create -n pi_trajectory_gen -c conda-forge bipedal-locomotion-framework jaxsim pytorch tensorboard adam-robotics jax2torch urdf-parser-py h5py ergocub-models meshcat-python
mamba activate pi_trajectory_gen
```

Next, within the conda environment, clone and install this repo:
```bash
git clone https://github.com/ami-iit/paper_delia_2025_iros_physics-informed_trajectory_generation.git
cd paper_delia_2025_iros_physics-informed_trajectory_generation
pip install .
```

You will also need to download the datasets necessary for running the code into this repo. This may take some time depending on your internet connection.
```bash
git clone git@hf.co:datasets/evelyd/paper_delia_2025_iros_physics-informed_trajectory_generation_dataset datasets/
cd datasets/
unzip D2.zip
```

Now you're ready to run the code.

## Running the code
To replicate our results, several scripts are available in the `scripts` folder.
```bash
cd ../scripts/
```

### Retargeting the data
The raw dataset contains 5 subsets: forward walking, backward walking, side walking, diagonal walking, and mixed walking.
The raw data is recorded on a human and therefore needs to be retargeted onto the robot model. For example, the mirrored version of the forward walking subset is retargeted using:
```bash
python retargeting.py --KFWBGR --filename ../datasets/D2/1_forward_normal_step/data.log
```

### Extracting features from the data
With the retargeted data we can extract the features (data and labels) for training, for example:
```bash
python features_extraction.py --dataset D2 --portion 1
```

### Training the model
With the extracted features we can train the model. The weight given to the PI loss component can be specified as an argument.
```bash
python training.py --pi_weight 1.0
```

### Running the trajectory generator with a trained model
With a trained model, we can give dummy inputs and generate a trajectory based on the model's predictions.
```bash
python trajectory_generation.py
```
The various parameters used for the trajectory generation can be tuned in the files present in the `config` folder. For example, in `config_mann.toml`, the trained model can be changed by editing the `onnx_model_path` parameter, and the correction block gains can be updated by changing the `linear_pid_gain` and `rotational_pid_gain` parameters.

### Replicating plots in the paper
We also include a script to reproduce some of the plots from our paper.
```bash
python show_selected_plots.py
```

## Maintainer

<table align="left">
    <tr>
        <td><a href="https://github.com/evelyd"><img src="https://github.com/evelyd.png" width="40"></a></td>
        <td><a href="https://github.com/evelyd"> @evelyd</a></td>
    </tr>
</table>
