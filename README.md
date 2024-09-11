<h1 align="center">
Trajectory Generation with Physics-Informed Learning and Drift Mitigation
</h1>

<div align="center">
Evelyn D'Elia, Paolo Maria Viceconte, Lorenzo Rapetti, Diego Ferigo, Giulio Romualdi, Giuseppe L'Erario, Raffaello Camoriano, and Daniele Pucci
</div>

<br>


<div align="center">
    Submitted to IEEE Robotics and Automation Letters (RA-L).
</div>

<div align="center">
    <a href="#installation"> <b> Installation </b> | </a> <a href="https://huggingface.co/datasets/ami-iit/paper_romualdi_viceconte_2024_icra_dnn-mpc-walking_dataset"> <b> Paper </b>  | </a> <a href="https://huggingface.co/datasets/ami-iit/paper_romualdi_viceconte_2024_icra_dnn-mpc-walking_dataset"> <b> Dataset </b>  | </a> <a href="https://huggingface.co/datasets/ami-iit/paper_romualdi_viceconte_2024_icra_dnn-mpc-walking_dataset"> <b> Video </b>  </a>
</div>


## Installation

We recommend using conda/mamba, available [here](https://github.com/conda-forge/miniforge), for this installation.

First, create and activate a conda environment with the necessary dependencies.
```
mamba create -n pi_trajectory_gen -c conda-forge bipedal-locomotion-framework jaxsim pytorch tensorboard adam-robotics urdf-parser-py h5py
mamba activate pi_trajectory_gen
```

Next, within the conda environment, clone and install this repo:
```
git clone https://github.com/ami-iit/paper_delia_2024_ral_physics-informed_trajectory_generation.git
cd paper_delia_2024_ral_physics-informed_trajectory_generation
pip install .
```

Now you're ready to run the code.

## Running the code
To replicate our results, several scripts are available in the `scripts` folder.
```
cd scripts/
```

### Retargeting the data
The raw dataset contains 5 subsets: forward walking, backward walking, side walking, diagonal walking, and mixed walking.
The raw data is recorded on a human and therefore needs to be retargeted onto the robot model. For example, the mirrored version of the forward walking subset is retargeted using:
```
python retargeting.py --KFWBGR --filename ../datasets/mocap/D2/1_forward_normal_step/data.log
```

### Extracting features from the data
With the retargeted data we can extract the features (data and labels) for training, for example:
```
python features_extraction.py --dataset D2 --portion 1
```

### Training the model
With the extracted features we can train the model. The weight given to the PI loss component can be specified as an argument.
```
python training_pytorch.py --pi_weight 1.0
```

### Running the trajectory generator with a trained model
With a trained model, we can give dummy inputs and generate a trajectory based on the model's predictions.
```
python trajectory_generation.py
```
The various parameters used for the trajectory generation can be tuned in the files present in the `config` folder. For example, in `config_mann.toml`, the trained model can be changed by editing the `onnx_model_path` parameter, and the correction block gains can be updated by changing the `linear_pid_gain` and `rotational_pid_gain` parameters.

### Replicating plots in the paper
We also include a script to reproduce some of the plots from our paper.
```
python show_selected_plots.py
```

## Maintainer

<table align="left">
    <tr>
        <td><a href="https://github.com/evelyd"><img src="https://github.com/evelyd.png" width="40"></a></td>
        <td><a href="https://github.com/evelyd"> @evelyd</a></td>
    </tr>
</table>
