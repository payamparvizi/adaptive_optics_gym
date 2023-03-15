# Adaptive Optics Reinforcement Learning Gym

## Overview 

This code corresponds to our IJCAI 2023 AI And Social Good submission. This includes a simulated satellite-to-ground communication environment for training reinforcement learning to control a deformable mirror in a wavefront sensorless

<p align="center">
  <img src="https://user-images.githubusercontent.com/45127690/223493910-2f739b38-3fd7-4a8e-97ea-a13e39b044a3.png" align="center" width="305" height="350">
</p>

### RL Gym

Episodic training: each episode is 30 time steps long. The RL agent learns transform the DM from its neutral position to a formation that focuses the beam on the SMF.

Observation space: The observation space in the environment is a discretization of the focal plane into a sub-aperture array of 2x2 pixels that can be realized with a fast and relatively low-cost quadrant photodetector.

Action space: The environment includes a 64-dimensional continuous action space that simulates 64-actuator segmented DM that has roughly 8 actuators across a linear dimension.

Reward function: The reward function is calculated as the Strehl ratio of the optical system.

## Installation Instructions

In this work, we used Windows operating system with Python 3.9.7 version. First, the modules can be installed through [requirements.txt](requirements.txt) file through PyPI using the command below in Windows:

```
python -m venv ao_gym_env
ao_gym_env\scripts\activate
pip install -r requirements.txt
```

Or if using Linux:

```
python -m venv ao_gym_env
source ao_gym_env/bin/activate
pip install -r requirements.txt
```

<!---
The modules can also be installed through [requirements_conda.txt](requirements_conda.txt) file through Conda using the command:

```
conda create -n ao_gym_env python==3.9.7
conda activate ao_gym_env
conda install --file requirements_conda.txt
```
-->

The following commands can be executed to install all the required modules. Nevertheless, if there is an interest to install each module individually, the commands can be found below. 

The modules utilized in this work are as follows:

### HCIPy: High Contrast Imaging for Python

The Adaptive Optics Gym framework is developed utilizing the HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments ([Por et al. 2018](https://doi.org/10.1117/12.2314407)).

It is available for installation through PyPI using the command:

```
pip install hcipy --upgrade
```

Also, it is possible to install the latest deveopment version from Github by:

```
git clone https://github.com/ehpor/hcipy
cd hcipy
pip install -e .
```

For more comprehensive instructions on installing the HCIPy framework, please refer to the link provided here:

```
https://docs.hcipy.org/0.4.0/installation.html
```

### Pytorch

To develop our Reinforcement Learning algorithms, we employed the PyTorch framework, which is a machine learning framework built on the Torch library. In this work, pytorch==1.12.1 version is installed in Windows with CPU compute platform ([Paszke et al. 2019](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)).

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```
or installation through PyPI using the command:
```
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```


PyTorch provides users with a range of choices for selecting their operating system (Linux/Mac/Windows), the package (Conda/Pip) and compute platform (CUDA/CPU). To find the suitable Pytorch framework check the link:
```
https://pytorch.org/get-started/locally/
```

Previous versions of PyTorch can be found in the link below:
```
https://pytorch.org/get-started/previous-versions/
```

### Gym
The Gym library is a Python-based open-source framework designed by OpenAI to facilitate the development and comparison of reinforcement learning algorithms. The library offers a standardized Application Programming Interface (API) to enable communication between learning algorithms and environments ([Brockman et al. 2016](https://arxiv.org/abs/1606.01540)).

In this work, we used the version of '0.23.1'. OpenAI Gym is available on PyPI. 

```
pip install gym==0.23.1
```

installation through Conda using the command:

```
conda install -c conda-forge gym
```


Other versions of the Gym framework can be found in the link below:
```
https://pypi.org/project/gym/#history
```

### Other dependencies

There are other dependencies that have been utilized in this work as follows:
- **scipy** (for advanced linear algebra)
- **numpy** (for all numerical calculations)
- **pandas** (for data manipulation and analysis)
- **matplotlib** (for visualisations)

Modules that are part of the standard library in Python:
- **pickle** (for storing or loading data) 
- **argparse** (for command-line parsing)  
- **sys** (for manipulating the runtime environment)
- **os** (for creating the naming a directory)
- **random** (for generating random numbers)


## Running the RL environment 

In this work, Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Deep Deterministic Policy Gradient (DDPG) algorithms can be used in any Gym environment, along with Shack-Hartmann for Adaptive Optics environment (‘AO-v0’). 

It is worth to note that a portion of the code's foundation was derived from Yang Yu's [Github repository](https://github.com/ericyangyu/PPO-for-Beginners)

### Proximal Policy Optimization (PPO)
The default configuration for execution entails employing the Proximal Policy Optimization (PPO) algorithm in the Adaptive Optics environment.

For training PPO on Adaptive Optics from scratch:
```
python main.py
```
or
```
python main.py --mode train --algorithm_name PPO 
```

To continue training with the existing actor and value function:
```
python main.py --mode train --algorithm_name PPO --actor_model ppo_actor.pth --criticV_model ppo_Vcritic.pth 
```

For testing the actor of PPO:
```
python main.py --mode test --actor_model ppo_actor.pth
```

### Soft Actor-Critic (SAC)

For training SAC on Adaptive Optics from scratch:
```
python main.py --algorithm_name SAC 
```
or
```
python main.py --mode train --algorithm_name SAC 
```


To continue training with the existing actor and critics:
```
python main.py --mode train --algorithm_name SAC --actor_model sac_actor.pth --criticQ1_model sac_critic1.pth --criticQ2_model sac_critic2.pth  
```

For testing the actor of SAC:
```
python main.py --mode test --actor_model sac_actor.pth
```

### Deep Deterministic Policy Gradient (DDPG)

For training DDPG on Adaptive Optics from scratch:
```
python main.py --algorithm_name DDPG 
```
or
```
python main.py --mode train --algorithm_name DDPG 
```


To continue training with the existing actor and critic:
```
python main.py --mode train --algorithm_name DDPG --actor_model ddpg_actor.pth --criticQ1_model ddpg_critic.pth
```

For testing the actor of DDPG:
```
python main.py --mode test --actor_model ddpg_actor.pth
```

### Shack-Hartmann Wavefront Sensor

The Shack-Hartmann wavefront sensor method is utilized as a point of reference for comparative purposes. It is not for the purpose of training.

To illustrate how Shack-Hartmann wavefront sensor works:
```
python main.py --algorithm_name SHACK 
```

Note that Shack-Hartmann method does not work on other Gym environments.

### Changing Parameters

In order to modify the hyperparameters of the algorithms, make changes within the [main.py](main.py) file. Also, to adjust the parameters of the Adaptive Optics environment, it is advised to make modifications within the [AO_env.py](./gym_AO/envs/AO_env.py) file.

Also, if interested to use other Gym environments, you can add --environment_name parser in above commands.

For example: --environment_name Pendulum-v1


## Citation

For Bibtex citation:

```
@misc{parvizi2023reinforcement,
      title={Reinforcement Learning-based Wavefront Sensorless Adaptive Optics Approaches for Satellite-to-Ground Laser Communication}, 
      author={Payam Parvizi and Runnan Zou and Colin Bellinger and Ross Cheriton and Davide Spinello},
      year={2023},
      eprint={2303.07516},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

This research was supported by the National Science and Engineering Research Council (NSERC) of Canada through Discovery grant RGPIN-2022-03921, and by the National Research Council (NRC) of Canada's [AI for Design Challenge program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program) through the AI4D grant AI4D-135-2.
