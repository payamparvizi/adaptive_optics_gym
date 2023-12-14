# Adaptive Optics Reinforcement Learning Gym

## Overview 

This code corresponds to our MPDI Photonics publication. The related paper can be found [here](https://www.mdpi.com/2598538). It includes a simulated environment for satellite communications downlinks, used to train reinforcement learning models for controlling a deformable mirror within a wavefront sensorless Adaptive Optics system.

<p align="center">
  <img src="https://github.com/payamparvizi/adaptive_optics_gym/assets/45127690/6bef8b6a-7f15-4633-b44d-8fba1f29a80c" align="center" width="500">
</p>

### RL Gym

In the generated AO-RL environment, users have a choice between 'quasi_static,' 'semi_dynamic,' or 'dynamic' environments. Additionally, they can adjust the observation-space dimension, action-space dimension, and reward function to suit their preferences. Furthermore, users have the flexibility to change other parameters associated with the environment within the [AO_env.py](./gym_AO/envs/AO_env.py) file.

In this environment, we used the following settings:

Episodic training: Each episode is 30 time steps in a quasi-static environment and 20 time steps in a semi-dynamic one. The RL agent learns to transform the DM from its neutral position to a formation that focuses the beam on the SMF. Users have the flexibility to change the amount of time steps per episode.

Observation-space: The observation-space in the environment is a discretization of the focal plane into a sub-aperture array of either 2x2 or 5x5 pixels, achievable using a fast and relatively low-cost photodetector. Users can modify the dimension of the observation space.

Action-space: The environment offers an action-space where users can choose between two types, the number of actuators ('num_actuators') or Zernike polynomials ('zernike'). In this study, a 64-dimensional action space (64-actuators) and the first 6 modes of Zernike polynomials are used. Users have the capability to adjust the type and dimension of the action-space.

Reward function: The reward function is calculated as the Strehl ratio of the optical system or a new reward function ($r_2$) as detailed in the paper. Users have the freedom to implement their own reward function as desired.

Overall, through the available line of code in [main.py](main.py), users can modify parameters such as atmosphere type, velocity, and Fried parameter, alongside action-space type and dimension, observation-space dimension, and the chosen reward function, and the amount of time step per episode 


```
    env_AO = gym.make(args.env_name,
                    atm_type='quasi_static',              # atmospheric condition: 'quasi_static', 'semi_dynamic', 'dynamic'
                    atm_vel = 0,                          # atmosphere velocity
                    atm_fried = 0.20,                     # Fried parameter of the atmosphere
                    act_type = 'num_actuators',           # action type: 'num_actuators', 'zernike'
                    act_dim = 64,                         # action dimension
                    obs_dim = 2,                          # observation dimension
                    rew_type = 'strehl_ratio',            # reward type: 'strehl_ratio', 'smf_ssim'
                    rew_threshold = None,                 # Threshould of the reward value
                    timesteps_per_episode= 30,            # Number of timesteps per episode
                    flat_mirror_start_per_episode = True, # If we want each episode to start with flat mirror
                    SH_operation=False)                   # If we require Shack_Hartmann wavefront sensor operation
                    )
```

## Installation Instructions

In this work, we used Windows operating system with Python 3.9.6 version. First, the modules can be installed through [requirements.txt](requirements.txt) file through PyPI using the command below in Windows:

```
python -m venv ao_gym_env python=3.9.6
ao_gym_env\scripts\activate
pip install -r requirements.txt
```

Or if using Linux:

```
python -m venv ao_gym_env
source ao_gym_env/bin/activate
module load python/3.9.6
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

https://docs.hcipy.org/0.4.0/installation.html


### Pytorch

To develop our Reinforcement Learning algorithms, we employed the PyTorch framework, which is a machine learning framework built on the Torch library. In this work, pytorch==2.0.1 version is installed in Windows with CPU compute platform ([Paszke et al. 2019](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)).

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
or installation through PyPI using the command:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```


PyTorch provides users with a range of choices for selecting their operating system (Linux/Mac/Windows), the package (Conda/Pip) and compute platform (CUDA/CPU). To find the suitable Pytorch framework check the link:

https://pytorch.org/get-started/locally/


Previous versions of PyTorch can be found in the link below:

https://pytorch.org/get-started/previous-versions/


### Gym and Gymnasium
The Gym and Gymnasium libraries are Python-based open-source frameworks designed by OpenAI to facilitate the development and comparison of reinforcement learning algorithms. These libraries offers a standardized Application Programming Interface (API) to enable communication between learning algorithms and environments ([Brockman et al. 2016](https://arxiv.org/abs/1606.01540)).

OpenAI Gym and Gymnasium are available on PyPI. 

```
pip install gym==0.26.0
pip install gymnasium==0.29.1
```

installation through Conda using the command:

```
conda install -c conda-forge gym
conda install -c conda-forge gymnasium
```


Other versions of the Gym framework can be found in the link below:

https://pypi.org/project/gym/#history


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
- **asdf** (Advanced Scientific Data Format)


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
python main.py --mode test --algorithm_name PPO --actor_model ppo_actor.pth
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
python main.py --mode test --algorithm_name SAC --actor_model sac_actor.pth
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
python main.py --mode test --algorithm_name DDPG --actor_model ddpg_actor.pth
```

### Shack-Hartmann Wavefront Sensor

The Shack-Hartmann wavefront sensor method is utilized as a point of reference for comparative purposes. It is not for the purpose of training.

To illustrate how Shack-Hartmann wavefront sensor works:
```
python main.py --algorithm_name SHACK 
```

Note that Shack-Hartmann method does not work on other Gym environments.

## How it works

### [main.py](main.py)

The [main.py](main.py) file serves as the sole executable file responsible for parsing the arguments defined in [arguments.py](arguments.py). Also, the main file initializes both the environment and RL algorithms using the parameters specified within this file.

Within this file, there are two dictionaries and an environment function that contain default parameters that users can modify:

- The 'default_parameters' dictionary allows adjustments to the general parameters used such as the total number of training timesteps, rendering, and the frequency of showing reward, and logs, and saving actor/critic.
```
  default_parameters = {
      'algorithm_name': algorithm_name,        # The algorithm used
  
      'total_timesteps': 150_000,              # total timesteps for training
  
      'freq_log': 2,                           # Number of training iterations to display logger/information
      'freq_rew': 20,                          # Number of training iterations to display reward plots
      'freq_render': 20,                       # Number of training episodes to render environment
      'freq_ac_save': 20,                      # Number of training iterations to save actor/critic
  
      'render': True,                          # Rendering environment
      'seed': 10,                              # Seed for random number generators
      }
```

- The 'hyperparameters' dictionary contains parameters for each RL algorithm, as each algorithm has its own set of hyperparameters. Here, you can find the hyperparameters of PPO algorithm as an example:
  
```
  hyperparameters = {
  
  # Actor information
  'lr_actor': 1e-2,                  # Actor learning rate
  'weight_decay_actor': 0,           # Actor weight decay to prevent overfitting
  'hidden_dim_actor': 150,           # Actor hidden dimension size
  
  # Critic information
  'lr_critic': 5e-6,                 # Actor learning rate
  'weight_decay_critic': 0,          # Actor weight decay to prevent overfitting
  'hidden_dim_critic': 50,           # Actor hidden dimension size
  
  # other parameters
  'clip': 0.35,                      # Clipping in the policy objective
  'gamma': float(0.95),              # Discount factor (between 0 and 1)
  
  # Training information
  'epoch': 1,                        # Number of epochs
  'timesteps_per_episode': 30,       # Number of timesteps in episodes
  'episodes_per_iteration': 2,       # Number of episodes in iterations
  'updates_per_iteration': 20,       # Number of updates per iteration
  }
```

- The parameters within the 'env' function include atmospheric turbulence type, velocity, Fried parameter, alongside action-space type and size, observation space size, reward type, size, etc. These parameters define part of the environment’s characteristics.

```
    env_AO = gym.make(args.env_name,
                    atm_type='quasi_static',              # atmospheric condition: 'quasi_static', 'semi_dynamic', 'dynamic'
                    atm_vel = 0,                          # atmosphere velocity
                    atm_fried = 0.20,                     # Fried parameter of the atmosphere
                    act_type = 'num_actuators',           # action type: 'num_actuators', 'zernike'
                    act_dim = 64,                         # action dimension
                    obs_dim = 2,                          # observation dimension
                    rew_type = 'strehl_ratio',            # reward type: 'strehl_ratio', 'smf_ssim'
                    rew_threshold = None,                 # Threshould of the reward value
                    timesteps_per_episode= 30,            # Number of timesteps per episode
                    flat_mirror_start_per_episode = True, # If we want each episode to start with flat mirror
                    SH_operation=False)                   # If we require Shack_Hartmann wavefront sensor operation
                    )
```


### [arguments.py](arguments.py)

[arguments.py](arguments.py) file presents a set of arguments intended to be parsed at the command line. These arguments will be called by the [main.py](main.py) file. This file contains train/test modes, the Gym environment name, RL algorithm name, and actor/critic files to call. The instructions for using command lines are detailed in the 'Running the RL environment' section.

```
    parser.add_argument('--environment_name', dest='environment_name', type=str, default='AO-v0')
    parser.add_argument('--algorithm_name', dest='algorithm_name', type=str, default='PPO')
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')          # Actor for PPO, SAC and DDPG
    parser.add_argument('--criticQ1_model', dest='criticQ1_model', type=str, default='')    # Critic-1 of SAC and DDPG
    parser.add_argument('--criticQ2_model', dest='criticQ2_model', type=str, default='')    # Critic-2 of SAC
    parser.add_argument('--criticV_model', dest='criticV_model', type=str, default='')      # Value function of PPO
```

Using actor and critic (and value function) arguments helps in continuing training and testing the networks. Using the command line, when starting training from scratch, the actor/critic argument will remain empty. However, if we wish to continue training or testing actors and critics, we can specify the file names within the command line. 


### [algorithm.py](algorithm.py)

[algorithm.py](algorithm.py) contains the PPO, SAC, and DDPG algorithms and the training process. Within this file, we save the actor and critic networks and reward values per iteration with the frequency specified by 'freq_ac_save' and 'freq_rew' in the default_parameters dictionary in [main.py](main.py).

### [eval_policy.py](eval_policy.py)

[eval_policy.py](eval_policy.py) contains the code to evaluate the agent. After  [main.py](main.py) finishes training (or saves the actor during training), we can load the actor and test it using the command lines specified in the 'Running the RL Environment' section.

### [network.py](network.py)

[network.py](network.py) contains neural networks used to define actor and critic (value function) networks in PPO, SAC, and DDPG. It also contains the Ornstein-Uhlenbeck Noise function that is used for the noise added to the actor of DDPG. The parameters of the neural networks can be modified from the hyperparameters dictionary available in [main.py](main.py).

### [replay_buffer.py](replay_buffer.py)

[replay_buffer.py](replay_buffer.py) file contains the random replay buffer that is used for storing and random sampling tuples (observation, action, reward, next_observation, done) by off-policy SAC and DDPG algorithms in [algorithm.py](algorithm.py).

### [AO_env.py](./gym_AO/envs/AO_env.py)

[AO_env.py](./gym_AO/envs/AO_env.py) is the Gym RL environment that is used for the training and testing of RL models. It contains reset(), step(), and render() functions. It also contains the Shack-Hartmann wavefront sensor function which is used for comparison in the paper. There are some default parameters used in the RL environment (parameters_init function), which can be modified as wished, such as telescope, pupil, wavefront, deformable mirror, atmosphere, fiber, and wavefront sensor configurations.

```
        parameters = {
            # telescope configuration:
            'telescope_diameter': 0.5,                     # diameter of the telescope in meters

            # pupil configuration
            'num_pupil_pixels' : 240,                      # Number of pupil grid pixels

            # wavefront configuration
            'wavelength_wfs' : 1.5e-6,                     # wavelength of wavefront sensing in micro-meters
            'wavelength_sci' : 2.2e-6,                     # wavelength of scientific channel in micro-meters

            # deformable mirror configuration
            'num_modes' : act_dim,                         # Number of actuators in Deformable mirror

            # Atmosphere configuration
            'delta_t': 1e-3,                               # in seconds, for a loop speed of 1 kHz
            'max_steps': timesteps_per_episode,                               # Maximum number of timesteps in an episode
            'velocity': velocity_value,                    # the velocity of attmosphere
            'fried_parameter' : fried_parameter,                      # The Fried parameter in meters
            'outer_scale' : 10,                            # The outer scale of the phase structure function in meters

            # Fiber configuration
            'D_pupil_fiber' : 0.5,                         # Diameter of the pupil for fiber
            'num_pupil_pixels_fiber': 128,                 # Number of pupil grid pixels for fiber
            'num_focal_pixels_fiber' : 128,                # Number of focal grid pixels for fiber
            'num_focal_pixels_fiber_subsample' : obs_dim,  # Number of focal grid pixels for fiber subsampled for quadrant photodetector
            'multimode_fiber_core_radius' : 25 * 1e-6,     # the radius of the multi-mode fiber
            'singlemode_fiber_core_radius' : 4.5 * 1e-6,   # the radius of the single-mode fiber
            'fiber_NA' : 0.14,                             # Fiber numerical aperture 
            'fiber_length': 10,

            # wavefront sensor configuration
            'f_number' : 50,                               # F-ratio
            'num_lenslets' : 12,                           # Number of lenslets along one diameter
            'sh_diameter' : 5e-3,                          # diameter of the sensor in meters
            'stellar_magnitude' : -5,                      # measurement of brightness for stars and other objects in space
            }
```

## Citation

For MDPI and ACS Style citation:

```
Parvizi, P.; Zou, R.; Bellinger, C.; Cheriton, R.; Spinello, D. Reinforcement Learning Environment for Wavefront Sensorless Adaptive Optics in Single-Mode Fiber Coupled Optical Satellite Communications Downlinks. Photonics 2023, 10, 1371. https://doi.org/10.3390/photonics10121371
```

## Acknowledgement

This research was supported by the National Science and Engineering Research Council (NSERC) of Canada through Discovery grant RGPIN-2022-03921, and by the National Research Council (NRC) of Canada's [AI for Design Challenge program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program) through the AI4D grant AI4D-135-2.
