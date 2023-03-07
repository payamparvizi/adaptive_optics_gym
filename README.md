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

To develop our Reinforcement Learning algorithms, we employed the PyTorch framework, which is a machine learning framework built on the Torch library. In this work, pytorch==1.12.1 version is used. 

PyTorch provides users with a range of choices for selecting their operating system (Linux/Mac/Windows), the package (conda/pip) and compute platform (CUDA/CPU). To find the suitable Pytorch framework check the link:
```
https://pytorch.org/get-started/locally/
```

### Other dependencies
fsdf
## Running the RL environment 

## Training the models from the paper

## Citation

## Acknowledgement

This work was graciously funded by the [University of Ottawa](https://www.uottawa.ca) and the National Research Council of Canada's [AI for Design program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program). 
