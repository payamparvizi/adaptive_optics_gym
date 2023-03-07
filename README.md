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

The Adaptive Optics Gym framework is developed utilizing the HCIPy, an open-source object-oriented framework written in Python for performing end-to-end simulations of high-contrast imaging instruments (Por et al. 2018). 

### Dependencies
fsdf
## Running the RL environment 

## Training the models from the paper

## Citation

## Acknowledgement

## References

This work was graciously funded by the [University of Ottawa](https://www.uottawa.ca) and the National Research Council of Canada's [AI for Design program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program). 
