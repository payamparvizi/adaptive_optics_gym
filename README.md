# Adaptive Optics Reinforcement Learning Gym

## Overview 

This code corresponds to our IJCAI 2023 AI And Social Good submission. This includes a simulated satellite-to-ground communication environment for training reinforcement learning to control a deformable mirror in a wavefront sensorless

![](https://github.com/cbellinger27/adaptive_optics_gym/files/10845103/ao_system_2.pdf?raw=true)

### RL Gym

Episodic training: each episode is 30 time steps long. The RL agent learns transform the DM from its neutral position to a formation that focuses the beam on the SMF.

Observation space: The observation space in the environment is a discretization of the focal plane into a sub-aperture array of 2x2 pixels that can be realized with a fast and relatively low-cost quadrant photodetector.

Action space: The environment includes a 64-dimensional continuous action space that simulates 64-actuator segmented DM that has roughly 8 actuators across a linear dimension.

Reward function: The reward function is calculated as the Strehl ratio of the optical system.

## Installation Instructions

## Running the RL environment 

## Training the models from the paper

## Citation

## Acknowledgement

This work was graciously funded by the [University of Ottawa](https://www.uottawa.ca) and the National Research Council of Canada's [AI for Design program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program). 
