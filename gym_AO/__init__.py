# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:39:12 2023

@author: payam
"""
from gymnasium.envs.registration import register

register(
    id='AO-v0',
    entry_point='gym_AO.envs:AOEnv',
)