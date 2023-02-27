"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', dest='mode', type=str, default='train')                                # 'train'/'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')         # ''/'sac_actor.pth'/'ppo_actor.pth'
    parser.add_argument('--critic1_model', dest='critic1_model', type=str, default='')   # ''/'sac_critic1.pth.pth'
    parser.add_argument('--critic2_model', dest='critic2_model', type=str, default='')                   # ''/'sac_critic2.pth'
    parser.add_argument('--criticV_model', dest='criticV_model', type=str, default='')                   # ''/'ppo_Vcritic.pth'
    
    args = parser.parse_args()
    
    return args
