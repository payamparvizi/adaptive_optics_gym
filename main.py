
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:54:46 2022

@author: Payam Parvizi
"""

environment_name = 'AO-v0'                  # 'Pendulum-v1'/'AO-v0'/ any GYM
algorithm = 'SHACK'                           # 'PPO'/'SAC'/'DDPG'/'SHACK'

save_name = 'PPO_01'                # create a file to save figures and actor/critic


import gym
import gym_AO

import sys
import torch

from arguments import get_args
from algorithm import ALGORITHM
from network import Actor, SoftQCritic1, SoftQCritic2, SoftVCritic
from eval_policy import eval_policy
import pandas as pd


def train(env, hyperparameters, default_parameters, actor_model, critic1_model, critic2_model, criticV_model, env_name):

    model = ALGORITHM(policy_class1=Actor, critic_class1=SoftQCritic1, critic_class2=SoftQCritic2, critic_classV=SoftVCritic,  
                env=env, env_name=environment_name, default_parameters=default_parameters, **hyperparameters)
    
    if algorithm == 'SAC': 
        if actor_model == '' and critic1_model == '' and critic2_model == '':
            print(f"Running from scratch for SAC ...")
            
        elif actor_model != '' and critic1_model != '' and critic2_model != '':
            print(f"Loading in {actor_model} and {critic1_model} and {critic2_model} for SAC ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(critic1_model))
            model.softq_critic2.load_state_dict(torch.load(critic2_model))
            print(f"Successfully loaded for SAC.", flush=True)
            
        elif actor_model != '' or critic1_model != '' or critic2_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to SAC algorithm")
            sys.exit(0)
    
    
    elif algorithm == 'DDPG': 
        if actor_model == '' and critic1_model == '':
            print(f"Running from scratch for DDPG ...")
            
        elif actor_model != '' and critic1_model != '':
            print(f"Loading in {actor_model} and {critic1_model} and {critic2_model} for DDPG ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(critic1_model))
            print(f"Successfully loaded for DDPG.", flush=True)
            
        elif actor_model != '' or critic1_model != '' or critic2_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to DDPG algorithm")
            sys.exit(0)
            
            
    elif algorithm == 'PPO':
        if actor_model == '' and criticV_model == '':
            print(f"Running from scratch for PPO ...")
            
        elif actor_model != '' and criticV_model != '':
            print(f"Loading in {actor_model} and {criticV_model} for PPO ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softV_critic.load_state_dict(torch.load(criticV_model))
            print(f"Successfully loaded for PPO.", flush=True)
            
        elif actor_model != '' or criticV_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to PPO algorithm")
            sys.exit(0)
    
    model.learn(total_timesteps = 150_000)   # 36_300

def test(env, actor_model, env_name):
    
    hidden_dim_actor = 250
    
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
    
	# Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
        
	# Build our policy the same way we build our actor model in PPO/SAC
    policy = Actor(obs_dim, act_dim, hidden_dim_actor)

	# Load in the actor model saved by the PPO/SAC algorithm
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, render=True)


def main(args):
    
    default_parameters = {
                'algorithm': algorithm,
                'name_save': save_name,
                
                'epoch': 1,    # 100
                'max_timesteps_per_episode': 30,      # 30
                
                'gamma': float(0.95), 
                
                'freq_log': 2,           
                'freq_rew': 20, 
                'freq_env': 20, 
                'freq_ac_save': 20, 

                'seed': 10,
                
                'render': True,
                'patience_stopping': 40,    # number of batches to 
        }
    
    
    if algorithm == 'SAC':
        hyperparameters = {
                
                'buffer': int(1000000),
                'buffer_size': 128,
                
				'lr_actor': 5e-4,            
                'weight_decay_actor': 0,        
                
				'lr_critic': 1e-2,                     
                'weight_decay_critic': 0,        
                
                'alpha' : 1.0,
                'lr_alpha': 1e-1,
                'weight_decay_alpha': 0,
                'alpha_stop': 0.4,
                
                'hidden_dim_actor': 150,
                'hidden_dim_critic': 80,
                
                'number_of_episodes_per_batch': 1,          
                'n_updates_per_iteration': 20,   # 20
                
                'tau': 1e-2,   
			  }
        
        
    elif algorithm == 'DDPG':
        hyperparameters = {
                
                'buffer': int(1000000),
                'buffer_size': 256,
                
				'lr_actor': 5e-5,              
                'weight_decay_actor': 0,        
                
				'lr_critic': 1e-2,             
                'weight_decay_critic': 0,        
                
                'hidden_dim_actor': 250,
                'hidden_dim_critic': 65,
                
                'number_of_episodes_per_batch': 2,          
                'n_updates_per_iteration': 20,  
                
                'tau': 1e-2,
			  }
    
    
    elif algorithm == 'PPO':
        hyperparameters = {
                
				'lr_actor': 1e-2,                 
                'weight_decay_actor': 0,         
                
                'lr_criticV': 5e-6,         
                'weight_decay_criticV': 0,
                
				'clip': 0.35,
                
                'hidden_dim_actor': 150,
                'hidden_dim_Vcritic': 50,
                
                'number_of_episodes_per_batch': 2,          
                'n_updates_per_iteration': 20,   # 20
			  }
      
    elif algorithm == 'SHACK':
        hyperparameters = {
            'number_of_episodes_per_batch': 1
            }
        
        if environment_name != 'AO-v0':
            print(f"Error: Shack-Hartmann algorithm only works for Adaptive optics environment")
            print(f"Check the environment and try again")
            sys.exit(0)
    
    env = gym.make(environment_name)

	# Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, default_parameters=default_parameters, 
                  actor_model=args.actor_model, critic1_model=args.critic1_model, critic2_model=args.critic2_model, 
                  criticV_model=args.criticV_model, env_name=environment_name)
            
    elif args.mode == 'test':
        test(env=env, actor_model=args.actor_model, env_name=environment_name)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
