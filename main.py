"""
        The following file presents an executable file capable of executing 
        Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and 
        Deep Deterministic Policy Gradient (DDPG) algorithms for any 
        Gym environment, along with Shack-Hartmann for Adaptive Optics 
        environment.
"""


#----------------------------- Importing modules -----------------------------#
import sys
import gym
import gym_AO
import torch
import os

from arguments import get_args
from algorithm import ALGORITHM
from network import Actor, CriticQ1, CriticQ2, CriticV
from eval_policy import eval_policy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#--------------------------------- Training ----------------------------------#
def train(env, hyperparameters, default_parameters, algorithm_name, actor_model, 
          criticQ1_model, criticQ2_model, criticV_model):

    model = ALGORITHM(actor_model=Actor, criticQ1_model=CriticQ1, criticQ2_model=CriticQ2, 
                      criticV_model=CriticV,  env=env, default_parameters=default_parameters, 
                      **hyperparameters)


    # Running SAC from scratch or importing actor/critics to continue training
    if algorithm_name == 'SAC':
        if actor_model == '' and criticQ1_model == '' and criticQ2_model == '':
            print(f"Running SAC from scratch ...")

        elif actor_model != '' and criticQ1_model != '' and criticQ2_model != '':
            print(f"Loading in {actor_model} and {criticQ1_model} and {criticQ2_model} for SAC ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(criticQ1_model))
            model.softq_critic2.load_state_dict(torch.load(criticQ2_model))
            print(f"Successfully loaded for SAC.", flush=True)

        elif actor_model != '' or criticQ1_model != '' or criticQ2_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to SAC algorithm")
            sys.exit(0)


    # Running DDPG from scratch or importing actor/critic to continue training
    if algorithm_name == 'DDPG':
        if actor_model == '' and criticQ1_model == '':
            print(f"Running DDPG from scratch ...")

        elif actor_model != '' and criticQ1_model != '':
            print(f"Loading in {actor_model} and {criticQ1_model} for DDPG ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(criticQ1_model))
            print(f"Successfully loaded for DDPG.", flush=True)

        elif actor_model != '' or criticQ1_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to DDPG algorithm")
            sys.exit(0)


    # Running PPO from scratch or importing actor/critic to continue training
    if algorithm_name == 'PPO':
        if actor_model == '' and criticV_model == '':
            print(f"Running PPO from scratch ...")

        elif actor_model != '' and criticV_model != '':
            print(f"Loading in {actor_model} and {criticV_model} for PPO ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softV_critic.load_state_dict(torch.load(criticV_model))
            print(f"Successfully loaded for PPO.", flush=True)

        elif actor_model != '' or criticV_model != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to PPO algorithm")
            sys.exit(0)

    # Learning process ...
    model.learn(default_parameters.get('total_timesteps'))


#--------------------------------- Testing -----------------------------------#
def test(env, actor_model, hyperparameters, default_parameters):

    hidden_dim_actor = hyperparameters.get('hidden_dim_actor')

    if actor_model == '':
        print(f"Didn't specify model file in argument file", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Building our policy
    policy = Actor(obs_dim, act_dim, hidden_dim_actor)

    # Load in the actor model saved before
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate the policy
    eval_policy(policy=policy, env=env, render=True)


#----------------------------------- main ------------------------------------#
def main(args):

    algorithm_name = args.algorithm_name

    # The parameters used for determining the frequency of logger, reward plots, 
    # and rendering of the environment

    default_parameters = {
        'algorithm_name': algorithm_name,        # The algorithm used

        'total_timesteps': 150_000,    # total timesteps for training

        'freq_log': 2,                 # Number of training iterations to display logger/information
        'freq_rew': 20,                # Number of training iterations to display reward plots
        'freq_render': 20,             # Number of training iterations to render environment
        'freq_ac_save': 20,            # Number of training iterations to save actor/critic

        'render': True,                # Rendering environment
        'seed': 10,                    # Seed for random number generators
        }


    # Hyperparameters used for Soft Actor-Critic (SAC) algorithm
    if algorithm_name == 'SAC':
        hyperparameters = {

            # replay buffer information
            'replay_size': int(1_000_000),     # Maximum length of replay buffer
            'batch_size': 128,                 # Minibatch size for SGD

            # Actor information
            'lr_actor': 5e-4,                  # Actor learning rate
            'weight_decay_actor': 0,           # Actor weight decay to prevent overfitting
            'hidden_dim_actor': 150,           # Actor hidden dimension size

            # Critic information
            'lr_critic': 1e-2,                 # Actor learning rate
            'weight_decay_critic': 0,          # Actor weight decay to prevent overfitting
            'hidden_dim_critic': 80,           # Actor hidden dimension size

            # Temperature information
            'alpha': 1.0,                      # starting value of temperature
            'lr_alpha': 1e-1,                  # Temperature learning rate
            'weight_decay_alpha': 0,           # Temperature weight decay
            'alpha_min': 0.4,                  # Minimum value of the temperature

            # other parameters
            'gamma': float(0.95),              # Discount factor (between 0 and 1)
            'polyak': 1e-2,                    # Interpolation factor averaging for target networks

            # Training information
            'epoch': 1,                        # Number of epochs
            'timesteps_per_episode': 30,       # Number of timesteps in episodes
            'episodes_per_iteration': 1,       # Number of episodes in iterations
            'updates_per_iteration': 20,       # Number of updates per iteration
            }


    # Hyperparameters used for Deep Deterministic Policy Gradient (DDPG) algorithm
    if algorithm_name == 'DDPG':
        hyperparameters = {

            # replay buffer information
            'replay_size': int(1_000_000),     # Maximum length of replay buffer
            'batch_size': 256,                 # Minibatch size for SGD

            # Actor information
            'lr_actor': 5e-5,                  # Actor learning rate
            'weight_decay_actor': 0,           # Actor weight decay to prevent overfitting
            'hidden_dim_actor': 250,           # Actor hidden dimension size

            # Critic information
            'lr_critic': 1e-2,                 # Actor learning rate
            'weight_decay_critic': 0,          # Actor weight decay to prevent overfitting
            'hidden_dim_critic': 65,           # Actor hidden dimension size

            # other parameters
            'gamma': float(0.95),              # Discount factor (between 0 and 1)
            'polyak': 1e-2,                    # Interpolation factor averaging for target networks

            # Training information
            'epoch': 1,                        # Number of epochs
            'timesteps_per_episode': 30,       # Number of timesteps in episodes
            'episodes_per_iteration': 2,       # Number of episodes in iterations
            'updates_per_iteration': 20,       # Number of updates per iteration
            }


    # Hyperparameters used for Proximal Policy Optimization (PPO) algorithm
    if algorithm_name == 'PPO':
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


    # Hyperparameters used for Shack-Hartmann wavefront sensor method
    if algorithm_name == 'SHACK':
        hyperparameters = {

            # Training information
            'epoch': 1,                        # Number of epochs
            'timesteps_per_episode': 30,       # Number of timesteps in episodes
            'episodes_per_iteration': 1,       # Number of episodes in iterations
            }

        # Shack-Hartmann method only works for 'AO-v0' environment
        if args.environment_name != 'AO-v0':
            print(f"Error: Shack-Hartmann method only works for Adaptive optics environment")
            print(f"Change the environment to 'AO-v0' and try again")
            sys.exit(0)


    # Envrionment definition
    env = gym.make(args.environment_name)


    # training or testing depending on the mode
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, default_parameters=default_parameters, 
              algorithm_name=args.algorithm_name, actor_model=args.actor_model, 
              criticQ1_model=args.criticQ1_model, criticQ2_model=args.criticQ2_model, 
              criticV_model=args.criticV_model)

    elif args.mode == 'test':
        test(env=env, actor_model=args.actor_model, hyperparameters=hyperparameters, 
             default_parameters=default_parameters)


if __name__ == '__main__':
    args = get_args()           # actor/critic and train/test can be modified through arguments
    main(args)


#-----------------------------------------------------------------------------#
