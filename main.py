"""
        The following file presents an executable file capable of executing 
        Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and 
        Deep Deterministic Policy Gradient (DDPG) algorithms for any 
        Gym environment, along with Shack-Hartmann for Adaptive Optics 
        environment.
"""


#-------------------------------- User input ---------------------------------#
"""
        In this section, the selection of the Gym environment of interest is 
        undertaken, along with the identification of the algorithm to be 
        utilized for training purposes. Moreover, a folder is created for the 
        storage of actor, critic, and reward plot outputs.
"""

# Write the desired Gym environment (For example: 'Pendulum-v1')
# If the Adaptive Optics environment is preferred, please write 'AO-v0'.
environment_name = 'AO-v0'

# Write the desired algorithm. For:
# Proximal Policy Optimization (PPO) --> 'PPO'
# Soft Actor-Critic (SAC) --> 'SAC'
# Deep Deterministic Policy Gradient (DDPG) --> 'DDPG'
# Shack-Hartmann wavefront sensor --> 'SHACK'
algorithm = 'PPO'

# Create a folder to save reward plots and actor/critic
# Write any name you desire
save_name = 'PPO_01'


#----------------------------- Importing modules -----------------------------#
import sys
import gym
import gym_AO
import torch

from arguments import get_args
from algorithm import ALGORITHM
from network import Actor, CriticQ1, CriticQ2, CriticV
from eval_policy import eval_policy


#--------------------------------- Training ----------------------------------#
def train(env, hyperparameters, default_parameters, actor_model, critic_classQ1, 
          critic_classQ2, critic_classV, env_name):

    model = ALGORITHM(policy_class1=Actor, critic_classQ1=CriticQ1, critic_classQ2=CriticQ2, 
                      critic_classV=CriticV,  env=env, env_name=environment_name, 
                      default_parameters=default_parameters, **hyperparameters)


    # Running SAC from scratch or importing actor/critics to continue training
    if algorithm == 'SAC':
        if actor_model == '' and critic_classQ1 == '' and critic_classQ2 == '':
            print(f"Running SAC from scratch ...")

        elif actor_model != '' and critic_classQ1 != '' and critic_classQ2 != '':
            print(f"Loading in {actor_model} and {critic_classQ1} and {critic_classQ2} for SAC ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(critic_classQ1))
            model.softq_critic2.load_state_dict(torch.load(critic_classQ2))
            print(f"Successfully loaded for SAC.", flush=True)

        elif actor_model != '' or critic_classQ1 != '' or critic_classQ2 != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to SAC algorithm")
            sys.exit(0)


    # Running DDPG from scratch or importing actor/critic to continue training
    if algorithm == 'DDPG':
        if actor_model == '' and critic_classQ1 == '':
            print(f"Running DDPG from scratch ...")

        elif actor_model != '' and critic_classQ1 != '':
            print(f"Loading in {actor_model} and {critic_classQ1} for DDPG ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softq_critic1.load_state_dict(torch.load(critic_classQ1))
            print(f"Successfully loaded for DDPG.", flush=True)

        elif actor_model != '' or critic_classQ1 != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to DDPG algorithm")
            sys.exit(0)


    # Running PPO from scratch or importing actor/critic to continue training
    if algorithm == 'PPO':
        if actor_model == '' and critic_classV == '':
            print(f"Running PPO from scratch ...")

        elif actor_model != '' and critic_classV != '':
            print(f"Loading in {actor_model} and {critic_classV} for PPO ...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.softV_critic.load_state_dict(torch.load(critic_classV))
            print(f"Successfully loaded for PPO.", flush=True)

        elif actor_model != '' or critic_classV != '':
            print(f"Error: One of the networks is not added, or you picked the wrong algorithm!")
            print(f"The networks should belong to PPO algorithm")
            sys.exit(0)

    # Learning process ...
    model.learn(default_parameters.get('total_timesteps'))


#--------------------------------- Testing -----------------------------------#
def test(env, actor_model, env_name, hyperparameters, default_parameters):

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

    # The parameters used for determining the frequency of logger, reward plots, 
    # and rendering of the environment

    default_parameters = {
        'algorithm': algorithm,        # The algorithm used
        'save_name': save_name,        # The name of the folder to save reward plots and actor/critic

        'total_timesteps': 150_000,    # total timesteps for training

        'freq_log': 2,                 # Number of training iterations to display logger/information
        'freq_rew': 20,                # Number of training iterations to display reward plots
        'freq_render': 20,             # Number of training iterations to render environment
        'freq_ac_save': 20,            # Number of training iterations to save actor/critic

        'render': True,                # Rendering environment
        'seed': 10,                    # Seed for random number generators
        }


    # Hyperparameters used for Soft Actor-Critic (SAC) algorithm
    if algorithm == 'SAC':
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
    if algorithm == 'DDPG':
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
    if algorithm == 'PPO':
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
    if algorithm == 'SHACK':
        hyperparameters = {

            # Training information
            'epoch': 1,                        # Number of epochs
            'timesteps_per_episode': 30,       # Number of timesteps in episodes
            'episodes_per_iteration': 1,       # Number of episodes in iterations
            }

        # Shack-Hartmann method only works for 'AO-v0' environment
        if environment_name != 'AO-v0':
            print(f"Error: Shack-Hartmann method only works for Adaptive optics environment")
            print(f"Change the environment to 'AO-v0' and try again")
            sys.exit(0)


    # Envrionment definition
    env = gym.make(environment_name)


    # training or testing depending on the mode
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, default_parameters=default_parameters, 
              actor_model=args.actor_model, critic_classQ1=args.critic_classQ1, critic_classQ2=args.critic_classQ2, 
              critic_classV=args.critic_classV, env_name=environment_name)

    elif args.mode == 'test':
        test(env=env, actor_model=args.actor_model, env_name=environment_name, 
             hyperparameters=hyperparameters, default_parameters=default_parameters)


if __name__ == '__main__':
    args = get_args()           # actor/critic and train/test can be modified through arguments
    main(args)


#-----------------------------------------------------------------------------#