"""
        This file presents a set of arguments intended to be parsed at the 
        command line. These arguments will be called by the main.py file through 
        the function get_args.
"""


#----------------------------- Importing modules -----------------------------#
import argparse


#--------------------------------- Arguments ---------------------------------#
def get_args():

    parser = argparse.ArgumentParser()

    # Select train or test. For:
    # training --> 'train'
    # testing --> 'test'
    parser.add_argument('--mode', dest='mode', type=str, default='train')


    # Select the Actor. If:
    # training from scratch --> default=''
    # testing or continuing training actor for SAC --> default='sac_actor.pth'
    # testing or continuing training actor for DDPG --> default='ddpg_actor.pth'
    # testing or continuing training actor for PPO --> default='ppo_actor.pth'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')


    # Write the desired algorithm. For:
    # Proximal Policy Optimization (PPO) --> 'PPO'
    # Soft Actor-Critic (SAC) --> 'SAC'
    # Deep Deterministic Policy Gradient (DDPG) --> 'DDPG'
    # Shack-Hartmann wavefront sensor --> 'SHACK'
    parser.add_argument('--algorithm_name', dest='algorithm_name', type=str, default='PPO')


    # Write the desired Gym environment (For example: 'Pendulum-v1')
    # If the Adaptive Optics environment is preferred, please write 'AO-v0'.
    parser.add_argument('--environment_name', dest='environment_name', type=str, default='AO-v0')


    # Select the 1st critic for SAC and only critic for DDPG. If:
    # training from scratch --> ''
    # testing or continuing training 1st critic for SAC --> default='sac_critic1.pth'
    # testing or continuing training actor for DDPG --> default='ddpg_critic.pth'
    parser.add_argument('--criticQ1_model', dest='criticQ1_model', type=str, default='')


    # Select the 2nd critic for SAC. If:
    # training from scratch --> ''
    # testing or continuing training 2nd critic for SAC --> default='sac_critic2.pth'
    parser.add_argument('--criticQ2_model', dest='criticQ2_model', type=str, default='')


    # Select the only critic for PPO. If:
    # training from scratch --> ''
    # testing or continuing training the critic for PPO --> default='ppo_Vcritic.pth'
    parser.add_argument('--criticV_model', dest='criticV_model', type=str, default='')


    args = parser.parse_args()

    return args


#-----------------------------------------------------------------------------#