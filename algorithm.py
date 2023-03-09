"""
        This file comprises a set of algorithms used for the purpose of training
"""


#----------------------------- Importing modules -----------------------------#
import os
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

from replay_buffer import ReplayBuffer


#------------------------------ Initialization -------------------------------#
class ALGORITHM:
    def __init__(self, actor_model, criticQ1_model, criticQ2_model, criticV_model, 
                 env, default_parameters, **hyperparameters):

        # Calling the environment
        self.env = env

        # Calling default parameters and algorithm hyperparameters from main.py
        self._init_parameters(hyperparameters, default_parameters)

        # Extract out dimensions of observation and action spaces
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]


        # Initialize actor and critic networks and replay buffer for SAC
        if self.algorithm_name == 'SAC':

            # Initialize Actor network and optimizer
            self.actor = actor_model(self.obs_dim, self.act_dim, self.hidden_dim_actor)
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, 
                                            weight_decay=self.weight_decay_actor)

            # Initialize critic networks and optimizers
            self.softq_critic1 = criticQ1_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.softq_critic2 = criticQ2_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)

            self.q_optimizer1 = optim.Adamax(self.softq_critic1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)
            self.q_optimizer2 = optim.Adamax(self.softq_critic2.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)

            # Initialize target critic networks
            self.softq_critic_target1 = criticQ1_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.softq_critic_target2 = criticQ2_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)

            # Initialize replay buffer
            self.replaybuffer = ReplayBuffer(self.replay_size, a_dim=self.act_dim, 
                                             a_dtype=np.float32, s_dim=self.obs_dim, s_dtype=np.float32, store_mu=False)

            # Initialize temperature hyperparameter
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adamax([self.log_alpha], lr=self.lr_alpha, weight_decay=self.weight_decay_alpha)

            # Initialize target entropy parameter
            self.target_entropy = -self.act_dim


        # Initialize actor and critic networks and replay buffer for DDPG
        if self.algorithm_name == 'DDPG':

            # Initialize Actor network and optimizer
            self.actor = actor_model(self.obs_dim, self.act_dim, self.hidden_dim_actor)
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, 
                                            weight_decay=self.weight_decay_actor)

            # Initialize critic networks and optimizer
            self.softq_critic1 = criticQ1_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.q_optimizer1 = optim.Adamax(self.softq_critic1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)

            # Initialize target actor network
            self.actor_target = actor_model(self.obs_dim, self.act_dim, self.hidden_dim_actor) 

            # Initialize target critic network
            self.softq_critic_target1 = criticQ1_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)

            # Initialize replay buffer
            self.replaybuffer = ReplayBuffer(self.replay_size, a_dim=self.act_dim, 
                                             a_dtype=np.float32, s_dim=self.obs_dim, s_dtype=np.float32, store_mu=False)


        # Initialize actor and value function for PPO
        if self.algorithm_name == 'PPO':

            # Initialize Actor network and optimizer
            self.actor = actor_model(self.obs_dim, self.act_dim, self.hidden_dim_actor)
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, 
                                            weight_decay=self.weight_decay_actor)

            # Initialize value function and the optimizer
            self.softV_critic = criticV_model(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.V_optimizer = optim.Adamax(self.softV_critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)


        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # create logger to help printing out summaries of iterations
        self.logger = {
            'batch_ep_rew': [],         # initialize the logger for rewards per episode in iteration
            'i_so_far': 0,              # initialize the iterations ran so far
            }

        # other initial values
        self.reward_plot = []          # initialize the collection of average reward per iteration for reward plot


#---------------------------- Calling parameters -----------------------------#
# calling parameters from main.py
    def _init_parameters(self, hyperparameters, default_parameters):

        # Calling algorithm hyperparameters from main.py
        for param, val in hyperparameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))

        # Calling default parameters from main.py
        for param, val in default_parameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))

        self.render = default_parameters.get('render')
        self.algorithm_name = default_parameters.get('algorithm_name')

        self.timesteps_per_iteration = self.episodes_per_iteration * self.timesteps_per_episode    # Number of timesteps per iteration
        
        # Seed for random number generators
        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)


#--------------------------------- Training ----------------------------------#
    def learn(self, total_timesteps):

        # train the actor and critic networks

        for self.epoch_no in range(self.epoch):

            t_so_far = 0             # initialize the timestep so far
            i_so_far = 0             # initialize the iterations so far

            while t_so_far < total_timesteps:

                # collecting batch of actions, observations and rewards and etc. from the simulation
                batch_obs, batch_act, batch_log_probs, batch_rew, batch_next_obs, batch_done, batch_lens = self.rollout()

                t_so_far += np.sum(batch_lens)      # timesteps ran so far
                self.logger['t_so_far'] = t_so_far

                self.logger['i_so_far'] = i_so_far  # iterations ran so far

                # put the collection of data in algorithms
                # SAC algorithm:
                if self.algorithm_name == 'SAC':
                    self.SAC(batch_obs, batch_act, batch_rew, batch_next_obs, batch_done)

                    if self.batch_size < len(self.replaybuffer):
                        self._log_summary()

                    # saving the actor and critic networks for SAC
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './sac_actor.pth')
                        torch.save(self.softq_critic1.state_dict(), './sac_critic1.pth')
                        torch.save(self.softq_critic2.state_dict(), './sac_critic2.pth')


                # DDPG algorithm:
                elif self.algorithm_name == 'DDPG':
                    self.DDPG(batch_obs, batch_act, batch_rew, batch_next_obs, batch_done)

                    if self.batch_size < len(self.replaybuffer):
                        self._log_summary()

                    # saving the actor and critic networks for SAC
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './ddpg_actor.pth')
                        torch.save(self.softq_critic1.state_dict(), './ddpg_critic.pth')


                # PPO algorithm:
                elif self.algorithm_name == 'PPO':
                    self.PPO(batch_obs, batch_act, batch_log_probs, batch_rew, batch_next_obs, batch_done)
                    self._log_summary()

                    # saving the actor and critic networks for SAC
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './ppo_actor.pth')
                        torch.save(self.softV_critic.state_dict(), './ppo_Vcritic.pth')


                # Shack-Hartmann method
                elif self.algorithm_name == 'SHACK':
                    self._log_summary()


                i_so_far += 1          # iterations ran so far


#--------------------------------- Rollout -----------------------------------#
    # collecting batch of actions, observations and rewards and etc. from the simulation

    def rollout(self):

        # Initialize the batches
        batch_obs = np.zeros((self.timesteps_per_iteration,self.obs_dim))                     # initialization of collection of observations
        batch_act = np.zeros((self.timesteps_per_iteration,self.act_dim))                     # initialization of collection of actions
        batch_log_probs = np.zeros((self.timesteps_per_iteration))                            # initialization of collection of actions log probability
        batch_rew = np.zeros((self.timesteps_per_iteration))                                  # initialization of collection of rewards per iteration
        batch_ep_rew = np.zeros((self.episodes_per_iteration,self.timesteps_per_episode))    # initialization of collection of rewards per episode in iteration
        batch_next_obs = np.zeros((self.timesteps_per_iteration,self.obs_dim))                # initialization of collection of next observations
        batch_done = np.zeros((self.timesteps_per_iteration))                                 # initialization of collection of done
        batch_lens = np.zeros((self.timesteps_per_iteration))                                 # initialization of the length of each episode

        t = 0              # count how many timesteps we've run
        t_iteration = 0    # count how many iterations we've run

        # run for a maximum number of timesteps per iteration (episodes_per_iteration * timesteps_per_episode)
        while t < (self.timesteps_per_iteration):

            # rewards collected per episode
            ep_rew = np.zeros((self.timesteps_per_episode)) 

            # reset the environment for new episode
            obs = self.env.reset()
            done = False

            # run for a maximum number of timesteps per episode
            for ep_t in range(self.timesteps_per_episode):

                # render the environment
                if self.render and (self.logger['i_so_far'] % self.freq_render == 0) and batch_lens[0] == 0:
                    self.env.render()

                # Collect the observation from simulation
                batch_obs[t,:] = obs

                # The action of Shack-Hartmann is generated through the environment.
                if self.algorithm_name == 'SHACK':
                    action, log_prob = self.env.shack_get_action()                  # getting next action from Shack-Hartmann

                else:
                    action, log_prob = self.actor.get_action(obs, self.cov_mat)     # getting next action from actor

                # observation, reward and done from simulation
                obs, rew, done, _ = self.env.step(action)
                next_obs = obs

                batch_act[t,:] = action           # collection of the actions
                batch_log_probs[t] = log_prob     # collection of actions log probability
                batch_rew[t] = rew                # collection of rewards per iteration
                ep_rew[ep_t] = rew                # collection of rewards per episode
                batch_next_obs[t,:] = next_obs    # collection of the next observations
                batch_done[t] = done              # collection of done

                t += 1       # count how many timesteps we've run

                # if at the end of the episode, break:
                if done:
                    break

            batch_ep_rew[t_iteration, :] = ep_rew     # collection of rewards per episode in iteration
            batch_lens[t_iteration] = ep_t + 1    # collection of the length of each episode
            t_iteration += 1                      # count how many iterations we've run

        # reshape the batches to tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_act = torch.tensor(batch_act, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rew = torch.tensor(batch_rew, dtype=torch.float)
        batch_next_obs = torch.tensor(batch_next_obs, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.float)

        # add the batches to logger to give information about the training
        self.logger['batch_ep_rew'] = batch_ep_rew

        return batch_obs, batch_act, batch_log_probs, batch_rew, batch_next_obs, batch_done, batch_lens


#---------------------- reshape to tensors -----------------------------------#
    # reshape the batch of transition to tensors
    def reshape_to_tensor(self, s, a, r, next_s, done):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        next_s = torch.FloatTensor(next_s)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        return s, a, r, next_s, done


#----------------------------- Loss function ---------------------------------#
    def get_loss(self, val, next_val):
        
        criterion = nn.MSELoss()
        return criterion(val, next_val)


#------------------------------ algorithms -----------------------------------#
#----------------------------- SAC algorithm ---------------------------------#
    """
    This SAC algorithm is following the SAC pseudocode by OpenAI.
    The pseudocode can be found here --> https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg
    """

    def SAC(self, state, action, reward, next_state, done):

        # store state, action, reward, next action and done in replay buffer
        for i in range(len(state)):
            self.replaybuffer.add(state[i], action[i], reward[i], next_state[i], done[i])

        # Initial filling of the replay buffer
        if self.batch_size < len(self.replaybuffer):

            # Number of updates for each iteration
            for _ in range(self.updates_per_iteration):

                # randomly sample a batch of transitions from replay buffer
                s, a, r, next_s, d = self.replaybuffer.sample_batch(self.batch_size, self.seed)

                # reshape the batch of transition to tensors
                s, a, r, next_s, d = self.reshape_to_tensor(s, a, r, next_s, d)

                # Control the randomness by increasing it by one
                self.seed += 1

                with torch.no_grad():
                    # calculate next action and action log probability
                    next_state_action, next_state_log_pi, _ = self.actor.evaluate(next_s, a, self.cov_mat)
                    
                    # calculate the next target Q-networks
                    qf1_next_target = self.softq_critic_target1(next_s, next_state_action)
                    qf2_next_target = self.softq_critic_target2(next_s, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.unsqueeze(1)

                    # compute targets for the Q-networsks
                    next_q_value = r + self.gamma * (min_qf_next_target)

                # update the Q-networks
                qf1 = self.softq_critic1(s, a)
                qf2 = self.softq_critic2(s, a)

                qf1_loss = self.get_loss(qf1, next_q_value)
                qf2_loss = self.get_loss(qf2, next_q_value)
                
                self.softq_critic1.train(qf1_loss, self.q_optimizer1)
                self.softq_critic2.train(qf2_loss, self.q_optimizer2)

                # calculate next action and action log probability
                pi, log_pi, _ = self.actor.evaluate(s, a, self.cov_mat)

                # compute the updated Q-networks
                qf1_pi = self.softq_critic1(s, pi)
                qf2_pi = self.softq_critic2(s, pi)

                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # Update the policy
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                self.actor.train(actor_loss, self.actor_optim)

                # update the target Q-networks
                for target_param, param in zip(self.softq_critic_target1.parameters(), self.softq_critic1.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

                for target_param, param in zip(self.softq_critic_target2.parameters(), self.softq_critic2.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

            # Update the temperature hyperparameter. 
            # Below equation is taken from --> https://arxiv.org/abs/1812.05905
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

            # prevent the temperature to go below specific value for 'semi-learned temperature'
            if self.alpha.item() < self.alpha_min:
                self.alpha = torch.tensor([self.alpha_min], requires_grad=True)


#-----------------------------------------------------------------------------#
#---------------------------- DDPG algorithm ---------------------------------#
    """
    This DDPG algorithm is following the DDPG pseudocode by OpenAI.
    The pseudocode can be found here --> https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg
    """

    def DDPG(self, state, action, reward, next_state, done):

        # reward normalization (mean-std normalization)
        reward = (reward - reward.mean()) / (reward.std() + 1e-10)

        # store state, action, reward, next action and done in replay buffer
        for i in range(len(state)):
            self.replaybuffer.add(state[i], action[i], reward[i], next_state[i], done[i])

        # Initial filling of the replay buffer
        if self.batch_size < len(self.replaybuffer):

            # Number of updates for each iteration
            for _ in range(self.updates_per_iteration):

                # randomly sample a batch of transitions from replay buffer
                s, a, r, next_s, d = self.replaybuffer.sample_batch(self.batch_size, self.seed)

                # reshape the batch of transition to tensors
                s, a, r, next_s, d = self.reshape_to_tensor(s, a, r, next_s, d)

                # Control the randomness by increasing it by one
                self.seed += 1

                with torch.no_grad():
                    # calculate next action and action log probability from target actor
                    next_state_action, _, _ = self.actor_target.evaluate(next_s, a, self.cov_mat)
                    
                    # calculate the next target Q-network
                    qf1_next_target = self.softq_critic_target1(next_s, next_state_action)

                    # compute targets for the Q-networsk
                    next_q_value = r + self.gamma * (qf1_next_target)

                # update the Q-network
                qf1 = self.softq_critic1(s, a)
                qf1_loss = self.get_loss(qf1, next_q_value)
                self.softq_critic1.train(qf1_loss, self.q_optimizer1)

                # calculate next action and action log probability
                pi, _, _ = self.actor.evaluate(s, a, self.cov_mat)

                # Update the policy
                actor_loss = -self.softq_critic1(s, pi).mean()
                self.actor.train(actor_loss, self.actor_optim)

                # update the target actor and Q-network
                for target_param, param in zip(self.softq_critic_target1.parameters(), self.softq_critic1.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)


#-----------------------------------------------------------------------------#
#---------------------------- PPO algorithm ----------------------------------#
    """
    This PPO algorithm is following the PPO pseudocode by OpenAI.
    The pseudocode can be found here --> https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
    """

    def PPO(self, state, action, log_probs, reward, next_state, done):

        # compute the value function
        V = self.softV_critic(state).squeeze()

        # compute the advantage function
        A_k = reward - V.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)       # advantage function normalization

        # Number of updates for each iteration
        for _ in range(self.updates_per_iteration):

            # compute the value function
            V = self.softV_critic(state).squeeze()

            # compute the current action log probability
            _, _, curr_log_probs = self.actor.evaluate(state, action, self.cov_mat) 

            # calculate the ration from previous policy and current policy
            ratios = torch.exp(curr_log_probs - log_probs)

            # calculate the “surrogate” objective function
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k    # clipping the probability ratio

            # update the actor and critic networks
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = self.get_loss(V, reward)
            
            self.actor.train(actor_loss, self.actor_optim)
            self.softV_critic.train(critic_loss, self.V_optimizer)


#-----------------------------------------------------------------------------#
#-------------------------------logger summary--------------------------------#
    def _log_summary(self):

        # calculate the average reward per iteration
        avg_ep_rews = np.mean([np.sum(ep_rew) for ep_rew in self.logger['batch_ep_rew']])

        # collection of average reward per iteration for reward plot
        self.reward_plot.append(avg_ep_rews)

        # display the logger/information of the training process
        if self.logger['i_so_far'] % self.freq_log == 0:

            print(flush=True)
            print(f"----------------------------------------------------------", flush=True)
            print(f" timesteps so far:      {int(self.logger['t_so_far'])}", flush=True)
            print(f" episodes so far:       {(self.logger['i_so_far'] +1)*self.episodes_per_iteration}", flush=True)
            print(f" iterations so far:     {self.logger['i_so_far'] +1}", flush=True)
            print(f" epochs so far:         {self.epoch_no+1}", flush=True)
            print(f" Average cost:          {str(round(avg_ep_rews, 2))}", flush=True) 
            print(f"----------------------------------------------------------", flush=True)
            print(flush=True)


        # display reward plots
        if self.logger['i_so_far'] % self.freq_rew == 0 and (len(self.reward_plot) > 1):

            # create the folder for reward plots:
            try:
                os.makedirs('./reward_plot/epoch_' + str(self.epoch_no+1))
            except OSError:
                pass

            # plot the rewards
            plt.plot(self.reward_plot,'b')
            plt.title('cost plot per iteration')
            plt.xlabel('training iterations')
            plt.ylabel('Average cost: %f' % (np.round(self.reward_plot[-1],3)))
            plt.grid()
            plt.savefig('./reward_plot/epoch_' + str(self.epoch_no+1)+'/rewards_iteration_' + str(int(len(self.reward_plot))) + ".png")
            plt.show(block=False)
            plt.pause(1)
            plt.clf()
            plt.close()

            # save the data of the reward plot
            with open('rewards.pkl', 'wb') as f:
                pickle.dump(self.reward_plot, f)


#-----------------------------------------------------------------------------#