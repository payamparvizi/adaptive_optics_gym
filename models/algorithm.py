
import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam, Adamax
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import os
from replay_buffer import ReplayBuffer
import pickle
import statistics as st
import copy
import random
from torch.utils.data import DataLoader
import sys

class ALGORITHM:

#-----------------------------------------------------------------------------#
    def __init__(self, policy_class1, critic_class1, critic_class2, critic_classV, env, env_name, default_parameters, **hyperparameters):
        
        self.best_strehl = float('-inf')
        self.no_rew_improvements = 0
        
        self.rewards = np.zeros((100_000_000))
        self.delta_t_total = []
        self.env_name = env_name

        self.myfilename = default_parameters.get('name_save')
        self.algorithm = default_parameters.get('algorithm')

        try:
            os.makedirs('./' + self.myfilename)
        except OSError:
            pass
            
        # cuda_avail = torch.cuda.is_available()
        # self.device = torch.device("cuda" if cuda_avail else "cpu")
        
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        self._init_parameters(hyperparameters, default_parameters)
        
        self.timesteps_per_batch = self.number_of_episodes_per_batch * self.max_timesteps_per_episode
        self.env = env
        
        if self.algorithm == 'SAC':
            self.actor = policy_class1(self.obs_dim, self.act_dim, self.hidden_dim_actor)   
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay_actor)
            
            self.replaybuffer = ReplayBuffer(self.buffer, a_dim=self.act_dim, 
                                  a_dtype=np.float32, s_dim=self.obs_dim, s_dtype=np.float32, store_mu=False)

            self.softq_critic1 = critic_class1(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.softq_critic2 = critic_class2(self.act_dim, self.obs_dim, self.hidden_dim_critic)

            self.softq_critic_target1 = critic_class1(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.softq_critic_target2 = critic_class2(self.act_dim, self.obs_dim, self.hidden_dim_critic)
    
            self.q_optimizer1 = optim.Adamax(self.softq_critic1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)
            self.q_optimizer2 = optim.Adamax(self.softq_critic2.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)
            
            # entropy parameters
            self.target_entropy = -self.act_dim
            self.mu_bar = copy.copy(self.target_entropy)
            self.sigma_bar = 0
            self.target_bar = copy.copy(self.target_entropy)
            self.ii = 0
            
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adamax([self.log_alpha], lr=self.lr_alpha, weight_decay=self.weight_decay_alpha)
        
        
        if self.algorithm == 'DDPG':
            self.replaybuffer = ReplayBuffer(self.buffer, a_dim=self.act_dim, 
                                  a_dtype=np.float32, s_dim=self.obs_dim, s_dtype=np.float32, store_mu=False)

            self.actor = policy_class1(self.obs_dim, self.act_dim, self.hidden_dim_actor)   
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay_actor)
            self.actor_target = policy_class1(self.obs_dim, self.act_dim, self.hidden_dim_actor) 

            self.softq_critic1 = critic_class1(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.softq_critic_target1 = critic_class1(self.act_dim, self.obs_dim, self.hidden_dim_critic)
            self.q_optimizer1 = optim.Adamax(self.softq_critic1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)
            
        
        elif self.algorithm == 'PPO':
            self.actor = policy_class1(self.obs_dim, self.act_dim, self.hidden_dim_actor)   
            self.actor_optim = optim.Adamax(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay_actor)
            
            self.softV_critic = critic_classV(self.act_dim, self.obs_dim, self.hidden_dim_Vcritic)
            self.V_optimizer = optim.Adamax(self.softV_critic.parameters(), lr=self.lr_criticV, weight_decay=self.weight_decay_criticV)
        
        
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        for i in range(len(self.cov_mat)):
            for j in range(len(self.cov_mat)):
                if i != j:
                    self.cov_mat[i,j] = 1e-6

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
            'batch_costs': [],
			'actor_losses': [],     # losses of actor network in current iteration
		}
        

#-----------------------------------------------------------------------------#
    def learn(self, total_timesteps):
        
        for self.epoch_no in range(self.epoch):
            # print('epoch: ', epoch_no)
            # print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
            # print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
            t_so_far = 0 # Timesteps simulated so far
            i_so_far = 0 # Iterations ran so far
            
            while t_so_far < total_timesteps:                                                                       
                batch_obs, batch_next_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_done, batch_r = self.rollout()
                
                t_so_far += np.sum(batch_lens)
                
                self.logger['t_so_far'] = t_so_far
                self.logger['i_so_far'] = i_so_far
                
                if self.algorithm == 'SAC':
                    self.optimization_SAC(batch_obs, batch_acts, batch_rtgs, batch_next_obs, batch_done, batch_r)
                    
                    if self.buffer_size < len(self.replaybuffer):
                        self._log_summary()
                    
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './'+ self.myfilename + '/sac_actor.pth')
                        torch.save(self.softq_critic1.state_dict(), './'+ self.myfilename + '/sac_critic1.pth')
                        torch.save(self.softq_critic2.state_dict(), './'+ self.myfilename + '/sac_critic2.pth')
               
                if self.algorithm == 'DDPG':
                    self.optimization_DDPG(batch_obs, batch_acts, batch_rtgs, batch_next_obs, batch_done, batch_r)
                    
                    if self.buffer_size < len(self.replaybuffer):
                        self._log_summary()
                    
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './'+ self.myfilename + '/ddpg_actor.pth')
                        torch.save(self.softq_critic1.state_dict(), './'+ self.myfilename + '/ddpg_critic1.pth')
                
                
                elif self.algorithm == 'PPO':
                    self.optimization_PPO(batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_r, batch_lens)
                    self._log_summary()
                    
                    if i_so_far % self.freq_ac_save == 0:
                        torch.save(self.actor.state_dict(), './'+ self.myfilename + '/ppo_actor.pth')
                        torch.save(self.softV_critic.state_dict(), './'+ self.myfilename + '/ppo_Vcritic.pth')
                        
                elif self.algorithm == 'SHACK':
                    self._log_summary()
                 
                i_so_far += 1


#-----------------------------------------------------------------------------#
    def rollout(self):

        batch_obs = np.zeros((self.timesteps_per_batch,self.obs_dim))
        batch_r = np.zeros((self.timesteps_per_batch))
        batch_next_obs = np.zeros((self.timesteps_per_batch,self.obs_dim))
        batch_acts = np.zeros((self.timesteps_per_batch,self.act_dim))
        batch_log_probs = np.zeros((self.timesteps_per_batch))
        batch_done = np.zeros((self.timesteps_per_batch))
        batch_lens = np.zeros((self.timesteps_per_batch))
        batch_rews = np.zeros((self.number_of_episodes_per_batch,self.max_timesteps_per_episode))

        t = 0 # Keeps track of how many timesteps we've run so far this batch
        t_batch = 0

        while t < (self.timesteps_per_batch):
            ep_rews = np.zeros((self.max_timesteps_per_episode)) # rewards collected per episode

#           Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
                
                if self.render and (self.logger['i_so_far'] % self.freq_env == 0) and batch_lens[0] == 0:
                    self.env.render()
                
                batch_obs[t,:] = obs
                
                if self.algorithm == 'SHACK':
                    action, log_prob = self.env.shack_step()
                
                else:    
                    action, log_prob = self.actor.get_action(obs, self.cov_mat)
                    #print(log_prob)
                
                obs, rew, done, _ = self.env.step(action)
                next_obs = obs
                
                batch_r[t] = rew
                batch_next_obs[t,:] = next_obs
                batch_acts[t,:] = action
                batch_log_probs[t] = log_prob
                batch_done[t] = done
                
                ep_rews[ep_t] = rew
                
                t += 1 # Increment timesteps ran this batch so far
                
                if done:
                    break

            batch_lens[t_batch] = (ep_t + 1)
            batch_rews[t_batch, :] = ep_rews
            t_batch += 1

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)                 # torch.Size([300, 25])
        batch_r = torch.tensor(batch_r, dtype=torch.float)                     # torch.Size([300])
        batch_next_obs = torch.tensor(batch_next_obs, dtype=torch.float)       # torch.Size([300, 25])
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)               # torch.Size([300, 64])
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)     # torch.Size([300])
        batch_done = torch.tensor(batch_done, dtype=torch.float)               # torch.Size([300])
        
        batch_rtgs = self.compute_rtgs(batch_rews)                                                      # ALG STEP 4

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_next_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_done, batch_r
    
    
#-----------------------------------------------------------------------------#
    def compute_rtgs(self, batch_rews):
        
        batch_rtgs = np.zeros((self.timesteps_per_batch))
        j = self.timesteps_per_batch - 1

        for ep_rews in reversed(batch_rews):
            
            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs[j] = discounted_reward
                j -= 1

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs


#-----------------------------------------------------------------------------#
    def _init_parameters(self, hyperparameters, default_parameters):
        
        for param, val in hyperparameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))
                
        for param, val in default_parameters.items():
            if isinstance(val, str) == False:
                exec('self.' + param + ' = ' + str(val))
        
        self.render = default_parameters.get('render')
        
        if self.seed != None:

            assert(type(self.seed) == int)
            
            torch.manual_seed(self.seed)
            # print(f"Successfully set seed to {self.seed}")


#-----------------------------------------------------------------------------#
    def send_to_device(self, s, a, r, next_s, done):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        next_s = torch.FloatTensor(next_s)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)
        
        return s, a, r, next_s, done
    
    
#-----------------------------------------------------------------------------#   
    def get_loss(self, val, next_val):
        
        criterion = nn.MSELoss()
        return criterion(val, next_val)
    

#-----------------------------------------------------------------------------# 
    def target_entropy_evaluate(self, et, target_bar):
        
        # @article{xu2021target,
        #          title={Target Entropy Annealing for Discrete Soft Actor-Critic},
        #          author={Xu, Yaosheng and Hu, Dailin and Liang, Litian and McAleer, Stephen and Abbeel, Pieter and Fox, Roy},
        #          journal={arXiv preprint arXiv:2112.02852},
        #          year={2021}
        #          }
        
        discount = 0.999
        std_threshold = 0.05
        avg_threshold = 0.01
        k = 0.99
        T = 0
        
        phi = (et - self.mu_bar).detach().mean().item()

        self.mu_bar = self.mu_bar + (1 - discount) * phi
        if phi >= 0:    
            sigma_2 = discount * (self.sigma_bar**2 + (1 - discount) * phi**2)
            self.sigma_bar = np.sqrt(sigma_2)
        else:
            sigma_2 = discount * (self.sigma_bar**2 - (1 - discount) * phi**2)
            if sigma_2 >= 0:
                self.sigma_bar = np.sqrt(sigma_2)
            else:
                self.sigma_bar = - np.sqrt(abs(sigma_2))
        
        if not (target_bar - avg_threshold < self.mu_bar < target_bar + avg_threshold) or self.sigma_bar > std_threshold:
            return target_bar
        
        self.ii = self.ii + 1
        if self.ii >= T:
            self.ii = 0
            if phi < 0:
                target_bar = target_bar / k
            elif phi > 0:      
                target_bar = target_bar * k
            
        return target_bar
        

#-----------------------------------------------------------------------------#
    def optimization_PPO(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_r, batch_lens):
        
        reww = batch_r
        
        V = self.softV_critic(batch_obs).squeeze()
        A_k = reww - V.detach()                                                                       
        
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for i in range(self.n_updates_per_iteration):                                                       
            V = self.softV_critic(batch_obs).squeeze()
            _, _, curr_log_probs = self.actor.evaluate(batch_obs, batch_acts, self.cov_mat) 

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = self.get_loss(V, reww)
            
            self.actor.train(actor_loss, self.actor_optim)
            self.softV_critic.train(critic_loss, self.V_optimizer)
            
            self.logger['actor_losses'].append(actor_loss.detach())
            
            
#-----------------------------------------------------------------------------#
    def optimization_SAC(self, state, action, q, next_state, done, reward, evaluation=False):
        
        #reward = (reward - reward.min())/(reward.max() - reward.min())
        #state = (state - state.mean()) / (state.std() + 1e-10)
        #next_state = (next_state - next_state.mean()) / (next_state.std() + 1e-10)
        #action = (action - action.mean()) / (action.std() + 1e-10)
        
        if not evaluation:
            for i in range(len(state)): 
                self.replaybuffer.add(state[i], action[i], reward[i], next_state[i], done[i])
            
            if self.buffer_size < len(self.replaybuffer):
               
                for _ in range(self.n_updates_per_iteration):
                    
                    ss, aa, rr, next_ss, dd = self.replaybuffer.sample_batch(self.buffer_size, self.seed)
                    s, a, r, next_s, d = self.send_to_device(ss, aa, rr, next_ss, dd) 

                    self.seed += 1
                 
                    with torch.no_grad():
                        next_state_action, next_state_log_pi, _ = self.actor.evaluate(next_s, a, self.cov_mat)
                        qf1_next_target = self.softq_critic_target1(next_s, next_state_action)
                        qf2_next_target = self.softq_critic_target2(next_s, next_state_action)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.unsqueeze(1)
                        next_q_value = r + (1) * self.gamma * (min_qf_next_target)
                    
                    qf1 = self.softq_critic1(s, a)
                    qf2 = self.softq_critic2(s, a)  

                    qf1_loss = self.get_loss(qf1, next_q_value)
                    qf2_loss = self.get_loss(qf2, next_q_value)
                    
                    self.softq_critic1.train(qf1_loss, self.q_optimizer1)
                    self.softq_critic2.train(qf2_loss, self.q_optimizer2)
                    
                    pi, log_pi, _ = self.actor.evaluate(s, a, self.cov_mat)
                    
                    qf1_pi = self.softq_critic1(s, pi)
                    qf2_pi = self.softq_critic2(s, pi)
                    
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                    
                    self.logger['actor_losses'].append(actor_loss.detach())
                    
                    self.actor.train(actor_loss, self.actor_optim)
                    
                    #soft update (can be done using soft-update function in model)
                    for target_param, param in zip(self.softq_critic_target1.parameters(), self.softq_critic1.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                    
                    for target_param, param in zip(self.softq_critic_target2.parameters(), self.softq_critic2.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                       
                self.target_entropy = self.target_entropy_evaluate(log_pi, self.target_entropy)
                
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optim.zero_grad()
                alpha_loss.backward(retain_graph=True)
                self.alpha_optim.step()
                
                self.alpha = self.log_alpha.exp()
                
                if self.alpha.item() < self.alpha_stop:
                    self.alpha = torch.tensor([self.alpha_stop], requires_grad=True)
                        
                    


#-----------------------------------------------------------------------------#
    def optimization_DDPG(self, state, action, q, next_state, done, reward, evaluation=False):
        
        reward = (reward - reward.min())/(reward.max() - reward.min())
        #state = (state - state.mean()) / (state.std() + 1e-10)
        #next_state = (next_state - next_state.mean()) / (next_state.std() + 1e-10)
        #action = (action - action.mean()) / (action.std() + 1e-10)
        
        if not evaluation:
            for i in range(len(state)): 
                self.replaybuffer.add(state[i], action[i], reward[i], next_state[i], done[i])
            
            if self.buffer_size < len(self.replaybuffer):
               
                for _ in range(self.n_updates_per_iteration):
                    
                    ss, aa, rr, next_ss, dd = self.replaybuffer.sample_batch(self.buffer_size, self.seed)
                    s, a, r, next_s, d = self.send_to_device(ss, aa, rr, next_ss, dd) 

                    self.seed += 1
                 
                    with torch.no_grad():
                        next_state_action, _, _ = self.actor_target.evaluate(next_s, a, self.cov_mat)
                        qf1_next_target = self.softq_critic_target1(next_s,next_state_action)
                        next_q_value = r + (1) * self.gamma * (qf1_next_target)
                    
                    qf1 = self.softq_critic1(s,a) 
                    
                    qf1_loss = self.get_loss(qf1, next_q_value)
                    self.softq_critic1.train(qf1_loss, self.q_optimizer1)
                    
                    pi, _, _ = self.actor.evaluate(s, a, self.cov_mat)
                    
                    actor_loss = -self.softq_critic1(s, pi).mean()
                    self.actor.train(actor_loss, self.actor_optim)
                    
                    # soft update (can be done using soft-update function in model)
                    for target_param, param in zip(self.softq_critic_target1.parameters(), self.softq_critic1.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                    
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                        
                    self.logger['actor_losses'].append(actor_loss.detach())
        
        
#-----------------------------------------------------------------------------#
    def _log_summary(self):
        
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns() 
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9 
        
        self.delta_t_total.append(delta_t)
        
        sum_delta = sum(self.delta_t_total)
        sum_delta_hr = int(sum_delta/3600)
        sum_delta_2 = sum_delta - sum_delta_hr*3600
        sum_delta_min = int(sum_delta_2/60)
        sum_delta_3 = sum_delta_2 - sum_delta_min*60
        sum_delta_sec = sum_delta_3

        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        
        # perform stopping of the overfitting
        
        strehl = 100 + (avg_ep_rews/self.max_timesteps_per_episode)
        # if strehl > (self.best_strehl+1):
        #     self.best_strehl = strehl
        #     self.no_rew_improvements = 0
        # else:
        #     self.no_rew_improvements += 1
            
        # if self.no_rew_improvements >= self.patience_stopping:
        #     print('Stopping because no improvement in reward')
        #     sys.exit(0)
        
        if i_so_far % self.freq_log == 0:
            
            print(flush=True)
            print(f"-------------------- Epoch #{self.epoch_no+1}, Batch #{i_so_far+1} ------------------", flush=True)
            print(f"timestep so far:          {t_so_far}", flush=True)
            print(f"episodes so far:          {int((i_so_far+1)*self.number_of_episodes_per_batch)}", flush=True)
            # print(f"Average Loss:             {str(round(avg_actor_loss, 5))}", flush=True)
            print(f"Average cost:             {str(round(avg_ep_rews, 2))}", flush=True) 
            print(f"Average strehl:           {str(round(strehl, 2))}", flush=True) 
            # print(f"best average strehl:      {str(round(self.best_strehl, 2))}", flush=True)
            # print(f"Stopping patience (<={str(self.patience_stopping)}): {str(self.no_rew_improvements)}", flush=True)
            print(f"Total Iteration took:     {str(round(sum_delta_hr,0))} hr, {str(round(sum_delta_min,0))} min, {str(round(sum_delta_sec,2))} secs", flush=True)
            print(f"----------------------------------------------------------", flush=True)
            print(flush=True)
        
        self.rewards[i_so_far] = float(avg_ep_rews)
        
        if len(self.rewards[0:i_so_far]) > 0 and i_so_far % self.freq_rew == 0:    
            
            try:
                os.makedirs('./' + self.myfilename + '/costs_plot/epoch_' + str(self.epoch_no+1))
            except OSError:
                pass
            
            rewards_ = self.rewards[0:i_so_far+1]
            
            plt.plot(rewards_,'b')
            plt.title('reward at epoch %d, batch %d ' % (self.epoch_no+1, i_so_far+1))
            plt.ylabel('Average Strehl ratio: %f ' % (np.round(rewards_[-1],3)))
            plt.savefig(self.myfilename +'/costs_plot/epoch_' + str(self.epoch_no+1)+'/rewards_batch_' + str(i_so_far+1) + ".png")
            plt.show()
            plt.clf()
            plt.close()
            
            with open(self.myfilename + '/rewards.pkl', 'wb') as f:
                pickle.dump(rewards_, f)
            
    		# Reset batch-specific logging data
            self.logger['batch_lens'] = []
            self.logger['batch_rews'] = []
            self.logger['batch_costs'] = []
            self.logger['actor_losses'] = []
#-----------------------------------------------------------------------------#
