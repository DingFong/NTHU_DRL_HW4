import torch.nn.functional as F

from model.critic import Critic
from model.actor import Actor

import torch
import torch.optim as optim

from model.replay_buffer import Memory
import os
import numpy as np

# SAC (Soft-Actor-Critic)
class Agent:
    def __init__(self):
        action_space = np.zeros(22)
        self.action_space = action_space
        # self.gamma = args.gamma
        # self.tau = args.tau
        # self.alpha = args.alpha
        num_inputs = 339    
        hidden_size = 256
        

        # self.target_update_interval = args.target_update_interval
        # self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = "cpu"

        # self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        # self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        
        # if self.automatic_entropy_tuning == True:
        #     self.target_entropy = -torch.prod(torch.Tensor(action_space.shape[0]).to(self.device)).item()
        #     self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        #     self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr)

        self.actor = Actor(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.load_checkpoint()
        # self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)

        # self.memory = Memory(args.memory_capacity)
        # self.batch_size = args.batch_size

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, observation):
        observation = self.trans_observation(observation)
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)

        
        _, _, action = self.actor.sample(state)
        action = torch.clamp(input = action, min = 0, max = 1)
        

        return action.detach().cpu().numpy()[0].flatten()
    
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    
    def update_params(self, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic, self.critic_target, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def save_checkpoint(self, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}".format(suffix)
        print('Saving model to {}'.format(ckpt_path))
        torch.save({'model': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'critic_target': self.critic_target.state_dict(),
                    'actor_optim': self.actor_optim.state_dict(),
                    'critic_optim': self.critic_optim.state_dict()
                    }, ckpt_path)      
        
    def load_checkpoint(self):
        weight = torch.load("111034521_hw4_data")
        
        self.actor.load_state_dict(weight)

    def trans_observation(self, observation):
        res = []

        # target velocity field (in body frame)
        res += observation['v_tgt_field'].flatten().tolist()

        res.append(observation['pelvis']['height'])
        res.append(observation['pelvis']['pitch'])
        res.append(observation['pelvis']['roll'])
        res.append(observation['pelvis']['vel'][0])
        res.append(observation['pelvis']['vel'][1])
        res.append(observation['pelvis']['vel'][2])
        res.append(observation['pelvis']['vel'][3])
        res.append(observation['pelvis']['vel'][4])
        res.append(observation['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += observation[leg]['ground_reaction_forces']
            res.append(observation[leg]['joint']['hip_abd'])
            res.append(observation[leg]['joint']['hip'])
            res.append(observation[leg]['joint']['knee'])
            res.append(observation[leg]['joint']['ankle'])
            res.append(observation[leg]['d_joint']['hip_abd'])
            res.append(observation[leg]['d_joint']['hip'])
            res.append(observation[leg]['d_joint']['knee'])
            res.append(observation[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(observation[leg][MUS]['f'])
                res.append(observation[leg][MUS]['l'])
                res.append(observation[leg][MUS]['v'])
        return res
