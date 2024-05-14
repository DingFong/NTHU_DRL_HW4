import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class DisagreementExploration:
    def __init__(self, observation_shape, action_shape, n_state_predictor, hidden_sizes, args):
        self.observation_space = observation_shape
        self.action_space = action_shape
        self.n_state_predictor = n_state_predictor
        

        self.hiddle_sizes = hidden_sizes
        self.lr = args.lr_exploration
        self.bonus_scale = args.bonus_scale

        self.state_predictors = [
            nn.Sequential(
                nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], self.hiddle_sizes[0]),
                nn.ReLU(),
                *[nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)],
                nn.Linear(hidden_sizes[-1], self.observation_space.shape[0])
            ) for _ in range(self.n_state_predictor)
        ]

        self.optimizer = optim.Adam([param for predictor in self.state_predictors for param in predictor.parameters()], lr=self.lr)

    def forward(self, state, action):
        # print(torch.cat([state, action], dim=1).shape)
        norm_next_state_pred = torch.stack([predictor(torch.cat([state, action], dim=1)) for predictor in self.state_predictors], dim=1)
        return norm_next_state_pred
    
    def get_exploration_bonus(self, state, action):
        # Normalize
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        # print(state.shape, action.shape)
        obs_mean = torch.mean(state, dim=1)
        obs_var = torch.var(state, dim=1) + 1e-6
        norm_state = (state - obs_mean) / torch.sqrt(obs_var)
        # print(norm_state.shape)
        # next_state_mean = torch.mean(next_state, dim=1, keepdim=True)
        # next_state_var = torch.var(next_state, dim=1, keepdim=True) + 1e-6
        # norm_next_state = (next_state - next_state_mean) / torch.sqrt(next_state_var)

        with torch.no_grad():
            norm_next_state_pred = self.forward(norm_state, action)
        # print(norm_next_state_pred.shape)
        

        bonus = self.bonus_scale * torch.var(norm_next_state_pred.squeeze(0))
        print(bonus)
        return bonus

    def select_best_action(self, obs, actions):
        # Reshape inputs to (n_samples*n_env, *_dim)
        n_samples, n_env, action_dim = actions.shape
        obs = obs.repeat(n_samples, 1, 1).view(-1, obs.shape[-1])
        actions = actions.view(-1, action_dim)

        # Predict next observation from each state predictor
        normed_pred_next_obs = self.forward(obs, actions)
        
        # Compute rewards
        rewards = self.compute_rewards(normed_pred_next_obs)  # (n_samples*n_env, 1)

        # Reshape rewards and select best actions
        rewards = rewards.view(n_samples, n_env, -1, 1)  # (n_samples, n_env, n_state_predictors, 1)
        rewards = rewards.sum(2)  # sum over state predictors; out (n_samples, n_env, 1)
        best_actions = actions.take_along_axis(1, rewards.argmax(1, keepdim=True))  # out (n_samples, 1, action_dim)
        best_actions = best_actions.squeeze(1)  # remove extra dimension

        return best_actions
    
    def train(self, obs, actions, next_obs):
        # Normalize observations
        obs_mean = torch.mean(obs, dim=1, keepdim=True)
        obs_var = torch.var(obs, dim=1, keepdim=True) + 1e-8
        # print(obs_mean.shape, obs_var.shape)
        normed_obs = (obs - obs_mean) / torch.sqrt(obs_var)

        # Normalize next observations
        next_obs_mean = torch.mean(next_obs, dim=1, keepdim=True)
        next_obs_var = torch.var(next_obs, dim=1, keepdim=True) + 1e-8
        normed_next_obs = (next_obs - next_obs_mean) / torch.sqrt(next_obs_var)

        # Predict next observation from each state predictor
        normed_pred_next_obs = self.forward(normed_obs, actions)
        # print(normed_pred_next_obs.shape)
        
        # Calculate loss for each state predictor
        losses = []
        total_loss = 0
        for i in range(self.n_state_predictor):
            loss = torch.mean((normed_pred_next_obs[:, i, :] - normed_next_obs) ** 2)
            total_loss += loss
            losses.append(loss.item())

        # Update state predictors using optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
       
        return np.mean(losses)


        
        
