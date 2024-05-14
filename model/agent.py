import torch.nn.functional as F

from .critic import Critic
from .actor import Actor
from .exploration import DisagreementExploration

import torch
import torch.optim as optim

from .replay_buffer import Memory, PrioritizedReplayBuffer
import os

# SAC (Soft-Actor-Critic)
class Agent:
    def __init__(self, observation_space, action_space, args):
        self.action_space = action_space
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = Critic(observation_space.shape[0], action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(observation_space.shape[0], action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        if self.automatic_entropy_tuning == True:
            target = -torch.prod(torch.Tensor(action_space.shape))
            print(target)
            self.target_entropy = target.to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.lr)

        self.actor = Actor(observation_space.shape[0], action_space.shape[0], args.hidden_size, action_space=action_space).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)

        # Prioritized replay bufffer
        self.prioritized_replay = False
        if args.prioritized_replay == True:
            self.prioritized_replay = True
            self.momory = PrioritizedReplayBuffer(args.memory_capacity)
        else:
            self.memory = Memory(args.memory_capacity)


        self.batch_size = args.batch_size

        if args.exploration == True:
            # only model the pose information of the state space

            self.exploration = DisagreementExploration(observation_space, action_space, args.n_state_predictor, args.hidden_sizes, args)

    def train_exploration(self):
        # sample a batch of transitions from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        
        return self.exploration.train(state_batch, action_batch, next_state_batch)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, evaluate = False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if not evaluate:
            # Sample action from Gaussian policy
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
            action = torch.clamp(action, 0, 1)
        
        return action.detach().cpu().numpy()[0]
    
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    
    def update_params(self, updates):
        if self.prioritized_replay == False:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, indices = self.memory.sample(batch_size=self.batch_size)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, indices, weights = self.memory.sample(batch_size=self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
            

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
        qf1, qf2 = self.critic(state_batch, action_batch) 
        if self.prioritized_replay:
            qf1_loss = F.mse_loss(qf1, next_q_value)*weights
            qf2_loss = F.mse_loss(qf2, next_q_value)*weights
            priors = torch.abs(qf1_loss + qf2_loss + 1e-5).squeeze().cpu().numpy()
        else:
            qf1_loss = F.mse_loss(qf1, next_q_value) 
            qf2_loss = F.mse_loss(qf2, next_q_value)  
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

        if self.prioritized_replay == True:
            self.memory.update_priorities(indices, priors)


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