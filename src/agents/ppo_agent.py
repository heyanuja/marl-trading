import torch
import torch.optim as optim
import numpy as np
from .models import ActorCritic

class PPOAgent:
    def __init__(self, input_dim, action_dim, hidden_dim=64, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(input_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.actor_critic.get_action(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy(), value.cpu().numpy()
    
    def update(self, rollouts):
        states = torch.FloatTensor(rollouts['states']).to(self.device)
        actions = torch.FloatTensor(rollouts['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
        rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        masks = torch.FloatTensor(rollouts['masks']).to(self.device)
        
        #should be commputing returns and advantages
        with torch.no_grad():
            values = self.actor_critic(states)[2].squeeze()
            advantages = self._compute_advantages(rewards, values, masks)
            returns = advantages + values
        
        #the PPO update
        for _ in range(self.ppo_epochs):
            # Get current policy distributions
            action_mean, action_std, values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            #the policy ratio and clipped objective
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            #losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((values - returns) ** 2).mean()
            
            #updating network
            total_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def _compute_advantages(self, rewards, values, masks):
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * masks[t] * gae
            advantages[t] = gae
        
        return advantages