import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        #shared network layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        #actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        #critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        
        #getting action distribution parameters
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        #state value stuff
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self(state)
        
        if deterministic:
            return action_mean
            
        #action from normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob, value