import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, num_agents=2, num_resources=3, max_steps=100):
        super().__init__()
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.max_steps = max_steps
        
        #action how much to buy/sell of each resource (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(num_resources,),
            dtype=np.float32
        )
        
        #state [inventory, capital, market_prices, last_trades]
        obs_dim = num_resources * 3 + 1  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        #init inventories, capital, prices
        self.inventories = np.ones((self.num_agents, self.num_resources)) * 100
        self.capital = np.ones(self.num_agents) * 1000
        self.prices = np.random.uniform(8, 12, self.num_resources)
        self.last_trades = np.zeros(self.num_resources)
        
        return self._get_obs(), {}
    
    def step(self, actions):
        self.current_step += 1
        
        #doing actions to numpy if they're not already3
        if isinstance(actions[0], np.ndarray):
            actions = [actions[0]]  #handing the single agent case
        
        #processing trades
        trade_volumes = np.zeros(self.num_resources)
        for agent in range(self.num_agents):
            max_trade = self.inventories[agent] * 0.5
            agent_trades = np.array(actions[agent]) * max_trade
            
            #putting out the trades if affordable
            trade_cost = np.sum(agent_trades * self.prices)
            if trade_cost <= self.capital[agent]:
                self.inventories[agent] = self.inventories[agent] + agent_trades  # Explicit addition
                self.capital[agent] -= trade_cost
                trade_volumes += agent_trades
        
        #updating hte market
        self.last_trades = trade_volumes
        self._update_prices(trade_volumes)
        
        #calculating rewards, portfolio value change
        rewards = []
        for agent in range(self.num_agents):
            portfolio_value = np.sum(self.inventories[agent] * self.prices)
            reward = portfolio_value + self.capital[agent]
            rewards.append(reward)
        
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), rewards, done, False, {}
    
    def _get_obs(self):
        obs = []
        for agent in range(self.num_agents):
            agent_obs = np.concatenate([
                self.inventories[agent],
                [self.capital[agent]],
                self.prices,
                self.last_trades
            ])
            obs.append(agent_obs)
        return obs
    
    def _update_prices(self, trades):
        #price impact model
        price_impact = trades * 0.1
        self.prices *= (1 + price_impact)
        self.prices = np.clip(self.prices, 1, 100)