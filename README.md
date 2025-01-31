# Multi-Agent RL Trading System

This is a reinforcement learning framework for training agents to trade in simulated markets using PPO/Proximal Policy Optimization! The goal is to get the agent to learn to buy low, sell high, and manage a portfolio while considering market impact

![Training Dashboard](results/training_results.png)

## Overview

- a multi agent environment simulating market dynamics
- PPO implementation for continuous action spaces
- price impact modeling and portfolio management!
- a visualization suite

## Results

the agent learns stable trading strategies:

![Returns Over Time](results/analysis_dashboard.png)

some key findings are that:
- returns stabilize around 3 million after initial exploration
- theres a price correlation that shows market impact understanding
- looks like the agent adapts to market volatility

## Some Technical Details

### Environment Performance
![Price Evolution](results/price_evolution.gif)

### Environment
- custom OpenAI gym environment
- 3 tradeable resources
- orice impact modeling
- portfolio tracking

### Agent Architecture
- an actor critic network
- PPO with continuous actions
- moving window advantage estimation
- an experience replay!

## Installation
```bash
clone the repo etc
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```python
python train.py  #to train the agent
```

in `notebooks/analysis.ipynb` there's performance analysis info
