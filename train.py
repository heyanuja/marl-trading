from src.environment.trading_env import TradingEnv
from src.agents.ppo_agent import PPOAgent
from src.visualization.visualize import TradingVisualizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from collections import defaultdict

def collect_rollout(env, agent, num_steps):
    rollout = defaultdict(list)
    obs = env.reset()[0]
    obs = obs[0]
    
    for _ in range(num_steps):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step([action])
        next_obs = next_obs[0]
        
        rollout['states'].append(obs)
        rollout['actions'].append(action)
        rollout['log_probs'].append(log_prob)
        rollout['rewards'].append(reward[0])
        rollout['masks'].append(1.0 - done)
        
        obs = next_obs
        if done:
            break
    
    for k, v in rollout.items():
        rollout[k] = np.array(v)
    
    return rollout

#-----------
def create_analysis_plots(returns, all_prices, save_dir="results/"):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")

    # ----------------------
    #dashboard style plot static fig
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)

    #log scale return over time
    ax1 = fig.add_subplot(gs[0, :])
    log_returns = np.log10(np.abs(np.array(returns)) + 1e-8)  #CHECK LATER ANU +1e-8 to avoid log(0) most simple?
    ax1.plot(log_returns)
    ax1.set_title('Returns Over Time (Log Scale)', size=14)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Log Returns')

    #price evol first 100 episode
    ax2 = fig.add_subplot(gs[1, 0])
    prices_array = np.array(all_prices)
    #if you like ...... really only want the first 100 episodes
    #len(prices_array) > 100 or it'll slice up to len(prices_array)
    for i in range(prices_array.shape[1]):
        ax2.plot(prices_array[:100, i], label=f'Resource {i}')
    ax2.set_title('Price Evolution (First 100 Episodes)', size=14)
    ax2.legend()

    #log scale return dist
    ax3 = fig.add_subplot(gs[1, 1])
    finite_returns = log_returns[np.isfinite(log_returns)]
    sns.histplot(data=finite_returns, bins=50, ax=ax3)
    ax3.set_title('Return Distribution (Log Scale)', size=14)

    #price correlations
    ax4 = fig.add_subplot(gs[2, 0])
    if prices_array.shape[0] > 1:
        price_corr = np.corrcoef(prices_array.T)
        sns.heatmap(price_corr, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('Price Correlations', size=14)
    else:
        ax4.set_title('Price Correlations: Not enough data', size=14)

    #learning progress
    ax5 = fig.add_subplot(gs[2, 1])
    window = 50
    returns_series = pd.Series(returns)
    rolling_mean = returns_series.rolling(window=window).mean()
    rolling_std = returns_series.rolling(window=window).std()
    ax5.plot(rolling_mean, label='Mean Return')
    ax5.fill_between(range(len(returns)),
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2)
    ax5.set_title(f'Learning Progress ({window}-Episode Window)', size=14)
    ax5.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_dashboard.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------
    #cretaing and save animation
    #at least 2 data points for dis 
    if len(prices_array) > 1:
        anim_fig, anim_ax = plt.subplots(figsize=(10, 6))

        def animate(frame):
            """Update the line plots up to index `frame`."""
            anim_ax.clear()
            for j in range(prices_array.shape[1]):
                anim_ax.plot(prices_array[:frame+1, j], label=f'Resource {j}')
            anim_ax.set_xlim(0, len(prices_array))
            anim_ax.set_ylim(prices_array.min(), prices_array.max())
            anim_ax.set_title('Price Evolution')
            anim_ax.legend()
            #if using blit=True for artists here
            return []

        anim = animation.FuncAnimation(
            anim_fig,
            animate,
            frames=min(200, len(prices_array)),  #200 frame limit 
            interval=50,
            blit=False  #anim_ax.clear(), blit=True won't be v efficient
        )

        gif_path = os.path.join(save_dir, 'price_evolution.gif')
        anim.save(gif_path, writer='pillow')
        plt.close(anim_fig)
    else:
        print("Not enough data to create animation. Skipping GIF creation.")

    # ------------ save and stuffs meow
    np.save(os.path.join(save_dir, 'returns.npy'), returns)
    np.save(os.path.join(save_dir, 'prices.npy'), all_prices)


def train():
    env = TradingEnv(num_agents=1)
    agent = PPOAgent(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    num_episodes = 1000
    steps_per_episode = 100
    
    returns = []
    all_prices = []
    
    for episode in range(num_episodes):
        rollout = collect_rollout(env, agent, steps_per_episode)
        
        episode_return = sum(rollout['rewards'])
        returns.append(episode_return)
        all_prices.append(env.prices.copy())
        
        agent.update(rollout)
        
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(returns[-10:])
            print(f"Episode {episode + 1}, Average Return: {avg_return:.2f}")
    
    #create vis
    create_analysis_plots(returns, all_prices)
    
    return returns, all_prices

if __name__ == "__main__":
    train()