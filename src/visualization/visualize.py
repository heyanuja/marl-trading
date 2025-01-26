import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd

class TradingVisualizer:
    def __init__(self, style="darkgrid"):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        self.colors = sns.color_palette("husl", 8)
    
    def plot_training_dashboard(self, returns, prices, trades=None, save_path="trading_dashboard.png"):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2)
        
        #returns over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_returns(ax1, returns)
        
        #price evol
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_prices(ax2, prices)
        
        #return dist
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_return_distribution(ax3, returns)
        
        #price correlations
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_price_correlations(ax4, prices)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns(self, ax, returns):
        df = pd.DataFrame({'returns': returns})
        window = 50
        rolling_mean = df['returns'].rolling(window=window).mean()
        rolling_std = df['returns'].rolling(window=window).std()
        
        ax.plot(df.index, df['returns'], alpha=0.3, color=self.colors[0], label='Returns')
        ax.plot(df.index, rolling_mean, color=self.colors[1], label=f'{window}-Episode Moving Average')
        ax.fill_between(df.index, 
                       rolling_mean - rolling_std,
                       rolling_mean + rolling_std,
                       alpha=0.2, color=self.colors[1])
        
        ax.set_title('Agent Returns Over Time', fontsize=14, pad=20)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prices(self, ax, prices):
        prices_array = np.array(prices)
        for i in range(prices_array.shape[1]):
            ax.plot(prices_array[:, i], 
                   label=f'Resource {i+1}',
                   color=self.colors[i],
                   alpha=0.8)
        
        ax.set_title('Resource Price Evolution', fontsize=14, pad=20)
        ax.set_xlabel('Step')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distribution(self, ax, returns):
        sns.histplot(returns, kde=True, ax=ax, color=self.colors[0])
        ax.set_title('Return Distribution', fontsize=14, pad=20)
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
    
    def _plot_price_correlations(self, ax, prices):
        prices_df = pd.DataFrame(prices)
        corr = prices_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Price Correlations', fontsize=14, pad=20)
    
    def create_trading_animation(self, prices, save_path="trading_animation.gif"):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        prices_array = np.array(prices)
        lines = [ax.plot([], [], label=f'Resource {i+1}', 
                        color=self.colors[i])[0] 
                for i in range(prices_array.shape[1])]
        
        ax.set_xlim(0, len(prices))
        ax.set_ylim(prices_array.min() * 0.9, prices_array.max() * 1.1)
        ax.set_title('Resource Price Evolution', fontsize=14, pad=20)
        ax.set_xlabel('Step')
        ax.set_ylabel('Price')
        ax.legend()
        
        def animate(frame):
            for line, price_history in zip(lines, prices_array.T):
                line.set_data(range(frame), price_history[:frame])
            return lines
        
        anim = FuncAnimation(fig, animate, frames=len(prices),
                           interval=50, blit=True)
        anim.save(save_path, writer='pillow')