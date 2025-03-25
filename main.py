import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from traders import TradingAlgorithm, FTPLMomentumTrader, MomentumTrader
from environments import AdversarialStockEnvironment


def run_simulation(environment, algorithm):
    pnl_history = []
    n_steps = environment.n_steps
    price_history = []

    for step in range(n_steps):
        prices = environment.get_prices(step)
        trade = algorithm.trade(prices, step)
        pnl = algorithm.update_pnl(prices, trade)
        pnl_history.append({'Step': step, 'PnL': pnl})
        price_history.append(prices)

    return pd.DataFrame(pnl_history), np.array(price_history)


if __name__ == "__main__":
    seed = int(time.time())

    # Initialize environment and algorithm
    env = AdversarialStockEnvironment(n_stocks=5, n_steps=1000, seed=seed)
    # algo = TradingAlgorithm(n_stocks=5)
    nr_algo = FTPLMomentumTrader(env, debug=False)
    algo1 = MomentumTrader(env, alpha=0.2, debug=False)
    algo2 = MomentumTrader(env, alpha=0.4, debug=False)
    algo3 = MomentumTrader(env, alpha=0.6, debug=False)
    algo4 = MomentumTrader(env, alpha=0.8, debug=False)
    algo5 = MomentumTrader(env, alpha=1.0, debug=False)

    # Run the simulation
    results_nr, price_history = run_simulation(env, nr_algo)
    results1, _ = run_simulation(env, algo1)
    results2, _ = run_simulation(env, algo2)
    results3, _ = run_simulation(env, algo3)
    results4, _ = run_simulation(env, algo4)
    results5, _ = run_simulation(env, algo5)

    # Plot PnL for all algorithms
    plt.figure(figsize=(12, 6))
    plt.plot(results_nr['Step'], results_nr['PnL'],
             label='FTPL', color='blue', linestyle='-')
    plt.plot(results1['Step'], results1['PnL'],
             label='alpha=0.2', color='orange', linestyle='--')
    plt.plot(results2['Step'], results2['PnL'],
             label='alpha=0.4', color='green', linestyle='--')
    plt.plot(results3['Step'], results3['PnL'],
             label='alpha=0.6', color='red', linestyle='--')
    plt.plot(results4['Step'], results4['PnL'],
             label='alpha=0.8', color='purple', linestyle='--')
    plt.plot(results5['Step'], results5['PnL'],
             label='alpha=1.0', color='brown', linestyle='--')
    plt.title('Trading PnL over Time for All Algorithms', fontsize=16)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('PnL', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

    # Plot Stock Prices
    plt.figure(figsize=(12, 6))
    for stock in range(price_history.shape[1]):
        plt.plot(range(price_history.shape[0]), price_history[:, stock], 
                 label=f'Stock {stock + 1}')
    plt.title('Stock Prices over Time', fontsize=16)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()
