import numpy as np
from scipy.optimize import minimize


class TradingAlgorithm:
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
        self.position = np.zeros(n_stocks)
        self.cash = 0

    def trade(self, prices, step):
        trade = np.random.uniform(-1, 1, self.n_stocks)
        return trade

    def update_pnl(self, prices, trade):
        trade_cost = np.dot(trade, prices)
        self.position += trade
        self.cash -= trade_cost
        return self.cash + np.dot(self.position, prices)


class MomentumTrader:
    def __init__(self, env, k=30, temp=30., alpha=0.5, debug=True):
        self.n_stocks = env.n_stocks
        self.price_history = [env.get_prices(0)]
        self.increase_history = []
        self.position = np.zeros(self.n_stocks)
        self.cash = 1000
        self.temp = temp
        self.alpha = alpha
        self.k = k
        self.debug = debug

    def trade(self, prices, step):
        self.price_history.append(prices)
        self.increase_history.append(
            self.price_history[-1] / self.price_history[-2])

        allocation = np.zeros(self.n_stocks)
        for i in range(1, min(self.k, step + 1)):
            allocation += np.power(self.alpha, i) * \
                self.increase_history[step + 1 - i]
        ones = np.ones(self.n_stocks)

        # print(allocation)
        allocation = allocation - np.mean(allocation)
        allocation = np.exp(self.temp * allocation)
        allocation = allocation / np.dot(ones, allocation)
        # print(allocation)

        current_wealth = self.cash + np.dot(self.position, prices)

        new_position = current_wealth * allocation / prices

        # print(current_wealth)
        print('old position', self.position) if self.debug else ()
        print('new position', new_position) if self.debug else ()

        return new_position - self.position

    def update_pnl(self, prices, trade):
        trade_cost = np.dot(trade, prices)
        self.position += trade
        self.cash -= trade_cost
        return self.cash + np.dot(self.position, prices)


class FTPLMomentumTrader:
    def __init__(self, env, k=30, temp=30., eta=1, debug=True):
        self.n_stocks = env.n_stocks
        self.price_history = [env.get_prices(0)]
        self.increase_history = []
        self.position = np.zeros(self.n_stocks)
        self.cash = 1000
        self.eta = eta
        self.temp = temp
        self.k = k
        self.debug = debug

    def trade(self, prices, step):
        self.price_history.append(prices)
        self.increase_history.append(
            self.price_history[-1] / self.price_history[-2])

        sigma = np.random.exponential(self.eta)

        def objective(alpha):
            y = -sigma * alpha
            for s in range(step + 1):
                d_s = self.increase_history[s]
                allocation = np.zeros(self.n_stocks)
                for i in range(1, min(self.k, s + 1)):
                    allocation += np.power(alpha, i) * \
                        self.increase_history[s - i]
                ones = np.ones(self.n_stocks)
                allocation = allocation - np.mean(allocation)
                allocation = np.exp(self.temp * allocation)
                allocation = allocation / np.dot(ones, allocation)
                y -= np.log(np.dot(d_s, allocation))
            return y

        result = minimize(objective, 0.5, method='SLSQP', bounds=[(0.01, 1)])
        alpha_star = result.x.item()

        print(alpha_star, result.success) if self.debug else ()

        allocation = np.zeros(self.n_stocks)
        for i in range(1, min(self.k, step + 1)):
            allocation += np.power(alpha_star, i) * \
                self.increase_history[step + 1 - i]
        ones = np.ones(self.n_stocks)

        # print(allocation)
        allocation = allocation - np.mean(allocation)
        allocation = np.exp(self.temp * allocation)
        allocation = allocation / np.dot(ones, allocation)
        # print(allocation)

        current_wealth = self.cash + np.dot(self.position, prices)

        new_position = current_wealth * allocation / prices

        # print(current_wealth)
        print('old position', self.position) if self.debug else ()
        print('new position', new_position) if self.debug else ()

        return new_position - self.position

    def update_pnl(self, prices, trade):
        trade_cost = np.dot(trade, prices)
        self.position += trade
        self.cash -= trade_cost
        return self.cash + np.dot(self.position, prices)
