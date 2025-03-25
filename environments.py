import numpy as np


class AdversarialStockEnvironment:
    def __init__(self, n_stocks=10, n_steps=100, seed=None):
        self.n_stocks = n_stocks
        self.n_steps = n_steps
        self.seed = seed
        self.prices = None
        self._initialize_prices()

    def _initialize_prices(self):
        """
        Initialize stock prices with adversarial patterns, random starting values, 
        and a momentum component to simulate trends, using geometric changes.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate random starting prices for each stock between $50 and $150
        initial_prices = np.random.uniform(100, 100, self.n_stocks)
        self.prices = np.zeros((self.n_steps, self.n_stocks))
        self.prices[0] = initial_prices  # Set initial prices

        momentum_factor = 0.95  # Controls the strength of momentum (0 to 1)
        for stock in range(self.n_stocks):
            price = initial_prices[stock]
            momentum = 0  # Initialize momentum for each stock

            for t in range(1, self.n_steps):
                # Random noise simulating percentage fluctuations (scaled)
                noise = np.random.uniform(-0.02, 0.021)

                # Adversarial shift to create cyclic or irregular patterns
                adversarial_shift = np.sin(
                    # ±1% shift
                    2 * np.pi * t / self.n_steps) * np.random.uniform(-0.01, 0.01)

                # Update price geometrically with momentum
                percentage_change = (
                    1 + noise + adversarial_shift) * (1 + momentum_factor * momentum)
                price *= percentage_change

                # Update momentum for the next step
                momentum = noise + adversarial_shift

                # Prevent prices from going below $1
                price = max(price, 1)

                self.prices[t, stock] = price

    def get_prices(self, step):
        if step >= self.n_steps:
            raise ValueError("Step exceeds the number of simulated steps.")
        return self.prices[step]

    def reset(self):
        self._initialize_prices()
