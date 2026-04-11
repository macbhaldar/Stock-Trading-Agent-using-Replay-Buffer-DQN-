import numpy as np

class TradingEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = 10000
        self.shares = 0
        self.prev_value = 10000
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.step_idx]
        return np.array([
            row["price"],
            row["ma5"],
            row["ma10"],
            row["volume"],
            row["volatility"],
            self.cash,
            self.shares
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.step_idx]
        price = row["price"]

        # Execute action
        if action == 1 and self.cash >= price:  # Buy
            self.shares += 1
            self.cash -= price

        elif action == 2 and self.shares > 0:  # Sell
            self.shares -= 1
            self.cash += price

        self.step_idx += 1
        done = self.step_idx >= len(self.data) - 1

        next_state = self._get_state()

        # Portfolio value
        current_value = self.cash + self.shares * price

        # Reward = profit change (important improvement)
        reward = current_value - self.prev_value
        self.prev_value = current_value

        return next_state, reward, done
