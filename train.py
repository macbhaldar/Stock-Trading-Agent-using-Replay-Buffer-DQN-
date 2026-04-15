import torch
import torch.optim as optim
import pandas as pd
import random
from replay_buffer import ReplayBuffer
from dqn_agent import DQN
from environment import TradingEnv
import torch.nn as nn

# Generate dummy data if not exists
import os
if not os.path.exists("data/stock_data.csv"):
    import numpy as np
    n = 500
    price = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({"price": price})
    df["ma5"] = df["price"].rolling(5).mean().bfill()
    df["ma10"] = df["price"].rolling(10).mean().bfill()
    df.to_csv("data/stock_data.csv", index=False)

data = pd.read_csv("data/stock_data.csv")

env = TradingEnv(data)
model = DQN(5, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
buffer = ReplayBuffer(5000)

gamma = 0.99
batch_size = 32

def train_step():
    if buffer.size() < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.FloatTensor(list(states))
    next_states = torch.FloatTensor(list(next_states))
    actions = torch.LongTensor(list(actions))
    rewards = torch.FloatTensor(list(rewards))

    q_values = model(states)
    next_q_values = model(next_states)

    target = rewards + gamma * torch.max(next_q_values, dim=1)[0]
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    loss = nn.MSELoss()(current_q, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for ep in range(10):
    state = env.reset()
    total_reward = 0

    while True:
        if random.random() < 0.1:
            action = random.randint(0, 2)
        else:
            state_tensor = torch.FloatTensor(state)
            action = torch.argmax(model(state_tensor)).item()

        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {ep}, Reward: {total_reward}")
