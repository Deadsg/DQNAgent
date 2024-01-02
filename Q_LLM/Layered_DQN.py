import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from Reasoning_ import train_layered_dqn

# Definition of the QNetwork class (as used in the DQNAgent)
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = torch.relu(self.dense2(x))
        return self.output_layer(x)

# Definition of the DQNAgent class (as used in the main training loop)
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=64, replay_buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        experiences = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

        q_values = self.q_network(states).gather(1, actions)

        next_q_values = self.target_network(next_states).detach()
        max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True).values
        td_targets = rewards + self.gamma * (1 - dones) * max_next_q_values

        loss = nn.MSELoss()(q_values, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        tau = 0.001
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Definition of the ReplayBuffer class
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)

# Continue the main training loop
# ...

if __name__ == "__main__":
    train_layered_dqn()