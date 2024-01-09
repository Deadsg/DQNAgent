import json
from turtle import done
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
import gym

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, state, action, reward, next_state, done):
        transition = self.Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.buffer.memory) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        q_values = self.q_network(batch.state)
        next_q_values = self.target_network(batch.next_state).detach()

        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            if batch.done[i]:
                target_q_values[i][batch.action[i]] = batch.reward[i]
            else:
                target_q_values[i][batch.action[i]] = batch.reward[i] + self.gamma * torch.max(next_q_values[i])

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def load_training_data(training_data_path, encoding='utf-8'):
    with open(training_data_path, 'r', encoding=encoding) as file:
        training_data = json.load(file)
    return training_data

def train_dqn_agent(self, agent, training_data, episodes=1000):
        for episode in range(episodes):
            for data_point in training_data:
                role = data_point.get("role")
                content = data_point.get("content")

                # Convert content to a list of ASCII values
                processed_content = [ord(char) for char in content]

                # Assuming content is now in a format suitable for your Q-network
                state = torch.tensor(processed_content, dtype=torch.float32)

                # Choose an action using epsilon-greedy policy
                exploration_prob = max(self.min_epsilon, self.epsilon * self.epsilon_decay**episode)
                action = agent.select_action(state, exploration_prob)

                # Placeholder: Obtain next_state and reward based on your problem
                next_content = "..."  # Replace with your logic
                next_state = torch.tensor([ord(char) for char in next_content], dtype=torch.float32)
                reward = 1.0  # Placeholder, define the reward based on your problem

                # Update the Q-network
                agent.update_q_network(state, action, reward, next_state)

                # Update state and total reward
                state = next_state
                self.total_reward += reward

                if done:
                    break

def train_dqn():
    state_size = 4
    action_size = 2
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)
            agent.buffer.add(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    q_network = QNetwork
    optimizer = optim.Adam
    input_size = 10
    output_size = 20
    dqn_agent = DQNAgent(q_network, optimizer, input_size, output_size)
    training_data_path = "C:/Users/Mayra/Documents/AGI/Q_LLM/training_data/training_data.json"
    training_data = load_training_data(training_data_path)
    train_dqn_agent(dqn_agent, training_data, episodes=1000)
    train_dqn()