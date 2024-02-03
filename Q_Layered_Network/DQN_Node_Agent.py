from turtle import done
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.autograd import Variable
import numpy as np
from collections import namedtuple, deque
import random
import gym


def DQN_Node_Agent():
    QNetwork
    DQNAgent(self, q_network, optimizer, state_size, action_size, input_size, output_size, learning_rate=0.001, discount_factor=0.9, buffer_size=10000, batch_size=64, gamma=0.99, min_epsilon=0.01, epsilon_decay=0.995, target_update_frequency=100, epsilon=1.0)
    self = DQNAgent
    learning_rate = 0.001
    q_network = QNetwork
    optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)


class DQNAgent:
    def __init__(self, q_network, optimizer, state_size, action_size, input_size, output_size, learning_rate=0.001, discount_factor=0.9, buffer_size=10000, batch_size=64, gamma=0.99, min_epsilon=0.01, epsilon_decay=0.995, target_update_frequency=100, epsilon=1.0):
        self.q_network = q_network(input_size, output_size)  # Create an instance of QNetwork
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate)  # Create an instance of the optimizer
        self.discount_factor = discount_factor
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.total_steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

def export_to_onnx(self, input_example, onnx_file_path="dqn_node_model.onnx"):
        dummy_input = Variable(torch.randn(input_example).view(1, -1))
        torch.onnx.export(
            self.q_network,
            dummy_input,
            onnx_file_path,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )

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

# Define the replay buffer
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

# Define the DQN Agent
class DQNAgent:
    def __init__(self, q_network, optimizer, state_size, action_size, input_size, output_size, learning_rate=0.001, discount_factor=0.9, buffer_size=10000, batch_size=64, gamma=0.99, min_epsilon=0.01, epsilon_decay=0.995, target_update_frequency=100, epsilon=1.0):
        self.q_network = q_network(input_size, output_size)  # Create an instance of QNetwork
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate)  # Create an instance of the optimizer
        self.discount_factor = discount_factor
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.total_steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
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

        # Compute Q-values
        q_values = self.q_network(batch.state)
        next_q_values = self.target_network(batch.next_state).detach()

        # Compute target Q-values
        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            if batch.done[i]:
                target_q_values[i][batch.action[i]] = batch.reward[i]
            else:
                target_q_values[i][batch.action[i]] = batch.reward[i] + self.gamma * torch.max(next_q_values[i])

        # Compute loss and update Q-network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Main training loop
def train_dqn():
    # Environment parameters
    state_size = 4  # Replace with your environment's state size
    action_size = 2  # Replace with your environment's action size
    env = gym.make("CartPole-v1")  # Replace with your environment

    # DQN agent
    agent = DQNAgent(state_size, action_size, input_size=128, output_size=64)

    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Ensure the state has the correct length
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)

            # Choose action
            action = agent.select_action(state, epsilon)

            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)

            # Store experience in replay buffer
            agent.buffer.add(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

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

class DQNAgent:
    def __init__(self, q_network, optimizer, state_size, action_size, input_size, output_size, learning_rate=0.001, discount_factor=0.9, buffer_size=10000, batch_size=64, gamma=0.99, min_epsilon=0.01, epsilon_decay=0.995, target_update_frequency=100, epsilon=1.0):
        self.q_network = q_network(input_size, output_size)  # Create an instance of QNetwork
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate)  # Create an instance of the optimizer
        self.discount_factor = discount_factor
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.total_steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state, exploration_prob):
        if np.random.rand() < exploration_prob:
            return np.random.choice(self.q_network.output_layer.out_features)
        else:
            q_values = self.q_network(128)
            return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state):
        state = state.view(1, -1)
        next_state = next_state.view(1, -1)

        q_values_current = self.q_network(state)
        action_q_value_current = q_values_current[0, action]

        with torch.no_grad():
            next_q_values = self.target_network(next_state)
            max_next_q_value, _ = torch.max(next_q_values, dim=1, keepdim=True)
            target_q_value = reward + self.discount_factor * max_next_q_value

        td_error = target_q_value - action_q_value_current

        loss = self.loss_fn(action_q_value_current, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_frequency == 0:
            self.update_target_network()

        self.total_steps += 1

    def export_to_onnx(self, input_example, onnx_file_path="dqn_node_model.onnx"):
        dummy_input = Variable(torch.randn(input_example).view(1, -1))
        torch.onnx.export(
            self.q_network,
            dummy_input,
            onnx_file_path,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )

    def train_dqn_agent(self, agent, training_data, episodes=1000):
        for episode in range(episodes):
            for data_point in training_data:
                role = data_point.get("role")
                content = data_point.get("content")

                processed_content = [ord(char) for char in content]

                state = torch.tensor(processed_content, dtype=torch.float32)

                exploration_prob = max(self.min_epsilon, self.epsilon * self.epsilon_decay**episode)
                action = agent.select_action(state, exploration_prob)

                next_content = "..."
                next_state = torch.tensor([ord(char) for char in next_content], dtype=torch.float32)
                reward = 1.0

                agent.update_q_network(state, action, reward, next_state)

                state = next_state
                self.total_reward += reward

                if done:
                    break

    def load_training_data(training_data_path, encoding='utf-8'):
        with open(training_data_path, 'r', encoding=encoding) as file:
            training_data = json.load(file)
        return training_data

    @classmethod
    def train_dqn_agent(self, agent, training_data, episodes=1000):
        for episode in range(episodes):
            for data_point in training_data:
                role = data_point.get("role")
                content = data_point.get("content")

                processed_content = [ord(char) for char in content]

                state = torch.tensor(processed_content, dtype=torch.float32)

                next_content = "..."
                next_state = torch.tensor([ord(char) for char in next_content], dtype=torch.float32)


if __name__ == "__main__":
    action_size = 64
    state_size = 128
    min_epsilon = 0.01
    epsilon = 1.0
    input_size = 128
    output_size = 64
    episode = 0
    dqn_agent = DQNAgent(QNetwork, optim.Adam, state_size, action_size, input_size, output_size,
                         gamma=0.99, min_epsilon=min_epsilon, epsilon_decay=0.995, target_update_frequency=100, epsilon=epsilon)

    exploration_prob = max(dqn_agent.min_epsilon, dqn_agent.epsilon * dqn_agent.epsilon_decay**episode)
    print(exploration_prob)

    training_data_path = "C:/Users/Mayra/Documents/AGI/DQNAgent/Q_LLM/training_data/training_data.json"
    training_data = DQNAgent.load_training_data(training_data_path)

    exploration_prob = max(dqn_agent.min_epsilon, dqn_agent.epsilon * dqn_agent.epsilon_decay**episode)
    dqn_agent.train_dqn_agent(dqn_agent, training_data, episodes=1000)
    input_example = torch.randn(1, 128)
    dqn_agent.export_to_onnx(input_example.size())
