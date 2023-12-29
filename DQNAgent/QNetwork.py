import QLAgent
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
import gym

def QNetwork():
    pass

def dtype():
    dtype(reshape, dtype=np.float32)
    state = np.array(state, reshape, dtype=(object))

def convert_dtype(reshape_func, state):
    return np.array(reshape_func(state), dtype=np.float32)

# Reshape function to flatten the input state
reshape = lambda x: np.array(x).reshape(1, -1)

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.update_target_network()

    def build_q_network(self):
        model = models.Sequential([
            Dense(64, activation='relu', input_shape=self.observation_space.shape),
            Dense(64, activation='relu'),
            Dense(self.action_space.n, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space.n)

        state = convert_dtype(reshape, state)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done, gamma=0.99, batch_size=32):
        state = convert_dtype(reshape, state)
        next_state = convert_dtype(reshape, next_state)

        target = self.q_network.predict(state)

        if done:
            target[0][action] = reward
        else:
            next_q_values = self.target_network.predict(next_state)
            target[0][action] = reward + gamma * np.max(next_q_values)

        self.q_network.fit(state, target, epochs=1, verbose=0, batch_size=batch_size)

env = gym.make('CartPole-v1')
observation_space = env.observation_space
action_space = env.action_space
agent = DQNAgent(observation_space, action_space)

for episode in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act([0, 2, 3, 0], 0.1)
        next_state, reward, done, _, _ = env.step(action)

        agent.train([0.3, 0.3, 0.3, 0.3], action, reward, [0.3, 0.3, 0.3, 0.3], done)
        total_reward += reward
        state = next_state

        # Print the values of state and next_state
        print("State:", state)
        print("Next State:", next_state)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    print(agent.q_network.summary())
    for layer in agent.q_network.layers:
        print(layer.get_weights())