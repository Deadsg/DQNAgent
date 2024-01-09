import QLAgent
import numpy as np
import gym

class QNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.zeros(output_size)

    def predict(self, state):
        return np.dot(state, self.weights) + self.biases

    def update_weights(self, state, target, learning_rate=0.01):
        predicted_q_values = self.predict(state)
        td_errors = target - predicted_q_values

        # Gradient descent update for weights and biases
        gradient_weights = -2 * np.dot(state.T, td_errors) / state.shape[0]
        gradient_biases = -2 * np.mean(td_errors)

        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

        return np.mean(td_errors**2)

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_network = QNetwork(observation_space.shape[0], action_space.n)
        self.target_network = QNetwork(observation_space.shape[0], action_space.n)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.weights = np.copy(self.q_network.weights)
        self.target_network.biases = np.copy(self.q_network.biases)

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space.n)

        state = np.array(state).reshape(1, -1)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done, gamma=0.99, batch_size=32):
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)

        target = self.q_network.predict(state)

        if done:
            target[0][action] = reward
        else:
            next_q_values = self.target_network.predict(next_state)
            target[0][action] = reward + gamma * np.max(next_q_values)

        self.q_network.update_weights(state, target)

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
    print("Q Network Weights:", agent.q_network.weights)
    print("Q Network Biases:", agent.q_network.biases)
