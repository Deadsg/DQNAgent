import random
import numpy as np

class ChatBot:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.state_space = [i for i in range(vocab_size)]
        self.action_space = 5
        self.q_table = np.zeros((self.vocab_size, self.action_space))
        self.alpha = 0.1
        self.gamma = 0.8
        self.epsilon = 0.1

    def preprocess(self, sentence):
        words = sentence.split()
        word_vectors = [w for w in words]
        state = np.array([np.where(self.state_space == i)[0][0] for i in word_vectors])
        return state

    def choose_action(self, current_state):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            q_values = self.q_table[current_state]
            return np.argmax(q_values)

    def learn(self, current_state, action, reward, next_state):
        td_error = (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[current_state, action])
        self.q_table[current_state, action] += self.alpha * td_error

class QNetwork:
    def __init__(self, input_size, output_size):
        # Initialize your neural network weights and biases
        self.input_size = input_size
        self.output_size = output_size

        # Define your neural network architecture (for simplicity, a single-layer network is shown)
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.zeros(output_size)

    def predict(self, state):
        # Forward pass to get Q-values for each action
        q_values = np.dot(state, self.weights) + self.biases
        return q_values

    def update_weights(self, state, target, learning_rate=0.01):
        # Backpropagation to update weights based on the TD error
        predicted_q_values = self.predict(state)

        # Calculate the mean squared TD error
        td_errors = target - predicted_q_values
        mse_loss = np.mean(td_errors**2)

        # Gradient descent update for weights and biases
        gradient_weights = -2 * np.dot(state.T, td_errors) / state.shape[0]
        gradient_biases = -2 * np.mean(td_errors)

        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

        return mse_loss

class ChatBotWithNN:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.state_space = [i for i in range(vocab_size)]
        self.action_space = 5
        self.q_network = QNetwork(vocab_size, self.action_space)
        self.alpha = 0.1
        self.gamma = 0.8
        self.epsilon = 0.1

    def preprocess(self, sentence):
        words = sentence.split()
        word_vectors = [w for w in words]
        state = np.array([np.where(self.state_space == i)[0][0] for i in word_vectors])
        return state

    def choose_action(self, current_state):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            q_values = self.q_network.predict(current_state)
            return np.argmax(q_values)

    def learn(self, current_state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.q_network.predict(next_state))
        td_error = target - self.q_network.predict(current_state)[action]
        self.q_network.update_weights(td_error)
