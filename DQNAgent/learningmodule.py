import gym
import numpy as np
from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import copy
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import torch.optim as optim

class QNetworkAgent():
    pass

def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.transformer_model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.transformer_model.config.hidden_size, output_size)

    def forward(self, state):
        inputs = self.transformer_model(**state)
        x = self.flatten(inputs.last_hidden_state.mean(dim=1))
        return self.dense(x)

class QLearningAgent:
    def __init__(self, q_network, learning_rate=0.001, discount_factor=0.9, exploration_prob=0.1):
        self.q_network = q_network
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_prob
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.num_states = 0

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.q_network.dense.out_features)
        else:
            q_values = self.q_network(state)
            return np.argmax(q_values.detach().numpy())

    def update_q_network(self, state, action, reward, next_state):
        state_tensor = {k: torch.tensor([v]) for k, v in state.items()}
        next_state_tensor = {k: torch.tensor([v]) for k, v in next_state.items()}

        self.optimizer.zero_grad()
        q_values_current = self.q_network(state_tensor)
        q_value_next = np.max(self.q_network(next_state_tensor).detach().numpy())

        td_target = reward + self.discount_factor * q_value_next
        td_error = td_target - q_values_current[0, action].item()
        loss = self.loss_fn(q_values_current, torch.tensor([[td_target]]))
        loss.backward()
        self.optimizer.step()

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    # Handle different types of observation spaces
    if hasattr(env.observation_space, 'shape'):
        if isinstance(env.observation_space.shape, int):
            state_size = env.observation_space.shape
        else:
            state_size = env.observation_space.shape[0]
    else:
        state_size = 1

    Q = np.zeros((state_size, num_actions))

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            # Handle different types of observation spaces
            if hasattr(env.observation_space, 'shape'):
                if isinstance(env.observation_space.shape, int):
                    state_as_integer = state
                    next_state_as_integer = next_state
                else:
                    state_as_integer = np.ravel_multi_index(state, env.observation_space.shape)
                    next_state_as_integer = np.ravel_multi_index(next_state, env.observation_space.shape)
            else:
                state_as_integer = int(0)
                next_state_as_integer = int(0)

            action = int(action)
            action = np.clip(action, 0, num_actions - 1)

            Q[state_as_integer, action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state_as_integer]) - Q[state_as_integer, action])

            state = next_state

    return Q

def calculate_y_q_sl():
    # Placeholder implementation, replace with your logic
    return np.zeros((100,))  # Assuming a numpy array for illustration

class SupervisedLearningModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

def supervised_learning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SupervisedLearningModel()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)

class DQNAgent:
    def __init__(self, state_size, action_size, num_states):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tokenizer = AutoTokenizer
        self.model = AutoModel
        self.num_states = num_states

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.transformer_model = AutoModel.from_pretrained("EleutherAI/gpt-neox-20b")

        # Build the Q-network using the transformer
        self._build_model()

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def _encode_state(self, state):
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True, padding=True)
        return inputs

    def _build_model(self):
        # You may need to modify this based on the transformer model's architecture
        input_size = self.transformer_model.config.hidden_size
        output_size = self.action_size

        # Custom layers for adapting the transformer output to Q-values
        self.dense = nn.Linear(input_size, output_size)

        # Define the Q-network model
        self.model = nn.Sequential(
            self.transformer_model,
            nn.Flatten(),
            self.dense
        )

        if hasattr(env.observation_space, 'shape'):
            if isinstance(env.observation_space.shape, int):
                self.num_states = env.observation_space.shape
            else:
                self.num_states = env.observation_space.shape[0]
        else:
            self.num_states = 1

    def num_states(self):  # Corrected indentation to define a method
        return self.num_states

    def train(self, states, targets):
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

def get_num_episodes():
    return 100

def shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        return space.shape[0]

def observation_space():
    pass

def action_space():
    

    def QLearningAgent(_, self, num_actions, learning_rate, discount_factor, exploration_prob, num_states, action_space):

        def run_q_learning(agent, env, _):
            pass

        def num_actions():
            pass

        def learning_rate():
            pass

        def discount_factor():
            pass

        def exploration_prob():
            pass

        def num_states():
            pass

        def env(observation_space, action_space, n):
            observation_space = (4,)
            action_space = (2)

class DQNAgent:
    def __init__(self, state_size, action_size, num_states):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = AutoModel.from_pretrained("EleutherAI/gpt-neox-20b")
        self.num_states = num_states

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_input_shape(self, observation_space):
        input_shape = observation_space.shape[0]

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    if len(env.observation_space.shape) == 1:
        state_size = env.observation_space.shape[0]
    else:
        state_size = np.prod(env.observation_space.shape)

    Q = np.zeros((state_size, num_actions))

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[ 1, :])

            next_state, reward, done, _, _ = env.step(action)

            if len(env.observation_space.shape) == 1:
                state_as_integer = int(0)
                next_state_as_integer = int(0)
            else:
                state_as_integer = np.ravel_multi_index(state, env.observation_space.shape)
                next_state_as_integer = np.ravel_multi_index(next_state, env.observation_space.shape)

            action = int(action)
            action = np.clip(1, 0, num_actions - 1)
            
            # Use next_state_as_integer as an index directly
            Q[state_as_integer, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state_as_integer]) - Q[state_as_integer, action])

            state = next_state

    return Q

# Example usage for Q-learning
env_q = gym.make('CartPole-v1')
q_table_q = q_learning(env_q)

num_states = 0
env_dqn = gym.make('CartPole-v1')
state_size_dqn = env_dqn.observation_space.shape[0]
action_size_dqn = env_dqn.action_space.n
agent_dqn = DQNAgent(1, -1, 1)
agent_dqn.set_input_shape(env_dqn.observation_space)

state_dqn = env_dqn.reset()

state_dqn = env_dqn.reset()
#state_dqn = np.reshape(-1, (1, ))
state_dqn = np.array([-1])
for time in range(500):
    action_dqn = agent_dqn.act(state_dqn)
    next_state_dqn, reward_dqn, done_dqn, _, _ = env_dqn.step(action_dqn)
    reward_dqn = reward_dqn if not done_dqn else -10
    agent_dqn.remember(state_dqn, action_dqn, reward_dqn, next_state_dqn, done_dqn)
    state_dqn = next_state_dqn

    if done_dqn:
        break

    if len(agent_dqn.memory) > 32:
        agent_dqn.replay(32)

# Example usage for Q-learning with Supervised Learning
env_q_sl = gym.make('CartPole-v1')
q_table_q_sl = q_learning(env_q_sl)


# Use Q-learning data to train a supervised learning model
states_q_sl = np.random.uniform(env_q_sl.observation_space.low, env_q_sl.observation_space.high, size=(1000, env_q_sl.observation_space.shape[0]))
X_q_sl = states_q_sl.reshape(-1, 1)


dtype = object
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, num_states)
agent.set_input_shape(env.observation_space)

# Create an instance of the DQNAgent
dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, num_states)

# Training the DQN
state = env.reset()
state = np.reshape(0, [-1, 1])
for time in range(500):
    action = dqn_agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [-1, 1])
    dqn_agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
    if len(dqn_agent.memory) > 32:
        dqn_agent.replay(32)

class QLearningAgent:
    def __init__(self, q_table, observation_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.q_table = q_table
        self.num_actions = action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0]
        self.num_states = observation_space.shape[0]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_prob
        self.model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.num_states = num_states

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    Q = np.zeros((env.observation_space.shape[0], num_actions))

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            # Update Q-value
            Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

        return Q

env = gym.make('CartPole-v1')
observation_space = env.observation_space
action_space = env.action_space

if isinstance(action_space, gym.spaces.Discrete):
    q_table = np.zeros((observation_space.shape[0], action_space.n))
else:
    q_table = np.zeros((observation_space.shape[0], action_space.shape[0]))

    q_agent = QLearningAgent(q_table, observation_space, action_space)
    q_agent.update_q_table(state, action, reward, next_state) 

    env = gym.make('CartPole-v1')

    learning_rate_q = 0.1
    discount_factor_q = 0.9
    exploration_prob_q = 0.1
    num_episodes_q = 100

    q_table = q_learning(env, learning_rate=learning_rate_q, discount_factor=discount_factor_q, epsilon=exploration_prob_q, episodes=num_episodes_q)

    q_agent = QLearningAgent(q_table, env.observation_space, env.action_space, learning_rate_q, discount_factor_q, exploration_prob_q)

    num_episodes_q = 100

    states_q = np.arange(env.observation_space.n)
    actions_q = np.argmax(q_agent.q_table, axis=1)
    X_q = states_q.reshape(-1, 1)
    y_q = actions_q

def select_action(self, state):
    if np.random.rand() < self.exploration_rate:
        return np.random.choice(self.num_actions)
    else:
        return np.argmax(self.q_table[state, :])

def update_q_table(self, state, action, reward, next_state):
    best_next_action = np.argmax(self.q_table[int(next_state), :])
    td_target = reward + self.discount_factor * self.q_table[int(next_state), best_next_action]
    td_error = td_target - self.q_table[state, action]
    self.q_table[state, action] += self.learning_rate * td_error

def q_learning(env, learning_rate, discount_factor, epsilon, episodes):
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        target = reward + discount_factor * np.max(next_state)
        target_f = (state)
        target_f[0][action] = target
        state = next_state
        env_q_sl = gym.make('CartPole-v1')
        q_table_sl = q_learning(env_q_sl)
        q_agent_sl = QLearningAgent(q_table_sl, env_q_sl.observation_space, env_q_sl.action_space)
        q_agent_sl.run_q_learning(env_q_sl, 100)

        # Use Q-learning data to train a supervised learning model
        states_q_sl = np.arange(env_q_sl.observation_space.n)
        actions_q_sl = np.argmax(q_agent_sl.q_table, axis=1)

        print(f"State: {state}, Action: {action}, Next State: {next_state}")

       
def get_num_actions(self, action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    else:
        return action_space.shape[2]

if hasattr(env.observation_space, 'shape'):
    num_states = env.observation_space.shape[0] if isinstance(env.observation_space.shape, tuple) else env.observation_space.shape
    Q = np.zeros((num_states, env.action_space.n))
else:
    Q = np.zeros((1, env.action_space.n))
env = gym.make('CartPole-v1')
num_actions = agent.num_actions()
state_size = env.observation_space.shape[4]
action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[2]
q_table = np.zeros((env.observation_space.shape[4], action_size))  # Initialize q_table
agent = QLearningAgent(q_table, env.observation_space, env.action_space)
num_episodes = 100

class SupervisedLearningModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

def supervised_learning(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the SupervisedLearningModel
    model = SupervisedLearningModel()

    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model on the test set
    model.evaluate(X_test, y_test)

    y_q_sl = calculate_y_q_sl()

    supervised_learning(X_q_sl, y_q_sl)

class SupervisedLearningModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def supervised_learning(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SupervisedLearningModel()
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

class reinforcement_learning:
    def __init__(self, q_network_agent, observation_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.q_network_agent = q_network_agent
        self.num_actions = self.get_num_actions(action_space)
        self.num_states = self.get_num_states(observation_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_prob

    def get_num_actions(self, action_space):
        if isinstance(action_space, gym.spaces.Discrete):
            return action_space.n
        else:
            return action_space.shape[0]

    def get_num_states(self, observation_space):
        if hasattr(observation_space, 'shape'):
            if isinstance(observation_space.shape, int):
                return observation_space.shape
            else:
                return observation_space.shape[0]
        else:
            return 1

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_network_agent.predict(state)
            return np.argmax(q_values)

    def update_q_network(self, state, action, reward, next_state):
        # Q-network update logic based on Q-learning algorithm
        q_values_current = self.q_network_agent.predict(state)
        q_value_next = np.max(self.q_network_agent.predict(next_state))

        td_target = reward + self.discount_factor * q_value_next
        td_error = td_target - q_values_current[action]

        # Update Q-values
        q_values_current[action] += self.learning_rate * td_error

        # Train the Q-network with the updated Q-values
        self.q_network_agent.train(state, q_values_current)

    class ChatBot:
        def __init__(self, vocab_size=10000):
            self.vocab_size = vocab_size
            self.state_space = [i for i in range(vocab_size)]
            self.action_space = 5
            self.q_table = np.zeros((self.vocab_size, self.action_space))
            self.alpha = 0.1
            self.gamma = 0.8
            self.epsilon = 0.1

            self.llm_model = GPT2LMHeadModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

        def preprocess(self, sentence):
            words = sentence.split()
            word_vectors = [w for w in words]
            state = np.array([np.where(np.array(self.state_space) == i)[0][0] if i in self.state_space else 0 for i in word_vectors])
            return state

        def choose_action(self, current_state):
            if random.random() < self.epsilon:
                return np.random.randint(self.action_space)
            else:
                q_values = self.q_table[current_state]
                return np.argmax(q_values)

        def generate_response_llm(self, prompt):
            response = self.llm_generator(prompt, max_length=50, num_return_sequences=1)
            return response[0]['generated_text']

        def learn(self, current_state, action, reward, next_state):
            td_error = (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[current_state, action])
            self.q_table[current_state, action] += self.alpha * td_error

        def show_q_table(self):
            print("Q-Table:")
            print(self.q_table)

        def save_q_table(self, filename):
            np.save(filename, self.q_table)

def main():
    env_q = gym.make('CartPole-v1')
    q_table_q = q_learning(env_q, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000)
    q_agent_q = QLearningAgent(q_table_q, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1)

    env_dqn = gym.make('CartPole-v1')
    state_size_dqn = env_dqn.observation_space.shape[0]
    action_size_dqn = env_dqn.action_space.n
    agent_dqn = QLearningAgent(np.zeros((state_size_dqn, action_size_dqn)), learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1)

    env_q_sl = gym.make('CartPole-v1')
    q_table_sl = q_learning(env_q_sl, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000)
    q_agent_sl = QLearningAgent(q_table_sl, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1)

    states_q_sl = np.arange(env_q_sl.observation_space.n)
    actions_q_sl = np.argmax(q_agent_sl.q_table, axis=1)
    X_q_sl = states_q_sl.reshape(-1, 1)
    y_q_sl = actions_q_sl

    state = env_q_sl.reset()
    action = q_agent_sl.select_action(state)
    next_state, _, _, _ = env_q_sl.step(action)
    print(f"State: {state}, Action: {action}, Next State: {next_state}")

    # Perform Supervised Learning using Q-learning data
    supervised_learning(X_q_sl, y_q_sl)

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
    
    chatbot = ChatBot()

    with open('your_custom_data.txt', 'r') as file:
        conversation = file.readlines()

    for epoch in range(1000):
        for i, sentence in enumerate(conversation):
            current_state = chatbot.preprocess(sentence)

            action = chatbot.choose_action(current_state)

            reward = random.uniform(0, 1)

            next_state = chatbot.preprocess(conversation[i + 1]) if i + 1 < len(conversation) else current_state

            chatbot.learn(current_state, action, reward, next_state)

        chatbot.show_q_table()
        chatbot.save_q_table(f'q_table_epoch_{epoch}.npy')

    test_sentence = "How's it going?"
    test_state = chatbot.preprocess(test_sentence)
    predicted_action = chatbot.choose_action(test_state)

    print(f"Test Sentence: '{test_sentence}'")
    print(f"Predicted Action: {predicted_action}")

    llm_response = chatbot.generate_response_llm("Tell me more about yourself.")
    print(f"LLM Response: {llm_response}")

if __name__ == "__main__":
    main()
