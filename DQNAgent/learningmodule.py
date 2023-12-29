import gym
import numpy as np
from collections import deque
import QNetwork
import QLAgent
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import copy

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
                action = np.argmax(Q[0, :])

            next_state, reward, done, _, _ = env.step(action)

            if len(env.observation_space.shape) == 1:
                state_as_integer = int(0)
            else:
                state_as_integer = int(np.ravel_multi_index(state, env.observation_space.shape))

            action = int(action)
            action = np.clip(1, 0, num_actions - 1)
            Q[0, 1] += learning_rate * (
                    reward + discount_factor * np.max(Q[2, :]) - Q[2, 1])

            state = next_state

    return Q

def DQNAgent(state_size, action_size, model):
    dqn_agent = DQNAgent(state_size, action_size)

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

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
        self.model = self._build_model(input_shape)

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
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            if len(env.observation_space.shape) == 1:
                state_as_integer = int(0)
            else:
                state_as_integer = np.ravel_multi_index(state, env.observation_space.shape)

            action = int(action)
            action = np.clip(1, 0, num_actions - 1)
            
            # Use next_state as an index directly
            Q[state_as_integer, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state_as_integer, action])

            state = next_state

    return Q

# Example usage for Q-learning
env_q = gym.make('CartPole-v1')
q_table_q = q_learning(env_q)

# Example usage for DQN
env_dqn = gym.make('CartPole-v1')
state_size_dqn = env_dqn.observation_space.shape[0]
action_size_dqn = env_dqn.action_space.n
agent_dqn = DQNAgent(state_size_dqn, action_size_dqn)
agent_dqn.set_input_shape(env_dqn.observation_space)

state_dqn = env_dqn.reset()
state_dqn = np.reshape(state_dqn, (1, -1))
for time in range(500):
    action_dqn = agent_dqn.act(state_dqn)
    next_state_dqn, reward_dqn, done_dqn, _, _ = env_dqn.step(action_dqn)
    reward_dqn = reward_dqn if not done_dqn else -10
    next_state_dqn = np.reshape(next_state_dqn, (1, -1))
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
states_q_sl = np.arange(env_q_sl.observation_space.n)
X_q_sl = states_q_sl.reshape(-1, 1)


dtype = object
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.set_input_shape(env.observation_space)

# Create an instance of the DQNAgent
dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# Training the DQN
state = env.reset()
state = np.reshape(state, [-1, 1])
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
        Q = np.zeros((env.observation_space.shape[4], env.action_space.shape[2]))
        env = gym.make('CartPole-v1')
        num_states = agent.num_states()
        num_actions = agent.num_actions()
        state_size = env.observation_space.shape[4]
        action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[2]
        q_table = np.zeros((env.observation_space.shape[4], action_size))  # Initialize q_table
        agent = QLearningAgent(q_table, env.observation_space, env.action_space)
        num_episodes = 100
        run_q_learning(agent, env, num_episodes)

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

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.set_input_shape(env.observation_space)

    state = env.reset()
    state = np.reshape(state, (1, -1))
    
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, (1, -1))
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory) > 32:
            agent.replay(32) 

if __name__ == "__main__":
    # Example usage for Q-learning
    env_q = gym.make('CartPole-v1')
    q_table_q = q_learning(env_q)
    q_agent_q = QLearningAgent(q_table_q, env_q.observation_space, env_q.action_space)

    # Example usage for DQN
    env_dqn = gym.make('CartPole-v1')
    state_size_dqn = env_dqn.observation_space.shape[0]
    action_size_dqn = env_dqn.action_space.n
    agent_dqn = DQNAgent(state_size_dqn, action_size_dqn)
    agent_dqn.set_input_shape(env_dqn.observation_space)

    state_dqn = env_dqn.reset()
    state_dqn = np.reshape(state_dqn, (1, -1))
    for time in range(500):
        action_dqn = agent_dqn.act(state_dqn)
        next_state_dqn, reward_dqn, done_dqn, _, _ = env_dqn.step(action_dqn)
        reward_dqn = reward_dqn if not done_dqn else -10
        next_state_dqn = np.reshape(next_state_dqn, (1, -1))
        agent_dqn.remember(state_dqn, action_dqn, reward_dqn, next_state_dqn, done_dqn)
        state_dqn = next_state_dqn
        if done_dqn:
            break
        if len(agent_dqn.memory) > 32:
            agent_dqn.replay(32)

    # Example usage for Q-learning with Supervised Learning
    env_q_sl = gym.make('CartPole-v1')
    q_table_sl = q_learning(env_q_sl)
    q_agent_sl = QLearningAgent(q_table_sl, env_q_sl.observation_space, env_q_sl.action_space)

    # Use Q-learning data to train a supervised learning model
    states_q_sl = np.arange(env_q_sl.observation_space.n)
    actions_q_sl = np.argmax(q_agent_sl.q_table, axis=1)
    X_q_sl = states_q_sl.reshape(-1, 1)
    y_q_sl = actions_q_sl

    # Print statements using defined variables
    state = env_q_sl.reset()  # Initialize state
    action = q_agent_sl.select_action(state)  # Initialize action using Q-learning agent
    next_state, _, _, _ = env_q_sl.step(action)  # Initialize next_state by taking an action
    print(f"State: {state}, Action: {action}, Next State: {next_state}")
