import numpy as np

class CustomQLearning:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        best_next_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[state, action] = new_q

state_space_size = 10
action_space_size = 4

q_learning_agent = CustomQLearning(state_space_size, action_space_size)

for episode in range(1000):
    current_state = np.random.randint(state_space_size)

    for step in range(100):
        action = q_learning_agent.choose_action(current_state)

        next_state = np.random.randint(state_space_size)
        reward = np.random.randint(-10, 10)
        
        q_learning_agent.update_q_table(current_state, action, reward, next_state)

        current_state = next_state

def test_agent(agent, state):
    action = agent.choose_action(state)
    return action

test_state = 5
optimal_action = test_agent(q_learning_agent, test_state)

def print_q_table(q_table):
    print("Q-Table:")
    print(q_table)

state_space_size = 10
action_space_size = 4

q_learning_agent = CustomQLearning(state_space_size, action_space_size)

for episode in range(1000):
    current_state = np.random.randint(state_space_size)

    for step in range(100):
        action = q_learning_agent.choose_action(current_state)
        
        next_state = np.random.randint(state_space_size)
        reward = np.random.randint(-10, 10)

        q_learning_agent.update_q_table(current_state, action, reward, next_state)
        
        current_state = next_state

print_q_table(q_learning_agent.q_table)

print(f"Optimal action for state {test_state}: {optimal_action}")

class CustomQLearning:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        best_next_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[state, action] = new_q

def print_q_table(q_table):
    print("Q-Table:")
    print(q_table)

def respond_to_action(action):
    # Simple responses based on action
    responses = {
        0: "I don't know what to say.",
        1: "Hello!",
        2: "How are you?",
        3: "Goodbye!",
        # Add more responses as needed
    }
    return responses.get(action, "I don't understand.")

state_space_size = 10
action_space_size = 4

q_learning_agent = CustomQLearning(state_space_size, action_space_size)

print("Welcome to the Q-learning Chatbot!")
print_q_table(q_learning_agent.q_table)

# This is the while loop that handles user input and output
while True:
    # Ask the user to enter a message or type "exit" to exit
    user_input = input("You: ")

    # If the user types "exit", break the loop and end the chat
    if user_input.lower() == "exit":
        break

    # Convert the user input to a next state, which is an integer between 0 and state_space_size - 1
    # You can use any method to do this, such as hashing, encoding, or mapping
    # For simplicity, I will use a simple hash function that takes the modulo of the sum of the ASCII values of the characters
    next_state = sum(ord(c) for c in user_input) % state_space_size

    # Choose a random reward, which is an integer between -10 and 10
    reward = np.random.randint(-10, 10)

    # Choose an action based on the current state using the Q-learning agent
    action = q_learning_agent.choose_action(current_state)

    # Update the Q-table based on the current state, action, reward, and next state
    q_learning_agent.update_q_table(current_state, action, reward, next_state)

    # Convert the action to a response, which is a string
    agent_response = respond_to_action(action)

    # Print the agent response
    print("Q-Bot:", agent_response)

    # Print the Q-table
    print_q_table(q_learning_agent.q_table)

    # Set the current state to the next state
    current_state = next_state

print("Goodbye! Thanks for chatting with the Q-learning Chatbot.")
