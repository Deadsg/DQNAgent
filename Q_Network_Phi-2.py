import gym
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn, optim

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, state):
        x = torch.relu(self.dense1(1))
        x = torch.relu(self.dense2(x))
        return self.output_layer(x)

class QLearningAgent:
    def __init__(self, q_network, optimizer, state_size, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.q_network = q_network
        self.optimizer = optimizer(self.q_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_prob
        self.num_actions = q_network.output_layer.out_features
        self.state_size = state_size  # Corrected initialization
        self.loss_fn = nn.MSELoss()  # Added loss function

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_network(1)
            return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state):
        self.optimizer.zero_grad()

        q_values_current = self.q_network(state)
        q_value_next = torch.max(self.q_network(next_state).detach()).item()

        td_target = reward + self.discount_factor * q_value_next
        td_error = td_target - q_values_current[0, action].item()

        loss = self.loss_fn(q_values_current.squeeze(), torch.tensor([td_target], dtype=torch.float32))
        loss.backward()
        self.optimizer.step()

    def q_learning(env, q_network, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, episodes=1000):
      num_states = env.observation_space.n
      num_actions = env.action_space.n

      q_agent = QLearningAgent(q_network, num_states, learning_rate, discount_factor, epsilon)

      for episode in range(episodes):
          state = env.reset()
          done = False
          total_reward = 0

          while not done:
              action = q_agent.select_action(state)
              next_state, reward, done, _ = env.step(action)
              q_agent.update_q_table(state, action, reward, next_state)
              state = next_state
              total_reward += reward

          print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def update_q_network(self, state, action, reward, next_state):
        self.optimizer.zero_grad()

        q_values_current = self.q_network(state)
        q_value_next = torch.max(self.q_network(next_state).detach()).item()

        td_target = reward + self.discount_factor * q_value_next
        td_error = td_target - q_values_current[0, action].item()

        loss = self.loss_fn(q_values_current.squeeze(), torch.tensor([td_target], dtype=torch.float32))
        loss.backward()
        self.optimizer.step()

def chat_with_phi_2_q_loop():
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    q_network = QNetwork(state_size, num_actions)
    optimizer = optim.Adam  # Use Adam optimizer for Q-network

    q_learning_agent = QLearningAgent(q_network, optimizer, state_size)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        state_tensor = torch.tensor([1], dtype=torch.float32)

        done = False
        while not done:
            action = q_learning_agent.select_action(2)
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)

            q_learning_agent.update_q_network(state_tensor, action, reward, next_state_tensor)

            state = next_state
            state_tensor = next_state_tensor

    # Load Phi-2 model
    model_name = "microsoft/phi-2"
    phi_2_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi_2_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Tokenize and convert to tensor
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

        # Generate model output without beam search
        with torch.no_grad():
            output = phi_2_model.generate(input_ids, max_length=50, num_beams=1, temperature=0.7)

        # Decode and print the model's response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Phi-2:", response)

if __name__ == "__main__":
    chat_with_phi_2_q_loop()
