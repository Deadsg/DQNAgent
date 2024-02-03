import torch
import logging
from DQN_Node_Agent import DQNAgent, QNetwork
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stable-code-3b", torch_dtype="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-code-3b", trust_remote_code=True)

logging.basicConfig(filename='dqn_llm_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DQNMistralLLM:
    def __init__(self):
        self.state_size = 128
        self.action_size = 64
        self.dqn_agent = DQNAgent(QNetwork, torch.optim.Adam,
                                  self.state_size, self.action_size,
                                  input_size=128, output_size=64,
                                  gamma=0.99, min_epsilon=0.01,
                                  epsilon_decay=0.995,
                                  target_update_frequency=100,
                                  epsilon=1.0)

        self.training_data_path = "C:/Users/Mayra/Documents/AGI/Q_LLM/training_data/training_data.json"
        self.training_data = DQNAgent.load_training_data(
            self.training_data_path)

        model_path = "stabilityai/stable-code-3b"
        self.mistral_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                                  torch_dtype="auto",
                                                                  trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

    def train_dqn_agent(self, episodes=1000):
        self.dqn_agent.train_dqn_agent(self.dqn_agent,
                                       self.training_data, episodes=episodes)

    def generate_response(self, query):
        if "train" in query.lower():
            self.train_dqn_agent()
            logging.info("DQN agent training completed.")
            return "DQN agent training completed."

        elif "export" in query.lower():
            input_example = torch.randn(1, 128)
            self.dqn_agent.export_to_onnx(input_example.size())
            logging.info("DQN agent exported to ONNX.")
            return "DQN agent exported to ONNX."

        elif "generate text" in query.lower():
            generated_text = self.generate_text(query, max_length=100)
            logging.info(f"Generated text from your LLM: {generated_text}")
            return f"Generated text from your LLM: {generated_text}"

    def generate_text(self, query, max_length=1024):
        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.mistral_model.generate(input_ids,
                                                 max_length=max_length,
                                                 num_return_sequences=1,
                                                 no_repeat_ngram_size=2,
                                                 attention_mask=attention_mask)

        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True)
        return generated_text


def main():
    dqnmistral_llm = DQNMistralLLM()

    print("BATMAN_AI CLI INTERFACE")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            logging.info("Exiting BATMAN_AI CLI INTERFACE.")
            break

        self = dqnmistral_llm
        query = f"{user_input}"
        generated_text = self.generate_text(query)
        print(f"{generated_text}")
        logging.info(f"User query: {query}, Generated text: {generated_text}")


if __name__ == "__main__":
    main()
