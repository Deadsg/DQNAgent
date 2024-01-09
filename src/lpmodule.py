import random

def lpmodule():
    pass

def simple_chatbot(user_input):
    responses = {
        "How are you?": "I'm good, thank you!",
        "What's your name?": "I'm a simple chatbot.",
        "Default": "I'm not sure how to respond to that."
    }

    return responses.get(user_input, responses["Default"])