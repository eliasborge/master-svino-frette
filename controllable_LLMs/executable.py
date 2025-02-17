from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent

model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K_L"


##### USED FOR VERYFIYING THE SYSTEM #####
# example_agent = ExampleAgent(model)

# output = example_agent.__call__()
# print(output)

# Replace with actual messages from preprocessing script
messages = ["I was not aware of how stupid they can be! I wish someone would just get rid of them. "]


for message in messages:
    print(f"Message: {message}")
    emotion_agent = EmotionAgent(model)
    emotions = emotion_agent.__call__(message)
    print(emotions)
