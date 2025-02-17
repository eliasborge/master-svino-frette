from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent

model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q6_K_L"


##### USED FOR VERYFIYING THE SYSTEM #####
# example_agent = ExampleAgent(model)

# output = example_agent.__call__()
# print(output)

# Replace with actual messages from preprocessing script
messages = ["Jews are great, but muslims are the worst"]


for message in messages:
    print(f"Message: {message}")

    #Emotion analysis
    # emotion_agent = EmotionAgent(model)
    # emotions = emotion_agent.__call__(message)
    # print(emotions)

    #Target group analysis
    target_group_agent = TargetGroupAgent(model)
    target_group = target_group_agent.__call__(message)
    print(target_group)

