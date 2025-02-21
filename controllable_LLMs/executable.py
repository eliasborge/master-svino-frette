from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.aggression_agent import AggressionAgent

model = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q6_K_L"


##### USED FOR VERYFIYING THE SYSTEM #####
example_agent = ExampleAgent(model)

output = example_agent.__call__()
print(output)

# Replace with actual messages from preprocessing script
messages = ["That's hilarious.  The two fags that replied before me need to wash the sand outta their vaginas", "part of secrete globalist brainwashing plot to purify the west of slav slave race, pure mix new master race"]


for message in messages:
    print(f"Message: {message}")

    #Emotion analysis
    # emotion_agent = EmotionAgent(model)
    # emotions = emotion_agent.__call__(message)
    # print(emotions)

    #Target group analysis
    # target_group_agent = TargetGroupAgent(model)
    # target_group = target_group_agent.__call__(message)
    # print(target_group)

    #Otherness analysis
    otherness_agent = OthernessAgent(model)
    otherness = otherness_agent.__call__(message)
    print(otherness)
   
    if(otherness['othernessBoolean'] == "True"):
        aggression_agent = AggressionAgent(model)
        aggression = aggression_agent.__call__(message, otherness['targetGroup'])
        print(aggression)
