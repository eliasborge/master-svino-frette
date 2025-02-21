from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent

model = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q6_K_L"


##### USED FOR VERYFIYING THE SYSTEM #####
example_agent = ExampleAgent(model)

output = example_agent.__call__()
print(output)

# Replace with actual messages from preprocessing script
messages = ["There is a special place in hell for people who like cats! They smell weird and the look like dick licking cunts. Off with their heads!!!!!!!👿👿👿👿","That's hilarious.  The two fags that replied before me can go and off themselves. there are too many to begin with", "I don't hate Kamala Harris! She really did her best effort"]


for message in messages:
    print(f"Message: {message}")

    #Emotion analysis
    #emotion_agent = EmotionAgent(model)
    #emotions = emotion_agent.__call__(message)
    #print(emotions)

    #Target group analysis
    # target_group_agent = TargetGroupAgent(model)
    # target_group = target_group_agent.__call__(message)
    # print(target_group)

    #Otherness analysis
    otherness_agent = OthernessAgent(model)
    otherness = otherness_agent.__call__(message)
    print(otherness)
   
    if( otherness['othernessBoolean'] == "True" or "False"):
        intent_agent = IntentAgent(model)
        intent = intent_agent.__call__(message, otherness['targetGroup'])
        print(intent)
