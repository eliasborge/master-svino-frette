from controllable_LLMs.agents.framing_agent import FramingAgent
from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent

import pandas as pd

model = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q6_K_L"

data = pd.read_csv("data\kleinberg\grouped_stormfront_data.csv")


##### USED FOR VERYFIYING THE SYSTEM #####
example_agent = ExampleAgent(model)

output = example_agent.__call__()
print(output)

# Replace with actual messages from preprocessing script

data_random_3 = data.sample(n=3)

for index,row in data_random_3.iterrows():
    content = row['combined_content']
    topic = row['stormfront_topic']

    print(topic)
    print(len(topic))
    print(row['content_posts'])

    #Emotion analysis
    #emotion_agent = EmotionAgent(model)
    #emotions = emotion_agent.__call__(message)
    #print(emotions)

    #Target group analysis
    # target_group_agent = TargetGroupAgent(model)
    # target_group = target_group_agent.__call__(message)
    # print(target_group)

    #Otherness analysis
    # otherness_agent = OthernessAgent(model)
    # otherness = otherness_agent.__call__(message)
    # print(otherness)
   
    # if(otherness['othernessBoolean'] == "True"):
    
    #Framing agent
    framing_agent = FramingAgent(model)
    framing = framing_agent.__call__(content)
    print(framing)

    otherness_agent = OthernessAgent(model)
    otherness = otherness_agent.__call__(content)
    print(otherness)

    if( otherness['othernessBoolean'] == "True" or "False"):
        intent_agent = IntentAgent(model)
        intent = intent_agent.__call__(content, otherness['targetGroup'], framing)
        print(intent)
