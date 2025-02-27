from controllable_LLMs.agents.framing_agent import FramingAgent
from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.validation_agent import ValidationAgent

import pandas as pd

model = "mistral-small"

data = pd.read_csv("data/grouped_data_from_stormfront/grouped_stormfront_data_2014.csv")

data_random_3 = data.sample(n=3)


for index,row in data_random_3.iterrows():
    
    content_list = eval(row['content_list'])
    content = "".join(content_list)
    topic = row['stormfront_topic']
    print("------------------------------")
    print("topic: \n", topic)
   

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
    

    otherness_agent = OthernessAgent(model)
    otherness = otherness_agent.__call__(content)
    print(otherness)

    #Framing agent
    framing_agent = FramingAgent(model)
    framing = framing_agent.__call__(content)
    print(framing)

    if( otherness['othernessBoolean'] == "True" or "False"):
        intent_agent = IntentAgent(model)
        intent = intent_agent.__call__(content, otherness['targetGroup'], framing)
        print(intent)

        validation_agent = ValidationAgent(model)
        #output = validation_agent.__call__(content, otherness['othernessBoolean'], otherness['targetGroup'], framing['framingStyle'], framing['framingTool'], intent['intent_of_violence'])
        validation = validation_agent.__call__(content, otherness_boolean = otherness['othernessBoolean'], target_group = otherness['targetGroup'], framing_style = framing['framingStyle'], framing_tool = framing['framingTool'], intent_of_violence=intent)
        print(validation),

    #     if(intent.lower() == "high intent" or intent.lower() == "high" or intent.lower() == "moderate intent" or intent.lower() == "moderate"):
    #         print(" ------ ENTER THE THREAD ------")
    #         for post in content_list:
    #             print(" ------ NEW POST ------")
    #             print("\n" +post)
    #             specific_post_otherness = otherness_agent.__call__(post)
    #             print(specific_post_otherness)

    #             specific_post_framing = framing_agent.__call__(post)
    #             print(specific_post_framing)

    #             specific_post_intent = intent_agent.__call__(post, specific_post_otherness['targetGroup'], specific_post_framing)
    #             print(specific_post_intent)
