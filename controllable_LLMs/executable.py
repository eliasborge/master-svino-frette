from controllable_LLMs.agents.call_to_action_agent import CallToActionAgent
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

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
validation_agent = ValidationAgent(model)
call_to_action_agent = CallToActionAgent(model)

for index,row in data_random_3.iterrows():
    
    content_list = eval(row['content_list'])
    content = "".join(content_list)
    topic = row['stormfront_topic']
    print("------------------------------")
    print("topic: \n", topic)
    
    ### CHECKING FOR SIGNS OF 'OTHERNESS' ###
    otherness = otherness_agent.__call__(content)
    print(otherness)

    ### CHECKING FOR HIDDEN MEANINGS ###
    framing = framing_agent.__call__(content)
    print(framing)

    ### CHECKING FOR INTENT OF VIOLENCE ###
    intent = intent_agent.__call__(content, otherness['targetGroup'], framing)
    print(intent)

    ### CHECKING FOR CALL TO ACTION ###
    call_to_action = call_to_action_agent.__call__(content, otherness['targetGroup'], framing)
    print(call_to_action)

    validation = validation_agent.__call__(content, otherness_boolean = otherness['othernessBoolean'], target_group = otherness['targetGroup'], framing_style = framing['framingStyle'], framing_tool = framing['framingTool'], intent_of_violence=intent, call_to_action=call_to_action)
    print(validation)

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







##### OLD CODE ######
    #Emotion analysis
    #emotion_agent = EmotionAgent(model)
    #emotions = emotion_agent.__call__(message)
    #print(emotions)

    #Target group analysis
    # target_group_agent = TargetGroupAgent(model)
    # target_group = target_group_agent.__call__(message)
    # print(target_group)