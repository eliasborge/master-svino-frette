from controllable_LLMs.agents.call_to_action_agent import CallToActionAgent
from controllable_LLMs.agents.framing_agent import FramingAgent
from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.message_validation_agent import MessageValidationAgent
from .utils.rekey_dictionary import rekey_dict
from json import loads

import pandas as pd

model = "mistral-small"

df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")


### Due to the size of the topic threads, they haev been split into chunks ###

grouped_messages = grouped_df

###TESTING###
grouped_messages = grouped_messages.sample(n=3)
###TESTING###
### Due to the size of the topic threads, they haev been split into chunks ###

collected_data = pd.DataFrame(columns=['document_id','num_posts_in_conversation','conversation_length','violence_label','intent_label','call_to_action','flagged_issues'])

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
message_validation_agent = MessageValidationAgent(model)
call_to_action_agent = CallToActionAgent(model)

context ="" ## BECAUSE WE DO NOT TEST FOR CONTEXT HERE

for index,row in grouped_messages.iterrows():

    content_with_ids = df
    raw_content = row['content']
    content_list = raw_content.split("###---###")
    content = "\nNew message:\n".join(content_list)
    num_posts_in_conversation = row['num_posts']
    conversation_length = row['content_length']
    list_of_ids:list = row['id'].split(", ")

    print(list_of_ids)

    for index,post in df[df['id'].isin(list_of_ids)].iterrows():
        specific_post_content = post['content']

        specific_post_otherness = otherness_agent.__call__(specific_post_content, context)
        print(specific_post_otherness)

        specific_post_framing = framing_agent.__call__(specific_post_content, context)
        print(specific_post_framing)

        specific_post_intent = intent_agent.__call__(specific_post_content, specific_post_otherness['targetGroup'], specific_post_framing, context)
        print(specific_post_intent)

        specific_post_call_to_action = call_to_action_agent.__call__(specific_post_content, specific_post_otherness['targetGroup'], specific_post_framing, context)
        print(specific_post_call_to_action)

        ##TODO DO SOMETHING WITH THIS
        specific_post_validation = message_validation_agent.__call__(specific_post_content, otherness_boolean = specific_post_otherness['othernessBoolean'], target_group = specific_post_otherness['targetGroup'], framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent, call_to_action=specific_post_call_to_action, context=context)
        print(specific_post_validation)

        new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
        'conversation_length': conversation_length,  
        'violence_label': specific_post_validation['validated_label'], 'intent_label': specific_post_intent, 
        'call_to_action': specific_post_call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}

        # Convert new_row to a DataFrame and concatenate with the existing DataFrame
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

        
### COLLECTION OF DATA ###

    collected_data.to_csv("data/collected_with_agents_wizardllm.csv",index=False)

