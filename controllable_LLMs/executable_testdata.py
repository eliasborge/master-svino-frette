from controllable_LLMs.agents.call_to_action_agent import CallToActionAgent
from controllable_LLMs.agents.framing_agent import FramingAgent
from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.example_agent import ExampleAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.validation_agent import ValidationAgent
from .utils .rekey_dictionary import rekey_dict
from json import loads

import pandas as pd

model = "mistral-small"

df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")


### Due to the size of the topic threads, they haev been split into chunks ###

topic_data = grouped_df



collected_data = pd.DataFrame(columns=['document_id','num_posts_in_same_topic','topic_length','topic_violence_label','violence_label','intent_label','call_to_action','flagged_issues'])

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
validation_agent = ValidationAgent(model)
call_to_action_agent = CallToActionAgent(model)


for index,row in topic_data.iterrows():


    content_with_ids = df
    content_list = row['content']
    content = "\n\n".join(content_list)
    num_posts = row['num_posts']
    topic_length = row['content_length']
    list_of_ids:list = row['id'].split(", ")

    print(list_of_ids)

    ### AVOID DOUBLE ANALYSIS IF ONLY ONE POST IN TOPIC ###
    topicWasAnalysed = False
    if(num_posts > 1):
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

        topicWasAnalysed = True



    print(" ------ ENTER THE THREAD ------")
    for index,post in df[df['id'].isin(list_of_ids)].iterrows():
    
        print(" ------ NEW POST ------")
        specific_post_content = post['content']
        specific_post_otherness = otherness_agent.__call__(specific_post_content)
        print(specific_post_otherness)

        specific_post_framing = framing_agent.__call__(specific_post_content)
        print(specific_post_framing)

        specific_post_intent = intent_agent.__call__(specific_post_content, specific_post_otherness['targetGroup'], specific_post_framing)
        print(specific_post_intent)

        specific_post_call_to_action = call_to_action_agent.__call__(specific_post_content, otherness['targetGroup'], framing)
        print(specific_post_call_to_action)

        specific_post_validation = validation_agent.__call__(specific_post_content, otherness_boolean = specific_post_otherness['othernessBoolean'], target_group = specific_post_otherness['targetGroup'], framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent, call_to_action=specific_post_call_to_action)
        print(specific_post_validation)

        if(topicWasAnalysed):
            new_row = {'document_id': post['id'], 'num_posts_in_same_topic': num_posts, 
            'topic_length': topic_length, 'topic_violence_label': validation['validated_label'], 
            'violence_label': specific_post_validation['validated_label'], 'intent_label': intent, 
            'call_to_action': call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}
        else:
            new_row = {'document_id': post['id'], 'num_posts_in_same_topic': num_posts, 
            'topic_length': topic_length, 'topic_violence_label': None, 
            'violence_label': specific_post_validation['validated_label'], 'intent_label': intent, 
            'call_to_action': call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}

        # Convert new_row to a DataFrame and concatenate with the existing DataFrame
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

        
### COLLECTION OF DATA ###

    collected_data.to_csv("data/collected_testdata.csv",index=False)

