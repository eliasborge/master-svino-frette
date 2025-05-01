from .agents.legacy.call_to_action_agent import CallToActionAgent
from .agents.context_agent import ContextAgent
from .agents.framing_agent import FramingAgent
from .agents.legacy.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.classification_agent import ClassificationAgent

from datetime import datetime

import pandas as pd

model = "mistral"
#model = "mistral-nemo"
# model = "mistral-small"


df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")


### Due to the size of the topic threads, they haev been split into chunks ###

grouped_messages = grouped_df


mode="context"

collected_data = pd.DataFrame(columns=['document_id','num_posts_in_conversation','conversation_length','violence_label','intent_label','call_to_action','flagged_issues'])

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
classification_agent = ClassificationAgent(model)
call_to_action_agent = CallToActionAgent(model)
context_agent = ContextAgent(model)
print("starting context processing")


for index,row in grouped_messages.iterrows():

    content_with_ids = df
    raw_content = row['content']
    content_list = raw_content.split("###---###")
    content = "\nNew message:\n".join(content_list)
    num_posts_in_conversation = row['num_posts']
    conversation_length = row['content_length']
    list_of_ids:list = row['id'].split(", ")

    print(f"Processing row {index + 1} of {len(grouped_messages)}...")

    # print(list_of_ids)

    context = context_agent.__call__(content)
    topicWasAnalysed = True

    for index,post in df[df['id'].isin(list_of_ids)].iterrows():
        specific_post_content = post['content']

        specific_post_framing = framing_agent.__call__(specific_post_content, context,mode=mode)
        specific_post_intent = intent_agent.__call__(specific_post_content, specific_post_framing, context=content,mode=mode)
        specific_post_call_to_action = specific_post_intent['call_to_action']
        specific_post_intent_of_violence = specific_post_intent['intent_of_violence']

        specific_post_classification = classification_agent.__call__(specific_post_content, framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent_of_violence, call_to_action=specific_post_call_to_action, context=context,mode=mode)
        print("done")

        if(topicWasAnalysed):
            new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
            'conversation_length': conversation_length, 
            'violence_label': specific_post_classification['label'], 'intent_label': specific_post_intent_of_violence, 
            'call_to_action': specific_post_call_to_action, 'flagged_issues': specific_post_classification['flagged_issues']}
        else:
            new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
            'conversation_length': conversation_length,  
            'violence_label': specific_post_classification['label'], 'intent_label': specific_post_intent_of_violence, 
            'call_to_action': specific_post_call_to_action, 'flagged_issues': specific_post_classification['flagged_issues']}

        # Convert new_row to a DataFrame and concatenate with the existing DataFrame
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

        
### COLLECTION OF DATA ###

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
print("Context processing completed and results saved.")
collected_data.to_csv(f"data/testdata/test_results_from_idun/context/context_analysis_{model}_{timestamp}.csv", index=False)

