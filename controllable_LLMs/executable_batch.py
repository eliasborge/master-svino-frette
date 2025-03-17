from json import loads
from controllable_LLMs.agents.call_to_action_agent import CallToActionAgent
from controllable_LLMs.agents.context_agent import ContextAgent
from controllable_LLMs.agents.framing_agent import FramingAgent
from controllable_LLMs.agents.target_group_agent import TargetGroupAgent
from .agents.batch_agent import BatchAgent
from .agents.emotion_agent import EmotionAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.message_validation_agent import MessageValidationAgent


import pandas as pd

model = "mistral-small"

grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")
grouped_messages = grouped_df

# Filter grouped_messages to only include the message with id 8_3356
grouped_messages = grouped_messages[grouped_messages['id'].str.contains('8_3356')]


batch_agent = BatchAgent(model)

# Prepare DataFrame to store results
collected_data = pd.DataFrame(columns=['video_num','document_id', 'agent_flags'])

for index, row in grouped_messages.iterrows():
    content = row['content']
    list_of_ids:list = row['id'].split(", ")

    content_list = content.split("###---###")
    print(content_list)
    results = []
    for i in content_list:
        print(i)
        result = batch_agent.__call__(i)
        results.append(result)

    for i, flag in enumerate(results):
        new_row = {'video_num': list_of_ids[i].split('_')[0],'document_id':list_of_ids[i], 'agent_flags': flag}
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

    
#collected_data.to_csv("data/collected_batch_agents.csv", index=False)

