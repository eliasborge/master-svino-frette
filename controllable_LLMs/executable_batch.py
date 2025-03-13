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


### Due to the size of the topic threads, they haev been split into chunks ###

grouped_messages = grouped_df

###TESTING###
grouped_messages = grouped_messages.sample(n=3)
###TESTING###

batch_agent = BatchAgent(model)

# Prepare DataFrame to store results
collected_data = pd.DataFrame(columns=['document_id', 'extremism_labels'])

for index, row in grouped_messages.iterrows():
    content_list = row['content']
    content = "\n\n".join(content_list)
    results = []
    for i in content:
        print(i)
        result = batch_agent.__call__(i)
        results.append(result)

    print(results)



