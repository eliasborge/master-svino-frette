
from .agents.SinglePrompt_agent import SinglePromptAgent
from datetime import datetime
from pydantic import ValidationError
import pandas as pd
import time



model = "mistral"
# model = "mistral-nemo"
# model = "mistral-small"
# model = "gemma3:27b"
# model = "gemma3:12b"
#model = "qwen3:8b"

### Logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
start_time = time.time()
####

df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_messages = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")



singlePrompt_agent = SinglePromptAgent(model)

# Prepare DataFrame to store results
collected_data = pd.DataFrame(columns=['document_id', 'violence_label', 'row_duration_sec', 'flagged_issues'])


print("starting singlePrompt_agent processing")

for index, row in grouped_messages.iterrows():
    row_start_time = time.time()
    content = row['content']
    list_of_ids:list = row['id'].split(", ")
    print(f"Processing row {index + 1} of {len(grouped_messages)}...")  
    content_list = content.split("###---###")   
    results = []

    for index,post in df[df['id'].isin(list_of_ids)].iterrows():
        content = post['content']
        try:
            result = singlePrompt_agent.__call__(content)
            print(result)
        except ValidationError as e:
            print(f"Validation error for content: {content}, Error: {e}")
            result = {'violent_label': None, 'flagged_issues': [1]}
        results.append(result)

    row_duration_sec = time.time() - row_start_time

    for i, flag in enumerate(results):
        new_row = {'document_id':post['id'], 'violence_label': flag['violent_label'], 'flagged_issues': flag['flagged_issues'], 'row_duration_sec': row_duration_sec}
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)


    
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
collected_data.to_csv(f"data/testdata/test_results_from_idun/solo/solo{model}_{timestamp}.csv", index=False)
print("solo processing completed and results saved.")

