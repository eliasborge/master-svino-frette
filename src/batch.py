
from .agents.batch_agent import BatchAgent
from datetime import datetime


import pandas as pd

# model = "mistral"
# model = "mistral-nemo"
model = "mistral-small"

grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")
grouped_messages = grouped_df
grouped_messages = grouped_df.head(2)


batch_agent = BatchAgent(model)

# Prepare DataFrame to store results
collected_data = pd.DataFrame(columns=['video_num','document_id', 'violence_label'])
print("starting batch processing")

for index, row in grouped_messages.iterrows():
    content = row['content']
    list_of_ids:list = row['id'].split(", ")

    content_list = content.split("###---###")   
    results = []
    for i in content_list:
        # print(i)
        result = batch_agent.__call__(i)
        print(result)
        results.append(result)

    for i, flag in enumerate(results):
        new_row = {'video_num': list_of_ids[i].split('_')[0],'document_id':list_of_ids[i], 'violence_label': flag, 'flagged_issues': flag['flagged_issues']}
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

    
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
collected_data.to_csv(f"data/testdata/test_results_from_idun/batch/batch_{model}_{timestamp}.csv", index=False)
print("Batch processing completed and results saved.")

