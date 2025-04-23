
from .agents.batch_agent import BatchAgent
from datetime import datetime
import pandas as pd
import tracemalloc
import time
import psutil


# model = "mistral"
# model = "mistral-nemo"
model = "mistral-small"

### Logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
start_time = time.time()
tracemalloc.start()
process = psutil.Process()
####

grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")
grouped_messages = grouped_df


batch_agent = BatchAgent(model)

# Prepare DataFrame to store results
collected_data = pd.DataFrame(columns=['video_num','document_id', 'violence_label'])
efficiency_data = pd.DataFrame(columns=[
    'row', 'row_duration_sec',
    'memory_used_MB', 'peak_memory_MB',
    'cpu_user_time_sec', 'cpu_system_time_sec', 'cpu_total_time_sec'
])


print("starting batch processing")

for index, row in grouped_messages.iterrows():
    row_start_time = time.time()
    cpu_start = process.cpu_times()
    content = row['content']
    list_of_ids:list = row['id'].split(", ")
    print(f"Processing row {index + 1} of {len(grouped_messages)}...")  
    content_list = content.split("###---###")   
    results = []
    for i in content_list:
        result = batch_agent.__call__(i)
        results.append(result)

    for i, flag in enumerate(results):
        new_row = {'video_num': list_of_ids[i].split('_')[0],'document_id':list_of_ids[i], 'violence_label': flag['violent_label'], 'flagged_issues': flag['flagged_issues']}
        new_row_df = pd.DataFrame([new_row])
        collected_data = pd.concat([collected_data, new_row_df], ignore_index=True)

    cpu_end = process.cpu_times()
    cpu_user = cpu_end.user - cpu_start.user
    cpu_system = cpu_end.system - cpu_start.system
    cpu_total = cpu_user + cpu_system

    row_duration = time.time() - row_start_time
    current, peak = tracemalloc.get_traced_memory()
    mem_used = current / 1e6
    peak_mem = peak / 1e6

    efficiency_data = pd.concat([
        efficiency_data, 
        pd.DataFrame([{
            'row': index + 1,
            'row_duration_sec': row_duration,
            'memory_used_MB': mem_used,
            'peak_memory_MB': peak_mem,
            'cpu_user_time_sec': cpu_user,
            'cpu_system_time_sec': cpu_system,
            'cpu_total_time_sec': cpu_total
        }])
    ], ignore_index=True)

    
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
collected_data.to_csv(f"data/testdata/test_results_from_idun/batch/batch_{model}_{timestamp}.csv", index=False)
efficiency_data.to_csv(f"data/testdata/test_results_from_idun/batch/batch_efficiency_{model}_{timestamp}.csv", index=False)
print("Batch processing completed and results saved.")

