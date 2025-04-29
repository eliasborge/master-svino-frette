
from .agents.call_to_action_agent import CallToActionAgent
from .agents.context_agent import ContextAgent
from .agents.framing_agent import FramingAgent
from .agents.otherness_agent import OthernessAgent
from .agents.intent_agent import IntentAgent
from .agents.message_validation_agent import MessageValidationAgent

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

df = pd.read_csv("data/testdata/processed_VideoCommentsThreatCorpus.csv")
grouped_df = pd.read_csv("data/testdata/grouped_processed_VideoCommentsThreatCorpus.csv")


### Due to the size of the topic threads, they haev been split into chunks ###

grouped_messages = grouped_df.head(5)



collected_data = pd.DataFrame(columns=['document_id','num_posts_in_conversation','conversation_length','violence_label','intent_label','call_to_action','flagged_issues'])
efficiency_data = pd.DataFrame(columns=[
    'row', 'row_duration_sec',
    'memory_used_MB', 'peak_memory_MB',
    'cpu_user_time_sec', 'cpu_system_time_sec', 'cpu_total_time_sec', 'total_latency_sec'
])

otherness_agent = OthernessAgent(model)
framing_agent = FramingAgent(model)
intent_agent = IntentAgent(model)
message_validation_agent = MessageValidationAgent(model)
call_to_action_agent = CallToActionAgent(model)
context_agent = ContextAgent(model)


mode = "neighbor"
for index,row in grouped_messages.iterrows():
    row_start_time = time.time()
    cpu_start = process.cpu_times()
    row_total_latency = 0
    print(f"Processing row {index + 1} of {len(grouped_messages)}...")

    content_with_ids = df
    raw_content = row['content']
    content_list = raw_content.split("###---###")
    content = "\nNew message:\n".join(content_list)
    num_posts_in_conversation = row['num_posts']
    conversation_length = row['content_length']
    list_of_ids:list = row['id'].split(", ")

    # print(list_of_ids)


    for index, post in df[df['id'].isin(list_of_ids)].iterrows():
        specific_post_content = post['content']
        post_id = post['id']

        # Ensure post ID exists in list_of_ids before using index lookup
        if post_id not in list_of_ids:
            print(f"Warning: Post ID {post_id} not found in list_of_ids")
            continue

        neighbors_window = 1  # Number of neighboring posts before and after

        post_index = list_of_ids.index(post_id)

        # Get context messages before the current post
        context_before = "\n".join(
            df[df['id'] == list_of_ids[j]]['content'].values[0]
            if not df[df['id'] == list_of_ids[j]].empty else "[MISSING]"
            for j in range(max(0, post_index - neighbors_window), post_index)
        )

        # Get context messages after the current post
        context_after = "\n".join(
            df[df['id'] == list_of_ids[j]]['content'].values[0]
            if not df[df['id'] == list_of_ids[j]].empty else "[MISSING]"
            for j in range(post_index + 1, min(len(list_of_ids), post_index + 1 + neighbors_window))
        )

        content = f"History before\n{context_before}\nTHIS IS THE MESSAGE YOU SHOULD CLASSIFY\n{specific_post_content}\nHistory After\n{context_after}"

        start_agent_time = time.time()
        specific_post_otherness = otherness_agent.__call__(specific_post_content, context = content,  mode=mode)
        latency = time.time() - start_agent_time
        row_total_latency += latency

        start_agent_time = time.time()
        specific_post_framing = framing_agent.__call__(specific_post_content,context=content, mode=mode)
        latency = time.time() - start_agent_time
        row_total_latency += latency

        start_agent_time = time.time()
        specific_post_intent = intent_agent.__call__(specific_post_content, specific_post_otherness['targetGroup'], specific_post_framing, context=content,mode=mode)
        latency = time.time() - start_agent_time
        row_total_latency += latency

        start_agent_time = time.time()
        specific_post_call_to_action = call_to_action_agent.__call__(specific_post_content, specific_post_otherness['targetGroup'], specific_post_framing, context=content, mode=mode)
        latency = time.time() - start_agent_time
        row_total_latency += latency

        start_agent_time = time.time()
        specific_post_validation = message_validation_agent.__call__(specific_post_content, otherness_boolean = specific_post_otherness['othernessBoolean'], target_group = specific_post_otherness['targetGroup'], framing_style = specific_post_framing['framingStyle'], framing_tool = specific_post_framing['framingTool'], intent_of_violence=specific_post_intent, call_to_action=specific_post_call_to_action, context=content, mode=mode)
        latency = time.time() - start_agent_time
        row_total_latency += latency

        new_row = {'document_id': post['id'], 'num_posts_in_conversation': num_posts_in_conversation, 
        'conversation_length': conversation_length,  
        'violence_label': specific_post_validation['validated_label'], 'intent_label': specific_post_intent, 
        'call_to_action': specific_post_call_to_action, 'flagged_issues': specific_post_validation['flagged_issues']}

        # Convert new_row to a DataFrame and concatenate with the existing DataFrame
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

    efficiency_data_row = pd.concat([
        efficiency_data,
        pd.DataFrame([{
            'row': index + 1,
            'row_duration_sec': row_duration,
            'memory_used_MB': mem_used,
            'peak_memory_MB': peak_mem,
            'cpu_user_time_sec': cpu_user,
            'cpu_system_time_sec': cpu_system,
            'cpu_total_time_sec': cpu_total,
            'total_latency_sec': row_total_latency,
        }])
    ], ignore_index=True)

        
### COLLECTION OF DATA ###

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
collected_data.to_csv(f"data/testdata/test_results_from_idun/neighbors/neighbor_{model}_{timestamp}.csv", index=False)
efficiency_data.to_csv(f"data/testdata/test_results_from_idun/neighbors/neighbor_efficiency_{model}_{timestamp}.csv", index=False)

