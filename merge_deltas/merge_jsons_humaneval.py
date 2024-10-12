import pandas as pd

df1 = pd.read_json('humaneval_merged.jsonl', lines=True)
df2 = pd.read_json('delta_9_gpt_output.jsonl', lines=True)
# df3 = pd.read_json('humaneval/humaneval_trans_old.jsonl', lines=True)

task_ids = df1['task_id'].tolist()
task_ids = list(set(task_ids))
task_ids.sort()

# new df to store merged results
df = pd.DataFrame(columns=['task_id', 'delta', 'completion', 'all_code'])

for task in task_ids:
    # filter all the rows from each df to only include rows with task_id == task
    df1_filtered = df1[df1['task_id'] == task]
    df2_filtered = df2[df2['task_id'] == task]
    # df3_filtered = df3[df3['task_id'] == task]

    # reset index
    df1_filtered.reset_index(drop=True, inplace=True)
    df2_filtered.reset_index(drop=True, inplace=True)
    # df3_filtered.reset_index(drop=True, inplace=True)

    # add rows of df1_filtered to df
    for i in range(len(df1_filtered)):
        row = df1_filtered.iloc[i]
        delta = row['delta']
        completion = row['completion']
        all_code = row['all_code']
        row = pd.DataFrame([{'task_id': task, 'delta': delta, 'completion': completion, 'all_code': all_code}])
        df = pd.concat([df, row], ignore_index=True)

    # add rows of df2_filtered to df
    for i in range(len(df2_filtered)):
        row = df2_filtered.iloc[i]
        delta = row['delta']
        completion = row['completion']
        all_code = row['all_code']
        row = pd.DataFrame([{'task_id': task, 'delta': delta, 'completion': completion, 'all_code': all_code}])
        df = pd.concat([df, row], ignore_index=True)

    # # add rows of df3_filtered to df
    # for i in range(len(df3_filtered)):
    #     row = df3_filtered.iloc[i]
    #     delta = row['delta']
    #     completion = row['completion']
    #     all_code = row['all_code']
    #     row = pd.DataFrame([{'task_id': task, 'delta': delta, 'completion': completion, 'all_code': all_code}])
    #     df = pd.concat([df, row], ignore_index=True)

# save df to a json file
df.to_json('humaneval/humaneval_merged.jsonl', orient='records', lines=True)