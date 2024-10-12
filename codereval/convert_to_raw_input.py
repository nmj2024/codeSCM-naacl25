import pandas as pd

file_path_base = 'CoderEval4Python.json'
file_path_label = 'CEPythonHumanLabel.jsonl'
file_path_llm = 'gpt-4-turbo_codereval-python.jsonl'

with open(file_path_base, 'r') as file:
    df_base = pd.json_normalize(pd.read_json(file)['RECORDS'])

df_label = pd.read_json(file_path_label, lines=True)
df_llm = pd.read_json(file_path_llm, lines=True)

# truncate df_base to only include prompts with level as "self_contained"
df_base_filtered = df_base[df_base['level'] == 'self_contained']
# reset index
df_base_filtered.reset_index(drop=True, inplace=True)

# saved all the question_ids in df_base_filtered to a list
self_contained_ids = df_base_filtered['_id'].tolist()
print(self_contained_ids)

# filter df_label to only include rows with question_id in question_ids
df_label_filtered = df_label[df_label['question_id'].isin(self_contained_ids)]
# reset index
df_label_filtered.reset_index(drop=True, inplace=True)

prompt_to_id = {}
for i in range(len(self_contained_ids)):
    prompt_to_id[i] = self_contained_ids[i]

# create new df
df = pd.DataFrame(columns=['_id', 'generate_results'])

# create a list of delta_{i} from delta_0 to delta_7
delta_list = []
for i in range(7):
    delta_list.append('delta_' + str(i+1))

# create a df for each delta
df_delta = {}
for delta in delta_list:
    df_delta[delta] = pd.DataFrame(columns=['_id', 'generate_results'])

for delta in delta_list:
    df_llm_filtered = df_llm[df_llm['delta'] == delta]
    df_llm_filtered.reset_index(drop=True, inplace=True)
    for i in range(len(df_llm_filtered)):
        row = df_llm_filtered.iloc[i]
        prompt = row['prompt']
        code = [row['code']]
        _id = prompt_to_id[prompt]
        row = pd.DataFrame([{'_id': _id, 'generate_results': code}])
        df_delta[delta] = pd.concat([df_delta[delta], row], ignore_index=True)

# save each df_delta to a json file in a new directory
for delta in delta_list:
    df_delta[delta].to_json('gpt4t_python' + delta + '.jsonl', orient='records', lines=True)