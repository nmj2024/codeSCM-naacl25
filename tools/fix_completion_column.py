import pandas as pd
from adapters import general_adapter

file = 'OpenAI_4T_mbpp_2runs.jsonl'

# import file as a DataFrame
df = pd.read_json(file, lines=True)

# clear all values in the 'completion' column
df['completion'] = None

# go through each row in the DataFrame
for i in range(len(df)):
    all_code = df.loc[i, 'all_code']
    completion = general_adapter.extract_python_code(all_code)
    df.loc[i, 'completion'] = completion

# save the DataFrame as a new file
df.to_json('OpenAI_4T_mbpp_2runs_FIXED.jsonl', orient='records', lines=True)