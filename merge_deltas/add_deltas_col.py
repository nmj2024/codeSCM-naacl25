import pandas as pd

df = pd.read_json('OpenAI_4T_humaneval_chunk_all.jsonl', lines=True)

# insert new column after first column called delta and set all values to 0
df.insert(1, 'delta', None)

# iterate over rows and set delta value to 1 if the row is the first row
for i in range(len(df)):
    df.at[i, 'delta'] = (i%2)+1

# delete every row with a delta value of 2
# df = df[df['delta'] < 4]

# increase the delta value of all rows by 1
df['delta'] = df['delta'] + 3

# save the new dataframe to a new json file
df.to_json('humaneval_new_baseline.jsonl', orient='records', lines=True)