import pandas as pd

df = pd.read_json('humaneval_merged.jsonl', lines=True)

# remove all rows where delta is 1, 6, 7, 8, 9
df = df[df['delta'] > 1]
df = df[df['delta'] < 6]

# save the new dataframe to a new json file
df.to_json('humaneval_merged_DE.jsonl', orient='records', lines=True)