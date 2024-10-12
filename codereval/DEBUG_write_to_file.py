import pandas as pd

# import hi.jsonl as df
df = pd.read_json('gpt-4-turbo_codereval-python.jsonl', lines=True)

# write the contents of "code" column for row 2 to a file test.py
with open('test.py', 'w') as f:
    f.write(df['code'][2])
