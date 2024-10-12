import json
import re
from collections import defaultdict
import pandas as pd

# Function to extract Java code from the RESPONSE: portion of the LLM output
def extract_java_code(llm_output):
    """
    Extract Java code from a LLM output block.

    :param llm_output: A string representing LLM output containing a Java code block.
    :return: Extracted Java code as a string.
    """
    # Look for the Java code block within the RESPONSE section
    response_match = re.search(r'### Response:\n```Java\n(.*?)```', llm_output, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()
    return ''

# Read the JSONL file
file_path = 'CE_java_wzcoder.json'
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.extend(json.loads(line))

file_path_base = 'CoderEval4Java.json'
file_path_label = 'CEJavaHumanLabel.jsonl'

with open(file_path_base, 'r') as file:
    df_base = pd.json_normalize(pd.read_json(file)['RECORDS'])
df_label = pd.read_json(file_path_label, lines=True)

# truncate df_base to only include prompts with level as "self_contained"
df_base_filtered = df_base[df_base['level'] == 'self_contained']
# reset index
df_base_filtered.reset_index(drop=True, inplace=True)

# saved all the question_ids in df_base_filtered to a list
self_contained_ids = df_base_filtered['_id'].tolist()

prompt_to_id = {}
for i in range(len(self_contained_ids)):
    prompt_to_id[i] = self_contained_ids[i]

# Dictionary to hold extracted data based on delta number
delta_dict = defaultdict(list)

# Extract content and organize by delta number
for entry in data:
    if isinstance(entry, list) and len(entry) == 2:
        key, content = entry
        match = re.match(r'CE_java_\d+_(\d+)', key)
        # Extract the prompt number from match
        # strip away the 'CE_java_' prefix from key
        key = key.replace('CE_java_', '')
        # strip everything after _ in key
        key = key.split('_')[0]
        _id = prompt_to_id[int(key)]
        if match:
            delta_number = int(match.group(1))
            java_code = extract_java_code(content)
            delta_dict[delta_number].append({'_id': _id, 'generated_results': [java_code]})

# Save extracted content into separate JSONL files for each delta
for delta_number, contents in delta_dict.items():
    output_file = f'wizard_delta_{delta_number}.jsonl'
    with open(output_file, 'w') as f:
        for entry in contents:
            f.write(json.dumps(entry) + '\n')
