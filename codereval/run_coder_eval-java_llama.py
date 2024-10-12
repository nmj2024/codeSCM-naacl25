# %% [markdown]
# Imports

# %%
import pandas as pd
import general_adapter
from vllm import LLM, SamplingParams
import os
import re

# %% [markdown]
# Args

# %%
class Args():
    file_path_base = 'CoderEval4Java.json'
    file_path_label = 'CEJavaHumanLabel.jsonl'

    # prompt stuff
    prompt_num = 10

    # model stuff
    model = 'meta-llama-3-8b'
    model_path = '../../Meta-Llama-3-8B/'
    system_prompt = 'Please provide a self-contained Java script that solves the following problem in a markdown code block:'

args=Args()

# %% [markdown]
# Load Dataset and Print Columns

# %%


with open(args.file_path_base, 'r') as file:
    df_base = pd.json_normalize(pd.read_json(file)['RECORDS'])

df_label = pd.read_json(args.file_path_label, lines=True)

print(df_base.columns)
print(df_label.columns)

# %% [markdown]
# Filter Dataframes To Only Include All Self-Contained Prompts

# %%
# truncate df_base to only include prompts with level as "self_contained"
df_base_filtered = df_base[df_base['level'] == 'self_contained']
# reset index
df_base_filtered.reset_index(drop=True, inplace=True)
print(df_base_filtered.shape)
# saved all the question_ids in df_base_filtered to a list
self_contained_ids = df_base_filtered['_id'].tolist()

# filter df_label to only include rows with question_id in question_ids
df_label_filtered = df_label[df_label['question_id'].isin(self_contained_ids)]
# reset index
df_label_filtered.reset_index(drop=True, inplace=True)
print(df_label_filtered.shape)

# %% [markdown]
# Print Content of Selected Prompt (DEBUG)

# %%
# for col in df_base.columns:
#     print(f'{col}: {df_base[col][args.prompt_num]}')

for col in df_label_filtered.columns:
    print(f'{col}: {df_label_filtered[col][args.prompt_num]}')

# %% [markdown]
# Printing Delta Components

# %%
function_header = df_label_filtered['signature'][args.prompt_num]
docstring = df_label_filtered['docstring'][args.prompt_num]
print('Prompt: ', args.prompt_num)
print('-'*50)
print('FUNCTION HEADER')
print(function_header)
print('-'*50)
print('DOCSTRING')
print(docstring)

# %% [markdown]
# Make Deltas (Combinations)

# %%
pattern = r"\b(public|private|protected)?\s*(static)?\s*([\w\[\]<>, ]+)\s+(\w+)\s*\(([^)]*)\)\s*(throws\s+[\w\., ]+)?\s*{"

def extract_entry_point(function_header):
    match = re.search(pattern, function_header)
    if match:
        return match.group(4)

def normalized_function_header(function_header):
    entry_point = extract_entry_point(function_header)
    if entry_point:
        return re.sub(entry_point, "func", function_header)
    return function_header

def del_underscore_and_caps(entry_point):
    return entry_point.capitalize()

# %%
def create_deltas(function_header, docstring):
    deltas_dict = {}

    # Pre-Transfromation Deltas
    deltas_dict['delta_1'] = f'{function_header}\n"""\n{docstring}\n"""\n'
    deltas_dict['delta_2'] = f'{docstring}\nCreate a function named {extract_entry_point(function_header)}'
    deltas_dict['delta_3'] = f'{normalized_function_header(function_header)}\n"""\n{docstring}\n"""\n'
    deltas_dict['delta_4'] = f'{docstring}\n{function_header}'

    # Post-Transformation Deltas
    docstring_transform = docstring.title() # capitalize the first letter of each word in the docstring
    deadcode_transform = f'{function_header}\n\tif (False) {{\n\t\tint[] x = new int[42];\n\t\tfor (int i = 0; i < 42; i++) {{\n\t\t\tx[i] = i;\n\t\t}}\n\t}}'
    entry_point_transform = del_underscore_and_caps(extract_entry_point(function_header)) # transform the function name
    entry_point_function_header_transform = function_header.replace(extract_entry_point(function_header), entry_point_transform) # replace the function name with a new name
    entry_point_docstring_transform = docstring.replace(extract_entry_point(function_header), entry_point_transform) # replace the function name with a new name

    deltas_dict['delta_5'] = f'{function_header}\n"""\n{docstring_transform}\n"""\n'
    deltas_dict['delta_6'] = f'{deadcode_transform}\n"""\n{docstring}\n"""\n'
    deltas_dict['delta_7'] = f'{entry_point_function_header_transform}\n"""\n{entry_point_docstring_transform}\n"""\n'

    return deltas_dict


# %% [markdown]
# LLM Inference Helper Functions

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sampling_params = SamplingParams(temperature=0.001, top_p=0.95, max_tokens=2048)
llm = LLM(model=args.model_path, enforce_eager=True)

def prepend_system_prompt(deltas):
    return [f'{args.system_prompt}\n{delta}' for delta in deltas]

def generate_llama_output(deltas):
    deltas = prepend_system_prompt(deltas)
    responses = llm.generate(deltas, sampling_params)
    answer  = [response.outputs[0].text for response in responses]
    return answer

# %% [markdown]
# Generate LLM Output

# %%
df_results = pd.DataFrame(columns=['prompt', 'delta', 'code', 'llm_output'])

for prompt_index in range(df_label_filtered.shape[0]):
    print(f'Prompt: {prompt_index}')
    function_header = df_label_filtered['signature'][prompt_index]
    docstring = df_label_filtered['docstring'][prompt_index]
    deltas_dict = create_deltas(function_header, docstring)
    llm_output = generate_llama_output(list(deltas_dict.values()))
    for delta, llm_output in zip(deltas_dict.keys(), llm_output):
        code = general_adapter.extract_java_code(llm_output)
        row = pd.DataFrame([{'prompt': prompt_index, 'delta': delta, 'code': '', 'llm_output': llm_output}])
        df_results = pd.concat([df_results, row], ignore_index=True)

# %% [markdown]
# Save Results to JSON

# %%
# Save results to JSONL file
df_results.to_json(f"{args.model}_codereval-java.jsonl", orient='records', lines=True)


