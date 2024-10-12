import argparse
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
import pandas as pd
from tqdm import tqdm
from adapters import general_adapter, mbpp_adapter, humaneval_adapter
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
args = parser.parse_args()

# Get dataset problems and expected output
if args.dataset == 'mbpp':
     problems = get_mbpp_plus()
     dataset_hash = get_mbpp_plus_hash()
     expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
elif args.dataset == 'humaneval':
    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(problems, dataset_hash, [])

# Add expected output to problems
for problem in problems:
    expected_output_keys = [*expected_output[problem]]
    for key in expected_output_keys:
        problems[problem][key] = expected_output[problem][key]

# Create df from problems
df = pd.DataFrame.from_dict(problems, orient='index')
df = df.reset_index(drop=True)

# Run experiments
df_results = pd.DataFrame(columns=['task_id', 'delta_num', 'delta_val'])

for prompt_index in tqdm(range(len(df)), desc="Prompts completed"):
     task_id = df.iloc[prompt_index]['task_id']

     if args.dataset == 'mbpp':
        deltas = mbpp_adapter.generate_deltas(df, prompt_index, all_deltas=True)
     elif args.dataset == 'humaneval':
        deltas = humaneval_adapter.generate_deltas(df, prompt_index, all_deltas=True)

     for delta_index, delta in enumerate(deltas):
        row = pd.DataFrame([{'task_id': task_id, 'delta_num': delta_index, 'delta_val': delta}])
        df_results = pd.concat([df_results, row], ignore_index=True)

# # Save results to JSONL file
df_results.to_json(f"{args.dataset}_deltas_dump.jsonl", orient='records', lines=True)