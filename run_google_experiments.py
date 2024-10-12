import argparse
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
import pandas as pd
from tqdm import tqdm
from adapters import mbpp_adapter, humaneval_adapter
import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='Google_GemP',help="LLM model name")
parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt")
parser.add_argument("-c", "--chunk", default='all', help="Chunk of prompts to run")
parser.add_argument('-t', '--type', default='base', choices=['base', 'transform'], help='Type of deltas to generate')
parser.add_argument("-k", "--key", help="API key for Google")
args = parser.parse_args()

def generate_google_output(deltas):
    client = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=2048, temperature=0, google_api_key=args.key)
    batch_responses = client.batch(deltas)
    return batch_responses

def extract_python_code(llm_output):
    """
    Extract Python code from a LLM output block.

    :param llm_output: A string representing LLM output containing a Python code block.
    :return: Extracted Python code as a string.
    """
    code_block = []
    in_code_block = False
    for line in llm_output.split('\n'):
        if (line.strip() == '```python' or line.strip() == '```') and not in_code_block:
            in_code_block = True
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_block.append(line)
    return '\n'.join(code_block)

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

if args.chunk == 'all':
     chunk_start = 0
     chunk_end = len(df)-1
else:
     chunk_start = int(args.chunk.split(',')[0])
     chunk_end = int(args.chunk.split(',')[1])

# Run experiments
df_results = pd.DataFrame(columns=['task_id', 'run', 'completion', 'all_code'])

chunk = list(range(chunk_start, chunk_end+1))

print(f"Running prompts {chunk_start} to {chunk_end} for {args.dataset} dataset")

failed_prompts = []

for prompt_index in tqdm(chunk, desc="Prompts completed"):
     # Get task id
     task_id = df.iloc[prompt_index]['task_id']

     # Generate deltas
     if args.dataset == 'mbpp':
          if args.type == 'base':
               deltas = mbpp_adapter.generate_deltas(df, prompt_index, None, False, False)
          elif args.type == 'transform':
               deltas = mbpp_adapter.generate_deltas(df, prompt_index, None, False, True)
     elif args.dataset == 'humaneval':
          if args.type == 'base':
               deltas = humaneval_adapter.generate_deltas(df, prompt_index, None, False)
          elif args.type == 'transform':
               deltas = humaneval_adapter.generate_deltas(df, prompt_index, None, True)
     
     # Run model for each delta
     for run_index in range(int(args.num_runs)):
          print(f"Run {run_index + 1} of {args.num_runs} for Prompt {prompt_index}")
          try:
               all_code = generate_google_output(deltas)
               for response in all_code:
                    content = response.content
                    completion = extract_python_code(content)
                    row = pd.DataFrame([{'task_id': task_id, 'run': run_index, 'completion': completion, 'all_code': content}])
                    df_results = pd.concat([df_results, row], ignore_index=True)
          except:
               print(f"Run {run_index + 1} of {args.num_runs} for Prompt {prompt_index} failed")
               failed_prompts.append(prompt_index)
               row = pd.DataFrame([{'task_id': task_id, 'run': run_index, 'completion': '', 'all_code': ''}])
               for i in range(len(deltas)):
                    df_results = pd.concat([df_results, row], ignore_index=True)
               continue
     time.sleep(10)


# Save results to JSONL file
df_results.to_json(f"{args.model}_{args.dataset}_chunk {args.chunk}_{args.num_runs} runs.jsonl", orient='records', lines=True)

# Save failed prompts to JSON file
with open(f"failed_prompts_{args.model}_{args.dataset}_chunk {args.chunk}_{args.num_runs} runs.json", 'w') as f:
     json.dump(failed_prompts, f)