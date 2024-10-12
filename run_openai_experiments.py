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
parser.add_argument("-m", "--model", default='OpenAI_4T',help="LLM model name")
parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt")
parser.add_argument("-c", "--chunk", default='all', help="Chunk of prompts to run")
parser.add_argument('-t', '--type', default='base', choices=['base', 'transform'], help='Type of deltas to generate')
parser.add_argument("-k", "--key", help="API key for OpenAI")
args = parser.parse_args()

model_dict = {
    'OpenAI_3.5T': 'gpt-3.5-turbo',
    'OpenAI_4': 'gpt-4',
    'OpenAI_4T': 'gpt-4-0125-preview'
}

def generate_openai_output(delta):
    question = f"{delta}"
    client = OpenAI(api_key=args.key)
    response = client.chat.completions.create(
        model=model_dict[args.model],
        messages=[{'role': 'user', 'content': question}],
        max_tokens=2048,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return answer

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
df_results = pd.DataFrame(columns=['task_id', 'delta', 'completion', 'all_code'])

chunk = list(range(chunk_start, chunk_end+1))

print(f"Running prompts {chunk_start} to {chunk_end} for {args.dataset} dataset")

for prompt_index in tqdm(chunk, desc="Prompts completed"):
     # Get task id
     task_id = df.iloc[prompt_index]['task_id']

     # Generate deltas
     if args.dataset == 'mbpp':
          if args.type == 'base':
               deltas = mbpp_adapter.generate_deltas(df, prompt_index, all_deltas=True)
          elif args.type == 'transform':
               deltas = mbpp_adapter.generate_deltas(df, prompt_index, None, False, True)
     elif args.dataset == 'humaneval':
          if args.type == 'base':
               deltas = humaneval_adapter.generate_deltas(df, prompt_index, all_deltas=True)
          elif args.type == 'transform':
               deltas = humaneval_adapter.generate_deltas(df, prompt_index, None, True)
     
     # only keep the last 2 elements in the delta list (delta_10 and delta_11)
     deltas = deltas[-2:]

     # for delta in deltas:
     #      print('-'*50)
     #      print('Delta:')
     #      print(delta)

     # Run model for each delta
     for run_index in range(int(args.num_runs)):
          print(f"Run {run_index + 1} of {args.num_runs} for Prompt {prompt_index}")
          for delta_index, delta in enumerate(deltas):
               print(f"Generating Output for Delta {delta_index + 1} of {len(deltas)}")
               all_code = generate_openai_output(delta)
               completion = general_adapter.extract_python_code(all_code)
               row = pd.DataFrame([{'task_id': task_id, 'delta': int(delta_index+10), 'completion': completion, 'all_code': all_code}])
               df_results = pd.concat([df_results, row], ignore_index=True)


# Save results to JSONL file
df_results.to_json(f"{args.model}_{args.dataset}_delta_10_delta_11.jsonl", orient='records', lines=True)