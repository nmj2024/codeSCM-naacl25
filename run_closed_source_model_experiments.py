import pandas as pd
import random
import llm
import argparse
import os
import shutil
import json
import pickle
from tqdm import tqdm
from models import get_hf_model, get_hf_pipeline
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash, write_jsonl
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

import warnings
warnings.filterwarnings("ignore")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='hf_codellama_13B',help="LLM model name")
parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
parser.add_argument("-p", "--num_prompts", default=-1, help="Number of prompts to test or list of prompt numbers")
parser.add_argument("-n", "--num_runs", default=1, help="Number of runs for prompt")
parser.add_argument("-g", "--delta_grouping", default=None, help="Grouping for generating delta: permutations or combinations")
parser.add_argument("-e", "--evaluation", default='evalplus', help="Evaluate using evalplus or runtime")

parser.add_argument("-exp", "--experiment", default='humaneval_codellama_13B')

parser.add_argument("-t", "--temperature", type=float, default=0.01)
parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--greedy_decode", type=bool, default=True)
parser.add_argument("--decoding_style", type=str, default='sampling')

parser.add_argument("--save_embds", default=True, type=bool)
parser.add_argument("--save_modal_components", default=False, type=bool)
parser.add_argument("--modal_transformations", default=True, type=bool)

args = parser.parse_args()

# Function to parse the num_prompts argument
def parse_num_prompts(arg):
    if arg.startswith('[') and arg.endswith(']'):
        return json.loads(arg)
    else:
        return int(arg)


# Check if the folder 'generated_code_files' exists and delete if it does
print('Deleting generated_code_files folder...')
if os.path.exists('generated_code_files'):
    shutil.rmtree('generated_code_files')

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

# Determine prompt numbers to test
if args.num_prompts == -1:
    prompt_numbers = list(range(len(problems)))
elif isinstance(args.num_prompts, list):
    # Use the provided list of prompt numbers
    prompt_numbers = args.num_prompts
else:
    # Pick random prompt numbers to test
    prompt_numbers = []
    while len(prompt_numbers) < args.num_prompts:
        prompt_number = random.randint(0, len(df) - 1)
        if prompt_number not in prompt_numbers:
            prompt_numbers.append(prompt_number)

#print('Prompt numbers to test:', prompt_numbers)


if 'hf' in args.model:
    if 'llama' in args.model and not args.save_embds:
        model, tokenizer, generation_config = get_hf_pipeline(args.model,
                                        args.temperature,
                                        args.max_len,
                                        args.greedy_decode,
                                        args.decoding_style)
    else:
        model, tokenizer, generation_config = get_hf_model(args.model,
                                        args.temperature,
                                        args.max_len,
                                        args.greedy_decode,
                                        args.decoding_style)        

# Call the function from llm.py with the necessary arguments
print("Running llm tests...")
final_results = []
embeds = {}

for prompt_number in tqdm(prompt_numbers, desc="Prompts completed"):
    try:
        if args.evaluation == 'runtime':
            llm.run_llm_tests(args.model, args.dataset, prompt_number, args.num_runs, args.delta_grouping, df)
        else:
            if args.save_embds:
                embeds[prompt_number] = llm.gen_hf_model_embeds(model, tokenizer, args.dataset, prompt_number,args.delta_grouping, df)
            else:
                final_results+=llm.gen_hf_model_output(model, tokenizer, generation_config,
                                                    args.dataset, prompt_number, args.num_runs, 
                                                    args.delta_grouping, df, args.max_len, args.save_modal_components, 
                                                    args.model, args.modal_transformations)
    except:
        print("Error in Prompt: ", prompt_number)
        #assert False
        #write_jsonl(f'{args.experiment}_temp{prompt_number}.jsonl', final_results)
        pass

if args.save_embds:
    pickle.dump(embeds, open(f'{args.experiment}_embeds', 'wb'))
else:
    result_file = f'{args.experiment}_generated_code.jsonl'
    write_jsonl(result_file, final_results)