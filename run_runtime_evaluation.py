import argparse
import pandas as pd
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
from evalplus.evaluate import get_groundtruth
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from adapters import mbpp_adapter, humaneval_adapter
from llm import run_test_cases_for_file
from tqdm import tqdm
import multiprocessing
import queue  # Used for communication between processes

# Function to run in a separate process
def run_in_process(queue, file_name, test_list):
    result, error = run_test_cases_for_file(file_name, test_list)
    queue.put((result, error))

# Wrapper function to run test cases with timeout
def run_test_cases_with_timeout(file_name, test_list, timeout=180):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_in_process, args=(q, file_name, test_list))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()  # Terminate the process
        p.join()  # Wait for process to terminate
        print(f"Timeout exceeded!")
        return "Fail", "Timeout Error (Exceeded 3 Minutes)"
    else:
        return q.get()  # Get the result from the queue

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="JSONL file")
    parser.add_argument("-d", "--dataset", default='humaneval', help="Dataset", choices=['humaneval', 'mbpp'])
    args = parser.parse_args()

    # Read JSONL file into df
    df_llm_response = pd.read_json(args.file, lines=True)

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

    # Create df from dataset
    df_dataset = pd.DataFrame.from_dict(problems, orient='index')
    df_dataset = df_dataset.reset_index(drop=True)
    df_dataset = df_dataset[['task_id', 'entry_point', 'plus_input', 'plus']]

    # Add test_list to df_dataset
    test_list_dict = {}
    for i in range(len(df_dataset)):
        task_id = df_dataset.iloc[i]['task_id']
        entry_point = df_dataset.iloc[i]['entry_point']
        plus_input = df_dataset.iloc[i]['plus_input']
        plus = df_dataset.iloc[i]['plus']
        try:
            if args.dataset == 'mbpp':
                test_list = mbpp_adapter.extract_mbpp_test_list(entry_point, plus_input, plus)
            elif args.dataset == 'humaneval':
                test_list = humaneval_adapter.extract_humaneval_test_list(entry_point, plus_input, plus)
        except:
            print(f"Error generating test list for task_id {task_id}")
            continue
        test_list_dict[task_id] = test_list

    test_list_df = pd.DataFrame(list(test_list_dict.items()), columns=['task_id', 'test_list'])
    df_dataset = pd.merge(df_dataset, test_list_df, on='task_id', how='left')

    # Run Test Cases
    # df_llm_response['delta'] = None
    df_llm_response['result'] = None
    df_llm_response['error'] = None
    file_name = 'test.py'
    for i in tqdm(range(len(df_llm_response)), desc="Deltas completed"):
        delta = df_llm_response.iloc[i]['delta']
        print(f"Running Delta: {delta}")
        with open(file_name, 'w') as file:
            file.write(df_llm_response.iloc[i]['completion'])
        # get test_list by checking against task_id in df_dataset
        task_id = df_llm_response.iloc[i]['task_id']
        test_list = df_dataset[df_dataset['task_id'] == task_id]['test_list'].values[0]
        result, error = run_test_cases_with_timeout(file_name, test_list)
        # add results to df_llm_response
        df_llm_response.at[i, 'delta'] = delta
        df_llm_response.at[i, 'result'] = result
        df_llm_response.at[i, 'error'] = error

    # Save df_llm_response to jsonl
    output_file = f"{args.file}_RUNTIME.jsonl"
    df_llm_response.to_json(output_file, orient='records', lines=True)

# This check ensures the following code only runs when executing the script, not when importing it
if __name__ == '__main__':
    main()