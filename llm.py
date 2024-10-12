import pandas as pd
import os
import importlib.util
import timeout_decorator
import models
from adapters import general_adapter, mbpp_adapter, humaneval_adapter

# Define a timeout duration for test cases
TEST_CASE_TIMEOUT = 30  # Example: 30 seconds

def run_single_test_case(module, test):
    """
    Run a single test case.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    exec(test, globals(), module.__dict__)

@timeout_decorator.timeout(TEST_CASE_TIMEOUT)
def run_test_case_with_timeout(module, test):
    """
    Execute a test case with a specified timeout.

    :param module: The Python module in which the test will be executed.
    :param test: A string representing the test case to be executed.
    """
    run_single_test_case(module, test)

def run_test_cases(module, test_list):
    """
    Run a list of test cases for a given module, handling timeouts and assertions.

    :param module: The Python module to test.
    :param test_list: A list of test case strings to be executed.
    :return: True if all tests pass, False otherwise.
    """
    for test in test_list:
        try:
            run_test_case_with_timeout(module, test)
        except AssertionError:
            return False
        except timeout_decorator.TimeoutError:
            print("Test case timed out. Moving to next test case.")
    return True

def run_test_cases_for_file(file_path, test_list):
    """
    Run test cases for a Python file.

    :param file_path: Path to the Python file to be tested.
    :param test_list: A list of test case strings to be executed.
    :return: A tuple with the test result ('Pass' or 'Fail') and an error type if applicable.
    """
    try:
        # Check if the file is empty
        if os.path.getsize(file_path) == 0 or not open(file_path).read().strip():
            return ('Fail', 'No Code Generated')

        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if run_test_cases(module, test_list):
            return ('Pass', None)
        else:
            # If the test cases run but fail, it's a semantic error
            return ('Fail', 'Semantic Error (Test Case)')
    except SyntaxError:
        return ('Fail', 'Syntax Error')
    except AssertionError:
        # If an assertion fails, it's considered a semantic error
        return ('Fail', 'Semantic Error (Assertion)')
    except timeout_decorator.TimeoutError:
        return ('Fail', 'Timeout')
    except MemoryError:
        return ('Fail', 'Resource Error')
    except ImportError:
        return ('Fail', 'Dependency Error')
    except EnvironmentError:
        return ('Fail', 'Environment Error')
    except Exception as e:
        return ('Fail', f'Runtime Error - {e.__class__.__name__}')


def gen_hf_model_output(model, tokenizer, generation_config, dataset, prompt_index, 
                        num_runs, delta_method, df, max_len, save_modal_components, model_name, modal_transformations):
    
    all_results = []

    if dataset == 'mbpp':
        deltas= mbpp_adapter.generate_deltas(df, prompt_index, delta_method, save_modal_components, modal_transformations)
    elif dataset == 'humaneval':
        deltas = humaneval_adapter.generate_deltas(df, prompt_index, delta_method, save_modal_components, modal_transformations)    
    
    if save_modal_components:
        return [{f'delta_{i}':j for i,j in enumerate(deltas)}]
    
    for run_index in range(num_runs):
        for i, delta in enumerate(deltas):
            if 'wizardcoder' in model_name:
                trucated_seq, raw_seq = models.generate_wizardcode_output(delta, model, tokenizer, generation_config, max_len)
            elif 'codellama' in model_name:
                trucated_seq, raw_seq = models.generate_llama_output(delta, model, tokenizer, generation_config, max_len, model_name)
            all_results.append(dict(task_id = f'HumanEval/{prompt_index}', completion = trucated_seq, all_code = raw_seq))

    return all_results


def gen_hf_model_embeds(model, tokenizer, dataset, prompt_index, delta_method, df):
    
    all_results = []

    if dataset == 'mbpp':
        deltas= mbpp_adapter.generate_deltas(df, prompt_index, delta_method, True)
    elif dataset == 'humaneval':
        deltas = humaneval_adapter.generate_deltas(df, prompt_index, delta_method, True)    
    
    
    for i, delta in enumerate(deltas):
        delta_embedding = models.get_hf_model_embedding(delta, model, tokenizer)
        all_results.append(delta_embedding)

    return all_results


def run_llm_tests(model, dataset, prompt_index, num_runs, delta_method, df):
    """
    Run the Language Learning Model (LLM) tests.

    :param model: The model to use for generating code.
    :param prompt_index: Index of prompt to test.
    :param num_runs: Number of times to run the tests.
    :param delta_method: The method for generating deltas ('permutations' or 'combinations').
    :param evaluation: The evaluation method ('evalplus' or 'runtime').
    :param df: A DataFrame containing code and test cases.
    """
    output_directory = f'generated_code_files/prompt_{prompt_index}'
    os.makedirs(output_directory, exist_ok=True)

    # Get task id
    task_id = df.iloc[prompt_index]['task_id']

    # Generate deltas
    if dataset == 'mbpp':
        deltas, delta_components_info, test_list = mbpp_adapter.generate_deltas(df, prompt_index, delta_method)
    elif dataset == 'humaneval':
        deltas, delta_components_info, test_list = humaneval_adapter.generate_deltas(df, prompt_index, delta_method)

    # Initialize results dictionary
    results = {f'delta_{i}': [] for i in range(len(deltas))}

    for run_index in range(num_runs):
        print(f"Run {run_index + 1} of {num_runs} for prompt {prompt_index}")
        for i, delta in enumerate(deltas):
            print(f"Generating code for delta {i + 1} of {len(deltas)}")
            generated_output = models.generate_model_output(delta, model)
            generated_code = general_adapter.extract_python_code(generated_output)
            file_name = f"{output_directory}/delta_{run_index}_{i}.py"
            with open(file_name, 'w') as file:
                file.write(generated_code)
            print(f"Running tests for delta {i + 1} of {len(deltas)}")
            results[f'delta_{i}'].append(run_test_cases_for_file(file_name, test_list) + (delta_components_info[delta],))

    all_results = []
    for delta_key, test_results in results.items():
        for run_index, (test_result, error_type, components) in enumerate(test_results):
            file_name = f"{output_directory}/delta_{run_index}_{delta_key.split('_')[1]}.py"
            with open(file_name, 'r') as file:
                code = file.read()
            all_results.append({
                'Task ID': task_id,
                'Delta': delta_key,
                'Pass/Fail': 'Pass' if test_result == 'Pass' else 'Fail',
                'Error Type': error_type,
                'Run Index': run_index + 1,
                'Code': code,
                'Components': components
            })

    results_df = pd.DataFrame(all_results)
    results_df['Passed In All Runs'] = results_df.groupby('Delta')['Pass/Fail'].transform(lambda x: 'Yes' if all(x == 'Pass') else 'No')

    csv_filename = f'{output_directory}/results_prompt_{prompt_index}.csv'
    jsonl_filename = f'{output_directory}/results_prompt_{prompt_index}.jsonl'
    results_df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")
    results_df.to_json(jsonl_filename, orient='records', lines=True)
    print(f"JSONL file created: {jsonl_filename}")