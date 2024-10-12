from . import general_adapter
import itertools
from typing import Iterable, Dict
import gzip
import json
import random
from adapters.manual_prompts_comp import humaneval_manual_prompt_dict
import sys
sys.set_int_max_str_digits(0)

def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def extract_humaneval_examples(code, function_header, start_words):
    text = code.split(function_header)[1].strip()
    examples_text = ""
    recording = False

    for line in text.split('\n'):
        if any(start_word in line for start_word in start_words):
            recording = True  
        elif recording and (line.strip() == '' or line.strip().startswith('"""')):
            break
        if recording:
            examples_text += line + '\n'
    return examples_text.strip()

def extract_humaneval_docstring(code, function_header, stop_words):
    text = code.split(function_header)[1].strip()
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0]
    return text

def extract_humaneval_test_list(entry_point, plus_input, expected_output):
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    test_list = [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i,j in zip(plus_input, expected_output)]
    return test_list


def transform_func_name(entry_point):
    if '_' in entry_point:
        func_elements = entry_point.split('_')
        func_elements = [i.capitalize() for i in func_elements]
        func_elements = ''.join(func_elements)
        return func_elements
    return entry_point.capitalize()

def transform_io_pairs(examples, entry_point):
    return examples.replace('==', '>=') + examples.replace('==', '<=')


def generate_deltas(df, prompt_index, delta_method=None, return_modal_components=False, modal_transformations=False, all_deltas=False):
    """
    Generate deltas based on the provided DataFrame, prompt index, and delta method.

    :param df: DataFrame containing the necessary data.
    :param prompt_index: The index of the prompt in the DataFrame.
    :param delta_method: Method for generating deltas ('permutations' or 'combinations').
    :return: A tuple containing the list of deltas and a dictionary with delta components info.
    """
    df = df[['prompt', 'entry_point', 'test', 'plus_input', 'plus', 'canonical_solution']].copy()
    #plus_input = df.iloc[prompt_index]['plus_input']
    #expected_output = df.iloc[prompt_index]['plus']

    # Extracting and ensuring the data types
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])

    if prompt_index in humaneval_manual_prompt_dict.keys():
        function_header, docstring, examples = humaneval_manual_prompt_dict[prompt_index]
        docstring = docstring.strip().replace('"""', '').replace("'''", "")
        examples = examples.strip().replace('"""', '').replace("'''", "")
    else:
        function_header = str(general_adapter.extract_function_header(prompt, entry_point))
        docstring = extract_humaneval_docstring(prompt, function_header, ['For example', 'For Example', 'Example', 'example', '>>>', '>>', f'\n{entry_point}'])
        #examples = extract_humaneval_examples(prompt, function_header, ['Example', 'example', 'For example', 'For Example', '>>>', '>>', f'\n{entry_point}'])
        examples = prompt.split(docstring)[1].strip().replace('"""', '').replace("'''", "")
        docstring = docstring.strip().replace('"""', '').replace("'''", "")
    #test_list = extract_humaneval_test_list(entry_point, plus_input, expected_output)
    normalized_function_header = function_header.replace(entry_point, 'func')

    docstring_trans = docstring.title()
    function_header_deadcode = f'{function_header}\n\tif False:\n\t\tx=[_ for i in range(42)]'
    entry_point_trans = transform_func_name(entry_point)
    function_header_name = function_header.replace(entry_point, entry_point_trans)
    examples_trans = examples.replace(entry_point, entry_point_trans)

    if all_deltas:
        examples_normalized = examples.replace(entry_point, normalized_function_header)
        examples_equality_transformed = transform_io_pairs(examples, entry_point)
        prefix_docstring = f'DOCSTRING: {docstring}'
        prefix_entry_point = f'func_{entry_point}'
        function_header_prefix = function_header.replace(entry_point, prefix_entry_point)
        examples_prefix = examples.replace(entry_point, prefix_entry_point)
        
        return [f'{prompt}', # DELTA 1
                f'{function_header}\n"""\n{examples}\n"""\n', # DELTA 2 - remove NL
                f'{docstring}\nCreate a function named {entry_point}\n{examples}\n', # DELTA 3 remove codeAL
                f'{normalized_function_header}\n"""\n{docstring}\n{examples_normalized}\n"""\n', # DELTA 4 (DELTA 1 of new) removing codeNL 
                f'{function_header}\n"""\n{docstring}\n"""\n',  # DELTA 5 (DELTA 2 of new) remove I/0
                f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n', # DELTA 6
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n', # DELTA 7
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n', # DELTA 8
                f'{function_header}\n"""\n{docstring}\n{examples_equality_transformed}\n"""\n', #transforming I/O DELTA 9 (DELTA 3 of new) - dont have this
                f'{function_header}\n"""\n{prefix_docstring}\n{examples}\n"""\n', #prefix docstring (DELTA 10)
                f'{function_header_prefix}\n"""\n{docstring.replace(entry_point, prefix_entry_point)}\n{examples_prefix}\n"""\n' #prefix entry point (DELTA 11)
        ]

        return [f'{normalized_function_header}\n"""\n{docstring}\n{examples_normalized}\n"""\n', # DELTA 4 (DELTA 1 of new) removing codeNL 
                f'{function_header}\n"""\n{docstring}\n"""\n',  # DELTA 5 (DELTA 2 of new) remove I/0
                f'{function_header}\n"""\n{docstring}\n{examples_equality_transformed}\n"""\n', #transforming I/O DELTA 9 (DELTA 3 of new) - dont have this
        ]
    
        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n', # DELTA 6
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n', # DELTA 7
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n', # DELTA 8
        ]

        return [f'{prompt}', # DELTA 1
                f'{function_header}\n"""\n{examples}\n"""\n', # DELTA 2 - remove NL
                f'{docstring}\nCreate a function named {entry_point}\n{examples}\n', # DELTA 3 remove codeAL
                f'{normalized_function_header}\n"""\n{docstring}\n"""\n',
                f'\n{docstring}\n{examples}\n{function_header}\n',
                f'{docstring}\n{function_header}\n"""\n{examples}\n"""\n',
                f'{function_header}\n"""\n{examples}\n{docstring}\n"""\n',
            ]

    if return_modal_components:
        return [
            # prompt,
            # function_header,
            # docstring,
            # examples,
            str(df.iloc[prompt_index]['canonical_solution'])
        ]
    
    if modal_transformations:
        docstring_trans = docstring.title()
        function_header_deadcode = f'{function_header}\n\tif False:\n\t\tx=[_ for i in range(42)]'
        entry_point_trans = transform_func_name(entry_point)
        function_header_name = function_header.replace(entry_point, entry_point_trans)
        examples_trans = examples.replace(entry_point, entry_point_trans)

        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n',
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n',
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n',
        ]

    return [f'{prompt}',
            f'{function_header}\n"""\n{examples}\n"""\n',
            f'{docstring}\nCreate a function named {entry_point}\n{examples}\n',
            f'{normalized_function_header}\n"""\n{docstring}\n"""\n',
            f'\n{docstring}\n{examples}\n{function_header}\n',
            f'{docstring}\n{function_header}\n"""\n{examples}\n"""\n',
            f'{function_header}\n"""\n{examples}\n{docstring}\n"""\n',
        ]

    # Define delta components as a dictionary
    delta_components = {
        'docstring': docstring,
        'function_header': function_header,
        'examples': examples
    }

    # Choose between permutations and combinations
    delta_generator = itertools.permutations if delta_method == 'permutations' else itertools.combinations

    # Generate all permutations or combinations of the deltas
    delta_elements = ['docstring', 'function_header', 'examples']
    all_deltas = []
    for r in range(1, len(delta_elements) + 1):
        all_deltas.extend(delta_generator(delta_elements, r))

    deltas = []
    delta_components_info = {}  # To store components information
    for delta in all_deltas:
        delta_key = '\n'.join([delta_components[element] for element in delta])
        deltas.append(delta_key)
        delta_components_info[delta_key] = ', '.join(delta)  # Store the components for each delta

    return deltas, delta_components_info, test_list