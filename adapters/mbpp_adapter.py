import ast
import itertools
from . import general_adapter

def parse_function_inputs(input_str):
    # Get the substring before the first '=='
    input_str = input_str.split('==')[0].strip()

    # Extract the substring inside the outermost parentheses
    start_idx = input_str.find('(')
    end_idx = input_str.rfind(')')
    params_str = input_str[start_idx + 1:end_idx]

    # Use ast.parse to safely evaluate the structure and extract parameters
    tree = ast.parse(f"f({params_str})")

    # Extract the arguments from the function call
    args = tree.body[0].value.args
    
    # Convert the AST nodes back to Python objects
    inputs = [ast.literal_eval(arg) for arg in args]

    return inputs

def extract_mbpp_examples(prompt, start_word):
    prompt = prompt.replace('"""', '')
    start_pos = prompt.find(start_word)
    examples = prompt[start_pos:].strip()
    return examples

def extract_mbpp_docstring(prompt, stop_word):
    prompt = prompt.replace('"""', '')
    stop_pos = prompt.find(stop_word)
    docstring = prompt[:stop_pos].strip()
    return docstring

def extract_mbpp_test_list(entry_point, plus_input, expected_output):
    def prepare_input(inp):
        return ', '.join([str(i) for i in inp])
    test_list = [f'assert {entry_point}({prepare_input(i)}) == {str(j)}' for i,j in zip(plus_input, expected_output)]
    return test_list

def create_function_header(entry_point, canonical_solution):
    for line in canonical_solution.split('\n'):
        if entry_point in line:
            return line
    return f'def {entry_point():}'


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
    df = df[['prompt', 'entry_point', 'plus_input', 'plus', 'canonical_solution']].copy()
    #plus_input = df.iloc[prompt_index]['plus_input']
    #expected_output = df.iloc[prompt_index]['plus']
    prompt = str(df.iloc[prompt_index]['prompt'])
    entry_point = str(df.iloc[prompt_index]['entry_point'])
    canonical_solution = str(df.iloc[prompt_index]['canonical_solution'])

    # Extracting and ensuring the data types
    function_header = create_function_header(entry_point, canonical_solution)
    docstring = extract_mbpp_docstring(prompt, 'assert')
    examples = extract_mbpp_examples(prompt, 'assert')
    #test_list = extract_mbpp_test_list(entry_point, plus_input, expected_output)
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

        return [f'{function_header}\n{prompt.strip()}', # keep DELTA 1
        f'{function_header}\n"""\n{examples}\n"""\n', # keep DELTA 2
        f'{docstring}\nCreate a function named {entry_point}\n{examples}\n', # keep DELTA 3
        f'{normalized_function_header}\n"""\n{docstring}\n{examples_normalized}\n"""\n', #removing code AL ( DELTA 4)
        f'{function_header}\n"""\n{docstring}\n"""\n', # remving I/O ( DELTA 5)
        f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n', # OLD TRANS 1 transNL (DELTA 6)
        f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n', # OLD TRANS 2 trans codeAL (DELTA 7)
        f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n', # OLD TRANS 3 - trans codeNL (DELTA 8)
        f'{function_header}\n"""\n{docstring}\n{examples_equality_transformed}\n"""\n', #transforming I/O trans I/O (DELTA 9)
        f'{function_header}\n"""\n{prefix_docstring}\n{examples}\n"""\n', #prefix docstring (DELTA 10)
        f'{function_header_prefix}\n"""\n{docstring.replace(entry_point, prefix_entry_point)}\n{examples_prefix}\n"""\n' #prefix entry point (DELTA 11)
        ]



        return [f'{normalized_function_header}\n"""\n{docstring}\n{examples_normalized}\n"""\n', #removing code AL ( DELTA 4)
                f'{function_header}\n"""\n{docstring}\n"""\n', # remving I/O ( DELTA 5)
                f'{function_header}\n"""\n{docstring}\n{examples_equality_transformed}\n"""\n', #transforming I/O trans I/O (DELTA 9)
        ]        
    
        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n', # OLD TRANS 1 transNL (DELTA 6)
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n', # OLD TRANS 2 trans codeAL (DELTA 7)
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n', # OLD TRANS 3 - trans codeNL (DELTA 8)
        ]


        return [f'{function_header}\n{prompt.strip()}', # keep DELTA 1
                f'{function_header}\n"""\n{examples}\n"""\n', # keep DELTA 2
                f'{docstring}\nCreate a function named {entry_point}\n{examples}\n', # keep DELTA 3
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
        docstring_trans = docstring.replace('Write', 'Return')
        docstring_trans = docstring_trans.title()
        function_header_deadcode = f'{function_header}\n\tif False:\n\t\tx=[_ for i in range(42)]'
        entry_point_trans = transform_func_name(entry_point)
        function_header_name = function_header.replace(entry_point, entry_point_trans)
        examples_trans = examples.replace(entry_point, entry_point_trans)

        return [f'{function_header}\n"""\n{docstring_trans}\n{examples}\n"""\n',
                f'{function_header_deadcode}\n"""\n{docstring}\n{examples}\n"""\n',
                f'{function_header_name}\n"""\n{docstring.replace(entry_point, entry_point_trans)}\n{examples_trans}\n"""\n',
        ]


    return [f'{function_header}\n{prompt.strip()}',
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
        'examples': examples
    }

    # Choose between permutations and combinations
    delta_generator = itertools.permutations if delta_method == 'permutations' else itertools.combinations

    # Generate all permutations or combinations of the deltas
    delta_elements = ['docstring', 'examples']
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