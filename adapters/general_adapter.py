def extract_function_header(code, entry_point=None):
    """
    Extract the function header from a block of code.

    :param code: A string representing a block of Python code.
    :return: The first line starting with 'def' indicating a function definition, or an empty string if not found.
    """
    for line in code.split('\n'):
        if line.strip().startswith('def'):
            if entry_point:
                if entry_point in line:
                    return line
            else:
                return line
    return ''

def extract_python_code(llm_output):
    """
    Extract Python code from a LLM output block.

    :param llm_output: A string representing LLM output containing a Python code block.
    :return: Extracted Python code as a string.
    """
    code_block = []
    in_code_block = False
    for line in llm_output.split('\n'):
        if line.strip() == '```python':
            in_code_block = True
        elif line.strip() == '```' and in_code_block:
            break
        elif in_code_block:
            code_block.append(line)
    return '\n'.join(code_block)