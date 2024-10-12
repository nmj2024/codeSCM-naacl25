def combine_jsonl_files(file1_path, file2_path, file3_path, output_file_path):
    # Open the output file in write mode
    with open(output_file_path, 'w') as output_file:
        # Process each file in sequence
        for file_path in [file1_path, file2_path, file3_path]:
            # Open the current file in read mode
            with open(file_path, 'r') as input_file:
                # Read and write each line to the output file
                for line in input_file:
                    output_file.write(line)

# Paths
file1_path = 'Google_GemP_mbpp_chunk 0,150_1 runs.jsonl'
file2_path = 'Google_GemP_mbpp_chunk 151,300_1 runs.jsonl'
file3_path = 'Google_GemP_mbpp_chunk 301,398_1 runs.jsonl'
# file4_path = 'OpenAI_4T_mbpp_chunk 300,398_1 runs.jsonl'
output_file_path = 'Google_GemP_mbpp_1run_0131.jsonl'

combine_jsonl_files(file1_path, file2_path, file3_path, output_file_path)