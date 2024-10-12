import argparse
import pandas as pd
from tabulate import tabulate

def categorize_error(error):
    if error is None or pd.isna(error):
        return "No Error"
    if error.startswith("Syntax Error"):
        return "Syntax Errors"
    elif error.startswith("Semantic Error"):
        return "Semantic Errors"
    elif error.startswith("Runtime Error"):
        return "Runtime Errors"
    else:
        return "Other Errors"

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="JSONL file")
    args = parser.parse_args()

    # Read JSONL file into a DataFrame
    df = pd.read_json(args.file, lines=True)

    # Count the number of 'Pass' and 'Fail' in the 'result' column
    result_counts = df['result'].value_counts()
    result_counts_df = result_counts.reset_index()
    result_counts_df.columns = ['Result', 'Count']
    result_counts_df['Percent'] = (result_counts_df['Count'] / result_counts_df['Count'].sum()) * 100
    result_table = tabulate(result_counts_df, headers='keys', tablefmt='grid')
    print("Result Counts:")
    print(result_table)
    print("\n")

    # Count the number of each type of error
    error_counts = df['error'].value_counts()
    error_counts_df = error_counts.reset_index()
    error_counts_df.columns = ['Error Type', 'Count']
    error_counts_df['Percent'] = (error_counts_df['Count'] / error_counts_df['Count'].sum()) * 100
    error_table = tabulate(error_counts_df, headers='keys', tablefmt='grid')
    print("Error Counts:")
    print(error_table)
    print("\n")

    # Categorize errors and filter out 'No Error'
    df['error_category'] = df['error'].apply(categorize_error)
    filtered_df = df[df['error_category'] != "No Error"]

    # Calculate error frequencies and percentages for each delta with categories
    delta_error_category_counts = filtered_df.groupby(['delta', 'error_category']).size().unstack(fill_value=0)
    delta_error_category_percentages = delta_error_category_counts.div(delta_error_category_counts.sum(axis=1), axis=0) * 100
    categorized_delta_error_table = tabulate(delta_error_category_percentages, headers='keys', tablefmt='grid', showindex=True, floatfmt=".2f")
    print("Categorized Error Percentages per Delta:")
    print(categorized_delta_error_table)
    print("\n")

    # Calculate most common error per delta
    most_common_errors = filtered_df.groupby('delta')['error'].agg(lambda x:x.value_counts().idxmax())
    most_common_errors_counts = filtered_df.groupby('delta')['error'].agg(lambda x:x.value_counts().max())
    most_common_errors_df = pd.DataFrame({'Delta': most_common_errors.index, 
                                          'Most Common Error': most_common_errors.values, 
                                          'Count': most_common_errors_counts.values})
    most_common_errors_table = tabulate(most_common_errors_df, headers='keys', tablefmt='grid')
    print("Most Common Error per Delta and Frequency:")
    print(most_common_errors_table)
    print("\n")

    # Calculate Categorized Sensitivity per Delta
    sensitivity_per_delta = delta_error_category_percentages.subtract(delta_error_category_percentages.loc[1], axis=1)
    sensitivity_per_delta_table = tabulate(sensitivity_per_delta, headers='keys', tablefmt='grid', showindex=True, floatfmt=".2f")
    print("Categorized Sensitivity per Delta:")
    print(sensitivity_per_delta_table)

    # Write results to a file
    output_file = f"{args.file}_ANALYSIS.txt"
    with open(output_file, 'w') as file:
        file.write("Result Counts:\n")
        file.write(result_table)
        file.write("\n\nError Counts:\n")
        file.write(error_table)
        file.write("\n\nCategorized Error Percentages per Delta:\n")
        file.write(categorized_delta_error_table)
        file.write("\n\nMost Common Error per Delta and Frequency:\n")
        file.write(most_common_errors_table)
        file.write("\n\nCategorized Sensitivity per Delta:\n")
        file.write(sensitivity_per_delta_table)

if __name__ == "__main__":
    main()