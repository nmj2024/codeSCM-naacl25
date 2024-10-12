# CodeSCM for NAACL 2025

In our paper, we propose CodeSCM, a Structural Causal Model (SCM) for analyzing CodeSCM code generation using large language models (LLMs). By applying interventions to CodeSCM, we define the causal effects of different prompt modalities, such as natural language, code, and input-output examples, on the model. CodeSCM introduces latent mediator variables to separate out the code and natural language semantics of a CodeSCM prompt. Using the principles of Causal Mediation Analysis on these mediators, we define direct effects through targeted interventions, quantitatively representing the model's spurious leanings. We find that, in addition to natural language instructions, input-output examples significantly influence model generation, and total causal effects evaluations from CodeSCM also reveal the memorization of code generation benchmarks.


## Installation

To install the required packages, use the following command:

```pip install -r requirements.txt```

This will install all the necessary packages required for this project.

## File Descriptions and Process Flow

### Main Execution
- **run_runtime_evaluation.py**: This script is designed for evaluating the runtime performance of various models. It includes functionalities to run test cases within a process with timeout control, parse command-line arguments, and measure execution time for each test case.
- **run_openai_model_experiments.py**: Utilizes OpenAI's models to generate outputs for a given dataset. It is equipped with argument parsing for script configuration and functions for loading datasets and generating outputs using OpenAI's API.
- **run_closed_source_model_experiments.py**: Focuses on evaluating closed source LLM models on specific datasets. It includes an argument parser setup to configure the number of prompts and other execution parameters.
- **run_analysis.py**: Dedicated to analyzing the results from model experiments. It categorizes errors, reads and prepares data from experiment outputs, and computes statistical results to evaluate model performance.
- **codereval**: Contains relevant files to parse and generate output for self-contained Python and Java CoderEval prompts.

### Tools Submodule Files
- **evaluate_deltas.py**: Evaluates differences between model outputs and expected outputs, identifying discrepancies to understand model performance across datasets.
- **llm.py**: Provides an interface for interacting with LLMs, facilitating communication and response handling between experiment scripts and models.
- **models.py**: Defines and configures the LLMs used in experiments, enabling easy model selection and comparative analysis through standardized parameters.

### Adapters Submodule Files
- **mbpp_adapter.py**: Formats MBPP dataset prompts for compatibility with LLMs, automating the processing of test cases for coding task evaluations.
- **humaneval_adapter.py**: Converts HumanEval dataset challenges for LLM processing, assessing models' creative coding and problem-solving abilities.
- **general_adapter.py**: Provides a generalized approach to formatting prompts for a variety of datasets, ensuring flexibility in experiment design.
- **manual_prompts_comp.py**: Manages manually created prompts for specific experiments, allowing for customization in prompt design to analyze nuanced model responses.
