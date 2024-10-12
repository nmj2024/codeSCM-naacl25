#/bin/bash

echo "Running on Humaneval"
python run_experiments.py --dataset humaneval --experiment humaneval_codellama_13B


echo "Running on MBPP"
python run_experiments.py --dataset mbpp --experiment mbpp_codellama_13B


echo "Running LLaMa-2 on Humaneval"
python run_experiments.py --model hf_llama2_13B --dataset humaneval --experiment humaneval_codellama_13B


echo "Running LLaMa-2 on MBPP"
python run_experiments.py --model hf_llama2_13B--dataset mbpp --experiment mbpp_codellama_13B