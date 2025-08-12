shots=0
name_for_output_file=dense_llama_32_1b
model=meta-llama/Llama-3.2-1B # model path
python -u run_benchmarking.py \
    --output-path results/task-${shots}-${name_for_output_file}.jsonl \
    --model-name ${model} 
