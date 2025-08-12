shots=0
name_for_output_file=dense_llama_32_1b
model=meta-llama/Llama-3.2-1B
model_arch=llama3
for task in boolq rte hellaswag winogrande openbookqa arc_easy arc_challenge
do
python -u evaluate_task_result.py \
    --result-file results/${task}-${shots}-${name_for_output_file}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-type ${model_arch}
done
