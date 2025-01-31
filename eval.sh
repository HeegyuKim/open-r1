
# MODEL="models/deepseek-r1-distill-qwen-1.5b.yaml"
# MODEL_ARGS="$MODEL"

# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
MODEL="iknow-lab/EXAONE-3.5-2.4B-Instruct-Open-R1-Distill-16k"
NUM_GPUS=1
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"

TASK=hrm8k_ksm_kms,hrm8k_ksm_kjmo
OUTPUT_DIR=data/evals/$MODEL
SYSTEM="Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
# SYSTEM="Please reason step by step, and put your final answer within \boxed{}."
    # --system-prompt="$SYSTEM" \

# custom|hrm8k_ksm_kjmo|0|0
lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --temperature 0.6 --top-p 0.95 \
    --save-details --max-samples 5
