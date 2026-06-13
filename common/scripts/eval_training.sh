#!/bin/bash

# Parametri fissi
MODEL_TYPE="qwen" 
DATA_TYPE="train" 
MAX_NUM_EXAMPLES=100000000000000
EVAL_BATCH_SIZE=16 
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
SEED=42

MODEL_SIZES=("3b" "7b" "14b")
BENCHMARKS=("gsm8k" "math" "piqa" "boolq")

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    
    MODEL_SIZE_CAPITAL="${MODEL_SIZE^^}"
    MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE_CAPITAL}-Instruct"
    
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        
        OUTPUT_DIR="outputs/Qwen2.5-${MODEL_SIZE}-Instruct/${BENCHMARK}"
        mkdir -p "${OUTPUT_DIR}"

        echo "====================================================================="
        echo "Starting evaluation: MODEL=${MODEL_SIZE_CAPITAL} | BENCHMARK=${BENCHMARK}"
        echo "Saving to: ${OUTPUT_DIR}"
        echo "====================================================================="

        # Run the benchmark
        CUDA_VISIBLE_DEVICES=0,1 python ../evaluation.py \
            --output-dir "${OUTPUT_DIR}" \
            --model-path "${MODEL_PATH}" \
            --tokenizer-path "${MODEL_PATH}" \
            --model-size "${MODEL_SIZE}" \
            --model-type "${MODEL_TYPE}" \
            --data-type "${DATA_TYPE}" \
            --max_num_examples ${MAX_NUM_EXAMPLES} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --temperature ${TEMPERATURE} \
            --seed ${SEED} \
            --benchmark "${BENCHMARK}" \
            --use_vllm
            
        # --- ADDED FOR VRAM CLEANUP ---
        echo "Forcing VRAM cleanup..."
        # Kill all of the current user's zombie vLLM python processes
        pkill -u $USER -f "vllm" || true
        pkill -u $USER -f "multiproc_worker_utils" || true
        
        # 5-second pause to let the OS physically free the memory
        sleep 5
        echo "VRAM freed. Moving to the next step."
        echo "---------------------------------------------------------------------"
            
    done
done