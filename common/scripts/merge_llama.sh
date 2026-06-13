#!/bin/bash

# 1. Define the main paths
BASE_ADAPTER_DIR="../LlamaFactory/lora_saves"
BASE_OUTPUT_DIR="./Modelli_Fusi"

# 2. Define the variables (8B only for Llama 3.1)
MODELS=("8B")
TASKS=("boolq" "gsm8k" "math" "piqa")

echo "========================================================"
echo " STARTING AUTOMATIC LORA ADAPTER MERGE (LLAMA 3.1)"
echo "========================================================"

for SIZE in "${MODELS[@]}"; do
    
    HF_BASE_MODEL="meta-llama/Llama-3.1-${SIZE}-Instruct"
    
    for TASK in "${TASKS[@]}"; do
        echo "--------------------------------------------------------"
        echo " >> Processing: Model Llama-3.1 $SIZE | Task: $TASK"
        
        ADAPTER_PATH="${BASE_ADAPTER_DIR}/LLaMA-3.1-${SIZE}-Instruct/lora/${TASK}_test"
        OUTPUT_PATH="${BASE_OUTPUT_DIR}/Merged-Llama-3.1-${SIZE}-${TASK}"
        
        if [ -d "$ADAPTER_PATH" ]; then
            echo " >> Adapter found. Starting merge..."
            
            python3 ../merge.py \
              --base "$HF_BASE_MODEL" \
              --adapter "$ADAPTER_PATH" \
              --output "$OUTPUT_PATH"
              
            echo " >> Merge completed for ${SIZE} on ${TASK}!"
        else
            echo " >> WARNING: adapter folder not found in $ADAPTER_PATH. Skipping to the next..."
        fi
    done
done

echo "========================================================"
echo " ALL MERGES COMPLETED SUCCESSFULLY!"
echo " The merged models are in: $BASE_OUTPUT_DIR"
echo "========================================================"