#!/bin/bash
MODEL_TYPE="qwen" # "qwen", "llama3"
DATA_TYPE="test" # "test", "train"

# Generation Settings (Globali)
MAX_NUM_EXAMPLES=100000000000000
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

MERGED_MODELS_DIR="./Modelli_Fusi"

# Lists to iterate over
MODELS=("3B" "7B" "14B") 
BENCHMARKS=("boolq" "gsm8k" "math" "piqa") 

for SIZE in "${MODELS[@]}"; do
    
    MODEL_SIZE="${SIZE,,}" # Convert to lower case (e.g. "3b")
    
    BASE_MODEL_PATH="Qwen/Qwen2.5-${SIZE}-Instruct"
    
    for BENCHMARK in "${BENCHMARKS[@]}"; do

        echo " MODEL: Qwen2.5-${SIZE}-Instruct | BENCHMARK: ${BENCHMARK^^} "

        MERGED_MODEL_PATH="${MERGED_MODELS_DIR}/Merged-Qwen2.5-${SIZE}-${BENCHMARK}"

        # Set the tokens dynamically based on the benchmark
        if [ "$BENCHMARK" == "math" ] || [ "$BENCHMARK" == "boolq" ]; then
            MAX_NEW_TOKENS=1024
        else
            MAX_NEW_TOKENS=512
        fi
        
        echo " -> Max New Tokens base: ${MAX_NEW_TOKENS}"

        # Loop over the compression ratios
        for COMPRESSION_RATIO in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
            
            echo "   [Test] Ratio: $COMPRESSION_RATIO "

            OUTPUT_DIR="outputs/inference/Qwen2.5_${MODEL_SIZE}/${BENCHMARK}/ratio_${COMPRESSION_RATIO}/"
            mkdir -p "${OUTPUT_DIR}"

            RUN_NAME_ID="${MODEL_SIZE}_${BENCHMARK}_tok${MAX_NEW_TOKENS}_ratio${COMPRESSION_RATIO}_run${RUN}"

            echo "    >> Starting PDU & GPU MONITORING in background..."
            python3 ../monitor_pdu.py --run-name "$RUN_NAME_ID" --output-dir "$OUTPUT_DIR" --interval 0.5 &
            PDU_PID=$!
            
            python3 ../monitor_gpu.py --run-name "$RUN_NAME_ID" --output-dir "$OUTPUT_DIR" --interval 0.5 &
            GPU_PID=$!

            # --- MODEL HANDLING (MERGED vs BASELINE) ---
            if [ "$COMPRESSION_RATIO" != "1.0" ]; then
                echo "    >> Ratio < 1.0: Loading the MERGED model for maximum vLLM speed"
                CURRENT_MODEL_PATH="${MERGED_MODEL_PATH}"
            else
                echo "    >> Ratio 1.0: Loading the original BASELINE"
                CURRENT_MODEL_PATH="${BASE_MODEL_PATH}"
            fi

            CUDA_VISIBLE_DEVICES=0 python ../evaluation.py \
                --output-dir "${OUTPUT_DIR}" \
                --model-path "${CURRENT_MODEL_PATH}" \
                --tokenizer-path "${CURRENT_MODEL_PATH}" \
                --model-size "${MODEL_SIZE}" \
                --model-type "${MODEL_TYPE}" \
                --data-type "${DATA_TYPE}" \
                --max_num_examples "${MAX_NUM_EXAMPLES}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --eval_batch_size "${EVAL_BATCH_SIZE}" \
                --temperature "${TEMPERATURE}" \
                --seed "${SEED}" \
                --benchmark "${BENCHMARK}" \
                --use_vllm \
                --compression_ratio "${COMPRESSION_RATIO}"

            echo "    >> Inferenza terminata. Chiusura monitor..."
            
            kill -2 $PDU_PID
            kill -2 $GPU_PID
            
            sleep 10
            
        done
        echo " All ratio completed for ${BENCHMARK} on ${SIZE}."
    done
done

echo " FINISH, maybe. "