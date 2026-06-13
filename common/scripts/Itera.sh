#!/bin/bash

# --- Configurazione Base ---
BENCHMARK="piqa"
MODEL_SIZE="8b"
MODEL_TYPE="llama3"
DATA_TYPE="test"

MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

ADAPTER_PATH="../LlamaFactory/lora_saves/LLaMA-3.1-8B-Instruct/lora/tokenskip_piqa"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

# ==============================================================================
# LOOP OVER MAX NEW TOKENS (32, 64, 128, 256, 512, 1024)
# ==============================================================================
for MAX_NEW_TOKENS in 32 64 128 256 512 1024; do

    echo "######################################################################"
    echo " STARTING EXPERIMENTS WITH MAX_NEW_TOKENS: $MAX_NEW_TOKENS "
    echo "######################################################################"

    # ==============================================================================
    # LOOP OVER THE COMPRESSION RATIOS (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    # ==============================================================================
    for COMPRESSION_RATIO in 0.5 0.6 0.7 0.8 0.9 1.0; do
        
        echo "======================================================================"
        echo " STARTING TEST FOR MAX_TOKENS: $MAX_NEW_TOKENS | COMPRESSION_RATIO: $COMPRESSION_RATIO "
        echo "======================================================================"

        # ==========================================================================
        # LOOP OVER THE 5 REPETITIONS (RUN)
        # ==========================================================================
        for RUN in {1..5}; do
            
            echo "---------------------------------------------------"
            echo " RUN $RUN of 5 for RATIO $COMPRESSION_RATIO (TOKENS: $MAX_NEW_TOKENS) "
            echo "---------------------------------------------------"

            OUTPUT_DIR="outputs/${MAX_NEW_TOKENS}/LLaMA-3.1-8B-Instruct/${BENCHMARK}/ratio_${COMPRESSION_RATIO}/run_${RUN}/"
            
            # Safety: physically create the directory before the Python scripts try to write to it
            mkdir -p "${OUTPUT_DIR}"

            # 4. Add the token count and RUN to the ID to avoid overwriting the monitor logs
            RUN_NAME_ID="${MODEL_SIZE}_${BENCHMARK}_tok${MAX_NEW_TOKENS}_ratio${COMPRESSION_RATIO}_run${RUN}"

            # START PYTHON ENERGY MONITORS
            echo " >> STARTING OF PDU & GPU MONITORING in background..."
            python3 ../monitor_pdu.py --run-name "$RUN_NAME_ID" --output-dir "$OUTPUT_DIR" --interval 0.5 &
            PDU_PID=$!
            
            python3 ../monitor_gpu.py --run-name "$RUN_NAME_ID" --output-dir "$OUTPUT_DIR" --interval 0.5 &
            GPU_PID=$!

            CUDA_VISIBLE_DEVICES=1 python ../evaluation.py \
                --output-dir "${OUTPUT_DIR}" \
                --model-path "${MODEL_PATH}" \
                --tokenizer-path "${TOKENIZER_PATH}" \
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
                --compression_ratio "${COMPRESSION_RATIO}" \
                --use_adapter \
                --adapter-path "${ADAPTER_PATH}"

            echo " >> INFERENCE FINISH, CLOSING MONITORS FOR RUN $RUN..."
            
            # Use SIGINT (-2) to stop the monitors cleanly (equivalent to Ctrl+C)
            kill -2 $PDU_PID
            kill -2 $GPU_PID
            
            # Pause to let the Python processes finish writing the logs to disk
            sleep 10
            
        done
    done
done

echo "======================================================================"
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY, hopefully! "
echo "======================================================================"