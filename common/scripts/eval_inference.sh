#!/bin/bash
BENCHMARK="math"
MODEL_SIZE="8b"
MODEL_TYPE="llama3"
DATA_TYPE="test"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH="../LlamaFactory/lora_saves/LLaMA-3.1-8B-Instruct/lora/math_test"
# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

MAX_NEW_TOKENS=1024
RUN=1

#CICLA SU 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1

for COMPRESSION_RATIO in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    
    echo " STARTING TEST FOR COMPRESSION_RATIO: $COMPRESSION_RATIO "

    OUTPUT_DIR="outputs/inference/${MODEL_TYPE}/${BENCHMARK}/ratio_${COMPRESSION_RATIO}/"
    
    mkdir -p "${OUTPUT_DIR}"

    RUN_NAME_ID="${MODEL_SIZE}_${BENCHMARK}_tok${MAX_NEW_TOKENS}_ratio${COMPRESSION_RATIO}_run${RUN}"

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

    echo " >> INFERENCE FINISH, CLOSING MONITORS FOR RATIO $COMPRESSION_RATIO..."
    
    kill -2 $PDU_PID
    kill -2 $GPU_PID
    
    sleep 10
    
done

echo " ALL COMPRESSION-RATIO TESTS COMPLETED SUCCESSFULLY, maybe! "