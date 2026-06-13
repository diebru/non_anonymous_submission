import os
import json
import random
import numpy as np
import re

def load_json(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def write_list_to_json(data_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def process_dataset(dataset_name, base_input_dir, output_dir, model_name, model_size):
    """
    Automatically handles the rules for piqa, boolq, math and gsm8k.
    """
    dataset_name = dataset_name.lower()
    
    # --- DATASET-SPECIFIC CONFIGURATIONS ---
    configs = {
        "piqa": {
            "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "math_mode": False,       
            "clean_regex": False,     
            "original_cot_key": "cot"
        },
        "boolq": {
            "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "math_mode": False,
            "clean_regex": True,      
            "original_cot_key": "cot"
        },
        "math": {
            "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "math_mode": True,        
            "clean_regex": False,
            "original_cot_key": "cot"
        },
        "gsm8k": {
            "ratios": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "math_mode": True,
            "clean_regex": False,
            "original_cot_key": "cot"
        }
    }

    if dataset_name not in configs:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Choose from: {list(configs.keys())}")
    
    config = configs[dataset_name]
    print(f"\n--- Processing dataset: {dataset_name.upper()} (Dir: {base_input_dir}) ---")

    # 1. Load data
    original_file = os.path.join(base_input_dir, "Original/train/samples/predictions_formatted.jsonl")
    
    if not os.path.exists(original_file):
        raise FileNotFoundError(f"Original file not found: {original_file}")

    all_data = [load_json(original_file)]
    
    compression_ratios = config["ratios"][1:] 
    for r in compression_ratios:
        path = os.path.join(base_input_dir, f"Compression/train_outputs_compressed_ratio_{r}.jsonl")
        if os.path.exists(path):
            all_data.append(load_json(path))
        else:
            print(f"WARNING: missing compressed file {path}. It will be ignored.")

    # 2. Data preparation
    datalines = []
    num_samples = min(len(d) for d in all_data)
    
    for i in range(num_samples):
        data_index = random.choice(range(len(all_data)))
        
        if data_index == 0:
            compression_ratio = 1.0
        else:
            compression_ratio = compression_ratios[data_index - 1]

        item = all_data[data_index][i]
        
        question_text = item.get('question', '')
        if not question_text and 'messages' in item:
            question_text = item['messages'][0]['content']
        question_text = str(question_text).strip()
        
        if compression_ratio == 1.0:
            input_data = question_text
            cot = item.get(config["original_cot_key"], item.get('cot', ''))
        else:
            input_data = f"{question_text}<|eot_id|>{compression_ratio}<|eot_id|>"
            cot = item.get('compressed_cot', '')

        answer = item.get('model_answer', item.get('prediction', item.get('answer', '')))
        if isinstance(answer, list):
            answer = str(answer[0]) if len(answer) > 0 else ""
        else:
            answer = str(answer)

        if config["clean_regex"]:
            cot = re.sub(r'The final answer is:\s*\$', '', cot, flags=re.IGNORECASE)
            cot = re.sub(r'final answer:\s*\$', '', cot, flags=re.IGNORECASE)
            cot = cot.replace('$', '').strip()

        if config["math_mode"]:
            output_data = f"{cot}\n\nThe final answer is: $\\boxed{{{answer}}}$"
        else:
            output_data = f"{cot}\n\nThe final answer is: \\boxed{{{answer}}}"

        data = {
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": input_data,
            "output": output_data
        }
        datalines.append(data)
        
    # 3. Save
    random.shuffle(datalines)
    output_path = os.path.join(output_dir, f'mydataset_{dataset_name}_{model_name}_{model_size}.json')
    write_list_to_json(datalines, output_path)
    print(f"Generated {len(datalines)} examples.")
    print(f"Saved successfully to: {output_path}")


if __name__ == '__main__':
    seed_everything(42)
    
    # Adapted for LLaMA-3.1
    dimensions = ["8B"] 
    benchmarks = ["boolq", "gsm8k", "math", "piqa"]
    
    for dim in dimensions:
        for bench in benchmarks:
            model_folder_name = f"LLaMA-3.1-{dim.upper()}-Instruct"
            
            dim_lower = dim.lower() 
            
            base_input = f"./outputs/{model_folder_name}/{bench}/{dim_lower}/"
            base_output = f"./outputs/{model_folder_name}/{bench}/"
            
            # Output file prefix
            model_base = "llama3.1"

            try:
                process_dataset(
                    dataset_name=bench,
                    base_input_dir=base_input,
                    output_dir=base_output,
                    model_name=model_base,      
                    model_size=dim_lower
                )
            except FileNotFoundError as e:
                print(f"  -> SKIP: {e}")