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

def load_all_data(input_dir="./outputs/llama3/8b/boolq/8b/"):
    original_data = load_json(os.path.join(input_dir, "Original/train/samples/predictions_formatted.jsonl"))
    
    compressed_datasets = [original_data]
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    for r in ratios:
        path = os.path.join(input_dir, f"Compression/train_outputs_compressed_ratio_{r}.jsonl")
        compressed_datasets.append(load_json(path))
        
    return compressed_datasets

def get_llamafactory_input():
    all_data = load_all_data()
    original_data = all_data[0]
    
    ratio_map = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1}
    datalines = []
    
    # Take the minimum length across all datasets to avoid IndexError (crash) 
    num_samples = min(len(d) for d in all_data)
    
    for i in range(num_samples):
        data_index = random.choice(list(ratio_map.keys()))
        compression_ratio = ratio_map[data_index]
        
        selected_dataset = all_data[data_index]
        item = selected_dataset[i]
        
        question_text = item.get('question', '')
        if not question_text and 'messages' in item:
            question_text = item['messages'][0]['content']
        question_text = question_text.strip()
        
        if compression_ratio == 1.0:
            input_data = question_text
            cot = item.get('cot', '') 
        else:
            # For compressed samples, append <|eot_id|> with the ratio
            input_data = f"{question_text}<|eot_id|>{compression_ratio}<|eot_id|>"
            cot = item.get('compressed_cot', '')

        # Extract the answer (True or False)
        answer = item.get('model_answer', item.get('prediction', item.get('answer', '')))
        if isinstance(answer, list):
            answer = str(answer[0]) if len(answer) > 0 else ""
        else:
            answer = str(answer)

        # --- KEY FIX TO CLEAN THE TEXT ---
        # Remove things like "The final answer is: $" or "final answer: $" from the end of the CoT
        cot = re.sub(r'The final answer is:\s*\$', '', cot, flags=re.IGNORECASE)
        cot = re.sub(r'final answer:\s*\$', '', cot, flags=re.IGNORECASE)
        cot = cot.replace('$', '').strip() # Toglie eventuali dollari rimasti
        
        # Now we can safely attach the clean block
        output_data = f"{cot}\n\nThe final answer is: \\boxed{{{answer}}}"

        # Format for LLaMA-Factory
        data = {
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": input_data,
            "output": output_data
        }
        datalines.append(data)
        
    print(f"Generated {len(datalines)} training examples for BoolQ.")
    random.shuffle(datalines)
    
    output_path = './outputs/llama3/8b/boolq/8b/mydataset_boolq.json'
    write_list_to_json(datalines, output_path)
    print(f"Dataset saved successfully to: {output_path}")

if __name__ == '__main__':
    seed_everything(42)
    get_llamafactory_input()