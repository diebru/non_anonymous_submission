import json
from datasets import load_dataset

def process_boolq():
    print("Downloading the BoolQ dataset")
    dataset = load_dataset("boolq", trust_remote_code=True)  
    
    # As with PIQA, the true test-set labels are often hidden or not included.
    # We use 'validation' as our 'test' set.
    splits = {'train': 'train', 'validation': 'test'} 
    
    for split_name, output_name in splits.items():
        formatted_data = []
        print(f"Processing split: {split_name} -> becomes {output_name}")
        
        for item in dataset[split_name]:
            passage = item['passage']
            question = item['question']
            
            # The original BoolQ labels are booleans (True or False). 
            correct_option = str(item['answer']) 
            
            prompt = (
                f"Passage: {passage}\n"
                f"Question: {question}\n"
            )
            
            # Build the structure expected by the evaluation
            formatted_item = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "answer": correct_option
            }
            formatted_data.append(formatted_item)
        
        output_file = f"boolq_{output_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Saved {output_file} successfully! ({len(formatted_data)} problems).")

if __name__ == "__main__":
    process_boolq()