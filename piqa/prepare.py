import json
from datasets import load_dataset

def process_piqa():
    print("Downloading the PIQA dataset")
    dataset = load_dataset("piqa", trust_remote_code=True)    
    splits = {'train': 'train', 'validation': 'test'} 
    
    for split_name, output_name in splits.items():
        formatted_data = []
        print(f"Processing split: {split_name} -> becomes {output_name}")
        
        for item in dataset[split_name]:
            goal = item['goal']
            sol1 = item['sol1']
            sol2 = item['sol2']
            
            # The original PIQA labels are 0 (sol1) and 1 (sol2). 
            # Convert them to "1" and "2" to make them easier to extract.
            correct_option = str(item['label'] + 1) 
            
            # Build the physical-reasoning prompt
            prompt = (
                f"Goal: {goal}\n"
                f"Option 1: {sol1}\n"
                f"Option 2: {sol2}\n"
            )
            
            # Build the structure expected by evaluation.py
            formatted_item = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "answer": correct_option
            }
            formatted_data.append(formatted_item)
        
        output_file = f"piqa_{output_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Saved {output_file} successfully! ({len(formatted_data)} problems).")

if __name__ == "__main__":
    process_piqa()
