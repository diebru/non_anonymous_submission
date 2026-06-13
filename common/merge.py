import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def merge_and_save(base_model_path, adapter_path, output_path):
    print(f"Starting merge for: {base_model_path} + {adapter_path}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Load the adapter and merge it
    print("Loading and merging the adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    
    print(f"Saving the merged model to: {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge completed successfully!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Destination folder for the merged model")
    args = parser.parse_args()
    
    merge_and_save(args.base, args.adapter, args.output)