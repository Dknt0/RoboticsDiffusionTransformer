"""
This script is used to pre-compute the language embeddings for all instructions 
in expanded_instruction_gpt-4-turbo.json of a dataset.

Run in the root directory of this project:

python -m scripts.encode_lang_batch

Dknt
"""

import os
import json

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
# TARGET_DIR = "data/datasets/agilex/tfrecords/"
TARGET_DIR = "datasets/test_set/"

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # Get all the task paths
    task_paths = []
    for sub_dir in os.listdir(TARGET_DIR):
        middle_dir = os.path.join(TARGET_DIR, sub_dir)
        if os.path.isdir(middle_dir):
            for task_dir in os.listdir(middle_dir):
                task_path = os.path.join(middle_dir, task_dir)
                if os.path.isdir(task_path):
                    task_paths.append(task_path)


    print(task_paths)

    # For each task, encode the instructions
    for task_path in tqdm(task_paths):
        # Load the instructions corresponding to the task from the directory
        with open(os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = [instruction_dict['instruction']] + instruction_dict['simplified_instruction'] + \
            instruction_dict['expanded_instruction']
    
        print(task_path)
        print(instructions)

        # Encode the instructions
        tokenized_res = tokenizer(
            instructions, return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)
        
        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()
        
        attn_mask = attn_mask.cpu().bool()

        # Save each instruction in a single .pt file
        # Save the embeddings for training use
        for i in range(len(instructions)):
            text_embed = text_embeds[i][attn_mask[i]]
            save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
            print(f"Saved to {save_path}")
            torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
