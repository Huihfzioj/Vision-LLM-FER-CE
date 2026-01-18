import pandas as pd
import json
import os
import random

# Mapping ID -> Compound Emotion Text
EMOTION_MAP = {
    0: 'happily surprised',
    1: 'happily disgusted',
    2: 'sadly fearful',
    3: 'sadly angry',
    4: 'sadly surprised',
    5: 'sadly disgusted',
    6: 'fearfully angry',
    7: 'fearfully surprised',
    8: 'fearfully disgusted',
    9: 'angrily surprised',
    10: 'angrily disgusted',
    11: 'disgustedly surprised',
    12: 'happily fearful',
    13: 'happily sad'
}

# Diverse prompts to improve generalization
PROMPTS = [
    "What is the emotion expressed by this person?",
    "Describe the facial expression.",
    "What compound emotion does this face show?",
    "Identify the emotion in the image.",
    "Analyze the facial features and determine the emotion.",
    "Question: What is the emotion? Answer:",
]

def create_instruction_data(csv_path, img_root_dir, output_path):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    dataset_data = []
    
    for _, row in df.iterrows():
        img_name = row['image']
        # Ensure filename correctness
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += ".jpg"
            
        label_id = int(row['compound_emotion'])
        emotion_text = EMOTION_MAP.get(label_id, "unknown emotion")
        
        # Randomly select a prompt
        instruction = random.choice(PROMPTS)
        
        # Create data sample
        # Format typical for Vision-LLM training
        sample = {
            "image": os.path.join(img_root_dir, img_name),
            "question": instruction,  # 'question' or 'text_input' depending on library
            "answer": emotion_text,   # 'answer' or 'text_output'
            "label_id": label_id
        }
        dataset_data.append(sample)
        
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(dataset_data, f, indent=4)
        
    print(f"Saved {len(dataset_data)} samples to {output_path}")

if __name__ == "__main__":
    # Base paths
    DATA_ROOT = "../../data"
    
    # Generate for Train
    create_instruction_data(
        csv_path=os.path.join(DATA_ROOT, "train_split.csv"),
        img_root_dir="../../data/train_images", 
        output_path=os.path.join(DATA_ROOT, "train_instructions.json")
    )
    
    # Generate for Val
    create_instruction_data(
        csv_path=os.path.join(DATA_ROOT, "val_split.csv"),
        img_root_dir="../../data/val_images",
        output_path=os.path.join(DATA_ROOT, "val_instructions.json")
    )
