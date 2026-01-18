import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class RAFCEInstructionDataset(Dataset):
    def __init__(self, json_path, processor):
        """
        json_path: Path to the train_instructions.json or val_instructions.json
        processor: The Blip2Processor instance
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.processor = processor
        # Root dir handling: The JSON has relative paths like "../data/train_images/..."
        # We assume the code is run from a location where these paths resolve, 
        # or we might need to adjust relative to the JSON file location.
        # Let's verify one path.
        self.root_dir = os.path.dirname(json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load Image
        # The JSON 'image' field is like "../data/train_images/img.jpg"
        # We need to resolve it relative to where this script might be run.
        # If absolute paths issues arise, we can fix here.
        # Assuming run from 'notebooks/' or project root.
        image_path = item['image']
        
        # If path doesn't exist, try resolving relative to the JSON file
        if not os.path.exists(image_path):
             # Try absolute resolution
             possible_path = os.path.abspath(image_path)
             if not os.path.exists(possible_path):
                 # Try relative to the data dir
                 # ../data/train_instructions.json -> ../data/../data/train_images ??? 
                 # Let's stick to the path in JSON for now and rely on user running from correct dir.
                 pass

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # Fallback for path issues - crucial for portability
            # Try removing the leading ".." if running from data root
            if image_path.startswith("../"):
                alt_path = image_path[3:] 
                if os.path.exists(alt_path):
                    image = Image.open(alt_path).convert("RGB")
                else:
                    raise FileNotFoundError(f"Could not find image at {image_path}")
            else:
                raise

        # 2. Prepare Prompt and Target
        # Input: Question/Instruction
        # Output: Answer/Emotion
        # BLIP-2 Processor expects:
        # - images
        # - text (the prompt)
        # - text_target (the answer/label) for conditional generation loss
        
        question = item['question']
        answer = item['answer']
        
        # The processor automatically handles:
        # - Image resizing/normalization
        # - Text tokenization
        # - Attention masks
        # - Label creation (for loss calculation, handling padding = -100)
        
        inputs = self.processor(
            images=image, 
            text=question, 
            return_tensors="pt"
        )
        
        # We need to tokenize the "target" separately or use the processor's capability if supported.
        # For HuggingFace BLIP-2, usually we tokenize input (prompt) and target (answer) together 
        # or rely on the model's structure.
        # Standard approach for CausalLM: Input is "Prompt", Label is "Prompt + Answer".
        # However, Blip2ForConditionalGeneration is an Encoder-Decoder or Decoder-only structure depending on LLM.
        # OPT (Decoder-only): We concat inputs. 
        # T5 (Enc-Dec): We pass `labels` separately.
        # Check model type: The user is using `blip2-opt-2.7b` (Decoder-only).
        
        # For OPT (Decoder-only), we need to feed the full text "Question: ... Answer: ..." as input_ids
        # and set labels such that the question part is -100 (ignored) and answer part is supervised.
        # The `processor` generally handles this if `text_target` is not supported directly for decoder-only.
        # Let's do manual tokenization for better control over masking if needed, 
        # OR essentially:
        
        # Prompt: "Question: ... Answer:"
        # Label: "happy"
        
        # Let's construct the full input text for Causal LM training
        full_text = f"{question} {answer}"
        
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt"
        )
        
        # Now we need to create labels.
        # Labels = input_ids, but with the "prompt" part masked out (-100).
        
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.pixel_values[0]
        
        # Create labels: clone input_ids
        labels = input_ids.clone()
        
        # Find where the answer starts.
        # This is a bit tricky with tokenization.
        # A simpler robust way: Tokenize just the prompt to find its length.
        prompt_tokens = self.processor.tokenizer(question, return_tensors="pt").input_ids[0]
        prompt_len = len(prompt_tokens)
        
        # Mask the prompt part in labels (except maybe the last token if it merges)
        # We'll mask up to `prompt_len` - but be careful of special tokens (BOS).
        # OPT tokenizer usually adds BOS.
        # Safest heuristic: mask the first Len(Prompt) tokens.
        
        # Ideally we verify this length.
        # If prompt_len < len(input_ids), mask [0 : prompt_len]
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
            
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    # Custom collate because input_ids might have different lengths
    # We need to pad them dynamically.
    
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad inputs and labels
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Use torch.nn.utils.rnn.pad_sequence
    # padding_value for input_ids is usually tokenizer.pad_token_id
    # padding_value for labels is -100
    
    # Note: We need the tokenizer to know the pad_token_id. 
    # We can pass it, or assume it's 1 (standard for OPT).
    # Let's enforce the user to pass the tokenizer or processor to collate if possible,
    # but `collate_fn` usually doesn't take extra args easily in DataLoader init.
    # We'll rely on a simple closure or just standard padding manually.
    
    max_len = max(len(x) for x in input_ids)
    
    # Basic manual padding
    input_ids_padded = []
    labels_padded = []
    attention_mask_padded = []
    
    pad_token_id = 1 # Default for OPT usually, but we should verify. 
                     # Ideally we fetch from processor.tokenizer.pad_token_id
    
    for i in range(len(batch)):
        diff = max_len - len(input_ids[i])
        
        # Pad Input IDs (Right padding is standard for Causal LM, though OPT can handle left)
        # OPT tokenizer often uses <pad>=1.
        input_ids_padded.append(torch.cat([input_ids[i], torch.full((diff,), pad_token_id, dtype=torch.long)]))
        
        # Pad Labels (-100)
        labels_padded.append(torch.cat([labels[i], torch.full((diff,), -100, dtype=torch.long)]))
        
        # Pad Attention Mask (0 for padded)
        attention_mask_padded.append(torch.cat([attention_mask[i], torch.full((diff,), 0, dtype=torch.long)]))
        
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_mask_padded)
    }
