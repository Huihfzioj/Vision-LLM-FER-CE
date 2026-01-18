import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

class Blip2Finetuner(nn.Module):
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", use_lora=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading BLIP-2 model: {model_name}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        # Load with 16-bit precision to save memory on Consumer GPUs
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Freeze frozen parameters (Vision Encoder & LLM)
        # In BLIP-2, usually we only train Q-Former or use LoRA on LLM.
        # Here, we will use LoRA on the LLM parts to allow text generation adaptation.
        if use_lora:
            self._setup_lora()
        else:
            # If not using LoRA, at least freeze the heavy vision encoder
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
                
    def _setup_lora(self):
        print("Setting up LoRA configuration...")
        # Define LoRA Config 
        # Target modules depend on the LLM backbone (OPT or T5).
        # For OPT, usually query_key_value or q_proj, v_proj
        # For BLIP-2 with OPT backbone:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"] # Common targets for attention layers
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Forward pass for training/inference.
        input_ids: The question/prompt
        labels: The answer (for calculating loss)
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=20):
        """
        Generate text for inference/evaluation.
        """
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
