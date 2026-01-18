import torch
import torch.optim as optim
from tqdm import tqdm
import os

def train_vision_llm(model_wrapper, train_loader, val_loader, num_epochs=5, learning_rate=1e-4, save_dir="checkpoints"):
    """
    Training loop for BLIP-2 with LoRA.
    model_wrapper: The Blip2Finetuner instance (which contains .model and .processor)
    """
    device = model_wrapper.device
    model = model_wrapper.model
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    os.makedirs(save_dir, exist_ok=True)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            # BLIP-2 forward automatically calculates loss if labels are provided
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Reset gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(save_dir, f"blip2_lora_epoch_{epoch+1}.pth")
        
        # Save ONLY LoRA weights to keep file small
        # model.save_pretrained(save_dir) handles this for PEFT models naturally
        # But here we are saving a .pth of the wrapper sometimes.
        # Let's use the native PEFT save method for best practice.
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_save_dir)
        print(f"Saved LoRA weights to {epoch_save_dir}")
        
    return history
