"""
Fine-tune BLIP-2 for German Food Recognition
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json

class GermanFoodDataset(Dataset):
    """Dataset for German food images with ingredient labels."""
    
    def __init__(self, data_dir="german_food_dataset", processor=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.samples = []
        
        # German dishes with their ingredients
        self.dish_ingredients = {
            'schnitzel': 'pork, breadcrumbs, lemon, butter',
            'bratwurst': 'sausage, mustard, bread roll',
            'sauerbraten': 'beef, red cabbage, gravy, dumplings',
            'schweinshaxe': 'pork knuckle, crispy skin, potatoes',
            'currywurst': 'sausage, curry sauce, french fries',
            'rouladen': 'beef rolls, bacon, pickles, onions, mustard',
            'kassler': 'smoked pork, sauerkraut, potatoes',
            'maultaschen': 'pasta pockets, meat, spinach, onions',
            'kartoffelpuffer': 'potato pancakes, potatoes, eggs, flour',
            'bratkartoffeln': 'fried potatoes, onions, bacon',
            'kartoffelkloesse': 'potato dumplings, potatoes, flour',
            'kartoffelsalat': 'potato salad, potatoes, vinegar, onions',
            'brezel': 'pretzel, dough, salt, butter',
            'leberkase': 'meatloaf, beef, pork, spices',
            'weisswurst': 'white sausage, veal, pork, parsley',
            'eintopf': 'stew, vegetables, meat, potatoes, broth',
            'gulaschsuppe': 'goulash soup, beef, paprika, onions',
            'kartoffelsuppe': 'potato soup, potatoes, vegetables, sausage',
            'sauerkraut': 'fermented cabbage, caraway seeds',
            'rotkohl': 'red cabbage, apples, vinegar, spices',
            'spaetzle': 'egg noodles, eggs, flour, butter',
            'knodel': 'bread dumplings, bread, eggs, parsley',
            'schwarzwalder_kirschtorte': 'chocolate cake, cherries, cream, kirsch',
            'apfelstrudel': 'apple strudel, apples, cinnamon, raisins, pastry',
            'bienenstich': 'bee sting cake, almonds, custard, honey',
            'kaiserschmarrn': 'shredded pancake, eggs, flour, raisins, sugar',
            'lebkuchen': 'gingerbread, spices, nuts, honey',
            'stollen': 'christmas bread, dried fruits, marzipan, butter',
        }
        
        # Load all images
        for dish_dir in self.data_dir.iterdir():
            if dish_dir.is_dir():
                dish_name = dish_dir.name
                ingredients = self.dish_ingredients.get(dish_name, 'german food')
                
                for img_path in dish_dir.glob('*.jpg'):
                    self.samples.append({
                        'image_path': str(img_path),
                        'dish_name': dish_name,
                        'ingredients': ingredients
                    })
        
        print(f"✅ Loaded {len(self.samples)} images from {len(self.dish_ingredients)} dishes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Create prompt and answer
        prompt = "Question: What are the individual food items and ingredients you can see in this image? List each ingredient separately. Answer:"
        answer = sample['ingredients']

        return {
            'image': image,
            'prompt': prompt,
            'answer': answer,
            'dish_name': sample['dish_name']
        }


def create_collate_fn(processor):
    """Create a collate function with the processor."""
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences."""
        # Extract images, prompts, and answers
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        answers = [item['answer'] for item in batch]

        # Process images and prompts together
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        )

        # Process answers (labels)
        labels = processor.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            max_length=64,
            truncation=True
        ).input_ids

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': inputs.pixel_values,
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels
        }

    return collate_fn


def setup_lora_model(model_name="Salesforce/blip2-opt-2.7b", device='cuda'):
    """Setup BLIP-2 with LoRA for efficient fine-tuning."""
    print(f"\n🔧 Setting up BLIP-2 with LoRA...")
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Load processor
    processor = Blip2Processor.from_pretrained(model_name)

    # Load model - don't use device_map="auto" to avoid meta device issues
    if device == 'cuda':
        print("Loading model to CUDA...")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        # Move model to GPU explicitly
        model = model.to(device)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Ensure model is on correct device and in training mode
    model = model.to(device)
    model.train()
    print(f"✅ Model loaded on {device}")

    return model, processor


def train_german_food(
    data_dir="german_food_dataset",
    output_dir="blip2_german_food_lora",
    epochs=3,
    batch_size=4,
    learning_rate=1e-4,
    device='cuda',
    gradient_accumulation_steps=1
):
    """Fine-tune BLIP-2 on German food dataset."""
    print("=" * 70)
    print("🇩🇪 BLIP-2 Fine-Tuning for German Food")
    print("=" * 70)
    
    # Setup model
    model, processor = setup_lora_model(device=device)

    # Load dataset
    dataset = GermanFoodDataset(data_dir, processor)
    # Create collate function with processor
    collate_fn = create_collate_fn(processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    print(f"\n🚀 Starting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total batches per epoch: {len(dataloader)}")

    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / gradient_accumulation_steps  # Scale loss
            epoch_loss += loss.item() * gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

        # Final optimizer step if there are remaining gradients
        if len(dataloader) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
    
    # Save model
    print(f"\n💾 Saving fine-tuned model to {output_dir}...")
    Path(output_dir).mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("\n" + "=" * 70)
    print("🎉 Fine-tuning complete!")
    print("=" * 70)
    print(f"Model saved to: {output_dir}")
    print(f"To use: Load with Blip2ForConditionalGeneration.from_pretrained('{output_dir}')")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune BLIP-2 for German food')
    parser.add_argument('--data_dir', default='german_food_dataset', help='Dataset directory')
    parser.add_argument('--output_dir', default='blip2_german_food_lora', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    train_german_food(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

