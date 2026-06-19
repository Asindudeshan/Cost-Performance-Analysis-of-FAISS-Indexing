import os
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer

# ------------------------------------------------------------
# 1. Load a small image‑text dataset (example: Flickr8k)
# ------------------------------------------------------------
# Replace "flickr8k" with your own dataset name or local path.
# The dataset should have columns: "image" (file path) and "caption" (text).

# Load a small image‑text dataset (example: COCO Captions)

try:
    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
except Exception as e:
    print(f"Dataset load failed ({e}), falling back to synthetic data.")
    from datasets import Dataset
    from PIL import Image as PilImage
    # Create a single dummy example
    dummy_image = PilImage.new("RGB", (224, 224), color="gray")
    dummy_data = {"image": [dummy_image], "text": ["a placeholder caption"]}
    dataset = Dataset.from_dict(dummy_data)

# For quick debugging/run use a subset:
dataset = dataset.select(range(min(200, len(dataset))))

# ------------------------------------------------------------
# 2. Initialise CLIP model and processor
# ------------------------------------------------------------
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------------------------------------
# 3. Pre‑process function – returns dict with pixel values & input ids
# ------------------------------------------------------------
def preprocess(example):
    # Load image (PIL) from local path – dataset provides path relative to download folder
    image = example["image"]  # already a PIL Image in many HF datasets
    text = example.get("caption", example.get("text", "a placeholder caption"))
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    # Move tensors to GPU/CPU now (they will be moved again in collate_fn, but this is fine)
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    return inputs

# Apply preprocessing – this creates new columns "pixel_values" and "input_ids"
processed = dataset.map(preprocess, remove_columns=dataset.column_names)
processed.set_format(type="torch")

# ------------------------------------------------------------
# 4. Custom data collator – batches the dicts correctly
# ------------------------------------------------------------
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    res = {"pixel_values": pixel_values, "input_ids": input_ids}
    if "attention_mask" in batch[0]:
        res["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    return res

# ------------------------------------------------------------
# 5. Define a simple training loop using HuggingFace Trainer
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./clip_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",  # disables wandb / comet by default
)

# The Trainer expects the model to return a loss when called with pixel_values & input_ids.
# For CLIP we can compute contrastive loss using the built‑in method.
class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass – model returns loss internally when both modalities are supplied
        outputs = model(**inputs)
        # outputs.loss is the contrastive loss (image‑text matching)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CLIPTrainer(
    model=model,
    args=training_args,
    train_dataset=processed,
    data_collator=collate_fn,
)
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed. Model saved to ./clip_finetuned")

# ------------------------------------------------------------
# 6. (Optional) Push the fine‑tuned model to the HuggingFace Hub
# ------------------------------------------------------------
# model.save_pretrained("./clip_finetuned")
# processor.save_pretrained("./clip_finetuned")
# Uncomment the lines above and run `huggingface-cli login` if you want to upload.
