import torch
from transformers import CLIPProcessor, CLIPModel

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text="a red car", return_tensors="pt")
features = model.get_text_features(**inputs)
print("Type of text features:", type(features))
if hasattr(features, "pooler_output"):
    pooler = features.pooler_output
    print("Pooler shape:", pooler.shape)
elif isinstance(features, torch.Tensor):
    print("It's a tensor of shape:", features.shape)
