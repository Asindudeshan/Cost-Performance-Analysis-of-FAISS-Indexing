import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = Image.new('RGB', (224, 224))
inputs = processor(images=[img], return_tensors="pt")
features = model.get_image_features(**inputs)
print(type(features))
if not isinstance(features, torch.Tensor):
    print(dir(features))
