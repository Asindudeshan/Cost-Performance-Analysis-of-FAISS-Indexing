import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def main():
    device = "cpu"
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print("Model/processor load failed, skipping script:", e)
        return

    img = Image.new('RGB', (224, 224))
    inputs = processor(images=[img], return_tensors="pt")
    features = model.get_image_features(**inputs)
    print("Type of features:", type(features))
    # `get_image_features` may return a Tensor or a model output depending on
    # transformers version. Handle both safely and avoid crashing on shape mismatch
    if isinstance(features, torch.Tensor):
        print("Features tensor shape:", features.shape)
    else:
        if hasattr(features, "pooler_output"):
            pooler = features.pooler_output
            print("Pooler shape:", pooler.shape)
            # Attempt projection only if shapes align; guard against runtime errors
            if hasattr(model, "visual_projection") and isinstance(pooler, torch.Tensor):
                try:
                    proj = model.visual_projection(pooler)
                    print("Projected shape:", proj.shape)
                except Exception as e:
                    print("Projection skipped or failed:", e)


if __name__ == "__main__":
    main()
