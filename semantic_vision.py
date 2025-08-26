"""
Semantic Vision — Zero-Shot Image Classifier using CLIP
CLI-only, CPU-friendly, customizable labels
"""

import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. Load model and processor

MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cpu"

print("Loading model... this may take a few seconds")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Model loaded. Semantic Vision ready!")

# 2. Helper function to load image

def load_image(path_or_url):
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            resp = requests.get(path_or_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
# 3. Helper function to predict labels

def predict(image, labels):
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    label_probs = list(zip(labels, probs))
    label_probs.sort(key=lambda x: x[1], reverse=True)
    return label_probs

# 4. Main CLI loop

print("\nType an image path or URL (or 'exit' to quit).")

while True:
    inp = input("\nEnter image path or URL: ").strip()
    if inp.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    img = load_image(inp)
    if img is None:
        continue

    labels_input = input("Enter comma-separated labels (or press Enter for defaults): ").strip()
    if labels_input == "":
        labels = ["cat", "dog", "car", "person", "food", "tree"]
    else:
        labels = [l.strip() for l in labels_input.split(",") if l.strip()]
        if not labels:
            print("No valid labels provided. Using default labels.")
            labels = ["cat", "dog", "car", "person", "food", "tree"]

    # Predict
    results = predict(img, labels)

    # Display results
    print("\nResults:")
    for label, prob in results:
    print(f"{label}: {prob:.2%}")
    best_label, best_prob = results[0]
    print(f"\nPrediction: {best_label} ({best_prob:.2%})")
