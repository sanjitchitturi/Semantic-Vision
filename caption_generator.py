from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    print("AI Image Caption Generator Ready! Type an image path/URL or 'exit'.\n")
    while True:
        img = input("Enter image path or URL: ").strip()
        if img.lower() in {"exit", "quit"}:
            break
        try:
            caption = generate_caption(img)
            print(f"\n🖼 Caption: {caption}\n")
        except Exception as e:
            print("Error:", e)
