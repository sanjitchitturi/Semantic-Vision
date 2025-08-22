# Semantic Vision  

An **AI-powered image caption generator** that converts pictures into human-like text descriptions using a pre-trained **vision-language transformer** (BLIP).  

This project shows how **multimodal AI** (vision + language) works in practice: you give the model an image, and it generates a natural-language caption like:  

> *“A cat sitting on a laptop keyboard.”*  

---

## Features
- Generate captions for **any image** (local file or online URL)  
- Powered by Hugging Face’s **BLIP (Bootstrapping Language-Image Pretraining)** model  
- Minimal code (only ~40 lines!) → perfect for beginners and portfolio projects  
- Extensible: can be turned into a **web app (Streamlit/Gradio)** or fine-tuned on a custom dataset  

---

## Project Structure
```

Semantic Vision/
│── caption\_generator.py   # Main script to run captioning
│── README.md              # Project documentation

````

---

## Quick Demo

```bash
$ python caption_generator.py
AI Image Caption Generator Ready! Type an image path/URL or 'exit'.

Enter image path or URL: https://huggingface.co/datasets/nateraw/image-captioning/resolve/main/example.jpg

🖼️ Caption: a cat sitting on a laptop keyboard
````

---

## Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/sanjitchitturi/Semantic-Vision.git
cd Semantic-Vision
```

### 2. Install dependencies

```bash
pip install transformers pillow requests
```

### 3. Run the script

```bash
python caption_generator.py
```

---

## How It Works

1. Loads the **BLIP model + processor** from Hugging Face
2. Takes an input image (from a file or URL)
3. Converts the image into embeddings and generates text using the model
4. Outputs a human-readable caption

The magic happens thanks to **vision-language transformers**, which combine image understanding with natural language generation.

---

## Example Captions

| Input Image                                                                                | Generated Caption                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------ |
| ![cat](https://huggingface.co/datasets/nateraw/image-captioning/resolve/main/example.jpg)  | *a cat sitting on a laptop keyboard* |
| ![dog](https://huggingface.co/datasets/nateraw/image-captioning/resolve/main/example2.jpg) | *a dog running across a field*       |

---

## Tech Stack

* **Python 3**
* **Hugging Face Transformers** (`blip-image-captioning-base`)
* **PIL (Pillow)** for image handling
* **Requests** for fetching online images

---

## Future Enhancements

* Add a **web app interface** (Streamlit or Gradio)
* Fine-tune on a **custom dataset** (e.g., domain-specific images)
* Export as a simple **REST API** with FastAPI/Flask

---
