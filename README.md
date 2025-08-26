# Semantic Vision

**Semantic Vision** is a lightweight, AI-powered **zero-shot image classifier** that interprets images and predicts their content using natural-language labels. Built with **OpenAI CLIP** and Python, this project demonstrates how multimodal AI (vision + language) works in practice — no GPU required.

---

## Features

* Classify images with **custom labels** or default ones (`cat`, `dog`, `car`, `person`, `food`, `tree`)
* Accepts **local image paths** or **online image URLs**
* **CLI-only**, lightweight, and runs on CPU
* Shows **probabilities** for each label, highlighting the top prediction
* Fully **zero-shot** — no model training required

---

## Demo

```
$ python semantic_vision.py
Semantic Vision — Zero-Shot Image Classifier
Type an image path or URL (or 'exit' to quit).

Enter image path or URL: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png
Enter comma-separated labels (or press Enter for defaults):

Results:
cat: 95.33%
dog: 1.22%
car: 0.55%
person: 2.90%
food: 0.00%
tree: 0.00%

Prediction: cat (95.33%)
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/sanjitchitturi/Semantic-Vision.git
cd Semantic-Vision
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the CLI script:

```bash
python semantic_vision.py
```

1. **Enter an image path** (local) or **image URL**.
2. **Enter comma-separated labels** to classify (or press Enter for defaults).
3. The script outputs probabilities for each label and highlights the top prediction.
4. Type `exit` or `quit` to close the program.

---

## Tech Stack

* **Python 3**
* **Transformers** (Hugging Face CLIP model: `openai/clip-vit-base-patch32`)
* **PyTorch** (CPU version)
* **Pillow** for image processing
* **Requests** for fetching online images

---
