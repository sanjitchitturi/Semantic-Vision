# Semantic Vision

A small, focused CLI tool for **zero-shot image classification** using OpenAI CLIP. Give it an image, tell it some labels, and it will rank how likely each label is for the image — no training required.

---

## Key features

- Zero-shot classification with custom labels.
- Prompt templates.  
- Interactive mode or single-shot CLI mode.  
- Batch/directory processing and optional CSV export.  
- Auto device selection.  
- CPU-friendly defaults and a single-file script.

---

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.\.venv\Scripts\activate         # Windows (PowerShell)
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` lists `torch`, but for best performance on GPU you should install the `torch` wheel that matches your CUDA version. See [https://pytorch.org](https://pytorch.org) for the appropriate install command.

3. Run the CLI:

* Interactive mode:

  ```bash
  python semantic_vision.py
  ```

* One-shot (single image):

  ```bash
  python semantic_vision.py --input "https://example.com/photo.jpg" \
    --labels "cat,dog,truck" --templates "a photo of a {},a drawing of a {}" --topk 3
  ```

* Batch (directory):

  ```bash
  python semantic_vision.py --input ./images_folder --csv results.csv --topk 5
  ```

---

## Examples

Interactive session example:

```
Image path / URL / directory: > https://images.example/cat.jpg
Labels (comma-separated, Enter for defaults): > cat,dog,fox
Templates (comma-separated, use '{}' for label, Enter for defaults): > a photo of a {},a close-up of a {}
Top-k to show (Enter for all): > 3

Image: https://images.example/cat.jpg
  cat: 95.32%
  dog: 2.01%
  fox: 0.45%
Prediction: cat (95.32%)
```

CLI example that writes CSV:

```bash
python semantic_vision.py --input "./sample_images" --labels "cat,dog,person" --csv out.csv --topk 3
```

---

## Recommended settings & tips

* **Templates:** Use 1–3 templates for CPU runs. Good defaults:

  * `a photo of a {}`
  * `a picture of a {}`
  * `there is a {} in this image`
* **Labels:** Keep label lists to a reasonable size on CPU (dozens, not thousands). For large label sets, consider batching or using fewer templates.
* **GPU:** If you have CUDA and want faster inference, install the correct `torch` + CUDA wheel from pytorch.org before installing other requirements.
* **Threads:** You can set CPU threads with `--threads` (or call `torch.set_num_threads(...)`) to tune performance.
* **CSV output:** Use `--csv output.csv` to save results. The CSV contains image paths and ranked labels/scores.

---

## CLI reference

```
usage: semantic_vision.py [-h] [--input INPUT] [--labels LABELS] [--templates TEMPLATES]
                          [--topk TOPK] [--csv CSV] [--device {auto,cpu,cuda}] [--threads THREADS]
```

* `--input, -i` : Image path / URL / directory. If omitted, starts interactive mode.
* `--labels, -l` : Comma-separated labels (defaults are provided).
* `--templates, -t` : Comma-separated templates using `{}` as placeholder for the label.
* `--topk` : Show top-k labels only.
* `--csv` : Path to write results as CSV.
* `--device` : `auto`, `cpu`, or `cuda`. `auto` uses CUDA if available.
* `--threads` : (Optional) Set number of CPU threads for torch.

---

## Troubleshooting

* **Model download hangs**: The first run downloads CLIP weights — ensure you have internet access and enough disk space.
* **Slow on CPU**: Reduce templates and number of labels. Use fewer CPU threads if the system is overloaded.
* **Unsupported image formats**: Supported extensions include `.jpg .jpeg .png .bmp .gif .tiff`. Convert other formats to one of these.
* **CUDA errors**: Make sure the `torch` version matches your CUDA driver. Reinstall the appropriate wheel from pytorch.org.

---
