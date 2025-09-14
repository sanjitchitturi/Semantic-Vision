from __future__ import annotations

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_LABELS = ["cat", "dog", "car", "person", "food", "tree"]
DEFAULT_TEMPLATES = ["a photo of a {}", "a picture of a {}", "there is a {} in this image"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))

def load_image(path_or_url: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        if is_url(path_or_url):
            r = requests.get(path_or_url, timeout=timeout)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        return Image.open(path_or_url).convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
        print(f"[error] failed to load '{path_or_url}': {exc}", file=sys.stderr)
        return None

def discover_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        return files
    if is_url(path):
        return [path]
    return []

def load_model(device_str: str | torch.device = "cpu") -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = torch.device(device_str)
    print(f"Loading model '{MODEL_NAME}' on {device} ...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device

def validate_templates(templates: Sequence[str]) -> List[str]:
    out = [t for t in templates if "{}" in t]
    return out if out else DEFAULT_TEMPLATES

def format_labels(labels_text: Optional[str]) -> List[str]:
    if not labels_text:
        return DEFAULT_LABELS.copy()
    labels = [l.strip() for l in labels_text.split(",") if l.strip()]
    return labels or DEFAULT_LABELS.copy()

def make_prompts(labels: Sequence[str], templates: Sequence[str]) -> List[str]:
    return [tpl.format(lbl) for lbl in labels for tpl in templates]

def predict_for_image(
    image: Image.Image,
    labels: Sequence[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    templates: Sequence[str],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    prompts = make_prompts(labels, templates)
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # shape (1, n_texts)
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    n_labels = len(labels)
    n_templates = len(templates)
    if probs.size != n_labels * n_templates:
        raise RuntimeError("unexpected logits size from model")

    scores = probs.reshape(n_labels, n_templates).mean(axis=1)
    ranked = sorted(zip(labels, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked

def write_csv(path: str, rows: Iterable[Sequence[str | float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for row in rows:
            w.writerow(row)

def process_images(
    inputs: Sequence[str],
    labels: Sequence[str],
    templates: Sequence[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    top_k: Optional[int],
    csv_out: Optional[str],
) -> None:
    csv_rows: List[List[str | float]] = []
    header_written = False

    for img_path in inputs:
        image = load_image(img_path)
        if image is None:
            continue

        results = predict_for_image(image, labels, model, processor, device, templates, top_k=top_k)
        print(f"\nImage: {img_path}")
        for label, score in results:
            print(f"  {label}: {score:.2%}")

        if results:
            best_label, best_score = results[0]
            print(f"Prediction: {best_label} ({best_score:.2%})")

        if csv_out:
            if not header_written:
                header = ["image", *(f"rank_{i+1}_label" for i in range(len(results))), *(f"rank_{i+1}_score" for i in range(len(results)))]
                csv_rows.append(header)
                header_written = True

            labels_cols = [r[0] for r in results]
            scores_cols = [r[1] for r in results]
            # pad if top_k smaller than other images
            csv_rows.append([img_path, *labels_cols, *scores_cols])

    if csv_out and csv_rows:
        write_csv(csv_out, csv_rows)
        print(f"\nSaved CSV to: {csv_out}")

def interactive_mode(model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            entry = input("Image path / URL / directory: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entry:
            continue
        if entry.lower() in {"exit", "quit"}:
            break

        inputs = discover_images(entry)
        if not inputs:
            print("No images found at that path or URL. Try again.")
            continue

        labels_text = input("Labels (comma-separated, Enter for defaults): ").strip()
        templates_text = input("Templates (comma-separated, use '{}' for label, Enter for defaults): ").strip()
        topk_text = input("Top-k to show (Enter for all): ").strip()

        labels = format_labels(labels_text)
        templates = validate_templates([t.strip() for t in templates_text.split(",")]) if templates_text else DEFAULT_TEMPLATES
        top_k = int(topk_text) if topk_text.isdigit() and int(topk_text) > 0 else None

        process_images(inputs, labels, templates, model, processor, device, top_k, csv_out=None)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic Vision — zero-shot CLIP classifier")
    p.add_argument("--input", "-i", help="Image path / URL / directory (interactive if omitted)")
    p.add_argument("--labels", "-l", help="Comma-separated labels (default preset used when omitted)")
    p.add_argument("--templates", "-t", help="Comma-separated templates using '{}' as placeholder")
    p.add_argument("--topk", type=int, help="Show top-k labels (default: all)")
    p.add_argument("--csv", help="Write results to CSV file")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to use")
    p.add_argument("--threads", type=int, help="Set torch CPU thread count (optional)")
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.threads:
        try:
            torch.set_num_threads(int(args.threads))
        except Exception:
            pass

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device

    model, processor, device_obj = load_model(device)

    if not args.input:
        interactive_mode(model, processor, device_obj)
        return

    inputs: List[str] = []
    # support comma-separated list or a single path/url
    if "," in args.input and not is_url(args.input):
        for part in args.input.split(","):
            part = part.strip()
            inputs.extend(discover_images(part))
    else:
        inputs = discover_images(args.input)

    if not inputs:
        print("No images found. Check --input path or URL.", file=sys.stderr)
        return

    labels = format_labels(args.labels)
    templates = validate_templates([t.strip() for t in args.templates.split(",")]) if args.templates else DEFAULT_TEMPLATES

    process_images(inputs, labels, templates, model, processor, device_obj, top_k=args.topk, csv_out=args.csv)

if __name__ == "__main__":
    main()
