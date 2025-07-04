import os
import sys
import json
import time
import random
import shutil
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, upload_folder
import torch
from PIL import Image, UnidentifiedImageError

# Optional: BLIP imports
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

# -----------------------------
# Utility Functions
# -----------------------------
def expand_keywords(prompt: str) -> List[str]:
    """
    Expand the user prompt into 5-10 semantically related phrases.
    Uses Hugging Face inference API for paraphrasing if available, else basic expansion.
    """
    try:
        from transformers import pipeline
        paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
        paraphrases = paraphraser(f"paraphrase: {prompt}", num_return_sequences=8, max_length=64)
        keywords = []
        if isinstance(paraphrases, list):
            for p in paraphrases:
                if isinstance(p, dict) and 'generated_text' in p and isinstance(p['generated_text'], str):
                    keywords.append(p['generated_text'])
        else:
            # fallback if paraphrases is not a list
            raise ValueError("Paraphraser did not return a list")
        keywords.append(prompt)
        return list(set(keywords))[:10]
    except Exception:
        # Fallback: simple rewordings
        base = prompt.lower()
        words = base.split()
        expansions = [
            base,
            f"{words[-1]} {' '.join(words[:-1])}" if len(words) > 1 else base,
            f"A photo of {base}",
            f"An image showing {base}",
            f"{base} scenery",
            f"{base} landscape",
            f"{base} view",
            f"{base} in nature",
            f"{base} outdoors",
        ]
        return list(set(expansions))[:8]


def filter_dataset(dataset, keywords: List[str], max_count: int) -> List[Dict]:
    """
    Filter the dataset for captions containing any of the expanded keywords.
    Returns up to max_count unique image-caption pairs.
    """
    pairs = []
    seen_urls = set()
    pbar = tqdm(total=max_count, desc="Filtering dataset", unit="img")
    for item in dataset:
        caption = str(item.get('caption') or item.get('text') or "").lower()
        url = item.get('url') or item.get('URL') or item.get('image_url')
        if not url or url in seen_urls:
            continue
        if any(kw.lower() in caption for kw in keywords):
            pairs.append({'url': url, 'caption': caption})
            seen_urls.add(url)
            pbar.update(1)
        if len(pairs) >= max_count:
            break
    pbar.close()
    return pairs


def download_image(url: str, path: str, retries: int = 2) -> bool:
    """
    Download a single image with retries. Returns True if successful.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.content:
                with open(path, 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
    return False


def download_images(pairs: List[Dict], output_dir: str, image_size: Optional[int] = None) -> Tuple[List[Dict], int]:
    """
    Download images in parallel, skipping broken URLs. Optionally resize. Returns list of successful pairs and count.
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    results = []
    pbar = tqdm(total=len(pairs), desc="Downloading images", unit="img")
    lock = threading.Lock()

    def task(idx_pair):
        idx, pair = idx_pair
        img_path = os.path.join(images_dir, f"{idx+1:06d}.jpg")
        ok = download_image(pair['url'], img_path)
        if ok:
            # Resize if needed
            if image_size is not None:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    img.save(img_path, format='JPEG')
                except Exception as e:
                    print(f"Resize failed for {img_path}: {e}")
            with lock:
                results.append({'image': f"images/{idx+1:06d}.jpg", 'text': pair['caption']})
        pbar.update(1)
        return ok

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(task, enumerate(pairs)))
    pbar.close()
    return results, len(results)


def generate_captions_blip(image_paths: List[str], output_dir: str, use_gpu: bool) -> List[Dict]:
    """
    Generate captions for images using BLIP. Overwrites captions.
    """
    if not BLIP_AVAILABLE:
        print("BLIP not installed. Skipping captioning.")
        return []
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(device)
    new_pairs = []
    pbar = tqdm(image_paths, desc="BLIP captioning", unit="img")
    for img_path in pbar:
        try:
            img = Image.open(os.path.join(output_dir, img_path)).convert('RGB')
            inputs = processor(img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            new_pairs.append({'image': img_path, 'text': caption})
        except (UnidentifiedImageError, ValueError, OSError) as e:
            print(f"Skipping {img_path}: {e}")
    return new_pairs


def write_captions_txt(pairs: List[Dict], output_dir: str):
    """
    Write captions.txt (image file <TAB> caption).
    """
    with open(os.path.join(output_dir, 'captions.txt'), 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(f"{pair['image']}\t{pair['text']}\n")


def write_dataset_json(pairs: List[Dict], output_dir: str):
    """
    Write dataset.json (list of {image, text}).
    """
    with open(os.path.join(output_dir, 'dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


def write_summary(
    output_dir: str,
    prompt: str,
    keywords: List[str],
    source: str,
    requested: int,
    downloaded: int,
    blip_status: str,
    upload_link: str,
    elapsed: float,
    image_size_str: str
):
    """
    Write summary.txt with all required info.
    """
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Original prompt: {prompt}\n")
        f.write(f"Expanded keywords: {', '.join(keywords)}\n")
        f.write(f"HF source: {source}\n")
        f.write(f"Images requested: {requested}\n")
        f.write(f"Images downloaded: {downloaded}\n")
        f.write(f"Image size: {image_size_str}\n")
        f.write(f"BLIP captioning: {blip_status}\n")
        f.write(f"Upload link: {upload_link}\n")
        f.write(f"Time taken: {elapsed:.2f} seconds\n")
        f.write(f"Author: Akshit (2025)\n")
        f.write(f"Generated by Scrapper Mk1\n")


def write_readme(output_dir: str, prompt: str, keywords: List[str], source: str, blip_status: str, count: int, image_size_str: str):
    """
    Write a simple README.md for Hugging Face.
    """
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"# Scrapper Dataset\n\n")
        f.write(f"- **Prompt:** {prompt}\n")
        f.write(f"- **Expanded keywords:** {', '.join(keywords)}\n")
        f.write(f"- **Source:** {source}\n")
        f.write(f"- **BLIP captioning:** {blip_status}\n")
        f.write(f"- **Total images:** {count}\n")
        f.write(f"- **Image size:** {image_size_str}\n\n")
        f.write(f"Generated by [Scrapper Mk1](ttps://github.com/GODELEV)\n")


def upload_to_huggingface(repo_name: str, output_dir: str) -> str:
    """
    Upload the dataset folder to Hugging Face Hub. Returns the repo URL.
    """
    api = HfApi()
    user = api.whoami()['name']
    repo_id = f"{user}/{repo_name}"
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        repo_type="dataset",
        commit_message="Add dataset from Scrapper Mk1"
    )
    return f"https://huggingface.co/datasets/{repo_id}"

# -----------------------------
# Main CLI Logic
# -----------------------------
def main():
    print("\n=== Scrapper Mk1: Hugging Face Dataset Builder ===\n")
    prompt = input("Enter your topic/prompt: ").strip()
    print("\nExpanding prompt...")
    keywords = expand_keywords(prompt)
    print(f"Expanded keywords: {keywords}\n")

    print("Choose Hugging Face dataset source:")
    sources = ["conceptual_captions", "laion/laion400m", "unsplash-lite"]
    for i, src in enumerate(sources, 1):
        print(f"  {i}. {src}")
    while True:
        try:
            src_idx = int(input("Enter number (1-3): ")) - 1
            if 0 <= src_idx < len(sources):
                source = sources[src_idx]
                break
        except Exception:
            pass
        print("Invalid input. Try again.")

    while True:
        try:
            n_images = int(input("How many images do you want? "))
            if n_images > 0:
                break
        except Exception:
            pass
        print("Please enter a positive integer.")

    # Ask for image size/quality
    print("\nChoose image size/quality:")
    sizes = [64, 128, 192, 256, 512]
    for i, sz in enumerate(sizes, 1):
        print(f"  {i}. {sz}x{sz}")
    print(f"  {len(sizes)+1}. No resize (original size)")
    while True:
        try:
            size_choice = int(input(f"Enter number (1-{len(sizes)+1}): "))
            if 1 <= size_choice <= len(sizes)+1:
                break
        except Exception:
            pass
        print("Invalid input. Try again.")
    if size_choice == len(sizes)+1:
        image_size = None
        image_size_str = "No resize (original size)"
    else:
        image_size = sizes[size_choice-1]
        image_size_str = f"{image_size}x{image_size}"

    out_dir = input("Output folder name (default: output/scrapper-dataset-001): ").strip() or "output/scrapper-dataset-001"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading dataset '{source}' (streaming)...")
    ds = load_dataset(source, split="train", streaming=True)
    pairs = filter_dataset(ds, keywords, n_images)
    print(f"\nFiltered {len(pairs)} image-caption pairs.")

    # Check if dataset already has captions
    has_captions = all(pair.get('caption') or pair.get('text') for pair in pairs)
    blip_status = "disabled"
    t0 = time.time()

    print("\nDownloading images...")
    downloaded_pairs, n_downloaded = download_images(pairs, out_dir, image_size)
    print(f"Downloaded {n_downloaded} images.")

    # Only prompt for BLIP if captions are missing
    if not has_captions and BLIP_AVAILABLE:
        blip = input("Dataset lacks captions. Do you want to generate captions using BLIP? (yes/no): ").strip().lower()
        if blip.startswith('y'):
            use_gpu = torch.cuda.is_available()
            blip_status = f"enabled ({'GPU' if use_gpu else 'CPU'})"
            print(f"\nGenerating captions with BLIP on {'GPU' if use_gpu else 'CPU'}...")
            image_paths = [pair['image'] for pair in downloaded_pairs]
            new_pairs = generate_captions_blip(image_paths, out_dir, use_gpu)
            if new_pairs:
                downloaded_pairs = new_pairs
    elif has_captions:
        print("\nDataset already contains captions. Skipping BLIP captioning.")
    elif not BLIP_AVAILABLE:
        print("BLIP not available. Skipping captioning.")

    write_captions_txt(downloaded_pairs, out_dir)
    write_dataset_json(downloaded_pairs, out_dir)
    write_readme(out_dir, prompt, keywords, source, blip_status, len(downloaded_pairs), image_size_str)

    upload_link = "-"
    upload = input("Do you want to upload this dataset to your HF account? (yes/no): ").strip().lower()
    if upload.startswith('y'):
        repo_name = input("Enter Hugging Face dataset name (e.g., 'shakti-dataset-v1'): ").strip()
        print("Uploading to Hugging Face Datasets Hub...")
        try:
            upload_link = upload_to_huggingface(repo_name, out_dir)
            print(f"Uploaded! View at: {upload_link}")
        except Exception as e:
            print(f"Upload failed: {e}")
            upload_link = "Upload failed"

    elapsed = time.time() - t0
    write_summary(
        out_dir, prompt, keywords, source, n_images, n_downloaded, blip_status, upload_link, elapsed, image_size_str
    )
    print(f"\nAll done! Dataset saved to: {out_dir}\n")

if __name__ == "__main__":
    main() 