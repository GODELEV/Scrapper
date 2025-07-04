# Scrapper Mk1

[![Author: GODELEV](https://img.shields.io/badge/GitHub-GODELEV-blue?logo=github)](https://github.com/GODELEV)

## Overview
Scrapper Mk1 is a CLI tool for building custom image-caption datasets from popular Hugging Face sources. It allows you to search, filter, download, optionally caption, and upload datasets to the Hugging Face Hub with ease.

## Features
- **Prompt Expansion:** Expands your search prompt into multiple semantically related keywords for broader dataset coverage.
- **Dataset Filtering:** Filters large datasets (e.g., LAION, Conceptual Captions, Unsplash) for images matching your keywords.
- **Parallel Image Downloading:** Downloads images in parallel with retry logic and optional resizing.
- **Caption Generation (BLIP):** Optionally generates captions for images using BLIP (if installed).
- **Dataset Formatting:** Outputs `captions.txt`, `dataset.json`, and a summary of the dataset.
- **Hugging Face Upload:** Uploads your dataset to the Hugging Face Hub directly from the CLI.
- **Interactive CLI:** Guides you through all steps, including prompt, dataset source, image count, and upload options.

## Requirements
- Python 3.7+
- `datasets`, `huggingface_hub`, `tqdm`, `requests`, `Pillow`, `torch`
- (Optional for BLIP captioning) `transformers`, `Salesforce/blip-image-captioning-base`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Run the main script:**
   ```bash
   python scrapper.py
   ```
2. **Follow the prompts:**
   - Enter your topic/prompt (e.g., "sunset beach").
   - Choose a dataset source (Conceptual Captions, LAION, Unsplash).
   - Specify the number of images to download.
   - Select image size (or keep original).
   - Choose output folder (default: `output/scrapper-dataset-001`).
   - Optionally generate captions with BLIP if available.
   - Optionally upload the dataset to your Hugging Face account.

3. **Upload to Hugging Face (standalone):**
   - You can also use `upload_to_hf.py` to upload any dataset folder interactively.
   ```bash
   python upload_to_hf.py
   ```

## Output
- `images/` — Downloaded images
- `captions.txt` — Tab-separated image-caption pairs
- `dataset.json` — List of `{image, text}` objects
- `summary.txt` — Dataset summary and stats
- `README.md` — Auto-generated dataset README

## Example Workflow
```
$ python scrapper.py
=== Scrapper Mk1: Hugging Face Dataset Builder ===
Enter your topic/prompt: cats in hats
Expanding prompt...
Expanded keywords: ['cats in hats', 'A photo of cats in hats', ...]
Choose Hugging Face dataset source:
  1. conceptual_captions
  2. laion/laion400m
  3. unsplash-dataset
Enter number (1-3): 2
How many images do you want? 50
Choose image size/quality:
  1. 64x64
  2. 128x128
  3. 192x192
  4. 256x256
  5. 512x512
  6. No resize (original size)
Enter number (1-6): 4
Output folder name (default: output/scrapper-dataset-001): my-cats-dataset
Loading dataset 'laion/laion400m' (streaming)...
Filtering dataset ...
Downloading images ...
Dataset already contains captions. Skipping BLIP captioning.
Do you want to upload this dataset to your HF account? (yes/no): yes
Enter Hugging Face dataset name (e.g., 'username/cats-in-hats-v1'): godelev/cats-in-hats-v1
Uploading to Hugging Face Datasets Hub...
Uploaded! View at: https://huggingface.co/datasets/godelev/cats-in-hats-v1
All done! Dataset saved to: my-cats-dataset
```

## Typical Time Estimates

| Images Requested | Filtering Only | Filtering + Downloading | + BLIP Captioning |
|------------------|---------------|-------------------------|-------------------|
| 10               | < 1 min       | 1-2 min                 | 2-5 min           |
| 100              | 1-3 min       | 3-10 min                | 10-30 min         |
| 1000             | 5-20 min      | 20-60+ min              | 1-2+ hours        |

- **Filtering**: Fast if keywords are common, slow if rare.
- **Downloading**: Each image can take 0.5–2 seconds (network dependent).
- **BLIP Captioning**: Slowest step if enabled, especially on CPU.

## Author
- GitHub: [GODELEV](https://github.com/GODELEV)

## License
See [LICENSE](LICENSE) for details. 