import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown


# config
BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = BASE_DIR / "images"
MODEL_PATH = "chandra-ocr-2/"

BATCH_SIZE = 8
PROMPT_TYPE = "ocr_layout"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@dataclass
class OCRResult:
    image_name: str
    markdown: Optional[str]
    error: Optional[str] = None

def load_model(model_path: str):
    """
    Load Chandra OCR model and processor.
    """

    print("[INFO] Loading model...")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    model.processor = processor

    print("[INFO] Model loaded successfully.")

    return model


# load all images
def get_image_paths(images_dir: Path) -> List[Path]:
    """
    Collect all valid image paths from directory.
    """

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted([
        path for path in images_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        raise ValueError(f"No valid images found in: {images_dir}")

    return image_paths


def load_image(image_path: Path) -> Image.Image:
    """
    Load image safely.
    """

    try:
        image = Image.open(image_path).convert("RGB")
        return image

    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path.name}: {e}")


# preparation of batches 
def create_batch(image_paths: List[Path]) -> List[BatchInputItem]:
    """
    Create Chandra batch input.
    """

    batch = []

    for image_path in image_paths:
        image = load_image(image_path)

        batch.append(
            BatchInputItem(
                image=image,
                prompt_type=PROMPT_TYPE
            )
        )

    return batch


def chunk_list(items: List, chunk_size: int):
    """
    Yield chunks from list.
    """

    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


#inference 
def run_batch_inference(
    model,
    image_paths: List[Path]
) -> List[OCRResult]:
    """
    Run OCR inference on batch of images.
    """

    results = []

    try:
        batch = create_batch(image_paths)

        print(f"[INFO] Running inference on batch of {len(batch)} images...")

        raw_outputs = generate_hf(batch, model)

        for image_path, output in zip(image_paths, raw_outputs):

            try:
                markdown = parse_markdown(output.raw)

                results.append(
                    OCRResult(
                        image_name=image_path.name,
                        markdown=markdown
                    )
                )

            except Exception as e:
                results.append(
                    OCRResult(
                        image_name=image_path.name,
                        markdown=None,
                        error=f"Markdown parsing failed: {e}"
                    )
                )

    except Exception as e:

        for image_path in image_paths:
            results.append(
                OCRResult(
                    image_name=image_path.name,
                    markdown=None,
                    error=str(e)
                )
            )

    return results

def save_results(results: List[OCRResult], output_dir: Path):
    """
    Save OCR outputs to markdown files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:

        output_file = output_dir / f"{Path(result.image_name).stem}.md"

        if result.error:
            content = f"# ERROR\n\n{result.error}"

        else:
            content = result.markdown

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"[INFO] Results saved to: {output_dir}")

def main():

    print("=" * 60)
    print("CHANDRA OCR BATCH INFERENCE")
    print("=" * 60)

    model = load_model(MODEL_PATH)

    image_paths = get_image_paths(IMAGES_DIR)

    print(f"[INFO] Found {len(image_paths)} images.")

    all_results = []

    for batch_idx, batch_paths in enumerate(
        chunk_list(image_paths, BATCH_SIZE),
        start=1
    ):

        print("\n" + "-" * 60)
        print(f"[INFO] Processing batch {batch_idx}")
        print("-" * 60)

        batch_results = run_batch_inference(model, batch_paths)

        all_results.extend(batch_results)

        for result in batch_results:

            if result.error:
                print(f"[ERROR] {result.image_name}: {result.error}")

            else:
                print(f"[SUCCESS] {result.image_name}")

    # Save outputs
    output_dir = BASE_DIR / "outputs"
    save_results(all_results, output_dir)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
