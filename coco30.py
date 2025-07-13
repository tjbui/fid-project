import os
import time

os.environ["HF_HOME"] = "/scratch/gilbreth/bui46/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/gilbreth/bui46/huggingface_cache/datasets"

from datasets import load_dataset
from PIL import Image


# Use Gilbreth scratch directory for cache
scratch_cache_dir = "/scratch/gilbreth/bui46/huggingface_cache/datasets"

# Load COCO-30 from HuggingFace (with fallback retries)
try:
    print("Attempting to load COCO dataset from scratch cache...")
    ds = load_dataset("sayakpaul/coco-30-val-2014", cache_dir=scratch_cache_dir)
    print("✅ Successfully loaded COCO dataset from scratch cache!")
except:
    print("⚠️ Cache not available, trying direct download...")
    try:
        ds = load_dataset("sayakpaul/coco-30-val-2014", cache_dir=scratch_cache_dir)
        print("✅ Successfully downloaded COCO dataset to scratch!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("⏳ Retrying after 60 seconds...")
        time.sleep(60)
        try:
            ds = load_dataset("sayakpaul/coco-30-val-2014", cache_dir=scratch_cache_dir)
            print("✅ Download succeeded on retry.")
        except:
            print("❌ Still failed. Try again later when rate limit resets.")
            quit()

# Pull the 'train' set and access one image
trainset = ds['train']
imageset = trainset[0]
image = imageset['image']

# Confirm it is a valid PIL image
if isinstance(image, Image.Image):
    print("✅ Successfully loaded one image from COCO-30")
else:
    print("❌ Error: Not a valid PIL image")

print(f"Dataset shape: {trainset.shape}")
