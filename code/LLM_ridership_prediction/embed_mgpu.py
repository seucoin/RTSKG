import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import torch
import time
import faiss
import pickle

# ================= Configuration =================
# Input file path
INPUT_CSV = "./CHI_data/relation_chi_rtskg_new.csv"
# Column name containing text descriptions
TEXT_COLUMN = "description"
# Column name for IDs
ID_COLUMN = "id"

# Output directory
OUTPUT_DIR = "./CHI_output"
# Model name
MODEL_NAME = 'all-MiniLM-L6-v2'

# GPU Control: None uses all visible devices
TARGET_DEVICES = None

# Batch Size: Recommended 256 or 512 depending on VRAM
BATCH_SIZE = 512


# ===========================================

def main():
    # 1. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Check GPU status
    if not torch.cuda.is_available():
        print("[ERROR] No GPU detected. Script cannot use multi-card acceleration.")
        return

    gpu_count = torch.cuda.device_count()
    print(f"[INFO] Detected {gpu_count} available GPUs.")
    for i in range(gpu_count):
        print(f"       GPU {i}: {torch.cuda.get_device_name(i)}")

    # 3. Read and cleanup data
    print(f"[INFO] Reading {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {INPUT_CSV}")
        return

    # Check column names
    if TEXT_COLUMN not in df.columns:
        print(f"[ERROR] CSV is missing column: {TEXT_COLUMN}")
        return

    # Data Cleaning: Fill NaN and convert to string
    print("[INFO] Cleaning data...")
    sentences = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    # Extract IDs
    if ID_COLUMN in df.columns:
        ids = df[ID_COLUMN].tolist()
    else:
        print("[WARN] ID column not found. Generating index IDs automatically.")
        ids = list(range(len(sentences)))

    data_count = len(sentences)
    print(f"[INFO] Data to encode: {data_count} entries")

    # 4. Load Model
    print(f"[INFO] Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # 5. Start Multi-Process Pool (Multi-GPU)
    print("[INFO] Starting multi-GPU process pool...")
    pool = model.start_multi_process_pool(target_devices=TARGET_DEVICES)

    start_time = time.time()

    # 6. Parallel Encoding
    print("[INFO] Starting parallel encoding (Batch Size: {})...".format(BATCH_SIZE))
    try:
        # Use pool parameter for multi-GPU encoding
        # normalize_embeddings=True is critical for L2 normalization,
        # making Inner Product equivalent to Cosine Similarity in FAISS.
        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            pool=pool,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    except Exception as e:
        print(f"[ERROR] Error during encoding: {e}")
        model.stop_multi_process_pool(pool)
        return

    # Stop process pool to release VRAM
    model.stop_multi_process_pool(pool)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[INFO] Encoding completed. Time: {duration:.2f}s (Speed: {data_count / duration:.2f} samples/s)")

    # 7. Build FAISS Index
    print("[INFO] Building FAISS index...")
    # Get vector dimension (all-MiniLM-L6-v2 is typically 384)
    dimension = embeddings.shape[1]

    # Use IndexFlatIP (Inner Product)
    # Since embeddings are normalized, this calculates cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to index
    index.add(embeddings)
    print(f"[INFO] Index built. Total vectors: {index.ntotal}")

    # 8. Save results to disk
    print("[INFO] Saving files to disk...")

    # A. Save .npy (Raw Embedding Backup)
    npy_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(npy_path, embeddings)
    print(f"       - Saved raw embeddings: {npy_path}")

    # B. Save .index (FAISS Index)
    index_path = os.path.join(OUTPUT_DIR, "knowledge_graph.index")
    faiss.write_index(index, index_path)
    print(f"       - Saved FAISS index: {index_path}")

    # C. Save .pkl (Metadata: ID and Text Mapping)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    metadata = {
        "ids": ids,
        "sentences": sentences,
        "model_name": MODEL_NAME
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"       - Saved metadata mapping: {meta_path}")

    print("-" * 50)
    print("[SUCCESS] All tasks completed successfully.")


if __name__ == "__main__":
    main()