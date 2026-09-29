import os
import faiss
import pickle
import numpy as np
import pandas as pd
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# ================= 1. Environment & Path Configuration =================
# Force offline mode & set mirror (Consistent with previous configuration)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Input CSV (Generated from previous step, contains only prompts)
INPUT_CSV_PATH = 'CHI_prompt.csv'

# Output CSV (New file containing retrieval results)
OUTPUT_CSV_PATH = './CHI_output/CHI_with_50_triplets.csv'

# Directory containing vector index (Ensure knowledge_graph.index exists here)
INDEX_DIR = "./CHI_output"
# Model name (or local path)
MODEL_NAME = 'all-MiniLM-L6-v2'

# Retrieval Configuration
TOP_K = 50  # Retrieve top 50 nearest neighbors
BATCH_SIZE = 256  # Batch size for encoding (Adjust based on available VRAM)


# ===================================================

def load_resources():
    """Load resources: Index, Metadata, and Model."""
    print(f"[INFO] Loading resources from {INDEX_DIR}...")

    # 1. Load FAISS Index
    index_path = os.path.join(INDEX_DIR, "knowledge_graph.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = faiss.read_index(index_path)

    # 2. Load Metadata (ID -> Text Mapping)
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # 3. Load Model
    print(f"[INFO] Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


def run_batch_retrieval():
    # 1. Initialize Resources
    index, metadata, model = load_resources()

    # 2. Read Input CSV
    print(f"[INFO] Reading input file: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[ERROR] Input file not found: {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)

    # Check if 'prompt' column exists
    if 'prompt' not in df.columns:
        print("[ERROR] CSV is missing 'prompt' column. Retrieval aborted.")
        return

    prompts = df['prompt'].tolist()
    print(f"[INFO] Data to process: {len(prompts)} entries")

    # 3. Batch Encoding
    # Enables GPU parallel computation for improved performance
    print("[INFO] Encoding Prompts (Batch Processing)...")
    query_vectors = model.encode(
        prompts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 4. Batch Retrieval (Batch Search)
    print(f"[INFO] Retrieving Top-{TOP_K} neighbors in FAISS...")
    # D: Distances (Scores), I: Indices (IDs)
    # FAISS is optimized for batch search
    D, I = index.search(query_vectors, TOP_K)

    # 5. Result Mapping (Indices -> Text)
    print("[INFO] Mapping retrieval results to text...")

    retrieved_contents = []

    # Use tqdm to display progress
    for i in tqdm(range(len(prompts)), desc="Mapping Results"):
        # Get 100 IDs for the current prompt
        indices = I[i]
        scores = D[i]

        # Extract corresponding text
        # Note: FAISS fills with -1 if insufficient results found
        context_list = []
        for idx in indices:
            if idx != -1:
                # Retrieve original triplet from metadata
                text = metadata['sentences'][idx]
                context_list.append(text)

        # Combine 100 triplets into a single string
        # Separated by newlines for LLM readability
        full_context_str = "\n".join(context_list)
        retrieved_contents.append(full_context_str)

    # 6. Save Results
    print("[INFO] Writing results to file...")
    # Append results as a new column to DataFrame
    df['retrieved_knowledge'] = retrieved_contents

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print("-" * 50)
    print(f"[SUCCESS] Task completed successfully.")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")
    print("New column: 'retrieved_knowledge' (Contains Top-50 relevant triplets)")
    print("-" * 50)

    # Preview sample
    print("Sample Preview (First 100 chars of first entry):")
    print(df['retrieved_knowledge'].iloc[0][:100] + "...")


if __name__ == "__main__":
    run_batch_retrieval()