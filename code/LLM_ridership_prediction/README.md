# RAG-Based LLM Experiment Workflow

This document delineates the sequential execution of the experimental pipeline, encompassing knowledge graph triple embedding, retrieval, context summarization, and the final Retrieval-Augmented Generation (RAG) prediction. A baseline procedure (without RAG augmentation) is also included for comparative analysis.

## Directory Structure

- `CHI_data/`: Directory containing raw knowledge graph triple data (must include `relation_chi_rtskg_new.csv`).
- `CHI_output/`: Directory for storing intermediate artifacts and final outputs (automatically generated during execution).
- `CHI_prompt.csv`: The foundational input file containing prompts for the experiment.

## Experimental Pipeline

Please execute the following scripts in the specified order:

### Step 1: Knowledge Graph Triple Embedding

Transforms textual descriptions of knowledge graph triples into dense vector representations and constructs an index for efficient downstream retrieval.

*   **Script**: [embed_mgpu.py](embed_mgpu.py)
*   **Function**: Reads triple data, generates embeddings using the SBERT model, and persists the data as a FAISS index.
*   **Input**: `./CHI_data/relation_chi_rtskg_new.csv` (Must contain `id` and `description` columns).
*   **Output**: 
    *   `./CHI_output/knowledge_graph.index` (Vector index file)
    *   `./CHI_output/metadata.pkl` (Metadata mapping file)
*   **Command**:
    ```bash
    python embed_mgpu.py
    ```

### Step 2: Batch Retrieval

Retrieves the top-K most relevant knowledge triples from the pre-constructed index for each query prompt.

*   **Script**: [batch_retrieve_to_csv.py](batch_retrieve_to_csv.py)
*   **Function**: Loads the FAISS index, encodes the prompts, and performs nearest neighbor search to retrieve relevant knowledge entries.
*   **Input**: 
    *   `CHI_prompt.csv` (List of prompts to query)
    *   `./CHI_output/knowledge_graph.index` (Generated in Step 1)
    *   `./CHI_output/metadata.pkl` (Generated in Step 1)
*   **Output**: `./CHI_output/CHI_with_50_triplets.csv` (Contains original prompts with the Top-50 retrieved triples)
*   **Command**:
    ```bash
    python batch_retrieve_to_csv.py
    ```

### Step 3: Context Summarization

Leverages a Large Language Model (LLM) to synthesize the fragmented retrieved triples into a coherent, query-relevant context abstract.

*   **Script**: [summarize_rag_context.py](summarize_rag_context.py)
*   **Function**: Invokes the OpenAI API to filter and summarize triple information specifically tailored to the question.
*   **Configuration**: ⚠️ **Ensure `API_KEY` and `BASE_URL` are correctly constrained in the script.**
*   **Input**: `./CHI_output/CHI_with_50_triplets.csv` (Generated in Step 2)
*   **Output**: `./CHI_output/CHI_rag_dataset.csv` (Contains the generated summaries/abstracts)
*   **Command**:
    ```bash
    python summarize_rag_context.py
    ```

### Step 4: RAG Prediction

Executes the final time-series forecasting using the prompt augmented with the synthesized context abstract.

*   **Script**: [main_rag.py](main_rag.py)
*   **Function**: Invokes the LLM for prediction, with the System Prompt configured for expertise in time-series forecasting.
*   **Configuration**: ⚠️ **Ensure `API_KEY` and `BASE_URL` are correctly constrained in the script.**
*   **Input**: `./CHI_output/CHI_rag_dataset.csv` (Generated in Step 3)
*   **Output**: `./CHI_output/CHI_rag_output.csv` (Final prediction results)
*   **Command**:
    ```bash
    python main_rag.py
    ```

### Step 5: Baseline: Raw Prediction (Comparative Analysis)

Performs prediction using the raw prompt without RAG augmentation to establish a performance baseline.

*   **Script**: [main_raw.py](main_raw.py)
*   **Function**: Directly calls the LLM with the original input prompt for prediction.
*   **Configuration**: ⚠️ **Ensure `API_KEY` and `BASE_URL` are correctly constrained in the script.**
*   **Input**: `CHI_prompt.csv`
*   **Output**: `CHI_raw_output.csv`
*   **Command**:
    ```bash
    python main_raw.py
    ```

## Environment Dependencies

Ensure the following Python libraries are installed prior to execution:

```bash
pip install pandas numpy torch sentence-transformers faiss-gpu openai tqdm
```
*(Note: Install `faiss-cpu` if a GPU is not available)*
