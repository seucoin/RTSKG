import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import os

# ================= Configuration =================
# Input File (Generated from previous step, contains retrieved_knowledge)
INPUT_CSV = 'CHI_output/CHI_with_50_triplets.csv'
# Output File (Will contain name, community, prompt, abstract)
OUTPUT_CSV = 'CHI_output/CHI_rag_dataset.csv'
START_INDEX = 0  # Resume execution from this index
MAX_RETRIES = 10

# API Configuration (Ensure to update with your actual key)
API_KEY = "sk-xxxxxxxxxxxxxxx"  # Your API Key
BASE_URL = "https://openai.com/v1"
MODEL_NAME = "gpt-4o"

# Initialize Client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Prompt Templates =================
# Note: {triples} and {question} are placeholders to be replaced
# SUMMARY_PROMPT_TEMPLATE = """You are an expert in urban rail transit data analysis.
# Your task is to read a list of facts in a knowledge graph (in natural language form) about rail transit stations and extract the key information that is relevant for answering a specific question.
#
# Each fact is expressed as a natural language sentence describing a triple. For example:
# "The Area/35_DOUGLAS lies within Borough/5_Bronzeville-Hyde Park-Woodlawn, indicating that the Area/35_DOUGLAS is fully contained within the administrative boundaries of Borough/5_Bronzeville."
# This expresses the triple: Area/35_DOUGLAS — Locates at — Borough/5_Bronzeville.
#
# Here are the facts:
# {triples}
#
# The question you need to prepare for is:
# {question}
#
# Please generate a concise and coherent paragraph summarizing the key information from the facts, specifically to answer the question. The paragraph should be written in clear, professional English and structured in a way that is suitable for downstream modeling or analysis."""


SUMMARY_PROMPT_TEMPLATE = """You are an expert in urban rail transit data analysis.
Your task is to review a list of facts (from a knowledge graph) and generate a concise summary of only those facts that are relevant to a provided context question.

Input Data:
-A list of natural language statements describing triples (e.g., "Station A — Connects to — Line B"). Each fact is expressed as a natural language sentence describing a triple. For example: "The Area/35_DOUGLAS lies within Borough/5_Bronzeville-Hyde Park-Woodlawn, indicating that the Area/35_DOUGLAS is fully contained within the administrative boundaries of Borough/5_Bronzeville." This expresses the triple: Area/35_DOUGLAS — Locates at — Borough/5_Bronzeville.
-A specific question defining the analysis target.

Instructions:
1. Analyze Relevance: Read the provided facts and identify which specific triples provide background knowledge or structural information useful for the Context Question.
2. Filter & Summarize: specific Generate a simple, coherent paragraph summarizing only the relevant facts identified in step 1.
3. Ignore Question Details: Do NOT summarize the specific events, time-series data, or prediction targets mentioned in the Context Question itself. The question is ONLY used to determine which facts are important.
4. Do Not Answer: Do NOT attempt to directly answer the Context Question. Your output must strictly be a summary of the provided facts.
5. Fallback: If none of the facts are relevant to the context, strictly output: "No relevant information provided."

Here are the facts:
{triples}

The question you need to prepare for is:
{question}

Please generate a concise and coherent paragraph summarizing the key information from the facts. The paragraph should be written in clear, professional English."""

# ================= Core Functions =================

def call_llm_for_summary(triples, question):
    """
    Call GPT-4o to generate summary.
    """
    # Construct full Prompt
    full_user_content = SUMMARY_PROMPT_TEMPLATE.format(
        triples=triples,
        question=question
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    # System Prompt can be simple as core instructions are in User Prompt
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": full_user_content}
                ],
                temperature=0.3,  # Lower temperature for more focused summary
                timeout=60  # Extended timeout for long inputs
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"Error: {str(e)}"
            # Exponential backoff retry
            time.sleep(2 ** attempt)

    return "Error: Unknown failure"


def run_summarization_pipeline():
    # 1. Read Data
    print(f"[INFO] Reading dataset: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    # Check required columns
    required_cols = ['name', 'community', 'prompt', 'retrieved_knowledge']
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Input CSV missing column: {col}")
            return

    print(f"[INFO] Starting summarization, total {len(df)} samples (Processing from index {START_INDEX})...")

    # 2. Initialize Output File (Write Header)
    # Required columns: name, community, prompt, abstract, label
    header = ["name", "community", "prompt", "abstract", "label"]

    # Resume Logic: If START_INDEX > 0, keep existing valid parts
    if START_INDEX == 0:
        print(f"[INFO] Initializing output file: {OUTPUT_CSV}")
        pd.DataFrame(columns=header).to_csv(OUTPUT_CSV, index=False)
    else:
        print(f"[INFO] Resuming from index {START_INDEX}, removing potentially corrupted data after this point...")
        if os.path.exists(OUTPUT_CSV):
            # Read existing file and keep up to START_INDEX (indices 0 to START_INDEX-1)
            try:
                temp_df = pd.read_csv(OUTPUT_CSV)
                # iloc[:START_INDEX] gets exactly the first START_INDEX elements (rows)
                temp_df.iloc[:START_INDEX].to_csv(OUTPUT_CSV, index=False)
            except Exception as e:
                print(f"[WARNING] Failed to read existing output file: {e}. Recreating file.")
                pd.DataFrame(columns=header).to_csv(OUTPUT_CSV, index=False)
        else:
            print(f"[WARNING] Output file {OUTPUT_CSV} not found. Starting from index {START_INDEX} may result in incomplete data.")
            pd.DataFrame(columns=header).to_csv(OUTPUT_CSV, index=False)

    # 3. Loop Processing
    success_count = 0
    # Process only data after the start index
    df_to_process = df.iloc[START_INDEX:]

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Summarizing"):

        triples_text = str(row['retrieved_knowledge'])  # Prevent NaN error
        question_text = str(row['prompt'])

        # If retrieved content is empty, no need to summarize
        if not triples_text or triples_text.lower() == 'nan':
            abstract = "No knowledge retrieved."
        else:
            # Call LLM
            abstract = call_llm_for_summary(triples_text, question_text)

        # Basic error check
        if not abstract.startswith("Error"):
            success_count += 1
        else:
            tqdm.write(f"!!! [Row {index}] Summary Failed: {abstract}")

        # 4. Real-time Save (Append Mode)
        current_result = {
            "name": row['name'],
            "community": row['community'],
            "prompt": row['prompt'],
            "abstract": abstract,
            "label": row['label']
        }

        # Append one row
        pd.DataFrame([current_result]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    print("-" * 50)
    print(f"[SUCCESS] Processing completed!")
    print(f"Success Rate: {success_count}/{len(df)}")
    print(f"Results saved to: {OUTPUT_CSV}")

    # Print a sample to check
    print("\n--- Sample Summary ---")
    result_df = pd.read_csv(OUTPUT_CSV)
    print(result_df['abstract'].iloc[0])


if __name__ == "__main__":
    run_summarization_pipeline()