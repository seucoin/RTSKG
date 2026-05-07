import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import os

# ================= Configuration =================
# Input File (Contains name, community, prompt, abstract)
INPUT_CSV = 'CHI_output/CHI_rag_dataset.csv'
# Output File (Will contain predictions)
OUTPUT_CSV = 'CHI_output/CHI_rag_output.csv'

# API Configuration
API_KEY = "sk-xxxxxxxxxxxxxxx"  # Your API Key
BASE_URL = "https://openai.com/v1"
MODEL_NAME = "gpt-4o"
# Experiment Configuration
MAX_RETRIES = 5  # Max retries per entry

# Initialize Client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ================= Core Functions =================

def get_llm_prediction(prompt):
    """
    Call LLM to get prediction results, including retry mechanism.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # === LLM API Call ===
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system",
                     "content": "You are an expert time-series forecaster. You must only output the predicted number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                timeout=30  # Timeout setting
            )
            return response.choices[0].message.content.strip()
            # ==============================

        except Exception as e:
            # If the last attempt still fails
            if attempt == MAX_RETRIES - 1:
                return f"Error: {str(e)}"

            # Exponential backoff (1s, 2s, 4s...)
            time.sleep(2 ** attempt)

    return "Error: Unknown failure"


def construct_rag_prompt(original_prompt, abstract):
    """
    Splice the abstract information into the specified position of the original Prompt.
    """
    # If abstract is empty or invalid, return original prompt directly
    if pd.isna(abstract) or str(abstract).lower() == 'nan' or not str(abstract).strip():
        return original_prompt

    # Define Anchor
    anchor_sentence = "Do not include any explanations or units."

    # Define insertion content
    # Add a newline character for clearer Prompt structure
    insertion_text = f"The following additional information might be available for your reference:{abstract}\n"

    # Execute replacement: Insert before anchor
    if anchor_sentence in original_prompt:
        new_prompt = original_prompt.replace(anchor_sentence, insertion_text + anchor_sentence)
    else:
        # If anchor not found, append to the end
        new_prompt = original_prompt + "\n" + insertion_text

    return new_prompt


def run_experiment():
    # 1. Read Data
    print(f"[INFO] Reading dataset: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    # Check required columns
    required_cols = ['name', 'community', 'prompt', 'abstract']
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Input CSV missing column: {col}")
            return

    print(f"[INFO] Starting prediction experiment, total {len(df)} samples...")

    # 2. Initialize Output File
    # Results include: Basic info, Final Prompt (optional for check), Prediction
    header = ["name", "community", "label", "final_prompt", "prediction"]

    # If original data has label (ground truth), keep it for error calculation
    # Assuming original CSV doesn't have label column, not saving here. Uncomment next line if needed.
    # if 'label' in df.columns: header = ["name", "community", "label", "prediction"]

    pd.DataFrame(columns=header).to_csv(OUTPUT_CSV, index=False)

    success_count = 0

    # 3. Loop Processing
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Predicting"):

        # A. Construct Final Prompt
        original_prompt = str(row['prompt'])
        abstract_text = row['abstract']

        final_prompt = construct_rag_prompt(original_prompt, abstract_text)

        # B. Call LLM
        prediction = get_llm_prediction(final_prompt)

        # C. Statistics & Logging
        if prediction.startswith("Error"):
            tqdm.write(f"!!! [Row {index}] Prediction Failed: {prediction}")
        else:
            success_count += 1

        # D. Prepare Result Data
        current_result = {
            "name": row['name'],
            "community": row['community'],
            # If original table has label column, use row['label'], else None
            "label": row.get('label', None),
            "final_prompt": final_prompt,  # Save final prompt for verification
            "prediction": prediction
        }

        # E. Real-time Write
        pd.DataFrame([current_result]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    # 4. Finish
    print("-" * 50)
    print(f"[SUCCESS] Experiment completed!")
    print(f"Success Rate: {success_count}/{len(df)}")
    print(f"Results saved to: {OUTPUT_CSV}")

    # Print a sample of the final prompt for confirmation
    print("\n[Sample] Final Prompt:")
    print(pd.read_csv(OUTPUT_CSV)['final_prompt'].iloc[0])


if __name__ == "__main__":
    run_experiment()