import pandas as pd
from tqdm import tqdm
import json
from evaluate_models import ModelEvaluator
import argparse

def generate_baseline(model_name, input_csv_path, output_jsonl_path):
    """
    Generates baseline results by evaluating patient messages from a CSV file.

    Args:
        model_name (str): The name of the model to use for evaluation.
        input_csv_path (str): The path to the input CSV file containing patient data.
        output_jsonl_path (str): The path to the output JSONL file to save results.
    """
    # Load the baseline dataset
    try:
        baseline_df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(baseline_df)} samples from {input_csv_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # Extract patient message
    if "whitespace_aug" in baseline_df.columns and "orig_input" in baseline_df.columns:
        def extract_augmented_message(row):
            orig = str(row['orig_input']).strip()
            aug = str(row['whitespace_aug'])
            idx = aug.find(orig)
            if idx != -1:
                after_orig = aug[idx + len(orig):]
                return after_orig.strip()
            return aug.strip()
        baseline_df['patient_message'] = baseline_df.apply(extract_augmented_message, axis=1)
    elif "Input" in baseline_df.columns:
        baseline_df['patient_message'] = baseline_df['Input'].apply(
            lambda x: x.split("Patient message:")[1].strip() if "Patient message:" in x else x
        )
    else:
        print("Error: Neither 'whitespace_aug' nor 'Input' column found in the CSV file.")
        return

    # Initialize model evaluator
    evaluator = ModelEvaluator(model_name=model_name)

    # Evaluate and save results
    print(f"Evaluating with model: {model_name}")
    with open(output_jsonl_path, "w") as f:
        for index, row in tqdm(baseline_df.iterrows(), total=baseline_df.shape[0], desc="Evaluating baseline data"):
            input_text = row['patient_message']
            print("input_text", input_text)
            result = evaluator.evaluate_triage(input_text)
            record = {
                "index": int(index),
                "patient_message": input_text,
                "result": result
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Results saved to {output_jsonl_path}")

def main():
    """Main function to parse arguments and run the baseline generation."""
    parser = argparse.ArgumentParser(description="Generate baseline data for medical question answering.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for evaluation."
    )
    parser.add_argument(
        "--input_csv_path",
        type=str,
        default="/workspace/sandbox/mats9/data/oncqa_colorful_perturb.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output_jsonl_path",
        type=str,
        default="/workspace/sandbox/mats9/data/llama3_oncqa_perturb_results.jsonl",
        help="Path to the output JSONL file."
    )
    args = parser.parse_args()
    generate_baseline(args.model_name, args.input_csv_path, args.output_jsonl_path)

if __name__ == "__main__":
    main()
