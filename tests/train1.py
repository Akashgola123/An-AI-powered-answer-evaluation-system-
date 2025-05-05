# --- Enhanced Imports ---
import json
import re
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    confusion_matrix
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import collections
import logging # Added for better logging
import time   # Added for potential timing info

from unsloth import FastLanguageModel # Import Unsloth
from transformers import AutoTokenizer # Import Tokenizer
from typing import Optional, Dict, Tuple

# --- Configuration Block ---
cfg = {
    # --- Paths ---
    "finetuned_model_path": "./llama-3.2-3b-qa-finetuned/final_model", # Path to your fine-tuned adapter/model
    "dataset_path": "/home/gola/GRAPH_RAG/Exam_Portal/An-AI-powered-answer-evaluation-system-/dataset.json",                         # Path to your evaluation dataset
    "output_csv_path": "./finetuned_enhanced_evaluation_results.csv", # Output CSV file
    "plot_output_dir": "./finetuned_enhanced_plots",          # Directory for plots

    # --- Model Loading ---
    "model_max_seq_length": 2048,
    "model_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Adjust dtype if needed
    "load_in_4bit": True,
    "hf_token": None, # Optional: Set your Hugging Face token if base model needs auth (e.g., "hf_...")

    # --- Generation Parameters ---
    "gen_max_new_tokens": 250,      # Max length for score+feedback+plagiarism
    "gen_do_sample": False,         # False for deterministic output (recommended for eval)
    "gen_temperature": 1.0,       # Only used if do_sample=True
    "gen_top_p": 1.0,             # Only used if do_sample=True

    # --- Evaluation Options ---
    "max_samples": None,            # Set to int for testing, None for full dataset
    "reinforce_prompt_format": True, # Add reminder about Score:/Feedback:/Plagiarism: format?
    "default_score_on_parse_fail": -1, # Score to assign if parsing fails (-1 indicates error)
    "verbose_logging": True,        # Print more detailed logs/warnings?
}

# --- Setup Logging ---
log_level = logging.INFO if cfg['verbose_logging'] else logging.WARNING
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables (Initialized in load function) ---
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# --- Function Definitions ---

def load_model_and_tokenizer():
    """Loads the fine-tuned model and tokenizer into memory."""
    global model, tokenizer
    if model is not None and tokenizer is not None:
        logging.info("Model and tokenizer already loaded.")
        return True

    model_load_start = time.time()
    logging.info(f"Loading model from {cfg['finetuned_model_path']}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg['finetuned_model_path'],
            max_seq_length=cfg['model_max_seq_length'],
            dtype=cfg['model_dtype'],
            load_in_4bit=cfg['load_in_4bit'],
            token=cfg.get("hf_token"), # Use .get for optional keys
        )
        model.to(device)
        model.eval() # Set to evaluation mode

        # Ensure pad token is set for tokenizer
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer pad_token not set, using eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Explicitly set pad_token_id is crucial for padding during batching or generation
            if tokenizer.pad_token_id is None:
                 tokenizer.pad_token_id = tokenizer.eos_token_id

        logging.info(f"Model and tokenizer loaded successfully in {time.time() - model_load_start:.2f} seconds.")
        return True
    except Exception as e:
        logging.exception(f"FATAL: Error loading model/tokenizer from {cfg['finetuned_model_path']}")
        model, tokenizer = None, None # Ensure they are None on failure
        return False


def prepare_evaluation_prompt(question: str, student_answer: str, model_answer: str) -> Optional[str]:
    """Formats the input into the Llama-3.2 Instruct prompt structure."""
    global tokenizer
    if tokenizer is None:
        logging.error("Tokenizer not loaded, cannot prepare prompt.")
        return None

    system_prompt = "You are an educational assistant that evaluates student answers."
    user_message = (
        f"Subject: [Optional: Add Subject]\nTopic: [Optional: Add Topic]\n"
        f"Question: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"
    )

    # Optionally reinforce format requirement
    if cfg['reinforce_prompt_format']:
        reinforcement = (
            "\n\nFormat your response strictly as follows:\n"
            "Score: [0-5]/5\nFeedback: [Your detailed feedback]\nPlagiarism: [Paraphrased/Quoted/Unique]"
        )
        user_message += reinforcement

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message},
    ]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted_prompt
    except Exception as e:
        logging.exception(f"Error applying chat template")
        return None


@torch.inference_mode() # Disable gradient calculations for inference
def get_raw_evaluation_output(prompt_text: str) -> Optional[str]:
    """Runs the model to generate the evaluation text."""
    global model, tokenizer, device
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not loaded, cannot generate output.")
        return None

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    except Exception as e:
        logging.exception(f"Error during tokenization for prompt starting with: {prompt_text[:100]}...")
        return None

    generation_config = {
        "max_new_tokens": cfg['gen_max_new_tokens'],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": cfg['gen_do_sample'],
        "temperature": cfg['gen_temperature'],
        "top_p": cfg['gen_top_p'],
    }

    try:
        gen_start = time.time()
        outputs = model.generate(**inputs, **generation_config)
        # Decode only the *newly generated* tokens
        prompt_len = inputs['input_ids'].shape[1]
        # Use atleast_1d to handle case where output is only prompt
        output_tokens = outputs[0, prompt_len:]
        # Need to handle pad tokens if generation stops early and pad_token != eos_token
        actual_output_tokens = output_tokens[output_tokens != tokenizer.pad_token_id]
        decoded_output = tokenizer.decode(actual_output_tokens, skip_special_tokens=True)
        gen_time = time.time() - gen_start
        if cfg['verbose_logging']:
            logging.info(f"Generation took {gen_time:.2f}s")
        return decoded_output.strip() # Remove leading/trailing whitespace
    except Exception as e:
        # Log the exception with stack trace for detailed debugging
        logging.exception(f"Error during model generation for prompt starting with: {prompt_text[:100]}...")
        return None

# Enhanced Parsing Function
def parse_evaluation_output(generated_text: str) -> Dict:
    """Parses the score, feedback, and plagiarism from the model's output (more robustly)."""
    # Initialize defaults (None means not found/parsed)
    parsed = {"score": None, "feedback": None, "plagiarism": None}

    if not generated_text or not generated_text.strip():
        logging.warning("Received empty generated text for parsing.")
        return parsed

    text_lower = generated_text.lower() # Use lowercase for robust matching
    score, feedback, plagiarism = None, None, None

    # --- 1. Extract Score ---
    score_match = re.search(r"score:\s*(\d)\s*/\s*5", text_lower)
    if score_match:
        try:
            score = int(score_match.group(1))
            score = max(0, min(5, score)) # Clamp to valid range
            parsed["score"] = score
        except ValueError:
            logging.warning(f"Found score pattern but failed integer conversion: '{score_match.group(0)}' in text: {generated_text[:150]}...")

    # --- 2. Extract Plagiarism ---
    # Search for the plagiarism tag, prioritize later occurrences if multiple exist
    best_plag_match = None
    for match in re.finditer(r"plagiarism:\s*(.*)", text_lower):
         best_plag_match = match # Last match is usually most reliable
    plagiarism_start_index = best_plag_match.start() if best_plag_match else -1

    if best_plag_match:
        # Take text after the tag, stop at newline
        plagiarism_text = best_plag_match.group(1).strip()
        plagiarism = plagiarism_text.split('\n')[0].strip() # Get first line after tag
        parsed["plagiarism"] = plagiarism

    # --- 3. Extract Feedback ---
    feedback_start_index = -1
    feedback_match = re.search(r"feedback:\s*(.*)", text_lower, re.DOTALL) # Use DOTALL to match across lines
    if feedback_match:
        feedback_start_index = feedback_match.start() # Remember where feedback tag starts
        potential_feedback = feedback_match.group(1).strip()

        # If plagiarism was found AFTER feedback tag, trim feedback
        if plagiarism_start_index > feedback_start_index:
            actual_feedback_end = plagiarism_start_index - feedback_start_index - len("feedback:") # Adjust index
            feedback = potential_feedback[:actual_feedback_end].strip()
        else:
            # Plagiarism tag not found or occurs before feedback tag
            feedback = potential_feedback
        parsed["feedback"] = feedback

    # --- Logging if Parsing Failed ---
    if parsed["score"] is None:
         logging.warning(f"Could not parse SCORE ('Score: X/5') from output: {generated_text[:150]}...")
    if parsed["feedback"] is None:
         logging.warning(f"Could not parse FEEDBACK ('Feedback: ...') from output: {generated_text[:150]}...")
         # Fallback: Assign raw if score wasn't found either? Maybe too risky.
    if parsed["plagiarism"] is None:
         logging.warning(f"Could not parse PLAGIARISM ('Plagiarism: ...') from output: {generated_text[:150]}...")

    # --- Assign Default Score if Parsing Failed ---
    if parsed["score"] is None:
         parsed["score"] = cfg['default_score_on_parse_fail']

    return parsed

# --- Main Script Execution ---
if __name__ == "__main__":
    # --- 1. Load Model ---
    if not load_model_and_tokenizer():
        logging.critical("Model loading failed. Exiting.")
        exit()

    # --- 2. Load Data ---
    logging.info(f"Loading dataset from: {cfg['dataset_path']}")
    try:
        with open(cfg['dataset_path'], 'r') as f:
            raw_data = json.load(f)
        logging.info(f"Loaded {len(raw_data)} examples.")
    except FileNotFoundError:
        logging.exception(f"FATAL: Dataset file not found at {cfg['dataset_path']}")
        exit()
    except Exception as e:
        logging.exception(f"FATAL: Error loading dataset from {cfg['dataset_path']}")
        exit()

    # Limit samples if specified
    if cfg['max_samples'] is not None and cfg['max_samples'] < len(raw_data):
        logging.info(f"Evaluating on a subset of {cfg['max_samples']} samples.")
        raw_data = raw_data[:cfg['max_samples']]

    # --- 3. Evaluation Loop ---
    logging.info(f"Starting evaluation using fine-tuned model...")
    predictions = []
    true_scores = []
    results_list = []
    processing_times = []

    for i, example in enumerate(tqdm(raw_data, desc="Evaluating Samples")):
        loop_start = time.time()
        try:
            # --- Extract Ground Truth ---
            input_data = example.get('input', {})
            output_data = example.get('output', {})
            true_score = output_data.get('score')
            question = input_data.get('question', 'N/A') # Get question for context

            if true_score is None:
                logging.warning(f"Skipping example {i} due to missing true score (Question: {question[:50]}...)")
                continue
            true_score = int(true_score)


            # --- Prepare Prompt ---
            prompt_text = prepare_evaluation_prompt(
                question,
                input_data.get('student_answer', ''),
                input_data.get('model_answer', '')
            )
            if prompt_text is None:
                logging.warning(f"Skipping example {i} due to prompt creation error.")
                continue # Skip this example

            # --- Get Model Prediction ---
            generated_text = get_raw_evaluation_output(prompt_text)
            if generated_text is None:
                logging.warning(f"Skipping example {i} due to model generation error (Question: {question[:50]}...)")
                # Append placeholders to ensure list alignment IF calculating metrics anyway
                # Or simply skip the sample for metrics by not appending true/pred scores here
                continue

            # --- Parse Prediction ---
            parsed_result = parse_evaluation_output(generated_text)
            predicted_score = parsed_result["score"]

            # --- Store Results ---
            # Append only if prediction was successful & parsing gave a score (or default)
            true_scores.append(true_score)
            predictions.append(predicted_score)
            processing_times.append(time.time() - loop_start)

            results_list.append({
                "example_index": i,
                "question": question,
                # "student_answer": input_data.get('student_answer', ''), # Keep if needed
                "true_score": true_score,
                "generated_text": generated_text,
                "predicted_score": predicted_score,
                "parsed_feedback": parsed_result["feedback"],
                "parsed_plagiarism": parsed_result["plagiarism"]
                # "prompt_text": prompt_text, # Optionally add prompt for debugging
            })

        except Exception as e:
            # Catch unexpected errors within the loop for a specific example
            logging.exception(f"Unexpected error processing example {i} (Question: {question[:50]}...). Skipping.")
            # Avoid appending potentially misaligned data if skipping


    # --- 4. Calculate and Display Metrics ---
    logging.info("Calculating metrics...")
    metrics = {}
    # Filter out placeholder scores (like -1) before calculating metrics
    valid_indices = [i for i, p in enumerate(predictions) if p is not None and p >= 0]
    if not valid_indices:
         logging.warning("No valid predictions found after filtering. Cannot calculate metrics.")
         true_np_filtered, pred_np_filtered = np.array([]), np.array([]) # Empty arrays
    else:
         true_np_filtered = np.array(true_scores)[valid_indices]
         pred_np_filtered = np.array(predictions)[valid_indices]
         num_valid_samples = len(true_np_filtered)
         num_parse_failures = len(predictions) - num_valid_samples

         if num_valid_samples > 0:
            metrics["strict_accuracy"] = accuracy_score(true_np_filtered, pred_np_filtered)
            metrics["within_one_point"] = np.mean(np.abs(true_np_filtered - pred_np_filtered) <= 1)
            metrics["rmse"] = np.sqrt(mean_squared_error(true_np_filtered, pred_np_filtered))
            metrics["mae"] = mean_absolute_error(true_np_filtered, pred_np_filtered)
            metrics["f1_weighted"] = f1_score(true_np_filtered, pred_np_filtered, average='weighted', zero_division=0)

            print("\n--- Fine-Tuned Model Evaluation Results ---")
            print(f"Model Path: {cfg['finetuned_model_path']}")
            print(f"Dataset: {cfg['dataset_path']}")
            print(f"Total Samples Processed: {len(predictions)}")
            print(f"Valid Samples for Metrics: {num_valid_samples}")
            print(f"Score Parsing Failures (resulted in score {cfg['default_score_on_parse_fail']}): {num_parse_failures}")
            avg_time = np.mean(processing_times) if processing_times else 0
            print(f"Average Processing Time per Sample: {avg_time:.3f}s")
            print("-" * 25)
            if metrics:
                for name, value in metrics.items():
                    print(f"{name.replace('_', ' ').title()}: {value:.4f}")
            else:
                print("No metrics calculated (check warnings).")
            print("-" * 25)
         else:
             print("\nNo valid samples remaining after filtering for metric calculation.")
             print(f"Total Samples Processed: {len(predictions)}")
             print(f"Score Parsing Failures (resulted in score {cfg['default_score_on_parse_fail']}): {len(predictions)}")


    # --- 5. Save Detailed Results ---
    try:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(cfg['output_csv_path'], index=False)
        logging.info(f"Detailed results saved to: {cfg['output_csv_path']}")
    except Exception as e:
        logging.exception(f"Error saving results to CSV")


    # --- 6. Generate and Save Plots ---
    if 'true_np_filtered' in locals() and 'pred_np_filtered' in locals() and len(true_np_filtered) > 0:
        logging.info("Generating plots...")
        plot_dir = Path(cfg['plot_output_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        possible_scores = list(range(6)) # Scores 0 to 5

        # Use the FILTERED numpy arrays for plotting
        true_plot_data = true_np_filtered
        pred_plot_data = pred_np_filtered

        # (Plotting code is the same as before, using true_plot_data and pred_plot_data)
        # 1. Score Distribution Plot
        try:
            plt.figure(figsize=(10, 6))
            true_counts=collections.Counter(true_plot_data); pred_counts=collections.Counter(pred_plot_data)
            true_values=[true_counts.get(i,0) for i in possible_scores]; pred_values=[pred_counts.get(i,0) for i in possible_scores]
            x = np.arange(len(possible_scores)); width = 0.35
            rects1=plt.bar(x-width/2, true_values, width, label='True Scores'); rects2=plt.bar(x+width/2, pred_values, width, label='Predicted Scores')
            plt.xlabel("Score"); plt.ylabel("Number of Valid Samples"); plt.title("Distribution of True vs. Predicted Scores (Valid Only)")
            plt.xticks(x, possible_scores); plt.legend(); plt.grid(axis='y', linestyle='--')
            plt.tight_layout(); plt.savefig(plot_dir/"score_distribution.png"); plt.close()
            logging.info(f"- Saved score distribution plot to {plot_dir/'score_distribution.png'}")
        except Exception as e: logging.exception(f"Error generating score distribution plot")

        # 2. Prediction Error Distribution Plot
        try:
            errors = np.abs(true_plot_data - pred_plot_data); error_counts = collections.Counter(errors)
            max_error = int(max(errors)) if len(errors)>0 else 0; possible_errors=list(range(max_error+1))
            error_values=[error_counts.get(i,0) for i in possible_errors]
            plt.figure(figsize=(8,5)); plt.bar(possible_errors, error_values, color='salmon')
            plt.xlabel("Absolute Prediction Error (|True - Predicted|)"); plt.ylabel("Number of Valid Samples"); plt.title("Distribution of Prediction Errors (Valid Only)")
            plt.xticks(possible_errors); plt.grid(axis='y', linestyle='--')
            plt.tight_layout(); plt.savefig(plot_dir/"error_distribution.png"); plt.close()
            logging.info(f"- Saved error distribution plot to {plot_dir/'error_distribution.png'}")
        except Exception as e: logging.exception(f"Error generating error distribution plot")

        # 3. Confusion Matrix Plot
        try:
            cm = confusion_matrix(true_plot_data, pred_plot_data, labels=possible_scores)
            plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=possible_scores, yticklabels=possible_scores)
            plt.xlabel("Predicted Score"); plt.ylabel("True Score"); plt.title("Confusion Matrix (Valid Samples Only)")
            plt.tight_layout(); plt.savefig(plot_dir/"confusion_matrix.png"); plt.close()
            logging.info(f"- Saved confusion matrix plot to {plot_dir/'confusion_matrix.png'}")
        except Exception as e: logging.exception(f"Error generating confusion matrix plot")

    else:
        logging.warning("Skipping plot generation due to lack of valid filtered evaluation data.")

    logging.info("Enhanced fine-tuned model evaluation complete.")