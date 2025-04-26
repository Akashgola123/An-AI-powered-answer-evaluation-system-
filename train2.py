# import torch
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer
# import re
# from typing import Dict, Optional, Tuple

# # --- Global Variables (Load once on application start if possible) ---
# FINETUNED_MODEL_PATH = "./llama-3.2-3b-qa-finetuned/final_model" # Your saved model path
# model = None
# tokenizer = None
# device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_model_and_tokenizer():
#     """Loads the fine-tuned model and tokenizer into memory."""
#     global model, tokenizer
#     if model is None or tokenizer is None:
#         print(f"Loading model from {FINETUNED_MODEL_PATH}...")
#         try:
#             model, tokenizer = FastLanguageModel.from_pretrained(
#                 model_name=FINETUNED_MODEL_PATH,
#                 max_seq_length=2048,
#                 dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
#                 load_in_4bit=True,
#                 # token=os.environ.get("HF_TOKEN"), # Optional: if base requires token
#             )
#             model.to(device)
#             model.eval() # Set to evaluation mode
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#                 tokenizer.pad_token_id = tokenizer.eos_token_id
#             print("Model and tokenizer loaded successfully.")
#         except Exception as e:
#             print(f"FATAL: Error loading model/tokenizer: {e}")
#             # Handle error appropriately (e.g., exit, disable feature)
#             raise # Re-raise to stop execution if loading fails

# # --- Helper Functions ---

# def prepare_evaluation_prompt(question: str, student_answer: str, model_answer: str) -> str:
#     """Formats the input into the Llama-3.2 Instruct prompt structure."""
#     global tokenizer
#     if tokenizer is None:
#         load_model_and_tokenizer() # Ensure tokenizer is loaded

#     system_prompt = "You are an educational assistant that evaluates student answers."
#     user_message = f"Subject: [Optional: Add Subject if available]\nTopic: [Optional: Add Topic if available]\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"

#     messages = [
#         {'role': 'system', 'content': system_prompt},
#         {'role': 'user', 'content': user_message},
#     ]
#     # Apply chat template *with* the generation prompt for inference
#     formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     return formatted_prompt

# @torch.inference_mode() # Disable gradient calculations for inference
# def get_raw_evaluation_output(prompt_text: str) -> Optional[str]:
#     """Runs the model to generate the evaluation text."""
#     global model, tokenizer, device
#     if model is None or tokenizer is None:
#         load_model_and_tokenizer()

#     inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

#     # Generation parameters (tune these!)
#     generation_config = {
#         "max_new_tokens": 512,      # Max length for score + feedback + plagiarism
#         "pad_token_id": tokenizer.pad_token_id,
#         "eos_token_id": tokenizer.eos_token_id,
#         "do_sample": False,        # Use greedy decoding for deterministic output
#         "temperature": 0.1,      # Optional: Sample slightly if needed
#         # "top_p": 0.9,            # Optional: Sample slightly if needed
#     }

#     try:
#         outputs = model.generate(**inputs, **generation_config)
#         # Decode only the *newly generated* tokens
#         prompt_len = inputs['input_ids'].shape[1]
#         decoded_output = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
#         return decoded_output.strip() # Remove leading/trailing whitespace
#     except Exception as e:
#         print(f"Error during model generation: {e}")
#         return None

# def parse_evaluation_output(generated_text: str) -> Dict[str, Optional[str] | Optional[int]]:
#     """Parses the score, feedback, and plagiarism from the model's output."""
#     if not generated_text:
#         return {"score": None, "feedback": None, "plagiarism": None}

#     score = None
#     feedback = None
#     plagiarism = None

#     # 1. Parse Score (expecting "Score: X/5")
#     score_match = re.search(r"Score:\s*(\d)\s*/\s*5", generated_text, re.IGNORECASE)
#     if score_match:
#         try:
#             score = int(score_match.group(1))
#             score = max(0, min(5, score)) # Clamp score
#         except ValueError:
#             print("Warning: Found score pattern but couldn't parse integer.")
#             score = None # Failed parsing

#     # 2. Parse Feedback (Assume feedback starts after score or 'Feedback:')
#     feedback_match = re.search(r"Feedback:\s*(.*)", generated_text, re.IGNORECASE | re.DOTALL)
#     if feedback_match:
#         feedback_text = feedback_match.group(1).strip()
#         # Remove potential plagiarism part if it follows directly
#         plagiarism_start_match = re.search(r"\nPlagiarism:", feedback_text, re.IGNORECASE)
#         if plagiarism_start_match:
#             feedback = feedback_text[:plagiarism_start_match.start()].strip()
#         else:
#             feedback = feedback_text
#     elif score_match:
#         # If score was found, assume feedback is text after it (simple fallback)
#         potential_feedback = generated_text[score_match.end():].strip()
#         plagiarism_start_match = re.search(r"\nPlagiarism:", potential_feedback, re.IGNORECASE)
#         if plagiarism_start_match:
#              feedback = potential_feedback[:plagiarism_start_match.start()].strip()
#         elif potential_feedback: # Check if there's any text left
#              feedback = potential_feedback
#     # Add more robust feedback parsing if needed

#     # 3. Parse Plagiarism (Assume it follows feedback or 'Plagiarism:')
#     plagiarism_match = re.search(r"Plagiarism:\s*(.*)", generated_text, re.IGNORECASE)
#     if plagiarism_match:
#         plagiarism = plagiarism_match.group(1).strip()
#         # Limit length or clean up if needed
#         plagiarism = plagiarism.split('\n')[0] # Take only the first line

#     # Fallback: If specific parsing fails but text exists, return the raw text?
#     if feedback is None and plagiarism is None and score is None and generated_text:
#          print("Warning: Could not parse structure, returning raw output as feedback.")
#          feedback = generated_text # Assign whole output as feedback as a last resort

#     return {"score": score, "feedback": feedback, "plagiarism": plagiarism}


# # --- Main Integration Function ---
# def evaluate_student_answer(question: str, student_answer: str, model_answer: str) -> Dict:
#     """
#     Takes raw inputs, runs them through the fine-tuned model,
#     and returns a dictionary with parsed score, feedback, and plagiarism.
#     """
#     # Ensure model is loaded (might call load_model_and_tokenizer)
#     if model is None or tokenizer is None:
#         load_model_and_tokenizer()
#         # Handle case where loading failed inside load_model_and_tokenizer
#         if model is None or tokenizer is None:
#              return {"error": "Model could not be loaded."}

#     # 1. Prepare the prompt
#     prompt = prepare_evaluation_prompt(question, student_answer, model_answer)

#     # 2. Get the raw model output
#     raw_output = get_raw_evaluation_output(prompt)
#     if raw_output is None:
#         return {"error": "Model generation failed."}

#     # 3. Parse the output
#     parsed_result = parse_evaluation_output(raw_output)
#     parsed_result["raw_output"] = raw_output # Optionally return the raw text too

#     # Handle cases where parsing failed (e.g., score is None)
#     if parsed_result["score"] is None:
#         print(f"Warning: Could not reliably parse score for question: {question[:50]}...")
#         # Decide on fallback: return error, default score, or just feedback?
#         # parsed_result["score"] = -1 # Indicate parsing error score?

#     return parsed_result


# # --- Example Usage (How your project would call it) ---
# if __name__ == "__main__":
#     # Load model when script starts (if running as standalone service)
#     load_model_and_tokenizer()

#     # Example Input (replace with actual data from your project)
#     test_question = "What is friction?"
#     test_student_answer = "The thing that makes stuff slow down when they rub together."
#     test_model_answer = "Friction is a force that opposes the relative motion between two surfaces in contact."

#     print("\n--- Evaluating Example ---")
#     evaluation_result = evaluate_student_answer(test_question, test_student_answer, test_model_answer)

#     print(f"\nQuestion: {test_question}")
#     print(f"Student Answer: {test_student_answer}")
#     print("\n--- Evaluation Result ---")
#     print(f"Score: {evaluation_result.get('score')}")
#     print(f"Feedback: {evaluation_result.get('feedback')}")
#     print(f"Plagiarism: {evaluation_result.get('plagiarism')}")
#     print(f"Raw Output: {evaluation_result.get('raw_output')}") # For debugging
#     if "error" in evaluation_result:
#          print(f"ERROR: {evaluation_result['error']}")




import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import re
import time
import logging
from typing import Dict, Optional, Tuple, Any # Added Any
import json

# --- Configuration ---
FINETUNED_MODEL_PATH = "./llama-3.2-3b-qa-finetuned/final_model" # Your saved model path
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
PASS_FAIL_THRESHOLD = 3 # Score >= 3 is considered Pass
GENERATION_MAX_NEW_TOKENS = 300 # Adjust based on expected feedback length

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
is_model_loaded = False # Flag to track loading status

def load_model_and_tokenizer():
    """Loads the fine-tuned model and tokenizer into memory ONCE."""
    global model, tokenizer, device, is_model_loaded
    # Check if already loaded or loading failed previously
    if is_model_loaded:
        if model is not None and tokenizer is not None:
             logging.info("Model and tokenizer already loaded.")
             return True
        else:
             logging.error("Model loading was attempted previously but failed.")
             return False

    load_start = time.time()
    logging.info(f"Loading model from {FINETUNED_MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=FINETUNED_MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=MODEL_DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        model.to(device)
        model.eval() # Set to evaluation mode

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            logging.warning("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None: # Crucial for padding/generation
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logging.info(f"Model and tokenizer loaded successfully to {device} in {time.time() - load_start:.2f} seconds.")
        is_model_loaded = True
        return True

    except Exception as e:
        logging.exception(f"FATAL: Error loading model/tokenizer from {FINETUNED_MODEL_PATH}")
        model, tokenizer = None, None
        is_model_loaded = True # Mark as attempted but failed
        return False

# --- Helper Functions ---

def prepare_evaluation_prompt(question: str, student_answer: str, model_answer: str) -> Optional[str]:
    """Formats the input into the Llama-3.2 Instruct prompt structure."""
    if not is_model_loaded or tokenizer is None:
        logging.error("Tokenizer not loaded, cannot prepare prompt.")
        # Optionally try loading here: if not load_model_and_tokenizer(): return None
        return None

    system_prompt = "You are an educational assistant that evaluates student answers."
    user_message = f"Subject: [Optional: Add Subject if available]\nTopic: [Optional: Add Topic if available]\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message},
    ]
    try:
        # Apply chat template *with* the generation prompt for inference
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted_prompt
    except Exception as e:
        logging.exception("Error applying chat template.")
        return None

@torch.inference_mode()
def get_raw_evaluation_output(prompt_text: str) -> Optional[str]:
    """Runs the fine-tuned model to generate the evaluation text."""
    if not is_model_loaded or model is None or tokenizer is None:
        logging.error("Model/tokenizer not loaded, cannot generate output.")
        # Optionally try loading here: if not load_model_and_tokenizer(): return None
        return None

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    except Exception as e:
        logging.exception(f"Tokenization error for prompt: {prompt_text[:100]}...")
        return None

    generation_config = {
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,       # Keep deterministic
        "temperature": 0.0,       # Temperature 0 for greedy decoding
    }

    try:
        gen_start = time.time()
        outputs = model.generate(**inputs, **generation_config)
        gen_time = time.time() - gen_start
        logging.info(f"Generation took {gen_time:.2f}s")

        # Decode only the *newly generated* tokens
        prompt_len = inputs['input_ids'].shape[1]
        output_tokens = outputs[0, prompt_len:]
        actual_output_tokens = output_tokens[output_tokens != tokenizer.pad_token_id] # Filter pad tokens
        decoded_output = tokenizer.decode(actual_output_tokens, skip_special_tokens=True)
        return decoded_output.strip()

    except Exception as e:
        logging.exception(f"Error during model generation for prompt: {prompt_text[:100]}...")
        return None

def parse_score_and_feedback(generated_text: str) -> Dict[str, Optional[Any]]:
    """Parses only the score and feedback from the model's output."""
    parsed = {"score": None, "feedback": None}
    if not generated_text or not generated_text.strip():
        logging.warning("Received empty generated text for parsing.")
        return parsed

    text_lower = generated_text.lower()
    score = None
    feedback = None

    # --- 1. Extract Score ---
    score_match = re.search(r"score:\s*(\d)\s*/\s*5", text_lower)
    score_line_end = -1
    if score_match:
        try:
            score = max(0, min(5, int(score_match.group(1))))
            parsed["score"] = score
            score_line_end = score_match.end()
        except ValueError:
            logging.warning(f"Found score pattern but failed integer conversion: '{score_match.group(0)}' in text: {generated_text[:150]}...")

    # --- 2. Extract Feedback ---
    # Look for 'Feedback:' tag first
    feedback_match = re.search(r"feedback:\s*(.*)", text_lower, re.DOTALL)
    if feedback_match:
        potential_feedback = feedback_match.group(1).strip()
        # Check if a known tag like Plagiarism comes after Feedback: and trim
        plag_match_in_feedback = re.search(r"\nplagiarism:", potential_feedback, re.IGNORECASE)
        if plag_match_in_feedback:
            feedback = potential_feedback[:plag_match_in_feedback.start()].strip()
        else:
            feedback = potential_feedback
    elif score_line_end != -1:
        # Fallback: If score was found, assume text after it is feedback
        potential_feedback = generated_text[score_line_end:].strip()
        # Simple check: If it starts with 'Plagiarism:', ignore it as feedback
        if potential_feedback and not potential_feedback.lower().startswith("plagiarism:"):
             # Also trim if plagiarism occurs later on a new line
             plag_match_later = re.search(r"\nplagiarism:", potential_feedback, re.IGNORECASE)
             if plag_match_later:
                  feedback = potential_feedback[:plag_match_later.start()].strip()
             else:
                  feedback = potential_feedback

    # Log if parts couldn't be parsed reliably
    if parsed["score"] is None: logging.warning(f"Could not parse SCORE from: {generated_text[:150]}...")
    if feedback is None: logging.warning(f"Could not parse FEEDBACK from: {generated_text[:150]}...")

    parsed["feedback"] = feedback # Assign found/or None feedback
    return parsed

def calculate_grade(score: Optional[int]) -> Optional[str]:
    """Converts numerical score (0-5) to letter grade."""
    if score is None or score < 0: # Handle None or parsing error (-1 etc)
        return None
    elif score == 5: return "A+"
    elif score == 4: return "A"
    elif score == 3: return "B"
    elif score == 2: return "C"
    elif score == 1: return "D"
    else:            return "F"

def determine_pass_fail(score: Optional[int], threshold: int = PASS_FAIL_THRESHOLD) -> Optional[str]:
    """Determines pass/fail status based on score and threshold."""
    if score is None or score < 0:
        return None
    return "Pass" if score >= threshold else "Fail"


# --- Main Integration Function ---
def evaluate_student_answer(question: str, student_answer: str, model_answer: str) -> Dict:
    """
    Orchestrates the evaluation: formats prompt, gets model output, parses,
    and returns the desired dictionary (score, grade, status, feedback).
    """
    start_time = time.time()

    # Try loading model only if needed (if first run or failed before)
    if not is_model_loaded:
        if not load_model_and_tokenizer():
            return {"error": "Model could not be loaded."} # Return error if load fails

    # 1. Prepare the prompt
    prompt = prepare_evaluation_prompt(question, student_answer, model_answer)
    if prompt is None:
        return {"error": "Failed to create evaluation prompt."}

    # 2. Get the raw model output
    raw_output = get_raw_evaluation_output(prompt)
    if raw_output is None:
        return {"error": "Model generation failed."}

    # 3. Parse the score and feedback
    parsed_data = parse_score_and_feedback(raw_output)
    parsed_score = parsed_data["score"]
    parsed_feedback = parsed_data["feedback"]

    # 4. Calculate derived info
    grade = calculate_grade(parsed_score)
    status = determine_pass_fail(parsed_score)

    # 5. Construct final result
    result = {
        "score": parsed_score, # Can be None or -1 if parsing failed
        "grade": grade,        # Can be None
        "status": status,      # Can be None
        "feedback": parsed_feedback if parsed_feedback else "No feedback could be parsed.", # Provide default message
        "raw_output": raw_output # Include raw for debugging
    }

    end_time = time.time()
    logging.info(f"Evaluation completed in {end_time - start_time:.2f}s")
    return result


# --- Example Usage (How your project would call it) ---
if __name__ == "__main__":
    # Ensure model loads on first run of this script
    if not load_model_and_tokenizer():
        print("Exiting because model could not be loaded.")
        exit()

    # Example Input 1
    q1 = "State Newton's First Law."
    sa1 = "Things stay still or keep moving unless something pushes them."
    ma1 = "An object remains at rest or in uniform motion unless acted upon by an external force."

    print("\n--- Evaluating Example 1 ---")
    eval1 = evaluate_student_answer(q1, sa1, ma1)
    print(json.dumps(eval1, indent=2)) # Pretty print JSON-like output

    print("\n" + "="*30 + "\n")

    # Example Input 2 (Expecting potentially lower score)
    q2 = "Define kinetic energy."
    sa2 = "Energy from moving."
    ma2 = "Kinetic energy is the energy possessed by an object due to its motion, given by KE=½mv²."

    print("--- Evaluating Example 2 ---")
    eval2 = evaluate_student_answer(q2, sa2, ma2)
    print(json.dumps(eval2, indent=2))

    # Example Input 3 (Good Answer)
    q3 = "What is Le Chatelier's Principle?"
    sa3 = "If you disturb a system in equilibrium, it shifts to counteract the change."
    ma3 = "Le Chatelier's Principle states that when a system at equilibrium is subjected to a change, the system will shift its equilibrium position to counteract the effect of the change."

    print("\n" + "="*30 + "\n")
    print("--- Evaluating Example 3 ---")
    eval3 = evaluate_student_answer(q3, sa3, ma3)
    print(json.dumps(eval3, indent=2))