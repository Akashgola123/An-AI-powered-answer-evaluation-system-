# ----- Imports and Initial Setup (Keep as is) -----
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, get_scheduler, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, mean_absolute_error
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re # Import regex for parsing

# ----- Model and Tokenizer Initialization -----
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# <<< CHANGE: Make dataset path a variable >>>
dataset_path = "/home/gola/GRAPH_RAG/Exam_Portal/An-AI-powered-answer-evaluation-system-/dataset.json"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# <<< CHANGE: Set pad_token_id explicitly for generation later if needed >>>
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Use consistent dtype - IMPORTANT: We're setting dtype explicitly to bfloat16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.bfloat16,  # Explicitly use bfloat16 throughout
    load_in_4bit=True,
    # <<< CHANGE: Remove redundant gradient checkpointing here >>>
    # use_gradient_checkpointing=True
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    # <<< CONFIRM: This is the correct place for Unsloth gradient checkpointing >>>
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ----- Dataset Preparation (prepare_dataset - Keep mostly as is) -----
def prepare_dataset(data_file_path): # <<< CHANGE: Pass path as argument >>>
    # Load dataset from local JSON file
    try:
        raw_dataset = load_dataset("json", data_files=data_file_path) # <<< CHANGE: Use variable >>>

        # If loaded as Dataset instead of DatasetDict, create train/test split
        if isinstance(raw_dataset, DatasetDict):
             # Check if the dataset already has train and validation splits
            if "train" not in raw_dataset:
                 raise ValueError("Dataset needs a 'train' split.")
            if "validation" not in raw_dataset:
                print("Validation split not found. Creating one from 10% of the training data.")
                # Create validation split from train data if it doesn't exist
                splits = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
                dataset = DatasetDict({
                    "train": splits["train"],
                    "validation": splits["test"]
                })
            else:
                dataset = raw_dataset
        elif isinstance(raw_dataset, load_dataset.Dataset): # Check if it's a single split Dataset object
            print("Only a single dataset split found. Creating train/validation split (90/10).")
            splits = raw_dataset.train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                "train": splits["train"],
                "validation": splits["test"]
            })
        else:
            # If the loaded object is not DatasetDict or Dataset (e.g. directly a split name)
             if "train" in raw_dataset: # Might load as {'train': Dataset(...)}
                 if len(raw_dataset.keys()) == 1:
                     print("Only a 'train' split found in the DatasetDict. Creating validation split.")
                     splits = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
                     dataset = DatasetDict({
                         "train": splits["train"],
                         "validation": splits["test"]
                     })
                 else: # Should have train/validation if multiple keys exist
                     dataset = raw_dataset
             else:
                 raise TypeError(f"Unexpected dataset format loaded: {type(raw_dataset)}. Expected DatasetDict or Dataset.")


        print(f"Dataset splits: {list(dataset.keys())}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")

        # ----- format_student_answer_evaluation (No major changes needed inside) -----
        def format_student_answer_evaluation(example):
            try:
                # Extract fields from the dataset
                instruction = example.get('instruction', '') # Keep even if unused later, part of original data struct

                # Extract input fields
                input_data = example.get('input', {})
                subject = input_data.get('subject', '')
                topic = input_data.get('topic', '')
                question = input_data.get('question', '')
                student_answer = input_data.get('student_answer', '')
                model_answer = input_data.get('model_answer', '')
                concepts = input_data.get('concepts', [])
                rubric = input_data.get('rubric', {}) # Keep for potential future use

                # Extract output fields
                output_data = example.get('output', {})
                score = output_data.get('score', 0)
                feedback = output_data.get('feedback', '')
                plagiarism = output_data.get('plagiarism', '')

                # Format in chat template for LLaMA 3.2 models
                # We need two versions: one for training (full dialogue) and one for inference prompt
                system_prompt = "You are an educational assistant that evaluates student answers."
                user_message = f"Subject: {subject}\nTopic: {topic}\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"
                assistant_message = f"Score: {score}/5\nFeedback: {feedback}\nPlagiarism: {plagiarism}" # This is the target generation

                # --- Training format (includes assistant response) ---
                messages_for_training = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
                formatted_input_for_training = tokenizer.apply_chat_template(
                    messages_for_training,
                    tokenize=False,
                    add_generation_prompt=False # Important for training: includes assistant part
                )
                tokenized_input = tokenizer(formatted_input_for_training, truncation=True, max_length=2048)

                # --- Inference prompt format (excludes assistant response) ---
                messages_for_inference = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    # NO assistant message here
                ]
                # We use add_generation_prompt=True to get the correct turn format for generation
                formatted_input_for_inference = tokenizer.apply_chat_template(
                    messages_for_inference,
                    tokenize=False,
                    add_generation_prompt=True # Creates the prompt ending with assistant turn marker
                )

                # Additional metrics (keep as is)
                student_words = set(student_answer.lower().split())
                model_words = set(model_answer.lower().split())
                word_overlap = len(student_words & model_words)
                similarity_score = word_overlap / len(model_words) if model_words else 0.0

                concepts_mentioned = sum(1 for concept in concepts if concept.lower() in student_answer.lower())
                concept_coverage = concepts_mentioned / len(concepts) if concepts else 0.0

                return {
                    # For Training
                    "input_ids": tokenized_input["input_ids"],
                    "attention_mask": tokenized_input["attention_mask"],
                    # For Inference
                    "inference_prompt": formatted_input_for_inference,
                    # For Evaluation & Analysis
                    "raw_score": int(score), # Ensure score is int for comparison
                    "raw_feedback": feedback,
                    "raw_plagiarism": plagiarism,
                    "subject": subject,
                    "question": question,
                    "similarity_score": float(similarity_score),
                    "concept_coverage": float(concept_coverage),
                    "word_overlap": int(word_overlap),
                }
            except Exception as e:
                print(f"Error processing example: {e}, Example: {example}") # Print example on error
                empty_encoding = tokenizer("", truncation=True, max_length=2048)
                return {
                    "input_ids": empty_encoding["input_ids"],
                    "attention_mask": empty_encoding["attention_mask"],
                    "inference_prompt": "",
                    "raw_score": 0,
                    "raw_feedback": "",
                    "raw_plagiarism": "",
                    "subject": "",
                    "question": "",
                    "similarity_score": 0.0,
                    "concept_coverage": 0.0,
                    "word_overlap": 0,
                }
        # ----- End of format_student_answer_evaluation -----

        # Apply preprocessing to all splits
        processed_dataset = DatasetDict()
        for split in dataset:
            # <<< CHANGE: Define columns to remove explicitly for clarity >>>
            original_columns = dataset[split].column_names
            processed_dataset[split] = dataset[split].map(
                format_student_answer_evaluation,
                remove_columns=original_columns
            )

        return processed_dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_file_path}")
        raise
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        raise  # Re-raise to see the full error


# ----- Custom collate function (Update to include inference_prompt) -----
def collate_fn(batch):
    """
    Custom collate function to pad sequences and handle different data types.
    """
    # --- Padding for training inputs ---
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=0 # Attention mask padding is 0
    )

    # --- Keep inference prompts as list of strings ---
    inference_prompts = [x["inference_prompt"] for x in batch]

    # --- Other metrics as tensors (if needed, otherwise keep as lists) ---
    # Example: similarity_scores = torch.tensor([x["similarity_score"] for x in batch], dtype=torch.float32)
    # Example: concept_coverage = torch.tensor([x["concept_coverage"] for x in batch], dtype=torch.float32)
    # Example: word_overlap = torch.tensor([x["word_overlap"] for x in batch], dtype=torch.int32)

    # --- Keep raw values for evaluation ---
    raw_scores = [x["raw_score"] for x in batch]
    raw_feedback = [x["raw_feedback"] for x in batch] # Keep if feedback eval is needed
    raw_plagiarism = [x["raw_plagiarism"] for x in batch] # Keep if plagiarism eval is needed
    subjects = [x["subject"] for x in batch] # Keep for analysis
    questions = [x["question"] for x in batch] # Keep for analysis

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "inference_prompts": inference_prompts, # Used for generation during evaluation
        "raw_score": raw_scores,
        "raw_feedback": raw_feedback,
        "raw_plagiarism": raw_plagiarism,
        "subject": subjects,
        "question": questions,
        # Include other metrics if needed, e.g., for logging/analysis
        # "similarity_score": similarity_scores,
    }

# ----- Function to evaluate model accuracy (<<< MAJOR REWRITE >>>) -----
def evaluate_model(model, eval_dataloader, device, tokenizer):
    model.eval()
    all_pred_scores = []
    all_true_scores = []
    eval_loss = 0
    total_samples = 0

    # Generation parameters
    # <<< TUNE THESE parameters based on desired output length and behavior >>>
    generation_config = {
        "max_new_tokens": 100,  # Max length of generated score/feedback/plagiarism
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False, # Use greedy decoding for consistent eval scores
        "temperature": 1.0, # Not used if do_sample=False
        "top_p": 1.0,       # Not used if do_sample=False
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            true_scores = batch["raw_score"]
            all_true_scores.extend(true_scores)
            total_samples += len(true_scores)

            # --- Calculate Loss (Optional but good practice) ---
            # Move tensors needed for loss calculation to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Create labels by cloning input_ids (standard Causal LM training)
            # TODO: Consider masking prompt tokens in labels for more focused loss calculation
            labels = input_ids.clone()

            # Forward pass for loss calculation
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()

            # --- Generate Predictions ---
            prompts = batch["inference_prompts"] # Get the prompts prepared earlier
            # Tokenize prompts *without padding* for generation (padding done internally)
            # Ensure tokenizer is set up correctly for batch generation if possible
            # Note: Batch generation might need careful handling of padding side
            # For simplicity, generating one by one if batch fails
            batch_pred_scores = []
            for i, prompt_text in enumerate(prompts):
                # Find the corresponding true score for error context
                current_true_score = true_scores[i]

                # Tokenize individual prompt
                prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

                try:
                    # Generate text
                    generated_ids = model.generate(
                        **prompt_inputs,
                        **generation_config
                    )

                    # Decode generated text (excluding prompt)
                    # generated_ids shape is [1, sequence_length]
                    # prompt_inputs['input_ids'].shape[1] is the length of the prompt
                    prompt_len = prompt_inputs['input_ids'].shape[1]
                    decoded_output = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

                    # --- Parse the generated text for the score ---
                    # Example using regex: looks for "Score: X/5"
                    score_match = re.search(r"Score:\s*(\d+)\s*/\s*5", decoded_output, re.IGNORECASE)

                    if score_match:
                        predicted_score = int(score_match.group(1))
                        # Clamp score to valid range (0-5) just in case
                        predicted_score = max(0, min(5, predicted_score))
                        batch_pred_scores.append(predicted_score)
                    else:
                        # Handle case where score is not found or format is wrong
                        print(f"Warning: Could not parse score from generation for sample {i}. Output: '{decoded_output[:100]}...' Using default score 0.")
                        batch_pred_scores.append(0) # Assign a default score

                except Exception as e:
                     print(f"Error during generation or parsing for sample {i}: {e}")
                     print(f"Prompt: {prompt_text}")
                     print(f"True Score: {current_true_score}")
                     batch_pred_scores.append(0) # Assign default on error

            all_pred_scores.extend(batch_pred_scores)

    # Calculate metrics (ensure lists are not empty)
    if not all_true_scores or not all_pred_scores:
         print("Warning: No scores collected during evaluation.")
         return {
             "eval_loss": eval_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0,
             "strict_accuracy": 0,
             "within_one_point": 0,
             "rmse": float('inf'),
             "mae": float('inf'),
             "f1_score": 0,
         }

    # Convert to numpy for sklearn metrics
    true_np = np.array(all_true_scores)
    pred_np = np.array(all_pred_scores)

    strict_accuracy = accuracy_score(true_np, pred_np)
    within_one_point = np.mean(np.abs(true_np - pred_np) <= 1)
    rmse = np.sqrt(mean_squared_error(true_np, pred_np))
    mae = mean_absolute_error(true_np, pred_np)
    # Calculate F1 score - ensure labels are appropriate for classification context if needed
    # Use 'weighted' average for potentially imbalanced score distribution
    f1 = f1_score(true_np, pred_np, average='weighted', zero_division=0)

    model.train() # Set back to train mode
    return {
        "eval_loss": eval_loss / len(eval_dataloader),
        "strict_accuracy": strict_accuracy,
        "within_one_point": within_one_point, # Often more informative than strict accuracy
        "rmse": rmse,
        "mae": mae,
        "f1_score": f1,
    }


# ----- Training arguments -----
# ----- Training arguments -----
training_args = TrainingArguments(
    output_dir="./llama-3.2-3b-qa-finetuned",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16, # Effective batch size = 16
    warmup_steps=100, # Or adjust based on total steps (e.g., 10% of 120 = 12)
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="steps",
    # <<< CHANGE: Evaluate more frequently >>>
    save_steps=50,      # Example: Save every 50 steps
    eval_strategy="steps",
    # <<< CHANGE: Evaluate more frequently >>>
    eval_steps=50,      # Example: Evaluate every 50 steps (will run at step 50 and 100)
    save_total_limit=3,
    load_best_model_at_end=True, # Requires evaluation to run!
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    optim="adamw_torch_fused",
    report_to="tensorboard",
    gradient_checkpointing=False,
)

# ----- Load and preprocess dataset -----
# <<< CHANGE: Pass the path variable >>>
processed_dataset = prepare_dataset(dataset_path)

# Check validation split existence (already handled within prepare_dataset)
if "validation" not in processed_dataset:
     raise RuntimeError("Failed to create or find a validation split in the dataset.")

# ----- Create DataLoaders -----
train_dataloader = DataLoader(
    processed_dataset["train"],
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

eval_dataloader = DataLoader(
    processed_dataset["validation"],
    batch_size=training_args.per_device_eval_batch_size, # Use eval batch size
    shuffle=False,
    collate_fn=collate_fn
)

# Initialize optimizer with weight decay
# Filter out parameters that should not have weight decay (biases, LayerNorm weights)
# This is often handled implicitly by HF Trainer or AdamW, but can be done explicitly
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

# Setup learning rate scheduler
# <<< ADJUST: Calculate num_training_steps based on dataloader length >>>
num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
num_training_steps = training_args.num_train_epochs * num_update_steps_per_epoch

scheduler = get_scheduler(
    training_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# Training metrics tracking
training_metrics = {
    "train_loss": [],
    "eval_metrics": [],
    "learning_rates": []
}
global_step = 0 # Track total optimization steps

# ----- Early stopping parameters (Now used with load_best_model_at_end)-----
# No need for manual early stopping if using Trainer's load_best_model_at_end feature
# Keeping variables for potential manual implementation if needed:
# early_stopping_patience = 5
# early_stopping_min_delta = 0.001 # For loss/error metrics
# early_stopping_counter = 0
# best_eval_metric = float('inf') # Use -inf if metric_for_best_model is accuracy/F1
# best_model_path = None

# Training loop with memory-efficient mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# <<< Ensure model is on the correct device (Unsloth usually handles this, but good to check) >>>
model.to(device)


# Print information about device and mixed precision
print(f"Using device: {device}")
print(f"Model dtype: {next(model.parameters()).dtype}") # Check actual parameter dtype
print(f"Using mixed precision: {training_args.bf16} (bf16)")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: {num_training_steps}")

# ----- Training Loop -----
for epoch in range(int(training_args.num_train_epochs)):
    model.train() # Ensure model is in training mode
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")

    for step, batch in enumerate(progress_bar):
        # Move batch tensors to device (collate_fn returns lists for non-tensors)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Create labels - standard Causal LM approach
        labels = input_ids.clone()
        # Optional: Mask prompt tokens in labels if desired
        # Example: labels[labels == tokenizer.pad_token_id] = -100 (ignore padding)
        # Example: Need logic to find where assistant response starts and mask before that

        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=training_args.bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Normalize loss for gradient accumulation
        loss = loss / training_args.gradient_accumulation_steps
        epoch_loss += loss.item() * training_args.gradient_accumulation_steps # Accumulate non-normalized loss for logging

        # Backward pass
        loss.backward()

        # Optimizer step (occurs every gradient_accumulation_steps)
        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            # Clip gradients (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Common max_norm value

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Log training loss
            current_lr = scheduler.get_last_lr()[0]
            training_metrics["learning_rates"].append({"step": global_step, "lr": current_lr})
            avg_step_loss = epoch_loss / (step + 1) # Average loss up to this point in epoch
            progress_bar.set_postfix({"loss": f"{loss.item() * training_args.gradient_accumulation_steps:.4f}", "avg_loss": f"{avg_step_loss:.4f}", "lr": f"{current_lr:.2e}"})

            # Log metrics to TensorBoard (or other integrations)
            # Placeholder: integrate with `training_args.report_to` if not using Trainer class

            # ----- Evaluation and Checkpointing -----
            if global_step % training_args.eval_steps == 0:
                # <<< CHANGE: Pass tokenizer to evaluate_model >>>
                eval_metrics = evaluate_model(model, eval_dataloader, device, tokenizer)
                current_eval_metric_value = eval_metrics[training_args.metric_for_best_model]

                eval_metrics_log = {f"eval_{k}": v for k, v in eval_metrics.items()} # Prefix metrics with "eval_"
                training_metrics["eval_metrics"].append({
                    "step": global_step,
                    **eval_metrics_log # Store prefixed metrics
                })
                progress_bar.write(f"\nStep {global_step} Eval Metrics: {eval_metrics_log}")


                # <<< Manual Saving Logic (if not using Trainer's saving) >>>
                # save_path = f"{training_args.output_dir}/checkpoint-step-{global_step}"
                # model.save_pretrained(save_path)
                # tokenizer.save_pretrained(save_path)
                # print(f"Checkpoint saved to {save_path}")

                # <<< Manual Best Model Tracking (if needed alongside Trainer's logic or if not using load_best...) >>>
                # is_better = (current_eval_metric_value < best_eval_metric) if not training_args.greater_is_better else (current_eval_metric_value > best_eval_metric)
                # if is_better:
                #     best_eval_metric = current_eval_metric_value
                #     best_model_path = f"{training_args.output_dir}/best_model"
                #     model.save_pretrained(best_model_path)
                #     tokenizer.save_pretrained(best_model_path)
                #     print(f"*** New best model saved at step {global_step} to {best_model_path} ({training_args.metric_for_best_model}: {best_eval_metric:.4f}) ***")
                #     early_stopping_counter = 0
                # else:
                #      early_stopping_counter +=1
                #      if early_stopping_counter >= early_stopping_patience:
                #          print(f"Early stopping triggered at step {global_step}")
                #          # break # Break inner loop
                #          # Need flag to break outer loop too


        progress_bar.update(1)

    # Log average epoch loss
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    training_metrics["train_loss"].append({
        "epoch": epoch + 1,
        "loss": avg_epoch_loss
    })
    print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f}")

    # --- Epoch End Actions ---
    # Perform final evaluation for the epoch if needed (often done based on steps)
    # If using manual saving, save checkpoint at end of epoch if desired
    # save_path = f"{training_args.output_dir}/checkpoint-epoch-{epoch+1}"
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)

    # --- Check for early stopping break signal (if manual implementation) ---
    # if early_stopping_flag:
    #     break


# ----- Save the final model -----
# <<< NOTE: If load_best_model_at_end=True, the 'best' model according to the metric
# is already loaded. Saving it here saves the potentially *best* checkpoint, not necessarily the *last* one. >>>
print("Training finished. Saving final model...")
final_save_path = f"{training_args.output_dir}/final_model"
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

# ----- Plot and save training metrics (Keep as is, but adjust metric names) -----
def plot_training_metrics(metrics, save_dir):
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_dir) # Use pathlib for robustness
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists


    if not metrics['eval_metrics']:
        print("No evaluation metrics found to plot.")
        return

    steps = [m['step'] for m in metrics['eval_metrics']]

    # --- Plot Accuracy Metrics ---
    plt.figure(figsize=(12, 6))
    # <<< Adjust metric keys based on eval_metrics structure (e.g., 'eval_strict_accuracy') >>>
    strict_accuracies = [m['eval_strict_accuracy'] for m in metrics['eval_metrics']]
    within_one = [m['eval_within_one_point'] for m in metrics['eval_metrics']]

    plt.plot(steps, strict_accuracies, 'g-', label='Strict Accuracy')
    plt.plot(steps, within_one, 'r--', label='Accuracy Within One Point')
    plt.title('Model Accuracy Metrics over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1) # Accuracy is between 0 and 1
    plt.savefig(output_dir / f"accuracy_metrics_{timestamp}.png")
    plt.close()

    # --- Plot Error Metrics ---
    plt.figure(figsize=(12, 6))
    # <<< Adjust metric keys >>>
    rmse_scores = [m['eval_rmse'] for m in metrics['eval_metrics']]
    mae_scores = [m['eval_mae'] for m in metrics['eval_metrics']]

    plt.plot(steps, rmse_scores, 'b-', label='RMSE')
    plt.plot(steps, mae_scores, 'r-', label='MAE')
    plt.title('Error Metrics (RMSE, MAE) over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    # Set y-axis lower bound to 0 for error plots
    min_error = min([min(rmse_scores), min(mae_scores)])
    max_error = max([max(rmse_scores), max(mae_scores)])
    plt.ylim(0, max_error * 1.1) # Start y-axis at 0
    plt.savefig(output_dir / f"error_metrics_{timestamp}.png")
    plt.close()

    # --- Plot F1 Score ---
    plt.figure(figsize=(12, 6))
    # <<< Adjust metric key >>>
    f1_scores = [m['eval_f1_score'] for m in metrics['eval_metrics']]
    plt.plot(steps, f1_scores, 'm-', label='Weighted F1 Score')
    plt.title('Weighted F1 Score over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1) # F1 is between 0 and 1
    plt.savefig(output_dir / f"f1_score_metrics_{timestamp}.png")
    plt.close()

    # --- Plot Training Loss ---
    if metrics['train_loss']:
        plt.figure(figsize=(12, 6))
        train_loss = [m['loss'] for m in metrics['train_loss']]
        epochs = [m['epoch'] for m in metrics['train_loss']] # Use epoch numbers
        plt.plot(epochs, train_loss, 'r-', label='Average Training Loss per Epoch')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        min_loss = min(train_loss)
        max_loss = max(train_loss)
        plt.ylim(0, max_loss * 1.1) # Start y-axis at 0
        plt.savefig(output_dir / f"loss_plot_{timestamp}.png")
        plt.close()

# ----- Save final training metrics and generate plots -----
# <<< Use the correct output directory >>>
final_metrics_path = Path(training_args.output_dir) / "final_training_metrics.json"
with open(final_metrics_path, "w") as f:
    json.dump(training_metrics, f, indent=2)
print(f"Final training metrics saved to {final_metrics_path}")

# Generate and save plots
plot_training_metrics(training_metrics, training_args.output_dir)

print(f"Training complete! Model saved in {training_args.output_dir}. Metrics plots saved.")