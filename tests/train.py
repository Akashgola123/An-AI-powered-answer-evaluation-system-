from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, get_scheduler
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Use consistent dtype - IMPORTANT: We're setting dtype explicitly to bfloat16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.bfloat16,  # Explicitly use bfloat16 throughout
    load_in_4bit=True,
    use_gradient_checkpointing=True
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
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load and prepare dataset
def prepare_dataset():
    # Load dataset from local JSON file
    try:
        raw_dataset = load_dataset("json", data_files="/home/gola/GRAPH_RAG/Exam_Portal/An-AI-powered-answer-evaluation-system-/dataset.json")
        
        # If loaded as Dataset instead of DatasetDict, create train/test split
        if "train" in raw_dataset:
            # Check if the dataset already has train and validation splits
            if "validation" not in raw_dataset:
                # Create validation split from train data if it doesn't exist
                splits = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
                dataset = DatasetDict({
                    "train": splits["train"],
                    "validation": splits["test"]
                })
            else:
                dataset = raw_dataset
        else:
            # If there's just a single Dataset object (not a DatasetDict)
            splits = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                "train": splits["train"],
                "validation": splits["test"]
            })
        
        print(f"Dataset splits: {list(dataset.keys())}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        
        def format_student_answer_evaluation(example):
            try:
                # Extract fields from the dataset
                instruction = example.get('instruction', '')
                
                # Extract input fields
                input_data = example.get('input', {})
                subject = input_data.get('subject', '')
                topic = input_data.get('topic', '')
                question = input_data.get('question', '')
                student_answer = input_data.get('student_answer', '')
                model_answer = input_data.get('model_answer', '')
                concepts = input_data.get('concepts', [])
                rubric = input_data.get('rubric', {})
                
                # Extract output fields
                output_data = example.get('output', {})
                score = output_data.get('score', 0)
                feedback = output_data.get('feedback', '')
                plagiarism = output_data.get('plagiarism', '')
                
                # Format in chat template for LLaMA 3.2 models
                system_prompt = "You are an educational assistant that evaluates student answers."
                user_message = f"Subject: {subject}\nTopic: {topic}\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"
                assistant_message = f"Score: {score}/5\nFeedback: {feedback}\nPlagiarism: {plagiarism}"
                
                # Format using the model's chat template
                formatted_input = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message}
                    ],
                    tokenize=False
                )
                
                # Tokenize the formatted input
                tokenized_input = tokenizer(formatted_input, truncation=True, max_length=2048)
                
                # Additional metrics
                student_words = set(student_answer.lower().split())
                model_words = set(model_answer.lower().split())
                word_overlap = len(student_words & model_words)
                similarity_score = word_overlap / len(model_words) if model_words else 0.0
                
                concepts_mentioned = sum(1 for concept in concepts if concept.lower() in student_answer.lower())
                concept_coverage = concepts_mentioned / len(concepts) if concepts else 0.0
                
                return {
                    "input_ids": tokenized_input["input_ids"],
                    "attention_mask": tokenized_input["attention_mask"],
                    "similarity_score": float(similarity_score),
                    "concept_coverage": float(concept_coverage),
                    "word_overlap": int(word_overlap),
                    "evaluation_score": int(score),
                    "raw_score": score,  # Keep raw score for evaluation
                    "raw_feedback": feedback,  # Keep raw feedback for evaluation
                    "raw_plagiarism": plagiarism,  # Keep raw plagiarism assessment
                    "subject": subject,  # Keep for analysis
                    "question": question  # Keep for analysis
                }
            except Exception as e:
                print(f"Error processing example: {e}")
                empty_encoding = tokenizer("", truncation=True, max_length=2048)
                return {
                    "input_ids": empty_encoding["input_ids"],
                    "attention_mask": empty_encoding["attention_mask"],
                    "similarity_score": 0.0,
                    "concept_coverage": 0.0,
                    "word_overlap": 0,
                    "evaluation_score": 0,
                    "raw_score": 0,
                    "raw_feedback": "",
                    "raw_plagiarism": "",
                    "subject": "",
                    "question": ""
                }
        
        # Apply preprocessing to all splits
        processed_dataset = DatasetDict()
        for split in dataset:
            processed_dataset[split] = dataset[split].map(
                format_student_answer_evaluation,
                remove_columns=dataset[split].column_names  # Remove original columns
            )
        
        return processed_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e  # Re-raise to see the full error

# Custom collate function
def collate_fn(batch):
    """
    Custom collate function to convert list inputs to tensors
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch], 
        batch_first=True, 
        padding_value=0
    )
    
    # Other metrics as tensors
    similarity_scores = torch.tensor([x["similarity_score"] for x in batch], dtype=torch.float32)
    concept_coverage = torch.tensor([x["concept_coverage"] for x in batch], dtype=torch.float32)
    word_overlap = torch.tensor([x["word_overlap"] for x in batch], dtype=torch.int32)
    evaluation_score = torch.tensor([x["evaluation_score"] for x in batch], dtype=torch.float32)
    
    # Keep raw values for evaluation
    raw_scores = [x["raw_score"] for x in batch]
    raw_feedback = [x["raw_feedback"] for x in batch]
    raw_plagiarism = [x["raw_plagiarism"] for x in batch]
    subjects = [x["subject"] for x in batch]
    questions = [x["question"] for x in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "similarity_score": similarity_scores,
        "concept_coverage": concept_coverage,
        "word_overlap": word_overlap,
        "evaluation_score": evaluation_score,
        "raw_score": raw_scores,
        "raw_feedback": raw_feedback,
        "raw_plagiarism": raw_plagiarism,
        "subject": subjects,
        "question": questions
    }

# Function to evaluate model accuracy
def evaluate_model(model, eval_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    eval_loss = 0
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Model inputs
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["input_ids"].clone()
            }
            
            # Forward pass
            outputs = model(**model_inputs)
            loss = outputs.loss
            eval_loss += loss.item()
            
            # Get the predicted scores from logits
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            predicted_token = torch.argmax(last_token_logits, dim=-1)
            
            # Convert predicted token to score (assuming score tokens are mapped)
            predicted_scores = [int(token.item()) % 6 for token in predicted_token]  # Map to 0-5 range
            
            # Get true scores
            true_scores = batch["evaluation_score"].cpu().numpy().tolist()
            
            # Update metrics
            total_samples += len(true_scores)
            correct_predictions += sum(1 for pred, true in zip(predicted_scores, true_scores) 
                                     if abs(pred - true) <= 1)  # Allow 1-point difference
            
            # Store predictions and labels for overall metrics
            all_preds.extend(predicted_scores)
            all_labels.extend(true_scores)
    
    # Calculate metrics
    accuracy = correct_predictions / total_samples  # Relaxed accuracy
    strict_accuracy = accuracy_score([round(l) for l in all_labels], 
                                   [round(p) for p in all_preds])  # Strict accuracy
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    f1 = f1_score([round(l) for l in all_labels], 
                  [round(p) for p in all_preds], 
                  average='weighted')
    
    # Calculate additional metrics
    mae = np.mean([abs(p - l) for p, l in zip(all_preds, all_labels)])
    within_one_point = sum(1 for p, l in zip(all_preds, all_labels) if abs(p - l) <= 1) / len(all_labels)
    
    model.train()
    return {
        "eval_loss": eval_loss / len(eval_dataloader),
        "relaxed_accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "rmse": rmse,
        "f1_score": f1,
        "mae": mae,
        "within_one_point": within_one_point
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-qa-finetuned",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=500,
    learning_rate=2e-4,
    lr_scheduler_type="cosine", 
    bf16=True,  # Using bfloat16
    fp16=False,  # Not using fp16
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    optim="adamw_torch_fused",
    report_to="tensorboard",
)

# Load and preprocess dataset
processed_dataset = prepare_dataset()

# Check to make sure we have validation data
if "validation" not in processed_dataset:
    # Create validation split if it doesn't exist
    splits = processed_dataset["train"].train_test_split(test_size=0.1, seed=42)
    processed_dataset = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"]
    })

# Create DataLoaders
train_dataloader = DataLoader(
    processed_dataset["train"], 
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

eval_dataloader = DataLoader(
    processed_dataset["validation"], 
    batch_size=training_args.per_device_eval_batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Initialize optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=0.01)

# Setup learning rate scheduler
num_training_steps = len(processed_dataset["train"]) * training_args.num_train_epochs
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# Choose between mixed precision training or full precision based on user preference
use_amp = training_args.bf16  # Using bf16 precision as specified in training args

# Training metrics tracking
training_metrics = {
    "train_loss": [],
    "eval_metrics": [],
    "learning_rates": []
}

# Training loop with memory-efficient mixed precision
device = model.device
model.train()

# Print information about device and mixed precision
print(f"Using device: {device}")
print(f"Model dtype: {model.dtype}")
print(f"Using mixed precision: {use_amp} (bf16)")

for epoch in range(int(training_args.num_train_epochs)):
    epoch_loss = 0
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(train_dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Extract only what the model needs
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["input_ids"].clone()
        }
        
        # Mixed precision training using torch.amp.autocast instead of torch.cuda.amp.autocast
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss / training_args.gradient_accumulation_steps
        else:
            outputs = model(**model_inputs)
            loss = outputs.loss / training_args.gradient_accumulation_steps
            
        # Backward pass (no scaler needed for bfloat16)
        loss.backward()
            
        epoch_loss += loss.item() * training_args.gradient_accumulation_steps

        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item() * training_args.gradient_accumulation_steps})
        
        # Evaluate and log metrics
        if (step + 1) % training_args.eval_steps == 0:
            eval_metrics = evaluate_model(model, eval_dataloader, device)
            training_metrics["eval_metrics"].append({
                "step": step + 1 + epoch * len(train_dataloader),
                **eval_metrics
            })
            
            print(f"\nStep {step+1} Eval Metrics:")
            print(f"Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"RMSE: {eval_metrics['rmse']:.4f}")
            print(f"F1 Score: {eval_metrics['f1_score']:.4f}\n")
    
    # Track epoch metrics
    training_metrics["train_loss"].append({
        "epoch": epoch + 1,
        "loss": epoch_loss / len(train_dataloader)
    })
    
    # Save checkpoint
    model.save_pretrained(f"./llama2-qa-finetuned/checkpoint-{epoch+1}")
    
    # Save training metrics
    with open(f"./llama2-qa-finetuned/training_metrics_epoch_{epoch+1}.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    # Run full evaluation at end of epoch
    eval_metrics = evaluate_model(model, eval_dataloader, device)
    print(f"\nEpoch {epoch+1} Complete Evaluation:")
    print(f"Loss: {eval_metrics['eval_loss']:.4f}")
    print(f"Accuracy: {eval_metrics['relaxed_accuracy']:.4f}")
    print(f"RMSE: {eval_metrics['rmse']:.4f}")
    print(f"F1 Score: {eval_metrics['f1_score']:.4f}\n")

# Save the final model
model.save_pretrained("./llama2-qa-finetuned-final")

# Plot and save training metrics
def plot_training_metrics(metrics, save_dir):
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot accuracies over time
    plt.figure(figsize=(12, 6))
    steps = [m['step'] for m in metrics['eval_metrics']]
    relaxed_accuracies = [m['relaxed_accuracy'] for m in metrics['eval_metrics']]
    strict_accuracies = [m['strict_accuracy'] for m in metrics['eval_metrics']]
    within_one = [m['within_one_point'] for m in metrics['eval_metrics']]
    
    plt.plot(steps, relaxed_accuracies, 'b-', label='Relaxed Accuracy')
    plt.plot(steps, strict_accuracies, 'g-', label='Strict Accuracy')
    plt.plot(steps, within_one, 'r--', label='Within One Point')
    plt.title('Model Accuracy Metrics over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy_metrics_{timestamp}.png")
    plt.close()
    
    # Plot error metrics
    plt.figure(figsize=(12, 6))
    rmse_scores = [m['rmse'] for m in metrics['eval_metrics']]
    mae_scores = [m['mae'] for m in metrics['eval_metrics']]
    
    plt.plot(steps, rmse_scores, 'b-', label='RMSE')
    plt.plot(steps, mae_scores, 'r-', label='MAE')
    plt.title('Error Metrics over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/error_metrics_{timestamp}.png")
    plt.close()
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    train_loss = [m['loss'] for m in metrics['train_loss']]
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_plot_{timestamp}.png")
    plt.close()

# Save final training metrics and generate plots
with open("./llama2-qa-finetuned/final_training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=2)

# Generate and save plots
plot_training_metrics(training_metrics, "./llama2-qa-finetuned")

print("Training complete! Metrics plots have been saved in the output directory.")