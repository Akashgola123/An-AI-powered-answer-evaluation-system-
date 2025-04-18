from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
from peft import LoraConfig
import torch


model_id = "llama3.2:latest"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


model, tokenizer = FastLanguageModel.from_pretrained(
    model_id=model_id,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)


lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
)

model = FastLanguageModel.get_peft_model(model, lora_config)

def prepare_dataset():
    dataset = load_dataset("chetan ka dataset") 
    
    def format_qa_with_copy_detection(example):
        context = example['context']
        question = example['question']
        answer = example['answers']['text'][0]

        original_text = f"Original text: {context}\n"
        copy_text = f"Potential copy: {answer}\n"
        similarity_task = "Task: Determine if the potential copy is derived from the original text.\n"
        

        text = f"{original_text}{copy_text}{similarity_task}Question: {question}\nAnswer: {answer}"
        

        similarity_score = float(len(set(context.split()) & set(answer.split())) / len(set(answer.split()))) if answer else 0.0
        
        return {
            "text": text,
            "similarity_score": similarity_score
        }
    
    return dataset.map(format_qa_with_copy_detection)


def compute_loss(model, inputs, similarity_scores):

    outputs = model(**inputs)
    lm_loss = outputs.loss
    
    hidden_states = outputs.hidden_states[-1][:, 0]  
    batch_size = hidden_states.shape[0]
    
    similarity_matrix = torch.matmul(hidden_states, hidden_states.t())
    similarity_matrix = similarity_matrix - torch.eye(batch_size).to(similarity_matrix.device)
   
    contrastive_loss = torch.mean(torch.abs(similarity_matrix - similarity_scores))
    
    # Combined loss
    total_loss = lm_loss + 0.1 * contrastive_loss
    return total_loss

training_args = TrainingArguments(
    output_dir="./llama2-qa-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
  
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

trainer = FastLanguageModel.get_trainer(
    model=model,
    train_dataset=prepare_dataset()["train"],
    args=training_args,
    tokenizer=tokenizer,
)


trainer.train()

trainer.save_model("./llama2-qa-finetuned-final")




"""torch>=2.0.0
transformers>=4.36.0
unsloth>=0.3.0
datasets>=2.15.0
tqdm>=4.66.0
peft>=0.7.0
accelerator>=0.25.0 """