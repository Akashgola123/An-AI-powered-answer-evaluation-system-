import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMManager:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the LLM model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_response(self, input_text, max_length=1024, temperature=0.7, top_p=0.9):
        """Generates text response using the LLM model."""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
