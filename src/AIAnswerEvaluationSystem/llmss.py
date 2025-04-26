
import torch
# Check if 'unsloth' is installed, fallback to standard transformers if not
try:
    from unsloth import FastLanguageModel
    # If using Unsloth, we might using a  4bit loading is handled efficiently
    print("Unsloth found. Using FastLanguageModel.")
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not found. Falling back to standard Transformers.")
    UNSLOTH_AVAILABLE = False
 
    from transformers import AutoModelForCausalLM as FastLanguageModel

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
import re
import logging
import time
from typing import Dict, Optional, Any, List 

# --- Configure Logging for this Module ---
log = logging.getLogger(__name__)

# --- Configuration (Defaults) ---
# These can be overridden when creating an instance
DEFAULT_MODEL_PATH = "/home/gola/GRAPH_RAG/Exam_Portal/llama-3.2-3b-qa-finetuned/final_model" # Default path to merged/adapted model
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LOAD_IN_4BIT = True # Recommended if Unsloth available or resource constrained
DEFAULT_MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

DEFAULT_GEN_MAX_NEW_TOKENS = 300 # Default max tokens for evaluation output (score+feedback)
DEFAULT_GEN_DO_SAMPLE = False    # Deterministic output by default
DEFAULT_GEN_TEMPERATURE = 0.0    # Greedy decoding (ignored if do_sample=False)

class FineTunedLLMEvaluator:
    """
    Handles loading and interacting with the fine-tuned language model
    for student answer evaluation. Encapsulates model loading, prompt formatting,
    text generation, and parsing of score/feedback from the specific
    fine-tuned output format.
    """
    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
                 dtype: Optional[torch.dtype] = None, # Allow user to specify or default
                 hf_token: Optional[str] = None,
                 device: Optional[str] = None
                 ):
        """
        Initializes the evaluator configuration. Model loading is done lazily.

        :param model_path: Path to the saved fine-tuned model/adapter directory.
        :param max_seq_length: Max sequence length for the model.
        :param load_in_4bit: Whether to load the base model in 4-bit precision (requires bitsandbytes).
                             Ignored if Unsloth handles it implicitly or if not using Unsloth/bnb.
        :param dtype: Data type for model loading (e.g., torch.bfloat16). Defaults based on support.
        :param hf_token: Optional Hugging Face token for gated base models if not cached.
        :param device: Optional device override ('cuda', 'cpu'). Defaults to cuda if available.
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit if UNSLOTH_AVAILABLE else False # Only rely on if Unsloth present or explicitly using bitsandbytes
        self.dtype = dtype if dtype else DEFAULT_MODEL_DTYPE # Use default if not specified
        self.hf_token = hf_token

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        # Determine device or use specified
        self.device: str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded: bool = False
        self._load_attempted: bool = False
        log.info(f"LLMEvaluator configured for model: {self.model_path} on device {self.device}")

    def _load_model_and_tokenizer(self) -> bool:
        """Internal method to load model/tokenizer. Returns True on success."""
        if self._is_loaded: return True
        if self._load_attempted: log.error("Skipping load attempt, previous failed."); return False

        self._load_attempted = True
        load_start = time.time()
        log.info(f"Attempting to load model and tokenizer from: {self.model_path}")
        try:
            # --- Loading Logic ---
            # Check if Unsloth is available and should be used
            if UNSLOTH_AVAILABLE:
                log.debug("Using Unsloth FastLanguageModel for loading.")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path, # Point to saved adapter/merged model
                    max_seq_length=self.max_seq_length,
                    dtype=self.dtype,
                    load_in_4bit=self.load_in_4bit, # Unsloth handles 4-bit internally
                    token=self.hf_token,
                    # device_map = "auto", # Unsloth often handles device mapping well
                )
            else:
                # --- Standard Transformers Loading ---
                # Note: Merged adapter recommended for standard loading. Loading LoRA separately is possible but adds complexity here.
                # Ensure 'bitsandbytes' is installed if using 4bit without Unsloth
                bnb_config = None
                if self.load_in_4bit:
                     try:
                         import bitsandbytes # noqa F401 - Check if installed
                         from transformers import BitsAndBytesConfig
                         bnb_config = BitsAndBytesConfig(
                              load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=self.dtype, # Use provided dtype
                              bnb_4bit_use_double_quant=True,
                         )
                         log.info("Configured BitsAndBytes for 4-bit loading.")
                     except ImportError:
                          log.error("BitsAndBytes not installed, cannot load in 4-bit without Unsloth. Loading in standard precision.")
                          self.load_in_4bit = False # Disable 4-bit flag


                log.debug("Using standard Transformers AutoModel/AutoTokenizer.")
                model = AutoModelForCausalLM.from_pretrained( # Use base class
                     self.model_path,
                     torch_dtype=self.dtype,
                     quantization_config=bnb_config, # Apply config if created
                     device_map="auto", # Let transformers handle device mapping
                     token=self.hf_token,
                )
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.hf_token)


            # Assign to instance variables
            self.model = model
            self.tokenizer = tokenizer

            # No explicit .to(device) needed if using device_map='auto' or Unsloth default
            # But good practice to set eval mode
            self.model.eval()

            # Ensure tokenizer has pad token configured
            if self.tokenizer.pad_token is None:
                 log.warning("Tokenizer missing pad token, setting to EOS token.")
                 self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                 log.debug(f"Set tokenizer pad_token_id to: {self.tokenizer.pad_token_id}")

            self._is_loaded = True
            load_time = time.time() - load_start
            log.info(f"Successfully loaded model and tokenizer (device assigned by framework) in {load_time:.2f}s.")
            return True

        except FileNotFoundError:
             log.exception(f"Model directory not found: {self.model_path}. Please check path.")
             self._is_loaded = False
             return False
        except ImportError as ie:
             log.exception(f"Import error during loading. Do you need to install 'bitsandbytes' for 4-bit? Error: {ie}")
             self._is_loaded = False
             return False
        except Exception as e:
            log.exception(f"FATAL Error loading model/tokenizer from {self.model_path}")
            self._is_loaded = False
            return False

    def _ensure_loaded(self) -> None:
        """Checks if model is loaded, attempts loading if not. Raises RuntimeError if loading failed."""
        if not self._is_loaded:
             if not self._load_attempted:
                  if not self._load_model_and_tokenizer():
                      raise RuntimeError(f"Failed to load LLM model from {self.model_path}. Check logs.")
             else: # Load was attempted previously but failed
                  raise RuntimeError(f"LLM Model at {self.model_path} failed to load previously and will not be re-attempted.")
        if self.model is None or self.tokenizer is None: # Safeguard check
             self._is_loaded = False
             raise RuntimeError("LLM model state inconsistent after loading attempt.")

    def format_prompt(self, question: str, student_answer: str, correct_answer: str) -> Optional[str]:
        """Formats prompt using Llama-3.2 Instruct chat template."""
        try:
            self._ensure_loaded() # Ensure tokenizer is ready
            system_prompt = "You are an educational assistant that evaluates student answers."
            user_message = f"Subject: [Optional Subject]\nTopic: [Optional Topic]\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {correct_answer}" # Use correct_answer here
            messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_message}]
            # Important for inference with chat models!
            return self.tokenizer.apply_chat_template(
                 messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            log.exception("Error formatting prompt.")
            return None

    @torch.inference_mode() # Crucial decorator for inference efficiency
    def generate_raw_output(self, prompt_text: str) -> Optional[str]:
        """Generates raw model output text given the formatted prompt."""
        if not prompt_text: log.warning("generate_raw_output received empty prompt."); return None
        try:
            self._ensure_loaded() # Ensure model and tokenizer are loaded and on device
            device = self.model.device # Get actual device model is on

            try:
                # Tokenize: Ensure padding strategy for single input is suitable
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
            except Exception as tok_e: log.exception("Tokenization failed"); return None

            generation_config = {
                "max_new_tokens": DEFAULT_GEN_MAX_NEW_TOKENS,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": DEFAULT_GEN_DO_SAMPLE,
                "temperature": DEFAULT_GEN_TEMPERATURE,
            }
            # Optional: Add other parameters if needed, e.g., repetition_penalty=1.1

            gen_start = time.time()
            # Note: Unsloth's generate might be faster if available
            outputs = self.model.generate(**inputs, **generation_config)
            gen_time = time.time() - gen_start
            log.info(f"Generation completed in {gen_time:.2f}s")

            prompt_len = inputs['input_ids'].shape[1]
            output_tokens = outputs[0, prompt_len:]
            # Robustly filter padding tokens ONLY if pad != eos
            if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                actual_output_tokens = output_tokens[output_tokens != self.tokenizer.pad_token_id]
            else:
                actual_output_tokens = output_tokens # If pad == eos, eos is handled by skip_special

            decoded_output = self.tokenizer.decode(actual_output_tokens, skip_special_tokens=True)
            return decoded_output.strip()

        except Exception as e:
            log.exception(f"Error during model generation")
            return None

    def parse_output(self, generated_text: Optional[str]) -> Dict[str, Any]:
        """
        Parses the fine-tuned model's output to extract score (0-5) and feedback.
        Expects specific format learned during fine-tuning (e.g., "Score: X/5", "Feedback: ...")
        """
        parsed: Dict[str, Any] = {"numeric_score": None, "feedback": None} # Changed key
        if not generated_text or not generated_text.strip(): log.warning("[Parser] Empty text received"); return parsed

        text_lower = generated_text.lower(); score, feedback = None, None

        # --- Score Extraction ---
        score_match = re.search(r"score:\s*(\d)\s*/\s*5", text_lower)
        score_line_end = -1
        if score_match:
            try: score = max(0, min(5, int(score_match.group(1)))); parsed["numeric_score"] = score; score_match_end_pos = score_match.end(); line_remainder = generated_text[score_match_end_pos:].split('\n',1)[0]; score_line_end = score_match_end_pos + len(line_remainder)
            except (ValueError, IndexError): log.warning(f"[Parser] Invalid int score: '{score_match.group(0)}'")

        # --- Feedback Extraction ---
        feedback_match = re.search(r"feedback:\s*(.*)", text_lower, re.DOTALL); plag_match = re.search(r"plagiarism:\s*(.*)", text_lower, re.DOTALL); plag_start = plag_match.start() if plag_match else -1
        if feedback_match:
            fb_start = feedback_match.start(); potential_fb = feedback_match.group(1).strip()
            if plag_start > fb_start: rel_plag_start = potential_fb.lower().find("plagiarism:"); feedback = potential_fb[:rel_plag_start].strip() if rel_plag_start != -1 else potential_fb
            else: feedback = potential_fb
        elif score_line_end != -1: # Fallback based on score position
             potential_fb = generated_text[score_line_end:].strip()
             if potential_fb: plag_start_fb = potential_fb.lower().find("plagiarism:"); feedback = potential_fb[:plag_start_fb].strip() if plag_start_fb != -1 else potential_fb

        parsed["feedback"] = feedback.strip() if feedback else None # Final cleanup

        if parsed["numeric_score"] is None: log.warning(f"[Parser] Score missing: {generated_text[:150]}...")
        if parsed["feedback"] is None: log.warning(f"[Parser] Feedback missing: {generated_text[:150]}...")
        return parsed
