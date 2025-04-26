# --- Imports ---
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import re
import json
import logging
import time
from typing import Dict, Optional, Tuple, Any, List # Added List, Any
from neo4j import GraphDatabase
import os
# tqdm is used in AnswerEvaluationSystem, make sure it's imported
try:
    from tqdm.auto import tqdm
except ImportError:
    # Provide a dummy tqdm if not installed, or raise error
    logging.warning("tqdm library not found. Progress bars will be disabled.")
    def tqdm(iterable, *args, **kwargs):
        return iterable


# --- Configure Logging ---
# Use a more specific logger name if preferred
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("EvaluationSystem") # Assign to a variable


# --- Neo4j Connection Class (with Corrected Fetch Query) ---
class Neo4jConnection:
    """Handles connections and queries to Neo4j database."""
    def __init__(self, uri: str, username: str, password: str):
        try:
            neo4j_logger = logging.getLogger("neo4j")
            neo4j_logger.setLevel(logging.WARNING)
            self._driver = GraphDatabase.driver(uri, auth=(username, password))
            self._driver.verify_connectivity()
            log.info(f"Neo4j connection initialized and verified for URI: {uri}") # Use 'log' variable
        except Exception as e:
            log.exception(f"FATAL: Failed to initialize/verify Neo4j connection to {uri}") # Use 'log' variable
            self._driver = None
            raise

    def close(self):
        if hasattr(self, '_driver') and self._driver:
            self._driver.close()
            log.info("Neo4j connection closed.") # Use 'log' variable

    def query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]: # Added Optional type hint
        """Executes a Cypher query and returns list of results, or empty list on error."""
        if parameters is None: parameters = {}
        if not hasattr(self, '_driver') or self._driver is None:
            log.error("Cannot execute query, driver not initialized.")
            return []
        try:
            # Recommended: Use specific database name if not default 'neo4j'
            # database_name = "neo4j" # Or get from config/env
            # with self._driver.session(database=database_name) as session:
            with self._driver.session() as session: # Uses default DB
                result = session.run(query, parameters)
                # Consume results within session context
                data = [record.data() for record in result]
                log.debug(f"Query successful, returned {len(data)} records for query: {query[:100]}...")
                return data
        except Exception as e:
            log.exception(f"Neo4j query failed. Query: {query}, Params: {parameters}")
            # log.error(f"Query: {query}") # Already included in exception log
            # log.error(f"Parameters: {parameters}")
            return [] # Return empty list on failure

    def fetch_student_answers(self, student_name: str = None, question_id: str = None) -> List[Dict]:
        """
        Fetches submitted answers for filtering criteria, including related Q&A text.
        Uses the ANSWERS_QUESTION relationship from Answer to Question.
        """
        # Base match includes Student -> Submission -> Answer -> Question
        # And Submission -> Subject
        match_clause = """
        MATCH (s:Student)-[:SUBMITTED]->(e:ExamSubmission)
              -[:HAS_ANSWER]->(a:Answer)-[:ANSWERS_QUESTION]->(q:Question)
        MATCH (e)-[:FOR_SUBJECT]->(sub:Subject)
        """

        # --- Build WHERE Clause ---
        where_clauses = []
        params = {}
        if student_name:
            # Make sure the property key matches your Student nodes ('id' or 'roll_no'?)
            # Assuming 'roll_no' based on previous context, but check your DB schema.
            # Let's use s.roll_no here assuming it's more standard
            where_clauses.append("s.roll_no = $p_student_name")
            params["p_student_name"] = student_name
        if question_id:
            where_clauses.append("q.id = $p_question_id")
            params["p_question_id"] = question_id

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        # --- Build RETURN Clause ---
        return_clause = """
        OPTIONAL MATCH (q)-[:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer) // Handle missing correct answers
        RETURN s.roll_no as student_roll_no, // Use roll_no consistently? Or s.id if that's the key
               s.name as student_name,      // Add student name
               sub.name as subject_name,      // Subject name
               q.id as question_id,
               q.text as question_text,        // Changed alias
               a.text as student_answer,      // Changed alias
               COALESCE(ca.text, "[No Correct Answer Defined]") as correct_answer, // Use correct alias
               COALESCE(q.max_marks, 'N/A') as max_marks // Consistent alias, better default
        ORDER BY e.submitted_at DESC, q.id // Order by submission time, then question
        """

        # --- Construct and Execute ---
        query = f"{match_clause} {where_clause} {return_clause}"
        log.info(f"Fetching student answers... (Student='{student_name or 'ALL'}', QID='{question_id or 'ALL'}')")
        return self.query(query, params) # Call the corrected query method


# --- Grading System Class (Keep As Is - Works with 0-5) ---
class GradingSystem:
    """Handles converting evaluation scores to letter grades and determining pass/fail status."""
    GRADE_SCALE = {(90, 100): 'A+',(80, 99.99): 'A',(60, 79.99): 'B',(40, 59.99): 'C',(20, 39.99): 'D',(0, 19.99): 'F'}
    PASSING_THRESHOLD_SCORE = 3 # Minimum score (0-5) to pass

    @classmethod
    def get_grade_details(cls, score: Optional[int], max_possible_score: int = 5) -> Dict[str, Any]:
        # ... (keep existing implementation) ...
        grade_details = {"percentage": None, "letter_grade": None, "passing": None}
        if score is None or score < 0 or max_possible_score <= 0:
             log.warning(f"Invalid grading input: score={score}, max_score={max_possible_score}")
             return grade_details
        percentage = round((score / max_possible_score) * 100, 2)
        grade_details["percentage"] = percentage
        letter_grade = 'N/A'
        for score_range, grade in cls.GRADE_SCALE.items():
             if score_range[0] <= percentage <= score_range[1]: letter_grade = grade; break
        # Handle edge case for 100% maybe if scale allows
        if percentage == 100 and (90,100) in cls.GRADE_SCALE: letter_grade = cls.GRADE_SCALE[(90,100)]
        grade_details["letter_grade"] = letter_grade
        grade_details["passing"] = score >= cls.PASSING_THRESHOLD_SCORE
        return grade_details

# --- Fine-Tuned Model Interaction (Keep As Is) ---
# Includes: model, tokenizer, device, is_model_loaded globals
# Includes: FINETUNED_MODEL_PATH, MAX_SEQ_LENGTH, LOAD_IN_4BIT, MODEL_DTYPE configs
# Includes: load_finetuned_model(), format_prompt_for_finetuned(),
#           generate_evaluation(), parse_score_and_feedback() functions
# (Make sure these are present and correct from previous examples)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
is_model_loaded = False
FINETUNED_MODEL_PATH = "./llama-3.2-3b-qa-finetuned/final_model" # Adjust path if needed
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_finetuned_model():
    global model, tokenizer, device, is_model_loaded
    if is_model_loaded: return model is not None and tokenizer is not None
    # ... (keep full loading logic from previous working example) ...
    load_start = time.time()
    log.info(f"[Model Loader] Loading model from {FINETUNED_MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained( FINETUNED_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH, dtype=MODEL_DTYPE, load_in_4bit=LOAD_IN_4BIT)
        model.to(device); model.eval()
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info(f"[Model Loader] Model loaded to {device} in {time.time() - load_start:.2f}s.")
        is_model_loaded = True; return True
    except Exception as e: log.exception(f"[Model Loader] FATAL Error"); model, tokenizer = None, None; is_model_loaded = True; return False

def format_prompt_for_finetuned(question: str, student_answer: str, model_answer: str) -> Optional[str]:
     if not is_model_loaded or tokenizer is None:
         if not load_finetuned_model(): return None
     system_prompt = "You are an educational assistant that evaluates student answers."
     user_message = f"Subject: [Sub]\nTopic: [Topic]\nQuestion: {question}\nStudent Answer: {student_answer}\nModel Answer: {model_answer}"
     messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_message}]
     try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
     except Exception as e: log.exception("[Prompt Formatter] Error"); return None

@torch.inference_mode()
def generate_evaluation(prompt_text: str) -> Optional[str]:
     if not is_model_loaded or model is None:
         if not load_finetuned_model(): return None
     # ... (keep generate_evaluation implementation from previous working example) ...
     try: inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
     except Exception as e: log.exception("Tokenization Error"); return None
     generation_config = {"max_new_tokens": 300, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "do_sample": False, "temperature": 0.0}
     try:
         outputs = model.generate(**inputs, **generation_config)
         prompt_len = inputs['input_ids'].shape[1]; output_tokens = outputs[0, prompt_len:]
         actual_output_tokens = output_tokens[output_tokens != tokenizer.pad_token_id]
         decoded_output = tokenizer.decode(actual_output_tokens, skip_special_tokens=True)
         return decoded_output.strip()
     except Exception as e: log.exception("Generation Error"); return None


def parse_score_and_feedback(generated_text: str) -> Dict[str, Any]:
     # ... (keep robust parse_score_and_feedback implementation from previous example) ...
     parsed = {"score": None, "feedback": None}; text_lower = generated_text.lower() if generated_text else ""
     score, feedback, score_line_end = None, None, -1
     if generated_text:
          score_match = re.search(r"score:\s*(\d)\s*/\s*5", text_lower)
          if score_match:
               try: score = max(0,min(5, int(score_match.group(1)))); parsed["score"] = score; score_line_end = score_match.end()
               except ValueError: log.warning("Invalid int in score pattern")
          feedback_match = re.search(r"feedback:\s*(.*)", text_lower, re.DOTALL)
          plag_match = re.search(r"plagiarism:\s*(.*)", text_lower)
          plag_start = plag_match.start() if plag_match else -1
          if feedback_match:
               fb_start = feedback_match.start(); potential_fb = feedback_match.group(1).strip()
               if plag_start > fb_start: feedback = potential_fb[:plag_start - fb_start - len("feedback:")].strip()
               else: feedback = potential_fb
          elif score_line_end != -1: # Fallback after score
               potential_fb = generated_text[score_line_end:].strip()
               if potential_fb:
                    plag_start_fb = potential_fb.lower().find("plagiarism:")
                    if plag_start_fb != -1: feedback = potential_fb[:plag_start_fb].strip()
                    else: feedback = potential_fb
          parsed["feedback"] = feedback
          if score is None: log.warning(f"No SCORE parsed: {generated_text[:150]}...")
          if feedback is None: log.warning(f"No FEEDBACK parsed: {generated_text[:150]}...")
     else: log.warning("Empty text to parse")
     return parsed

# --- AnswerEvaluationSystem Class (Relies on above functions/classes) ---
class AnswerEvaluationSystem:
    def __init__(self,
                 neo4j_uri: str, # Make required
                 neo4j_user: str, # Make required
                 neo4j_password: str # Make required
                ):
        if not load_finetuned_model(): # Ensure model loads during init
             raise RuntimeError("Fine-tuned model loading failed.")
        self.neo4j = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password) # Connects on init
        log.info("AnswerEvaluationSystem initialized successfully.")

    def __del__(self):
        if hasattr(self, 'neo4j'):
            self.neo4j.close()

    def evaluate_single_answer(self,
                           question: str,
                           correct_answer: str,
                           student_answer: str) -> Dict[str, Any]:
        # ... (keep implementation from previous example, which uses the grading system) ...
        start_time = time.time()
        final_result = {"score": None, "percentage": None, "letter_grade": None, "status": None, "feedback": "Evaluation incomplete.", "raw_output": None, "error": None }
        prompt = format_prompt_for_finetuned(question, student_answer, correct_answer)
        if prompt is None: final_result["error"] = "Prompt format error."; return final_result
        raw_output = generate_evaluation(prompt)
        final_result["raw_output"] = raw_output
        if raw_output is None: final_result["error"] = "Model generation failed."; return final_result
        parsed_data = parse_score_and_feedback(raw_output); score = parsed_data.get("score"); feedback = parsed_data.get("feedback")
        final_result["score"] = score; final_result["feedback"] = feedback if feedback else "Feedback parsing failed."
        if score is not None and score >= 0:
            grade_details = GradingSystem.get_grade_details(score=score, max_possible_score=5)
            final_result.update({k:v for k,v in grade_details.items() if k != 'passing'}) # Update, exclude passing
            final_result['status'] = "Pass" if grade_details['passing'] else "Fail" # Use passing for status
        else:
             final_result["error"] = "Score parsing failed."; final_result["letter_grade"] = "N/A"; final_result["status"] = "N/A"
        log.info(f"Single eval done ({time.time()-start_time:.2f}s). Score: {score}")
        return final_result

    def fetch_and_evaluate_student_answers(self,
                                          student_name: str = None,
                                          question_id: str = None) -> List[Dict]:
        # ... (Keep implementation BUT make sure it uses the correct fetch_student_answers from Neo4jConnection) ...
        processed_results = []; log.info(f"Fetching/evaluating for student='{student_name}', qid='{question_id}'")
        try:
            # Calls the fetch method WITH THE CORRECTED QUERY using ANSWERS_QUESTION
            submissions = self.neo4j.fetch_student_answers(student_name, question_id)
            log.info(f"Fetched {len(submissions)} submissions from Neo4j.")
            if not submissions: return []

            # Use tqdm for progress if available
            submission_iterator = tqdm(submissions, desc="Evaluating DB Submissions") if 'tqdm' in globals() else submissions

            for sub in submission_iterator:
                 try:
                     # Extract data using .get with defaults
                     eval_result = self.evaluate_single_answer(
                        question=sub.get('question_text', ''), # Use correct key from corrected query
                        correct_answer=sub.get('correct_answer', ''), # Use correct key
                        student_answer=sub.get('student_answer', '') # Use correct key
                    )
                     processed_results.append({
                        "student_name": sub.get("student_name", "unknown"), # Add if needed
                        "student_roll_no": sub.get("student_roll_no", student_name or "unknown"), # Use roll_no key
                        "subject_name": sub.get("subject_name", "unknown"),
                        "question_id": sub.get("question_id", "unknown"),
                        "question_text": sub.get("question_text", ""),
                        "db_max_marks": sub.get("max_marks", None), # Marks from DB
                        "evaluation": eval_result # Nested results dict
                     })
                 except Exception as eval_err:
                     log.exception(f"Error evaluating submission for student {sub.get('student_roll_no')}, qid {sub.get('question_id')}")
                     # Add error placeholder to results
                     processed_results.append({"student_roll_no": sub.get("student_roll_no"), "question_id": sub.get("question_id"), "error": f"Eval failed: {eval_err}"})

            log.info(f"Finished evaluating {len(processed_results)} submissions.")
            return processed_results
        except Exception as fetch_err:
             log.exception("Error during fetch_and_evaluate_student_answers")
             return [{"error": f"Fetch/Process failed: {fetch_err}"}]

    def generate_grade_report(self, student_name: str) -> Dict:
        # ... (keep implementation, it uses the results from fetch_and_evaluate) ...
        # Needs `student_roll_no` key now from results
        log.info(f"Generating grade report for student: {student_name}")
        evaluation_results = self.fetch_and_evaluate_student_answers(student_name=student_name)
        # ... (rest of report generation logic, ensuring keys match processed_results items) ...
        # Adjust calculation based on score = eval_data.get("score") being 0-5
        if not evaluation_results or "error" in evaluation_results[0]: ... # Handle errors
        total_obtained, total_possible, valid_count = 0, 0, 0; question_details = []
        for result in evaluation_results:
             eval_data = result.get("evaluation", {}); score = eval_data.get("score")
             q_detail = {"question_id": result.get("question_id", "?"), "score": "Error", "status": "N/A", "grade": "N/A", "feedback_parsed": False}
             if score is not None and score >= 0:
                  total_obtained += score; total_possible += 5; valid_count += 1 # Use fixed max score of 5
                  q_detail.update({"score":score, "status":eval_data.get('status'), "grade":eval_data.get('letter_grade'), "feedback_parsed": bool(eval_data.get('feedback'))})
             question_details.append(q_detail)
        # ... Calculate overall and return report dict ...
        if valid_count == 0: return {"student_name": student_name, "error": "No valid evaluations."}
        overall_perc = round((total_obtained / total_possible) * 100, 2)
        overall_grade_details = GradingSystem.get_grade_details(score=total_obtained, max_possible_score=total_possible) # Grade overall sum

        report = {"student_name": student_name, "overall_summary": {"total_questions_evaluated": len(evaluation_results),"validly_scored_questions": valid_count,"total_score_obtained": total_obtained,"total_score_possible": total_possible,"overall_percentage": overall_perc,"overall_letter_grade": overall_grade_details.get("letter_grade"),"overall_status": "Pass" if overall_grade_details.get('passing') else "Fail"},"details_per_question": question_details}
        log.info(f"Grade report generated for {student_name}.")
        return report


# --- Example Usage ---
if __name__ == "__main__":
    # Use environment variables for credentials
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password") # Make sure this is set in .env

    if not NEO4J_PASSWORD:
        log.error("NEO4J_PASSWORD not found in environment. Please set it in .env file.")
        exit()

    try:
        evaluation_system = AnswerEvaluationSystem(
            neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD
        )
    except Exception as init_err:
        print(f"\n--- SYSTEM INITIALIZATION FAILED ---")
        print(f"Error: {init_err}")
        exit()

    # Example 1: Evaluate student "Akash"
    print("\n--- Example 1: Evaluate all answers for student 'Akash' ---")
    # Note: Assumes 'Akash' is the roll_no identifier used in your DB
    akash_results = evaluation_system.fetch_and_evaluate_student_answers(student_name="Akash")
    print(f"Retrieved and evaluated {len(akash_results)} results for Akash.")
    if akash_results: print("First Result Detail:", json.dumps(akash_results[0], indent=2))

    # ... (Keep other examples, ensure student_name parameter matches roll_no if needed) ...

    print("\n--- Example 2: Generate Grade Report for student 'Akash' ---")
    report = evaluation_system.generate_grade_report("Akash") # Assumes 'Akash' is roll_no
    print(json.dumps(report, indent=2))

    # ...

    print("\n--- Example 5: Direct evaluation ---")
    direct_eval = evaluation_system.evaluate_single_answer(
        "What is photosynthesis?",
        "Plants use sunlight to make food.",
        "Photosynthesis is the process used by plants..."
    )
    print(json.dumps(direct_eval, indent=2))

    # Optional explicit close (or rely on __del__)
    # evaluation_system.neo4j.close()
    log.info("Evaluation script finished.")