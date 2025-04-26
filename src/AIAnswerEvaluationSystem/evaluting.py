
import logging
import time
from typing import Dict, Optional, Any, List
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager # Adjust path
from AIAnswerEvaluationSystem.llmss import FineTunedLLMEvaluator # Adjust path
from AIAnswerEvaluationSystem.logger import logger # Adjust path
import os
import json

PASS_FAIL_THRESHOLD_PERCENTAGE = 60.0 # Passing threshold based on percentage
DEFAULT_QUESTION_MAX_MARKS = 5.0 # Default max marks if missing from DB


# --- Standalone Grading Logic Helper Class ---
class GradingLogic:
    """Provides static methods for grading based on scores and percentages."""
    GRADE_SCALE = { # Map score/percentage range start to letter grade
        90: 'A+',
        80: 'A',
        70: 'B',
        60: 'C', # Assuming C is lowest pass
        50: 'D',
        0: 'F' # Catch-all for < 50%
    }

    @staticmethod
    def calculate_grade_and_status(marks_obtained: Optional[float], max_marks_possible: float) -> Dict[str, Any]:
        """
        Calculates percentage, letter grade, and pass/fail status based on actual marks.

        :param marks_obtained: Marks achieved by the student.
        :param max_marks_possible: Maximum possible marks for the question.
        :return: Dict with 'percentage', 'letter_grade', 'status'.
        """
        results = {"percentage": None, "letter_grade": "N/A", "status": "N/A"}
        if marks_obtained is None or marks_obtained < 0 or max_marks_possible <= 0:
            logger.warning(f"Grading skipped due to invalid inputs: Marks={marks_obtained}, Max={max_marks_possible}.")
            return results

        percentage = round((marks_obtained / max_marks_possible) * 100, 2)
        results["percentage"] = percentage

        # Determine Letter Grade
        grade = 'F' # Default grade
        # Iterate sorted thresholds (high to low)
        for threshold, letter in sorted(GradingLogic.GRADE_SCALE.items(), key=lambda item: item[0], reverse=True):
            if percentage >= threshold:
                grade = letter
                break
        results["letter_grade"] = grade

        # Determine Pass/Fail Status
        results["status"] = "Pass" if percentage >= PASS_FAIL_THRESHOLD_PERCENTAGE else "Fail"
        return results


# --- Main Processor Class ---
class EvaluationProcessor:
    """
    Orchestrates fetching submissions, running LLM evaluation, calculating
    final marks/grades, storing results to Neo4j, and fetching stored results.
    """
    def __init__(self, neo4j_manager: Neo4jManager, llm_evaluator: FineTunedLLMEvaluator):
        """
        Initializes the processor with dependencies.

        :param neo4j_manager: An initialized instance of Neo4jManager.
        :param llm_evaluator: An initialized instance of FineTunedLLMEvaluator.
        """
        if not isinstance(neo4j_manager, Neo4jManager):
             log.error(f"Received invalid type for neo4j_manager: {type(neo4j_manager)}")
             raise TypeError("EvaluationProcessor requires a valid Neo4jManager instance.")
        if not isinstance(llm_evaluator, FineTunedLLMEvaluator):
            log.error(f"Received invalid type for llm_evaluator: {type(llm_evaluator)}")
            raise TypeError("EvaluationProcessor requires a valid FineTunedLLMEvaluator instance.")

        self.db = neo4j_manager
        self.llm = llm_evaluator
        logger.info("EvaluationProcessor initialized successfully.")

    # === Private Helper Methods for NEW Evaluations ===

    def _fetch_submission_details(self, student_roll_no: str, question_id: str) -> Optional[Dict]:
        """Fetches the latest specific answer submission with needed details for evaluation."""
        query = """
        MATCH (s:Student {roll_no: $p_roll_no})-[:SUBMITTED]->(e:ExamSubmission)
              -[:HAS_ANSWER]->(a:Answer)-[:ANSWERS_QUESTION]->(q:Question {id: $p_question_id})
        OPTIONAL MATCH (q)-[:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer)
        WITH s, e, a, q, ca
        ORDER BY e.submitted_at DESC // Get the latest submission for this student/question pair
        LIMIT 1
        RETURN
            q.id AS question_id,
            q.text AS question_text,
            // Ensure max_marks is treated as number, provide default if missing
            COALESCE(q.max_marks, $p_default_max_marks) AS max_marks_for_question,
            a.id AS answer_node_id, // ID of the Answer node to store results on
            a.text AS submitted_answer,
            COALESCE(ca.text, "[No Correct Answer Provided]") AS correct_answer_text, // Default if no CA node
            s.roll_no AS student_roll_no // Include student roll_no
        """
        params = {
            "p_roll_no": student_roll_no,
            "p_question_id": question_id,
            "p_default_max_marks": DEFAULT_QUESTION_MAX_MARKS # Use configured default
            }
        logger.debug(f"Executing fetch for submission details: S='{student_roll_no}', Q='{question_id}'")
        results = self.db.execute_read(query, params) # Assumes Neo4jManager handles errors/returns []
        if results:
            logger.debug(f"Found submission details: {results[0]}")
            return results[0]
        else:
            logger.warning(f"No submission detail found for S:'{student_roll_no}', Q:'{question_id}'")
            return None

    def _convert_score_to_marks(self, llm_score_0_5: Optional[int], max_marks_for_question: float) -> Optional[float]:
        """Converts the LLM's 0-5 score to the actual marks based on question's max marks."""
        if llm_score_0_5 is None or not isinstance(llm_score_0_5, int) or not (0 <= llm_score_0_5 <= 5):
            logger.warning(f"Cannot convert invalid LLM score: {llm_score_0_5}")
            return None
        if max_marks_for_question <= 0:
            logger.warning(f"Cannot convert score with invalid max_marks: {max_marks_for_question}")
            return None

        # Linear scaling
        marks_obtained = (llm_score_0_5 / 5.0) * max_marks_for_question
        return round(marks_obtained, 1) # Round to 1 decimal place

    def _store_evaluation_result(self, answer_node_id: str, eval_data: Dict) -> bool:
        """Stores the computed evaluation results onto the Answer node in Neo4j."""
        if not answer_node_id:
            logger.error("Cannot store evaluation without valid answer_node_id.")
            return False

        query = """
        MATCH (a:Answer {id: $p_answer_node_id})
        WHERE a.id IS NOT NULL // Ensure node exists
        // Use SET for overwriting/adding properties from the map
        SET a += $p_properties
        SET a.evaluated_at = datetime() // Add/Update timestamp
        RETURN a.id // Confirm update happened
        """
        # Prepare properties, filter Nones, add prefix
        properties_to_set = {}
        for key in [ "numeric_score", "marks_obtained", "percentage",
                     "letter_grade", "status", "feedback",
                     "llm_error", "storage_error", "score_str", # Added score_str
                     # Add max_marks_possible to know the scale later?
                     "max_marks_possible"]:
            value = eval_data.get(key)
            # Ensure feedback is stored even if empty string, but exclude None
            if value is not None:
                 properties_to_set[f"evaluation_{key}"] = value

        if not properties_to_set:
            logger.warning(f"No valid properties generated to store for Answer ID: {answer_node_id}. Eval Data: {eval_data}")
            return False

        params = {"p_answer_node_id": answer_node_id, "p_properties": properties_to_set}
        logger.debug(f"Attempting to store evaluation properties for Answer ID {answer_node_id}")

        try:
            summary = self.db.execute_write(query, params) # Use the optimized manager method
            # Summary can be None on error. If it exists, check counters.
            # Neo4j returns propertiesSet=0 if node wasn't found, or N properties if update ok.
            # We check for properties_set > 0 to be sure.
            success = summary and hasattr(summary,'properties_set') and summary.properties_set > 0
            if success: logger.info(f"Successfully stored evaluation results for Answer ID: {answer_node_id}")
            else: logger.error(f"Failed to store evaluation results for Answer ID: {answer_node_id}. Summary: {summary}")
            return success
        except Exception as e:
            logger.exception(f"Neo4j write transaction failed while storing evaluation for Answer ID: {answer_node_id}")
            return False

    # === Public Method to Perform and Store Evaluation ===
    def evaluate_and_store_answer(self, student_roll_no: str, question_id: str) -> Optional[Dict]:
        """
        Fetches submission, runs LLM, calculates results, stores results to Neo4j.
        """
        logger.info(f"Processing evaluation request: S='{student_roll_no}', Q='{question_id}'")
        eval_start = time.time()

        final_result: Dict[str, Any] = {
            # Initialize with keys expected in the return value
            "student_roll_no": student_roll_no, "question_id": question_id,
            "numeric_score": None, "score_str": "N/A", "marks_obtained": None,
            "max_marks_possible": None, "percentage": None, "letter_grade": "N/A",
            "status": "N/A", "feedback": None, "raw_output": None,
            "answer_node_id": None, "error": None, "storage_error": None
        }

        # 1. Fetch data from DB
        submission_data = self._fetch_submission_details(student_roll_no, question_id)
        if not submission_data: final_result["error"] = "Submission details not found"; return final_result

        # Extract and validate crucial fields
        answer_node_id = submission_data.get('answer_node_id')
        question_text = submission_data.get('question_text')
        correct_answer = submission_data.get('correct_answer_text', "") # Use empty if missing
        student_answer = submission_data.get('submitted_answer')
        max_marks_db_raw = submission_data.get('max_marks_for_question') # Comes from COALESCE(..., 5.0)
        final_result["answer_node_id"] = answer_node_id

        if not all([answer_node_id, question_text, student_answer]): final_result["error"]="Missing critical data from DB"; return final_result
        try: max_marks_db = float(max_marks_db_raw) if max_marks_db_raw else DEFAULT_QUESTION_MAX_MARKS
        except (ValueError, TypeError): logger.warning(f"Using default max marks; invalid value from DB: {max_marks_db_raw}"); max_marks_db=DEFAULT_QUESTION_MAX_MARKS
        if max_marks_db <= 0: final_result["error"]=f"Invalid max marks ({max_marks_db})"; return final_result
        final_result["max_marks_possible"] = max_marks_db


        # 2. Generate LLM output
        prompt = self.llm.format_prompt(question_text, student_answer, correct_answer)
        if not prompt: final_result["error"]="Prompt formatting failed"; return final_result
        raw_output = self.llm.generate_raw_output(prompt)
        final_result["raw_output"] = raw_output # Store raw output
        if not raw_output: final_result["error"]="LLM generation failed"; return final_result


        # 3. Parse LLM Output
        parsed_llm = self.llm.parse_output(raw_output)
        numeric_score = parsed_llm.get("numeric_score") # This is the 0-5 score
        feedback = parsed_llm.get("feedback")
        final_result["numeric_score"] = numeric_score # Store numeric LLM score
        final_result["feedback"] = feedback if feedback else "[Feedback not parsed]"


        # 4. Calculate final scores/grades if LLM score is valid
        if numeric_score is not None and 0 <= numeric_score <= 5:
             marks_obtained = self._convert_score_to_marks(numeric_score, max_marks_db)
             grade_info = GradingLogic.calculate_grade_and_status(marks_obtained, max_marks_db)

             final_result["marks_obtained"] = marks_obtained
             final_result["score_str"] = f"{numeric_score}/5" # Keep LLM score format
             final_result.update({k:v for k,v in grade_info.items()}) # Add percent, grade, status
        else:
             # Keep calculated fields as None/N/A, set error flag
             final_result["error"] = "LLM score parsing failed or invalid"
             logger.warning(f"{final_result['error']} for S:{student_roll_no}, Q:{question_id}")


        # 5. Store ALL calculated/parsed results back to Neo4j Answer node
        store_success = self._store_evaluation_result(answer_node_id, final_result)
        if not store_success:
            final_result["storage_error"] = "Failed to save results to DB"
            logger.error(final_result["storage_error"] + f" for Answer ID {answer_node_id}")


        eval_time = time.time() - eval_start
        logger.info(f"Processed evaluation request ({'stored' if store_success else 'STORE FAILED'}) S:{student_roll_no}, Q:{question_id} in {eval_time:.2f}s. Grade: {final_result['letter_grade']}")
        return final_result


    # === *** NEW Public Method to FETCH Stored Evaluations *** ===
    def get_student_evaluations(self, roll_no: str, subject: Optional[str] = None) -> List[Dict]:
        """
        Retrieves previously stored evaluation results for a student from Neo4j.

        :param roll_no: The student's roll number.
        :param subject: Optional subject name to filter results.
        :return: List of dictionaries, each representing an evaluated answer.
        """
        logger.info(f"Processor: Fetching stored evaluation results for S:{roll_no}, Sub:{subject or 'ALL'}")
        try:
            # Call the corresponding fetch method in Neo4jManager
            # Ensure this method exists and has the correct query
            raw_evaluations = self.db.fetch_evaluation_results_for_student(roll_no, subject)

            # Basic processing/cleanup (e.g., stringify datetimes)
            processed_results = []
            for row in raw_evaluations:
                 if row.get("evaluated_time") and not isinstance(row.get("evaluated_time"), str):
                      row["evaluated_time"] = str(row["evaluated_time"]) # Convert for JSON
                 # Ensure essential keys exist before adding (optional)
                 if row.get("question_id") and row.get("student_roll_no"):
                      processed_results.append(row)
                 else:
                      logger.warning(f"Skipping fetched evaluation due to missing IDs: {row}")

            logger.info(f"Processor: Returning {len(processed_results)} stored evaluations for S:{roll_no}")
            return processed_results

        except Exception as e:
            logger.exception(f"Error within processor while getting student evaluations for {roll_no}")
            return [] # Return empty list if processor step fails
