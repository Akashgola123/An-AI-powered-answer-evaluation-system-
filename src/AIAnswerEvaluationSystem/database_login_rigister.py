

import time
import uuid
import logging
from fastapi import HTTPException 
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.evaluting import EvaluationProcessor 
from pydantic import BaseModel
from typing import Dict, Any
from fastapi import status 


log = logging.getLogger(__name__)

class ExamSubmissionRequest(BaseModel):
    roll_no: str
    subject: str
    answers: Dict[str, str] 

class ExamService:
    def __init__(self, evaluation_processor: EvaluationProcessor):
        """
        Initialize ExamService with an EvaluationProcessor instance.
        """
        if not isinstance(evaluation_processor, EvaluationProcessor):
             raise TypeError("ExamService requires a valid EvaluationProcessor instance.")
        self.processor = evaluation_processor 
        
        self.neo4j_manager = evaluation_processor.db
        log.info("ExamService initialized.")


    def submit_exam(self, submission_data: ExamSubmissionRequest):
        """
        Submits exam answers and triggers automatic evaluation for each answer.
        """
        start_time_perf = time.perf_counter()
        submission_time_unix = int(time.time())
        submission_id = f"SUB_{submission_data.roll_no}_{submission_data.subject}_{submission_time_unix}_{uuid.uuid4().hex[:6]}"
        answers_list = [{"qid": qid, "answer_text": text} for qid, text in submission_data.answers.items()]

        
        cypher_query_submit = """
        MATCH (s:Student {roll_no: $p_roll_no})
        WITH s MATCH (sub:Subject {name: $p_subject}) // Find subject after student
        CREATE (e:ExamSubmission { id: $p_submission_id, submitted_at_unix: $p_submission_time_unix, submitted_at: datetime(), question_count: $p_question_count, roll_no: s.roll_no, subject_name: sub.name })
        CREATE (s)-[:SUBMITTED]->(e) CREATE (e)-[:FOR_SUBJECT]->(sub)
        WITH e, sub // Pass nodes needed for answer creation
        UNWIND $p_answers_list AS answer_data
        MATCH (q:Question {id: answer_data.qid}) // Ensure question exists
        CREATE (a:Answer { id: $p_submission_id + "_" + answer_data.qid, text: answer_data.answer_text, submitted_at: datetime() })
        CREATE (e)-[:HAS_ANSWER]->(a) CREATE (a)-[:ANSWERS_QUESTION]->(q) CREATE (a)-[:FOR_SUBJECT]->(sub)
        RETURN count(a) AS answers_created
        """
        parameters = {
            "p_roll_no": submission_data.roll_no, "p_subject": submission_data.subject,
            "p_submission_id": submission_id, "p_submission_time_unix": submission_time_unix,
            "p_question_count": len(submission_data.answers), "p_answers_list": answers_list,
        }

        try:
            
            log.info(f"Attempting to save submission {submission_id}...")
            write_summary = self.neo4j_manager.execute_write(cypher_query_submit, parameters)

            
            if write_summary is None or write_summary.nodes_created == 0:
                 # Check if student/subject missing is likely cause
                 check_s = self.neo4j_manager.execute_read("MATCH(n:Student{roll_no:$r}) RETURN n",{'r':submission_data.roll_no})
                 check_sub = self.neo4j_manager.execute_read("MATCH(n:Subject{name:$s}) RETURN n",{'s':submission_data.subject})
                 error_detail = "Submission DB write failed."
                 if not check_s: error_detail += " Reason: Student not found."
                 elif not check_sub: error_detail += " Reason: Subject not found."
                 else: error_detail += " Reason: Unknown DB issue or question mismatch."
                 log.error(error_detail)
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail) # Changed to 400 for missing entities

            log.info(f"Submission {submission_id} saved successfully to DB.")
            submission_duration = time.perf_counter() - start_time_perf

            #
            # This happens AFTER the submission is successfully saved.
            log.info(f"Starting automatic evaluation for {len(answers_list)} answers in submission {submission_id}...")
            evaluation_results = {}
            eval_errors = 0
            for answer_detail in answers_list:
                qid = answer_detail["qid"]
                try:
                    # Use the EvaluationProcessor to handle the fetch->LLM->grade->store cycle
                    result = self.processor.evaluate_and_store_answer(
                         student_roll_no=submission_data.roll_no,
                         question_id=qid
                    )
                    evaluation_results[qid] = result if result else {"error": "Evaluation process failed"}
                    if not result or result.get("error") or result.get("storage_error"):
                         eval_errors += 1
                except Exception as eval_e:
                     log.exception(f"Error during background evaluation trigger for QID {qid}, SubID {submission_id}")
                     evaluation_results[qid] = {"error": f"Trigger failed: {eval_e}"}
                     eval_errors += 1

            eval_duration = time.perf_counter() - start_time_perf - submission_duration
            log.info(f"Automatic evaluation finished for {submission_id}. Errors: {eval_errors}. Total time: {eval_duration:.2f}s")

            # Return success response for the SUBMISSION itself.
            # Evaluation happens afterward (or potentially in background in production).
            return {
                "status": "success",
                "message": f"Exam submitted successfully. Evaluation triggered for {len(answers_list)} questions ({eval_errors} errors).",
                "submission_id": submission_id,
                
            }

        except HTTPException as http_exc: raise http_exc
        except Exception as e:
            log.exception(f"Error during exam submission process for student {submission_data.roll_no}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during exam submission.")