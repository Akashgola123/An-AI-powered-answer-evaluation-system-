# # import time
# # import logging
# # from pydantic import BaseModel
# # from typing import Dict
# # from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager

# # class ExamSubmissionModel(BaseModel):
# #     roll_no: str
# #     subject: str
# #     answers: Dict[str, str]  # {question_id: answer_text}

# # class ExamService:
# #     def __init__(self):
# #         self.db = Neo4jManager.get_instance()
# #         self.logger = logging.getLogger(__name__)

# #     def check_existing_submission(self, roll_no: str, subject: str) -> bool:
# #         """
# #         Checks if a student has already submitted an exam for a subject.
# #         """
# #         query = """
# #         MATCH (s:Student {roll_no: $roll_no})-[:SUBMITTED]->(exam:ExamSubmission {subject: $subject}) 
# #         RETURN exam.id LIMIT 1
# #         """
# #         result = self.db.execute_query(query, {"roll_no": roll_no, "subject": subject})
# #         return bool(result)

# #     def create_exam_submission(self, roll_no: str, subject: str) -> str:
# #         """
# #         Creates a new exam submission node linked to a student and subject.
# #         """
# #         submission_id = f"{roll_no}_{subject}_{int(time.time())}"
# #         query = """
# #         MATCH (s:Student {roll_no: $roll_no}), (sub:Subject {name: $subject})
# #         CREATE (exam:ExamSubmission {id: $submission_id, subject: $subject, submitted_at: timestamp()})
# #         CREATE (s)-[:SUBMITTED]->(exam)
# #         CREATE (exam)-[:FOR_SUBJECT]->(sub)
# #         RETURN exam.id
# #         """
# #         self.db.execute_query(query, {"roll_no": roll_no, "subject": subject, "submission_id": submission_id})
# #         return submission_id

# #     def store_answers(self, submission_id: str, subject: str, answers: Dict[str, str]):
# #         """
# #         Stores student answers linked to the submission and relevant questions.
# #         """
# #         for qid, answer in answers.items():
# #             query = """
# #             MATCH (exam:ExamSubmission {id: $submission_id})
# #             MATCH (q:Question {id: $qid})-[:HAS_QUESTION]->(sub:Subject {name: $subject}) 
# #             CREATE (a:Answer {
# #                 id: $answer_id,
# #                 text: $answer, 
# #                 submitted_at: timestamp()
# #             })
# #             CREATE (exam)-[:HAS_ANSWER]->(a)
# #             CREATE (a)-[:FOR_QUESTION]->(q)
# #             """
# #             self.db.execute_query(query, {
# #                 "submission_id": submission_id,
# #                 "qid": qid,
# #                 "subject": subject,
# #                 "answer_id": f"{submission_id}_{qid}",
# #                 "answer": answer
# #             })

# #     def submit_exam(self, submission: ExamSubmissionModel) -> dict:
# #         """
# #         Handles the full exam submission process.
# #         """
# #         try:
# #             if self.check_existing_submission(submission.roll_no, submission.subject):
# #                 return {"error": "Exam already submitted for this subject!"}

# #             submission_id = self.create_exam_submission(submission.roll_no, submission.subject)
# #             self.store_answers(submission_id, submission.subject, submission.answers)

# #             self.logger.info(f"✅ Exam submitted successfully for {submission.roll_no} in {submission.subject}")
# #             return {"message": "Exam submitted successfully!", "submission_id": submission_id}

# #         except Exception as e:
# #             self.logger.error(f"❌ Error submitting exam: {str(e)}")
# #             return {"error": f"Error submitting exam: {str(e)}"}



# import time
# import logging
# from pydantic import BaseModel
# from typing import Dict
# from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager

# class ExamSubmissionModel(BaseModel):
#     roll_no: str
#     subject: str
#     answers: Dict[str, str]

# class ExamService:
#     def __init__(self):
#         self.db = Neo4jManager.get_instance()
#         self.logger = logging.getLogger(__name__)

#     def check_existing_submission(self, roll_no: str, subject: str) -> bool:
#         query = """
#         MATCH (s:Student {roll_no: $roll_no})-[:SUBMITTED]->(exam:ExamSubmission {subject: $subject}) 
#         RETURN exam.id LIMIT 1
#         """
#         result = self.db.execute_query(query, {"roll_no": roll_no, "subject": subject})
#         return bool(result)

#     def create_exam_submission(self, roll_no: str, subject: str) -> str:
#         if not self.db.execute_query("MATCH (s:Student {roll_no: $roll_no}) RETURN s", {"roll_no": roll_no}):
#             return {"error": "Student not found!"}
#         if not self.db.execute_query("MATCH (sub:Subject {name: $subject}) RETURN sub", {"subject": subject}):
#             return {"error": "Subject not found!"}

#         submission_id = f"{roll_no}_{subject}_{int(time.time())}"
#         query = """
#         OPTIONAL MATCH (s:Student {roll_no: $roll_no})
#         OPTIONAL MATCH (sub:Subject {name: $subject})
#         WITH s, sub
#         WHERE s IS NOT NULL AND sub IS NOT NULL
#         CREATE (exam:ExamSubmission {id: $submission_id, subject: $subject, submitted_at: timestamp()})
#         CREATE (s)-[:SUBMITTED]->(exam)
#         CREATE (exam)-[:FOR_SUBJECT]->(sub)
#         RETURN exam.id
#         """
#         result = self.db.execute_query(query, {"roll_no": roll_no, "subject": subject, "submission_id": submission_id})
#         return submission_id if result else {"error": "Exam submission failed!"}

#     def store_answers(self, submission_id: str, subject: str, answers: Dict[str, str]):
#         for qid, answer in answers.items():
#             valid_question = self.db.execute_query(
#                 "MATCH (q:Question {id: $qid})-[:HAS_QUESTION]->(sub:Subject {name: $subject}) RETURN q.id",
#                 {"qid": qid, "subject": subject}
#             )
#             if not valid_question:
#                 self.logger.error(f"❌ Invalid question ID {qid} for subject {subject}. Skipping...")
#                 continue  # Skip invalid questions

#             query = """
#             MATCH (exam:ExamSubmission {id: $submission_id})
#             MATCH (q:Question {id: $qid})-[:HAS_QUESTION]->(sub:Subject {name: $subject}) 
#             CREATE (a:Answer {
#                 id: $answer_id,
#                 text: $answer, 
#                 submitted_at: timestamp()
#             })
#             CREATE (exam)-[:HAS_ANSWER]->(a)
#             CREATE (a)-[:FOR_QUESTION]->(q)
#             """
#             self.db.execute_query(query, {
#                 "submission_id": submission_id,
#                 "qid": qid,
#                 "subject": subject,
#                 "answer_id": f"{submission_id}_{qid}",
#                 "answer": answer
#             })

#     def submit_exam(self, submission: ExamSubmissionModel) -> dict:
#         try:
#             if self.check_existing_submission(submission.roll_no, submission.subject):
#                 return {"error": "Exam already submitted for this subject!"}

#             submission_id = self.create_exam_submission(submission.roll_no, submission.subject)
#             if isinstance(submission_id, dict):  # If error is returned
#                 return submission_id

#             self.store_answers(submission_id, submission.subject, submission.answers)

#             self.logger.info(f"✅ Exam submitted successfully for {submission.roll_no} in {submission.subject}")
#             return {"message": "Exam submitted successfully!", "submission_id": submission_id}

#         except Exception as e:
#             self.logger.error(f"❌ Error submitting exam: {str(e)}")
#             return {"error": f"Error submitting exam: {str(e)}"}


from pydantic import BaseModel
from typing import Dict

class ExamSubmission(BaseModel):
    roll_no: str
    subject: str
    answers: Dict[str, str]

# File: app/services/exam_service.py
import time
from fastapi import HTTPException
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.logger import logger

class ExamService:
    def __init__(self):
        self.neo4j_manager = Neo4jManager.get_instance()
    
    def submit_exam(self, submission_data):
        """
        Submit exam answers for evaluation.
        
        Creates Answer nodes linked to Student, Question, and Subject nodes.
        """
        try:
            driver = self.neo4j_manager.get_connection()
            submission_time = int(time.time())
            
            # First verify student and subject exist
            with driver.session() as session:
                student_result = session.run(
                    "MATCH (s:Student {roll_no: $roll_no}) RETURN s", 
                    {"roll_no": submission_data.roll_no}
                )
                if not student_result.single():
                    raise HTTPException(status_code=400, detail="Student not found")
                    
                subject_result = session.run(
                    "MATCH (s:Subject {name: $subject}) RETURN s", 
                    {"subject": submission_data.subject}
                )
                if not subject_result.single():
                    raise HTTPException(status_code=400, detail="Subject not found")
            
            # Create an exam submission node
            submission_id = f"{submission_data.roll_no}_{submission_data.subject}_{submission_time}"
            
            with driver.session() as session:
                session.run("""
                    MATCH (s:Student {roll_no: $roll_no})
                    MATCH (sub:Subject {name: $subject})
                    CREATE (e:ExamSubmission {
                        id: $submission_id,
                        submitted_at: timestamp(),
                        question_count: $question_count
                    })
                    CREATE (s)-[:SUBMITTED]->(e)
                    CREATE (e)-[:FOR_SUBJECT]->(sub)
                    """, {
                        "roll_no": submission_data.roll_no,
                        "subject": submission_data.subject,
                        "submission_id": submission_id,
                        "question_count": len(submission_data.answers)
                    })
            
            # Store each answer
            for qid, answer_text in submission_data.answers.items():
                with driver.session() as session:
                    session.run("""
                        MATCH (e:ExamSubmission {id: $submission_id})
                        MATCH (q:Question {id: $qid})
                        MATCH (sub:Subject {name: $subject})
                        CREATE (a:Answer {
                            id: $answer_id,
                            text: $answer_text,
                            submitted_at: timestamp()
                        })
                        CREATE (e)-[:HAS_ANSWER]->(a)
                        CREATE (a)-[:FOR_QUESTION]->(q)
                        CREATE (a)-[:IN_SUBJECT]->(sub)
                        """, {
                            "submission_id": submission_id,
                            "qid": qid,
                            "subject": submission_data.subject,
                            "answer_id": f"{submission_id}_{qid}",
                            "answer_text": answer_text
                        })
            
            logger.info(f"Exam submitted successfully: {submission_id} with {len(submission_data.answers)} answers")
            return {
                "status": "success", 
                "message": "Exam submitted successfully!",
                "submission_id": submission_id
            }
            
        except HTTPException as e:
            # Re-throw HTTP exceptions
            raise e
        except Exception as e:
            logger.error(f"Error submitting exam: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")
