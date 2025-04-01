# from langchain_ollama import OllamaLLM
# from AIAnswerEvaluationSystem.logger import logging

# class LLMSS:
#     def __init__(self,llm=None):
#         if llm:
#             self.llm = llm
#         else:
#             self.llm = OllamaLLM(model="llama3.2:latest") 
#             logging.info("LLMSS initialized")



# from typing import Optional
# from langchain_ollama import OllamaLLM
# from AIAnswerEvaluationSystem.logger import logging

# class LLMSS:
#     def __init__(self, llm: Optional[OllamaLLM] = None):
#         """
#         Initialize the LLMSS with an optional LLM.
        
#         :param llm: Optional pre-configured OllamaLLM instance
#         """
#         try:
#             self.llm = llm if llm is not None else OllamaLLM(model="llama3.2:latest")
#             logging.info("LLMSS initialized successfully")
#         except Exception as e:
#             logging.error(f"Failed to initialize LLMSS: {e}")
#             raise


\
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Pydantic models
# class EvaluationRequest(BaseModel):
#     roll_no: str
#     subject: str
#     generate_corrections: bool = False

# class BatchEvaluationRequest(BaseModel):
#     roll_numbers: List[str]
#     subject: str
#     generate_corrections: bool = False

# class EvaluationResponse(BaseModel):
#     question_scores: Dict
#     feedback: Dict
#     grade_report: Dict
#     message: Optional[str] = None

# Neo4j Database Manager







# Import section - group imports logically
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
import uuid
from neo4j import GraphDatabase

# Configuration imports
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.logger import logger

# ================ DATA MODELS =================
# Define clear data models for all entities

class EvaluationResult(BaseModel):
    """Data model for evaluation results"""
    question_scores: Dict
    feedback: Dict
    grade_report: Dict
    message: Optional[str] = None

class StudentResult(BaseModel):
    """Data model for student results"""
    status: str
    message: Optional[str] = None
    results: List[Dict] = []

# ================ REPOSITORY LAYER =================

class ExamRepository:
    """Repository class for database operations related to exams"""
    
    def __init__(self):
        self.db_manager = Neo4jManager.get_instance()
    
    def get_submission_id(self, roll_no: str, subject: str) -> Optional[str]:
        """
        Get the latest submission ID for a student in a subject
        
        Args:
            roll_no: Student roll number
            subject: Subject name
            
        Returns:
            Submission ID or None if not found
        """
        driver = self.db_manager.get_connection()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Student {roll_no: $roll_no})
                MATCH (s)-[:SUBMITTED]->(e:ExamSubmission)-[:FOR_SUBJECT]->(sub:Subject {name: $subject})
                RETURN e.id as submission_id
                ORDER BY e.submitted_at DESC
                LIMIT 1
                """, {
                    "roll_no": roll_no,
                    "subject": subject
                })
            
            record = result.single()
            return record["submission_id"] if record else None
    
    def check_submission_exists(self, roll_no: str, subject: str) -> bool:
        """
        Check if a submission exists for a student in a subject
        
        Args:
            roll_no: Student roll number
            subject: Subject name
            
        Returns:
            Boolean indicating if submission exists
        """
        driver = self.db_manager.get_connection()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Student {roll_no: $roll_no})
                MATCH (s)-[:SUBMITTED]->(e:ExamSubmission)-[:FOR_SUBJECT]->(sub:Subject {name: $subject})
                RETURN count(e) as submission_count
                """, {
                    "roll_no": roll_no,
                    "subject": subject
                })
            
            record = result.single()
            return record and record["submission_count"] > 0
    
    def get_student_results(self, roll_no: str, subject: Optional[str] = None) -> List[Dict]:
        """
        Get evaluation results for a student
        
        Args:
            roll_no: Student roll number
            subject: Optional subject name to filter results
            
        Returns:
            List of evaluation results
        """
        driver = self.db_manager.get_connection()
        
        # Construct the query based on whether a subject is specified
        query = self._build_results_query(subject is not None)
        
        params = {"roll_no": roll_no}
        if subject:
            params["subject"] = subject
            
        with driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def _build_results_query(self, has_subject: bool) -> str:
        """
        Helper method to build the query for getting student results
        
        Args:
            has_subject: Whether to include subject filter
            
        Returns:
            Neo4j query string
        """
        query = """
        MATCH (s:Student {roll_no: $roll_no})
        MATCH (s)-[:SUBMITTED]->(e:ExamSubmission)
        """
        
        if has_subject:
            query += "MATCH (e)-[:FOR_SUBJECT]->(sub:Subject {name: $subject})"
        else:
            query += "MATCH (e)-[:FOR_SUBJECT]->(sub:Subject)"
            
        query += """
        OPTIONAL MATCH (e)-[:HAS_EVALUATION]->(eval:Evaluation)
        RETURN sub.name as subject, 
               e.id as submission_id,
               e.submitted_at as submitted_at,
               eval.overall_score as score,
               eval.letter_grade as grade,
               eval.passing as passing,
               eval.created_at as evaluated_at
        ORDER BY e.submitted_at DESC
        """
        return query
    
    def store_evaluation(self, submission_id: str, evaluation: Dict) -> bool:
        """
        Store evaluation results in Neo4j
        
        Args:
            submission_id: ID of the exam submission
            evaluation: Evaluation data dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            driver = self.db_manager.get_connection()
            
            # Extract data from evaluation results
            overall_results = evaluation.get("grade_report", {}).get("overall_results", {})
            eval_data = self._prepare_evaluation_data(submission_id, overall_results)
            
            with driver.session() as session:
                result = session.run(self._get_store_evaluation_query(), eval_data)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error storing evaluation results: {str(e)}")
            return False
    
    def _prepare_evaluation_data(self, submission_id: str, overall_results: Dict) -> Dict:
        """
        Prepare data for storing evaluation
        
        Args:
            submission_id: ID of the exam submission
            overall_results: Overall results from evaluation
            
        Returns:
            Dictionary of parameters for Neo4j query
        """
        return {
            "submission_id": submission_id,
            "eval_id": f"{submission_id}_eval",
            "overall_score": overall_results.get("total_marks_earned", 0),
            "percentage": overall_results.get("percentage", 0),
            "letter_grade": overall_results.get("letter_grade", "N/A"),
            "passing": overall_results.get("passing", False)
        }
    
    def _get_store_evaluation_query(self) -> str:
        """
        Get query for storing evaluation
        
        Returns:
            Neo4j query string
        """
        return """
        MATCH (e:ExamSubmission {id: $submission_id})
        CREATE (eval:Evaluation {
            id: $eval_id,
            overall_score: $overall_score,
            percentage: $percentage,
            letter_grade: $letter_grade,
            passing: $passing,
            created_at: datetime()
        })
        CREATE (e)-[:HAS_EVALUATION]->(eval)
        RETURN eval.id as evaluation_id
        """
    
    def get_question_data(self, subject: str) -> List[Dict]:
        """
        Get question data for a subject
        
        Args:
            subject: Subject name
            
        Returns:
            List of question data dictionaries
        """
        driver = self.db_manager.get_connection()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Subject {name: $subject})-[:HAS_QUESTION]->(q:Question)
                RETURN q.id as id, 
                       q.text as text,
                       q.max_marks as max_marks,
                       q.model_answer as model_answer,
                       q.keywords as keywords
                ORDER BY q.id
                """, {
                    "subject": subject
                })
            
            return [record.data() for record in result]
    
    def get_student_answers(self, submission_id: str) -> List[Dict]:
        """
        Get student answers for a submission
        
        Args:
            submission_id: ID of the exam submission
            
        Returns:
            List of student answer dictionaries
        """
        driver = self.db_manager.get_connection()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (e:ExamSubmission {id: $submission_id})-[:HAS_ANSWER]->(a:Answer)-[:FOR_QUESTION]->(q:Question)
                RETURN q.id as question_id,
                       a.text as answer_text
                ORDER BY q.id
                """, {
                    "submission_id": submission_id
                })
            
            return [record.data() for record in result]


# ================ DOMAIN LAYER =================

class AnswerEvaluator:
    """Class for evaluating student answers"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the answer evaluator
        
        Args:
            openai_api_key: Optional API key for OpenAI
        """
        self.openai_api_key = openai_api_key
        # Initialize any AI models or components here
    
    def evaluate_submission(self, questions: List[Dict], answers: List[Dict]) -> EvaluationResult:
        """
        Evaluate a student's submission
        
        Args:
            questions: List of question data
            answers: List of student answers
            
        Returns:
            EvaluationResult object with scores and feedback
        """
        # Map answers to questions
        answer_map = {a["question_id"]: a["answer_text"] for a in answers}
        
        # Evaluate each question
        question_scores = {}
        feedback = {}
        total_marks = 0
        total_max_marks = 0
        
        for question in questions:
            question_id = question["id"]
            answer_text = answer_map.get(question_id, "")
            max_marks = question["max_marks"]
            total_max_marks += max_marks
            
            # Evaluate the answer
            score, feedback_text = self._evaluate_answer(
                question_text=question["text"],
                model_answer=question["model_answer"],
                student_answer=answer_text,
                max_marks=max_marks,
                keywords=question.get("keywords", [])
            )
            
            question_scores[question_id] = score
            feedback[question_id] = feedback_text
            total_marks += score
        
        # Calculate overall results
        percentage = (total_marks / total_max_marks) * 100 if total_max_marks > 0 else 0
        letter_grade = self._calculate_letter_grade(percentage)
        passing = percentage >= 50  # Assuming 50% is passing
        
        grade_report = {
            "overall_results": {
                "total_marks_earned": total_marks,
                "total_max_marks": total_max_marks,
                "percentage": percentage,
                "letter_grade": letter_grade,
                "passing": passing
            }
        }
        
        return EvaluationResult(
            question_scores=question_scores,
            feedback=feedback,
            grade_report=grade_report,
            message="Evaluation completed successfully"
        )
    
    def _evaluate_answer(self, question_text: str, model_answer: str, 
                         student_answer: str, max_marks: int, keywords: List[str]) -> tuple:
        """
        Evaluate a single answer
        
        Args:
            question_text: The question text
            model_answer: The model answer
            student_answer: The student's answer
            max_marks: Maximum marks for this question
            keywords: List of keywords to look for
            
        Returns:
            Tuple of (score, feedback)
        """
        # This is a placeholder for the actual AI evaluation logic
        # In a real implementation, this would use NLP or an AI model to compare answers
        
        # Simple keyword matching for demonstration
        score = 0
        matched_keywords = []
        
        # Check for keywords
        for keyword in keywords:
            if keyword.lower() in student_answer.lower():
                score += 1
                matched_keywords.append(keyword)
        
        # Normalize score to max_marks
        normalized_score = min(score, max_marks)
        
        feedback = f"Your answer included {len(matched_keywords)} key concepts: {', '.join(matched_keywords)}"
        if score < max_marks:
            feedback += f". You could improve by including more of these concepts: {', '.join(set(keywords) - set(matched_keywords))}"
        
        return normalized_score, feedback
    
    def _calculate_letter_grade(self, percentage: float) -> str:
        """
        Calculate letter grade based on percentage
        
        Args:
            percentage: Percentage score
            
        Returns:
            Letter grade
        """
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        elif percentage >= 50:
            return "E"
        else:
            return "F"


# ================ APPLICATION LAYER =================

class Neo4jAnswerEvaluationSystem:
    """Core system that integrates repository and evaluation logic"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the evaluation system
        
        Args:
            openai_api_key: Optional API key for OpenAI
        """
        self.repository = ExamRepository()
        self.evaluator = AnswerEvaluator(openai_api_key)
    
    def evaluate_student_submission(self, roll_no: str, subject: str) -> EvaluationResult:
        """
        Evaluate a student's submission
        
        Args:
            roll_no: Student roll number
            subject: Subject name
            
        Returns:
            EvaluationResult object
        """
        # Check if submission exists
        if not self.repository.check_submission_exists(roll_no, subject):
            raise ValueError(f"No submission found for student {roll_no} in subject {subject}")
        
        # Get submission ID
        submission_id = self.repository.get_submission_id(roll_no, subject)
        if not submission_id:
            raise ValueError(f"Cannot retrieve submission ID for student {roll_no} in subject {subject}")
        
        # Get questions and answers
        questions = self.repository.get_question_data(subject)
        answers = self.repository.get_student_answers(submission_id)
        
        # Evaluate answers
        evaluation_result = self.evaluator.evaluate_submission(questions, answers)
        
        # Store evaluation results
        success = self.repository.store_evaluation(submission_id, evaluation_result.dict())
        if not success:
            logger.warning(f"Failed to store evaluation results for submission {submission_id}")
        
        return evaluation_result
    
    def get_student_results(self, roll_no: str, subject: Optional[str] = None) -> StudentResult:
        """
        Get student results
        
        Args:
            roll_no: Student roll number
            subject: Optional subject name to filter results
            
        Returns:
            StudentResult object
        """
        try:
            results = self.repository.get_student_results(roll_no, subject)
            
            if not results:
                return StudentResult(
                    status="success",
                    message=f"No results found for student {roll_no}" + (f" in subject {subject}" if subject else ""),
                    results=[]
                )
            
            return StudentResult(
                status="success",
                results=results
            )
        except Exception as e:
            logger.error(f"Error fetching student results: {str(e)}")
            return StudentResult(
                status="error",
                message=f"Error fetching results: {str(e)}"
            )


# ================ SERVICE LAYER =================

class EvaluationService:
    """Service layer for handling evaluation requests"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the evaluation service
        
        Args:
            openai_api_key: Optional API key for OpenAI
        """
        self.system = Neo4jAnswerEvaluationSystem(openai_api_key)
    
    async def evaluate_submission(self, roll_no: str, subject: str) -> Dict:
        """
        Evaluate a student's submission
        
        Args:
            roll_no: Student roll number
            subject: Subject name
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            result = self.system.evaluate_student_submission(roll_no, subject)
            return {
                "status": "success",
                "message": "Evaluation completed successfully",
                "data": result.dict()
            }
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error evaluating submission: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def get_student_results(self, roll_no: str, subject: Optional[str] = None) -> Dict:
        """
        Get student results
        
        Args:
            roll_no: Student roll number
            subject: Optional subject name to filter results
            
        Returns:
            Dictionary with student results
        """
        try:
            result = self.system.get_student_results(roll_no, subject)
            return result.dict()
        except Exception as e:
            logger.error(f"Error fetching student results: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")