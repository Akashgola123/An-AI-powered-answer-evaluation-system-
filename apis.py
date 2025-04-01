from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os
import bcrypt
import time
import uvicorn

# Import from AIAnswerEvaluationSystem
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.Fetching import QuestionFetcher
from AIAnswerEvaluationSystem.ocr import GeminiTextExtractor
from AIAnswerEvaluationSystem.logger import logger
from AIAnswerEvaluationSystem.database_login_rigister import ExamService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Answer Evaluation System", 
              description="API for student exam submission and evaluation")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_secure_password")

# Initialize database and services
neo4j_manager = Neo4jManager.get_instance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
fetcher = QuestionFetcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
extractor = GeminiTextExtractor()
router = APIRouter()
exam_service = ExamService()

# ================ DATA MODELS =================

class ExamSubmission(BaseModel):
    roll_no: str
    subject: str
    answers: Dict[str, str]

class StudentRegister(BaseModel):
    name: str
    roll_no: str
    password: str
    department: str
    semester: int

class StudentLogin(BaseModel):
    roll_no: str
    password: str

class EvaluationRequest(BaseModel):
    roll_no: str
    subject: str
    generate_corrections: bool = False

class BatchEvaluationRequest(BaseModel):
    roll_numbers: List[str]
    subject: str
    generate_corrections: bool = False

class EvaluationResponse(BaseModel):
    question_scores: Dict
    feedback: Dict
    grade_report: Dict
    message: Optional[str] = None

# ================ HELPER FUNCTIONS =================

def check_neo4j():
    """Check if Neo4j is connected before allowing requests."""
    try:
        driver = neo4j_manager.get_connection()
        if driver is None:
            raise HTTPException(status_code=500, detail="Database connection error. Try again later.")
        return driver
    except Exception as e:
        logger.error(f"Neo4j connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Neo4j connection error: {str(e)}")

# ================ EVALUATION SERVICE CLASS =================

class EvaluationService:
    def __init__(self):
        from AIAnswerEvaluationSystem.llmss import Neo4jAnswerEvaluationSystem
        self.evaluation_system = Neo4jAnswerEvaluationSystem()
        self.exam_repo = self.evaluation_system.exam_repo
    
    def evaluate_exam(self, roll_no: str, subject: str, generate_corrections: bool = False) -> Dict:
        """Evaluate a student's exam"""
        # Check if the student has a submission
        if not self.exam_repo.check_submission_exists(roll_no, subject):
            raise HTTPException(
                status_code=404, 
                detail=f"No exam submission found for student {roll_no} in subject {subject}"
            )
        
        # Evaluate the exam
        return self.evaluation_system.evaluate_student_subject(
            student_roll_no=roll_no,
            subject_name=subject,
            generate_corrections=generate_corrections
        )
    
    def batch_evaluate_exams(self, roll_numbers: List[str], subject: str, 
                           generate_corrections: bool = False) -> Dict:
        """Batch evaluate multiple students' exams"""
        return self.evaluation_system.batch_evaluate_multiple_students(
            student_roll_numbers=roll_numbers,
            subject_name=subject,
            generate_corrections=generate_corrections
        )
    
    def evaluate_and_store_exam(self, roll_no: str, subject: str, 
                              generate_corrections: bool = False) -> Dict:
        """Evaluate and store exam results"""
        # Get the submission ID
        submission_id = self.exam_repo.get_submission_id(roll_no, subject)
        
        if not submission_id:
            raise HTTPException(
                status_code=404, 
                detail=f"No exam submission found for student {roll_no} in subject {subject}"
            )
        
        # Evaluate the exam
        evaluation_result = self.evaluation_system.evaluate_student_subject(
            student_roll_no=roll_no,
            subject_name=subject,
            generate_corrections=generate_corrections
        )
        
        # Store the results
        store_success = self.exam_repo.store_evaluation(
            submission_id=submission_id,
            evaluation=evaluation_result
        )
        
        if store_success:
            evaluation_result["message"] = "Evaluation completed and results stored successfully"
        else:
            evaluation_result["message"] = "Evaluation completed but failed to store results"
            
        return evaluation_result
    
    def get_student_results(self, roll_no: str, subject: Optional[str] = None) -> Dict:
        """Get evaluation results for a student"""
        results = self.exam_repo.get_student_results(roll_no, subject)
        
        if not results:
            if subject:
                message = f"No evaluation results found for student {roll_no} in subject {subject}"
            else:
                message = f"No evaluation results found for student {roll_no}"
            
            return {"status": "warning", "message": message, "results": []}
        
        return {"status": "success", "results": results}

# Initialize evaluation service
evaluation_service = EvaluationService()

# ================ API ENDPOINTS =================

@app.get("/neo4j_status")
def neo4j_status():
    """API to check if Neo4j is connected."""
    try:
        check_neo4j()
        return {"status": "success", "message": "Neo4j connection is active"}
    except HTTPException as e:
        return {"status": "error", "message": str(e.detail)}

@app.post("/register_student")
def register_student(data: StudentRegister):
    """Register a new student in the system."""
    try:
        driver = check_neo4j()
        
        # Check if student already exists
        with driver.session() as session:
            result = session.run(
                "MATCH (s:Student {roll_no: $roll_no}) RETURN s", 
                {"roll_no": data.roll_no}
            )
            if result.single():
                raise HTTPException(status_code=400, detail="Roll number already registered")

        # Hash password
        hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
        
        # Create student node
        query = """
        MERGE (d:Department {name: $department})
        CREATE (s:Student {
            name: $name, 
            roll_no: $roll_no, 
            password: $password, 
            department: $department,
            semester: $semester,
            created_at: timestamp()
        })
        CREATE (s)-[:BELONGS_TO]->(d)
        RETURN s.roll_no as roll_no
        """
        
        with driver.session() as session:
            result = session.run(query, {
                "name": data.name,
                "roll_no": data.roll_no,
                "password": hashed_password,
                "department": data.department,
                "semester": data.semester
            })
            
            if not result.single():
                raise HTTPException(status_code=500, detail="Failed to create student record")
                
        logger.info(f"Student registered successfully: {data.roll_no}")
        return {"status": "success", "message": "Student registered successfully!"}
    
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error registering student: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering student: {str(e)}")

@app.post("/login_student")
def login_student(data: StudentLogin):
    """Login student by verifying credentials."""
    try:
        driver = check_neo4j()
        
        with driver.session() as session:
            result = session.run(
                "MATCH (s:Student {roll_no: $roll_no}) RETURN s.password AS stored_password, s.name AS name", 
                {"roll_no": data.roll_no}
            )
            record = result.single()
            
            if not record:
                raise HTTPException(status_code=400, detail="User not found")

            stored_password = record["stored_password"]
            if not bcrypt.checkpw(data.password.encode(), stored_password.encode()):
                raise HTTPException(status_code=401, detail="Invalid credentials")

        logger.info(f"Student login successful: {data.roll_no}")
        return {
            "status": "success", 
            "message": "Login successful!",
            "student_name": record["name"]
        }
    
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

@app.get("/subjects")
def get_subjects():
    """Fetch available subjects"""
    try:
        driver = check_neo4j()
        
        with driver.session() as session:
            result = session.run("MATCH (s:Subject) RETURN s.name AS name, s.department AS department")
            subjects = [{"name": r["name"], "department": r["department"]} for r in result]
            
        logger.info(f"Retrieved {len(subjects)} subjects")
        return subjects
    
    except Exception as e:
        logger.error(f"Error fetching subjects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching subjects: {str(e)}")

@app.get("/questions")
def get_questions(subject: str, limit: int = 50):
    """
    Fetch exam questions for a given subject.
    
    Parameters:
    - subject: The subject name for which to fetch questions
    - limit: Number of questions to fetch (default: 50)
    """
    if not subject:
        raise HTTPException(status_code=400, detail="Subject parameter is required")

    try:
        questions = fetcher.fetch_questions(subject, limit)
        
        if not questions:
            logger.warning(f"No questions found for subject: {subject}")
            return {"questions": []}
            
        logger.info(f"Retrieved {len(questions)} questions for subject: {subject}")
        return {"questions": questions}
    
    except Exception as e:
        logger.error(f"Error fetching questions for {subject}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching questions: {str(e)}")

@app.post("/upload-answer")
async def upload_answer(file: UploadFile = File(...)):
    """
    Upload an answer sheet (image/PDF), extract text, and return it.
    """
    try:
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types are: {', '.join(allowed_types)}"
            )
            
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        extracted_text = extractor.extract_text(content, file.content_type)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Text extraction failed")

        logger.info(f"Successfully extracted text from uploaded file: {file.filename}")
        return {
            "status": "success", 
            "extracted_text": extracted_text,
            "file_name": file.filename
        }

    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/submit_exam")
def submit_exam(submission: ExamSubmission):
    """
    Submit exam answers for evaluation.
    """
    return exam_service.submit_exam(submission)

@app.post("/evaluate/student", response_model=EvaluationResponse)
async def evaluate_student(request: EvaluationRequest):
    """Evaluate a student's exam in a subject"""
    try:
        check_neo4j()
        result = evaluation_service.evaluate_exam(
            roll_no=request.roll_no,
            subject=request.subject,
            generate_corrections=request.generate_corrections
        )
        logger.info(f"Evaluation completed for student {request.roll_no} in subject {request.subject}")
        return result
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error evaluating exam: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating exam: {str(e)}")

@app.post("/evaluate/batch", response_model=Dict)
async def evaluate_batch(request: BatchEvaluationRequest):
    """Batch evaluate multiple students' exams"""
    try:
        check_neo4j()
        result = evaluation_service.batch_evaluate_exams(
            roll_numbers=request.roll_numbers,
            subject=request.subject,
            generate_corrections=request.generate_corrections
        )
        logger.info(f"Batch evaluation completed for {len(request.roll_numbers)} students in subject {request.subject}")
        return result
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error evaluating batch exams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating batch exams: {str(e)}")

@app.post("/evaluate/store", response_model=EvaluationResponse)
async def evaluate_and_store(request: EvaluationRequest):
    """Evaluate and store exam results"""
    try:
        check_neo4j()
        result = evaluation_service.evaluate_and_store_exam(
            roll_no=request.roll_no,
            subject=request.subject,
            generate_corrections=request.generate_corrections
        )
        logger.info(f"Evaluation stored for student {request.roll_no} in subject {request.subject}")
        return result
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error evaluating and storing exam: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating and storing exam: {str(e)}")

@app.get("/results/{roll_no}")
async def get_student_results(roll_no: str, subject: Optional[str] = None):
    """Get evaluation results for a student"""
    try:
        check_neo4j()
        result = evaluation_service.get_student_results(roll_no, subject)
        logger.info(f"Retrieved results for student {roll_no}")
        return result
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error retrieving student results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving student results: {str(e)}")

@app.get("/")
def serve_frontend():
    """Serve frontend HTML page"""
    try:
        with open("templates/index.html", "r") as f:
            content = f.read()
            return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>AI Answer Evaluation System API</h1><p>Frontend template not found.</p></body></html>", status_code=200)
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving frontend")

# ================ RUN SERVER =================

if __name__ == "__main__":
    # Initialize database before starting the server
    logger.info("Starting AI Answer Evaluation System API")
    uvicorn.run(app, host="0.0.0.0", port=8000)