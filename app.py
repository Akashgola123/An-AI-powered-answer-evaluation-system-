

# from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import os
# import bcrypt
# from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
# from AIAnswerEvaluationSystem.Fetching import QuestionFetcher
# from AIAnswerEvaluationSystem.ocr import GeminiTextExtractor
# from AIAnswerEvaluationSystem.logger import logger
# # from AIAnswerEvaluationSystem.database_login_rigister import ExamService,ExamSubmissionModel


# from typing import List, Dict
# import time

# # Load environment variables
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI()

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure Neo4j
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_secure_password")

# # Initialize database and services
# neo4j_manager = Neo4jManager.get_instance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
# fetcher = QuestionFetcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
# extractor = GeminiTextExtractor()
# # exam_service = ExamService()


# class ExamSubmission(BaseModel):
#     roll_no: str
#     subject: str
#     answers: Dict[str, str]

# class StudentRegister(BaseModel):
#     name: str
#     roll_no: str
#     password: str
#     department: str
#     semester: int


# class StudentLogin(BaseModel):
#     roll_no: str
#     password: str




# def check_neo4j():
#     """Check if Neo4j is connected before allowing requests."""
#     try:
#         driver = neo4j_manager.get_connection()
#         if driver is None:
#             raise HTTPException(status_code=500, detail="Database connection error. Try again later.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Neo4j connection error: {str(e)}")


# @app.get("/neo4j_status")
# def neo4j_status():
#     """API to check if Neo4j is connected."""
#     try:
#         check_neo4j()
#         return {"status": "success", "message": "Neo4j connection is active"}
#     except HTTPException as e:
#         return {"status": "error", "message": str(e.detail)}


# @app.post("/register_student")
# def register_student(data: StudentRegister):
#     """Register a new student in the system."""
#     check_neo4j()  # Ensure Neo4j is connected

#     try:
#         query_check = "MATCH (s:Student {roll_no: $roll_no}) RETURN s"
#         with neo4j_manager.get_connection().session() as session:
#             result = session.run(query_check, {"roll_no": data.roll_no})
#             if result.single():
#                 raise HTTPException(status_code=400, detail="Roll number already registered")

#         hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
#         query_insert = """
#         MERGE (d:Department {name: $department})
#         CREATE (s:Student {
#             name: $name, 
#             roll_no: $roll_no, 
#             password: $password, 
#             department: $department,
#             semester: $semester
#         })
#         CREATE (s)-[:BELONGS_TO]->(d)
#         """
#         with neo4j_manager.get_connection().session() as session:
#             session.run(query_insert, {
#                 "name": data.name,
#                 "roll_no": data.roll_no,
#                 "password": hashed_password,
#                 "department": data.department,
#                 "semester": data.semester
#             })
#         return {"status": "success", "message": "Student registered successfully!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/login_student")
# def login_student(data: StudentLogin):
#     """Login student by verifying credentials."""
#     check_neo4j()  # Ensure Neo4j is connected

#     try:
#         query = "MATCH (s:Student {roll_no: $roll_no}) RETURN s.password AS stored_password"
#         with neo4j_manager.get_connection().session() as session:
#             result = session.run(query, {"roll_no": data.roll_no})
#             record = result.single()
#             if not record:
#                 raise HTTPException(status_code=400, detail="User not found")

#             stored_password = record["stored_password"]
#             if not bcrypt.checkpw(data.password.encode(), stored_password.encode()):
#                 raise HTTPException(status_code=401, detail="Invalid credentials")

#         return {"status": "success", "message": "Login successful!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/subjects")
# def get_subjects():
#     """Fetch available subjects"""
#      # Ensure Neo4j is connected
#     check_neo4j()
#     neo4j_manager = Neo4jManager.get_instance()
#     query = "MATCH (s:Subject) RETURN s.name AS name, s.department AS department"
#     try:
        
#         results = neo4j_manager.execute_query(query)
#         return [{"name": r["name"], "department": r["department"]} for r in results]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching subjects: {str(e)}")


# @app.get("/questions")
# def get_questions(subject: str, limit: int = 50):  # Default limit = 50
#     """
#     Fetch exam questions for a given subject.

#     - `subject`: The subject name for which to fetch questions.
#     - `limit`: Number of questions to fetch (default: 50).
#     """
#     if not subject:
#         raise HTTPException(status_code=400, detail="❌ Subject parameter is required")

#     try:
#         questions = fetcher.fetch_questions(subject, limit)

#         if not questions:
#             return {"questions": []}  # Return an empty list if no questions are found

#         return {"questions": questions}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Error fetching questions: {str(e)}")


# # @app.post("/extract-text")
# # def extract_text(file: UploadFile = File(...)):
# #     """Extract text from uploaded image"""
# #     check_neo4j()  # Ensure Neo4j is connected

# #     try:
# #         extracted_text = extractor.extract_text(file.file.read(), file.content_type)
# #         return {"text": extracted_text}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload-answer")
# async def upload_answer(file: UploadFile = File(...)):
#     """
#     Upload an answer sheet (image/PDF), extract text, and return it.
#     """
#     try:
#         content = await file.read()  # Read file content
#         extracted_text = extractor.extract_text(content, file.content_type)

#         if not extracted_text:
#             raise HTTPException(status_code=400, detail="Text extraction failed")

#         return {"status": "success", "extracted_text": extracted_text}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    


# # @app.post("/submit_exam")
# # def submit_exam(submission: ExamSubmission):
# #     try:
# #         for qid, answer in submission.answers.items():
# #             query = """
# #             MATCH (s:Student {roll_no: $roll_no})
# #             MATCH (q:Question {id: $qid})
# #             MATCH (sub:Subject {name: $subject})
# #             CREATE (a:Answer {
# #                 text: $answer, 
# #                 submitted_at: timestamp()
# #             })
# #             CREATE (s)-[:SUBMITTED]->(a)
# #             CREATE (a)-[:FOR_QUESTION]->(q)
# #             CREATE (a)-[:IN_SUBJECT]->(sub)
# #             """

# #             neo4j_manager = Neo4jManager.get_instance()
# #             neo4j_manager.execute_query(query, {
# #                 "roll_no": submission.roll_no, 
# #                 "qid": qid, 
# #                 "subject": submission.subject,
# #                 "answer": answer
# #             })
# #         return {"message": "Exam submitted successfully!"}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")

# # @app.post("/submit_exam")
# # def submit_exam(submission: ExamSubmission):
# #     try:
# #         query = """
# #         MATCH (s:Student {roll_no: $roll_no})
# #         MATCH (sub:Subject {name: $subject})
# #         MERGE (exam:ExamSubmission {id: $exam_id})  // Unique exam submission node
# #         ON CREATE SET exam.created_at = timestamp()

# #         MERGE (s)-[:SUBMITTED]->(exam)
# #         MERGE (exam)-[:FOR_SUBJECT]->(sub)

# #         FOREACH (qid IN keys($answers) | 
# #             MERGE (q:Question {id: qid})
# #             MERGE (a:Answer {id: $roll_no + "_" + qid, text: $answers[qid]})
# #             ON CREATE SET a.submitted_at = timestamp()

# #             MERGE (exam)-[:HAS_ANSWER]->(a)
# #             MERGE (a)-[:FOR_QUESTION]->(q)
# #             MERGE (a)-[:IN_SUBJECT]->(sub)
# #         )
# #         """

# #         neo4j_manager = Neo4jManager.get_instance()
# #         neo4j_manager.execute_query(query, {
# #             "roll_no": submission.roll_no, 
# #             "subject": submission.subject,
# #             "exam_id": f"{submission.roll_no}_{submission.subject}_{int(time.time())}",  
# #             "answers": submission.answers
# #         })

# #         return {"message": "Exam submitted successfully!"}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")
# # @app.post("/submit_exam")
# # def submit_exam(submission: ExamSubmission):
# #     try:
# #         submission_id = f"{submission.roll_no}_{submission.subject}_{int(time.time())}"  # Unique ID


        
# #         # Step 2: Create a new exam submission node linked only to the selected subject
# #         submission_id = f"{submission.roll_no}_{submission.subject}_{int(time.time())}"
# #         exam_query = """
# #         MATCH (s:Student {roll_no: $roll_no})
# #         MATCH (sub:Subject {name: $subject})
# #         CREATE (exam:ExamSubmission {id: $submission_id, subject: $subject, submitted_at: timestamp()})
# #         CREATE (s)-[:SUBMITTED]->(exam)
# #         CREATE (exam)-[:FOR_SUBJECT]->(sub);
# #         """
# #         neo4j_manager.execute_query(exam_query, {
# #             "roll_no": submission.roll_no, 
# #             "subject": submission.subject,
# #             "submission_id": submission_id
# #         })

# #         # Step 3: Ensure Answers Are Only for Selected Subject
# #         for qid, answer in submission.answers.items():
# #             answer_query = """
# #             MATCH (exam:ExamSubmission {id: $submission_id})
# #             MATCH (q:Question {id: $qid})-[:HAS_QUESTION]->(sub:Subject {name: $subject}) 
# #             CREATE (a:Answer {
# #                 id: $answer_id,
# #                 text: $answer, 
# #                 submitted_at: timestamp()
# #             })
# #             CREATE (exam)-[:HAS_ANSWER]->(a)
# #             CREATE (a)-[:FOR_QUESTION]->(q);
# #             """
# #             neo4j_manager.execute_query(answer_query, {
# #                 "submission_id": submission_id,
# #                 "qid": qid,
# #                 "subject": submission.subject,  # Added Subject Validation
# #                 "answer_id": f"{submission_id}_{qid}",
# #                 "answer": answer
# #             })
# #         logger.info(f"Exam submitted successfully for {submission.roll_no} in {submission.subject}")
# #         return {"message": "Exam submitted successfully!", "submission_id": submission_id}

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")
        
#     #     # Create a new exam submission node
#     #     exam_query = """
#     #     MATCH (s:Student {roll_no: $roll_no})
#     #     MATCH (sub:Subject {name: $subject})
#     #     CREATE (exam:ExamSubmission {id: $submission_id, subject: $subject, submitted_at: timestamp()})
#     #     CREATE (s)-[:SUBMITTED]->(exam)
#     #     CREATE (exam)-[:FOR_SUBJECT]->(sub);
#     #     """

#     #     neo4j_manager = Neo4jManager.get_instance()
#     #     neo4j_manager.execute_query(exam_query, {
#     #         "roll_no": submission.roll_no, 
#     #         "subject": submission.subject,
#     #         "submission_id": submission_id
#     #     })

#     #     # Store each answer linked to the exam submission
#     #     for qid, answer in submission.answers.items():
#     #         answer_query = """
#     #         MATCH (exam:ExamSubmission {id: $submission_id})
#     #         MATCH (q:Question {id: $qid})
#     #         CREATE (a:Answer {
#     #             id: $answer_id,
#     #             text: $answer, 
#     #             submitted_at: timestamp()
#     #         })
#     #         CREATE (exam)-[:HAS_ANSWER]->(a)
#     #         CREATE (a)-[:FOR_QUESTION]->(q);
#     #         """

#     #         neo4j_manager.execute_query(answer_query, {
#     #             "submission_id": submission_id,
#     #             "qid": qid,
#     #             "answer_id": f"{submission_id}_{qid}",
#     #             "answer": answer
#     #         })

#     #     return {"message": "Exam submitted successfully!", "submission_id": submission_id}

#     # except Exception as e:
#     #     raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")




# # ✅ API Endpoint to Submit Exam
# @app.post("/submit_exam")
# def submit_exam(submission: ExamSubmission):
#     try:
#         for qid, answer in submission.answers.items():
#             query = """
#             MATCH (s:Student {roll_no: $roll_no})
#             MATCH (q:Question {id: $qid})
#             MATCH (sub:Subject {name: $subject})
#             CREATE (a:Answer {
#                 text: $answer, 
#                 submitted_at: timestamp()
#             })
#             CREATE (s)-[:SUBMITTED]->(a)
#             CREATE (a)-[:FOR_QUESTION]->(q)
#             CREATE (a)-[:IN_SUBJECT]->(sub)
#             """
#             Neo4jManager.execute_query(query, {
#                 "roll_no": submission.roll_no, 
#                 "qid": qid, 
#                 "subject": submission.subject,
#                 "answer": answer
#             })
#         return {"message": "Exam submitted successfully!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")


# @app.get("/")
# def serve_frontend():
#     """Serve frontend HTML page"""
#     with open("templates/index.html", "r") as f:
#         return HTMLResponse(content=f.read(), status_code=200)


# if __name__ == "__main__":
#     # Initialize database before starting the server
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, HTTPException, UploadFile, File,APIRouter,Depends,Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
import bcrypt
from typing import Dict,List, Optional,Any
from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.Fetching import QuestionFetcher
from AIAnswerEvaluationSystem.ocr import GeminiTextExtractor
from AIAnswerEvaluationSystem.logger import logger
from AIAnswerEvaluationSystem.database_login_rigister import ExamService, ExamSubmission
# from AIAnswerEvaluationSystem.question_answer_store import DatabaseConnection
# from AIAnswerEvaluationSystem.question_answer_store import SubjectRepository
# from AIAnswerEvaluationSystem.question_answer_store import QuestionRepository
# from AIAnswerEvaluationSystem.question_answer_store import AnswerRepository
# from AIAnswerEvaluationSystem.question_answer_store import ContextRepository
# from AIAnswerEvaluationSystem.question_answer_store import ExamService
# from AIAnswerEvaluationSystem.question_answer_store import DataInitializer
# from AIAnswerEvaluationSystem.question_answer_store import DatabaseConfig,DatabaseDiagnostic
# import time

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

# Data models
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

class TeacherRegister(BaseModel):
    name: str
    password: str
    department: str

class TeacherLogin(BaseModel):
    name: str
    password: str


# class SubjectCreate(BaseModel):
#     subject_id: str = Field(..., description="Unique ID for the subject (e.g., 'MATH001')")
#     subject_name: str = Field(..., description="Name of the subject")

# class SubjectResponse(BaseModel):
#     id: str
#     name: str

# class QuestionCreate(BaseModel):
#     subject_id: str = Field(..., description="ID of the subject")
#     qid: str = Field(..., description="User-defined question ID (e.g., 'Q001')")
#     question_text: str = Field(..., description="Text of the question")
#     marks: int = Field(..., description="Number of marks for the question")
#     answer_text: str = Field(..., description="Text of the answer")
#     context_text: Optional[str] = Field(None, description="Optional context for the question")

# class QuestionResponse(BaseModel):
#     id: str
#     qid: str
#     question: str
#     marks: int
#     answer: Optional[str] = None
#     context: Optional[str] = None
#     subject_id: Optional[str] = None
#     subject_name: Optional[str] = None



# def get_db_connection():
#     config = DatabaseConfig()
#     db_connection = DatabaseConnection(config.uri, config.username, config.password)
#     try:
#         yield db_connection
#     finally:
#         db_connection.close()

# def get_exam_service(db_connection: DatabaseConnection = Depends(get_db_connection)):
#     subject_repo = SubjectRepository(db_connection)
#     question_repo = QuestionRepository(db_connection)
#     answer_repo = AnswerRepository(db_connection)
#     context_repo = ContextRepository(db_connection)
#     return ExamService(subject_repo, question_repo, answer_repo, context_repo)


# Helper function to check Neo4j connection
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

@app.get("/neo4j_status")
def neo4j_status():
    """API to check if Neo4j is connected."""
    try:
        check_neo4j()
        return {"status": "success", "message": "Neo4j connection is active"}
    except HTTPException as e:
        return {"status": "error", "message": str(e.detail)}
    

@app.post("/register_teacher")
def register_teacher(data: TeacherRegister):
    """Register a new teacher in the system."""
    try:
        driver = check_neo4j()
        
        # Check if teacher already exists
        with driver.session() as session:
            result = session.run(
                "MATCH (t:Teacher {name: $name}) RETURN t", 
                {"name": data.name}
            )
            if result.single():
                raise HTTPException(status_code=400, detail="Teacher already registered")

        # Hash password
        hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
        
        # Create teacher node and relationship with department
        query = """
        MERGE (d:Department {name: $department})
        CREATE (t:Teacher {
            name: $name, 
            password: $password,
            department: $department,
            created_at: timestamp()
        })
        CREATE (t)-[:TEACHES_IN]->(d)
        RETURN t.name as name
        """
        
        with driver.session() as session:
            result = session.run(query, {
                "name": data.name,
                "password": hashed_password,
                "department": data.department
            })
            
            if not result.single():
                raise HTTPException(status_code=500, detail="Failed to create teacher record")
                
        logger.info(f"Teacher registered successfully: {data.name}")
        return {"status": "success", "message": "Teacher registered successfully!"}
    
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error registering teacher: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering teacher: {str(e)}")

@app.post("/login_teacher")
def login_teacher(data: TeacherLogin):
    """Login teacher by verifying credentials."""
    try:
        driver = check_neo4j()
        
        with driver.session() as session:
            result = session.run(
                "MATCH (t:Teacher {name: $name}) RETURN t.password AS stored_password", 
                {"name": data.name}
            )
            record = result.single()
            
            if not record:
                raise HTTPException(status_code=400, detail="Teacher not found")

            stored_password = record["stored_password"]
            if not bcrypt.checkpw(data.password.encode(), stored_password.encode()):
                raise HTTPException(status_code=401, detail="Invalid credentials")

        logger.info(f"Teacher login successful: {data.name}")
        return {
            "status": "success", 
            "message": "Login successful!",
            "teacher_name": data.name
        }
    
    except HTTPException as e:
        # Re-throw HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error during teacher login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during teacher login: {str(e)}")


# @app.post("/initialize", tags=["Admin"])
# async def initialize_database(db_connection: DatabaseConnection = Depends(get_db_connection),
#                              exam_service: ExamService = Depends(get_exam_service)):
#     """Initialize the database with sample data"""
#     initializer = DataInitializer(db_connection, exam_service)
#     initializer.initialize_database()
#     return {"message": "Database initialized with sample data"}

# @app.get("/diagnostic", tags=["Admin"])
# async def run_diagnostic(db_connection: DatabaseConnection = Depends(get_db_connection)):
#     """Run database diagnostic to verify content"""
#     subject_repo = SubjectRepository(db_connection)
#     question_repo = QuestionRepository(db_connection)
#     diagnostic = DatabaseDiagnostic(subject_repo, question_repo)
    
#     # We can't directly return the output of verify_database as it prints to console
#     # Instead, we'll collect the data it would check and return it
#     subjects = subject_repo.get_all_subjects()
#     questions = question_repo.get_all_questions()
#     qa_pairs = question_repo.get_questions_with_answers()
#     qc_pairs = question_repo.get_questions_with_context()
    
#     return {
#         "subjects_count": len(subjects),
#         "questions_count": len(questions),
#         "qa_pairs_count": len(qa_pairs),
#         "qc_pairs_count": len(qc_pairs),
#         "status": "Database diagnostic complete"
#     }

# @app.get("/subjects", response_model=List[SubjectResponse], tags=["Subjects"])
# async def get_subjects(exam_service: ExamService = Depends(get_exam_service)):
#     """Get all subjects"""
#     subjects = exam_service.list_all_subjects()
#     return subjects

# @app.post("/subjects", response_model=Dict[str, Any], tags=["Subjects"])
# async def create_subject(subject: SubjectCreate, exam_service: ExamService = Depends(get_exam_service)):
#     """Create a new subject"""
#     result = exam_service.create_new_subject(subject.subject_id, subject.subject_name)
#     return {"success": True, "subject_id": result}

# @app.get("/subjects/{subject_id}/questions", response_model=List[QuestionResponse], tags=["Questions"])
# async def get_questions_by_subject(
#     subject_id: str = Path(..., description="ID of the subject"),
#     exam_service: ExamService = Depends(get_exam_service)
# ):
#     """Get all questions for a specific subject"""
#     questions = exam_service.list_questions_by_subject(subject_id)
#     return questions

# @app.get("/questions/{qid}", response_model=QuestionResponse, tags=["Questions"])
# async def get_question_by_qid(
#     qid: str = Path(..., description="User-defined question ID"),
#     exam_service: ExamService = Depends(get_exam_service)
# ):
#     """Get question details by its qid"""
#     question = exam_service.get_question_by_qid(qid)
#     if not question:
#         raise HTTPException(status_code=404, detail=f"Question with QID {qid} not found")
#     return question

# @app.post("/questions", response_model=Dict[str, Any], tags=["Questions"])
# async def create_question(
#     question: QuestionCreate,
#     exam_service: ExamService = Depends(get_exam_service)
# ):
#     """Create a new question with its answer and context for a specific subject"""
#     result = exam_service.create_question_and_answer(
#         question.subject_id,
#         question.qid,
#         question.question_text,
#         question.marks,
#         question.answer_text,
#         question.context_text
#     )
    
#     if not result.get("success", False):
#         raise HTTPException(status_code=400, detail=result.get("message", "Failed to create question"))
    
#     return result


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

# @app.post("/submit_exam")
# def submit_exam(submission: ExamSubmission):
#     """
#     Submit exam answers for evaluation.
    
#     Creates Answer nodes linked to Student, Question, and Subject nodes.
#     """
#     try:
#         driver = check_neo4j()
#         submission_time = int(time.time())
        
#         # First verify student and subject exist
#         with driver.session() as session:
#             student_result = session.run(
#                 "MATCH (s:Student {roll_no: $roll_no}) RETURN s", 
#                 {"roll_no": submission.roll_no}
#             )
#             if not student_result.single():
#                 raise HTTPException(status_code=400, detail="Student not found")
                
#             subject_result = session.run(
#                 "MATCH (s:Subject {name: $subject}) RETURN s", 
#                 {"subject": submission.subject}
#             )
#             if not subject_result.single():
#                 raise HTTPException(status_code=400, detail="Subject not found")
        
#         # Create an exam submission node
#         submission_id = f"{submission.roll_no}_{submission.subject}_{submission_time}"
        
#         with driver.session() as session:
#             session.run("""
#                 MATCH (s:Student {roll_no: $roll_no})
#                 MATCH (sub:Subject {name: $subject})
#                 CREATE (e:ExamSubmission {
#                     id: $submission_id,
#                     submitted_at: timestamp(),
#                     question_count: $question_count
#                 })
#                 CREATE (s)-[:SUBMITTED]->(e)
#                 CREATE (e)-[:FOR_SUBJECT]->(sub)
#                 """, {
#                     "roll_no": submission.roll_no,
#                     "subject": submission.subject,
#                     "submission_id": submission_id,
#                     "question_count": len(submission.answers)
#                 })
        
#         # Store each answer
#         for qid, answer_text in submission.answers.items():
#             with driver.session() as session:
#                 session.run("""
#                     MATCH (e:ExamSubmission {id: $submission_id})
#                     MATCH (q:Question {id: $qid})
#                     MATCH (sub:Subject {name: $subject})
#                     CREATE (a:Answer {
#                         id: $answer_id,
#                         text: $answer_text,
#                         submitted_at: timestamp()
#                     })
#                     CREATE (e)-[:HAS_ANSWER]->(a)
#                     CREATE (a)-[:FOR_QUESTION]->(q)
#                     CREATE (a)-[:IN_SUBJECT]->(sub)
#                     """, {
#                         "submission_id": submission_id,
#                         "qid": qid,
#                         "subject": submission.subject,
#                         "answer_id": f"{submission_id}_{qid}",
#                         "answer_text": answer_text
#                     })
        
#         logger.info(f"Exam submitted successfully: {submission_id} with {len(submission.answers)} answers")
#         return {
#             "status": "success", 
#             "message": "Exam submitted successfully!",
#             "submission_id": submission_id
#         }
        
#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error submitting exam: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error submitting exam: {str(e)}")

@app.post("/submit_exam")
def submit_exam(submission: ExamSubmission):
    """
    Submit exam answers for evaluation.
    """
    return exam_service.submit_exam(submission)


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

if __name__ == "__main__":
    # Initialize database before starting the server
    logger.info("Starting AI Answer Evaluation System API")
    uvicorn.run(app, host="0.0.0.0", port=8000)