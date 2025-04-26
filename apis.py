


# from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Depends, Path
# from fastapi import status
# from fastapi.responses import HTMLResponse
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# import os
# import bcrypt
# from typing import Dict, List, Optional, Any
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder
# import logging
# from contextlib import asynccontextmanager
# from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
# from AIAnswerEvaluationSystem.Fetching import QuestionFetcher
# from AIAnswerEvaluationSystem.ocr import GeminiTextExtractor
# from AIAnswerEvaluationSystem.logger import logger
# from AIAnswerEvaluationSystem.database_login_rigister import ExamService
# from AIAnswerEvaluationSystem.question_answer_store import Neo4jConnection, QuestionUploader
# from AIAnswerEvaluationSystem.llmss import FineTunedLLMEvaluator # Important!
# from AIAnswerEvaluationSystem.evaluting import EvaluationProcessor

# # Load environment variables
# load_dotenv()



# # --- DEFINE Grading Logic Constants and Function ---
# PASS_FAIL_THRESHOLD_PERCENTAGE = 60.0 # Define threshold here

# GRADE_SCALE_RULES = { # Use tuples for range checks if preferred, or direct mapping
#     90: 'A+',
#     80: 'A',
#     70: 'B',
#     60: 'C', # Assuming C is the lowest pass grade at 60% threshold
#     50: 'D',
#     0: 'F'  # Catch-all for below D
# }

# def calculate_grade_and_status(marks_obtained: Optional[float], max_marks_possible: float) -> Dict[str, Any]:
#     """Calculates percentage, grade, and status based on actual marks."""
#     results = {"percentage": None, "letter_grade": "N/A", "status": "N/A"}
#     if marks_obtained is None or marks_obtained < 0 or max_marks_possible <= 0:
#          logger.warning(f"Grading skipped: Invalid marks_obtained ({marks_obtained}) or max_marks ({max_marks_possible}).")
#          return results # Return defaults if marks invalid

#     percentage = round((marks_obtained / max_marks_possible) * 100, 2)
#     results["percentage"] = percentage

#     grade = 'F' # Default to F
#     # Iterate through sorted thresholds (high to low)
#     for threshold, letter in sorted(GRADE_SCALE_RULES.items(), key=lambda item: item[0], reverse=True):
#         if percentage >= threshold:
#              grade = letter
#              break # Stop at the first matching (highest) grade
#     results["letter_grade"] = grade

#     # Determine status based on percentage threshold
#     results["status"] = "Pass" if percentage >= PASS_FAIL_THRESHOLD_PERCENTAGE else "Fail"
#     return results
# # --- END Grading Logic ---

# # Configure Neo4j
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_secure_password")

# # Global app state to store initialized components
# app_state: Dict[str, Any] = {"neo4j_manager": None, "question_uploader": None, "startup_error": None}

# # Application lifecycle management
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("App startup: Initializing resources...")
#     manager = None
#     llm_eval = None
#     processor = None
#     exam_service = None
#     uploader = None
    
#     try:
#         # Initialize Neo4j manager
#         manager = Neo4jManager.get_instance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
#         app_state["neo4j_manager"] = manager
#         logger.info("Neo4jManager initialized.")

#         # Initialize Neo4jConnection for QuestionUploader
#         neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
#         uploader = QuestionUploader(neo4j_connection=neo4j_conn)
#         app_state["question_uploader"] = uploader
#         logger.info("QuestionUploader initialized.")

#         # Initialize LLM evaluator
#         llm_eval = FineTunedLLMEvaluator()
#         app_state["llm_evaluator"] = llm_eval
#         logger.info("FineTunedLLMEvaluator configured.")

#         # Initialize Processor only if BOTH dependencies are available
#         if manager and llm_eval:
#             processor = EvaluationProcessor(neo4j_manager=manager, llm_evaluator=llm_eval)
#             app_state["evaluation_processor"] = processor
#             logger.info("EvaluationProcessor initialized.")
#         else:
#             raise RuntimeError("Cannot initialize EvaluationProcessor due to missing Neo4jManager or LLMEvaluator.")

#         # Initialize ExamService using the processor
#         if processor:
#             exam_service = ExamService(evaluation_processor=processor)
#             app_state["exam_service"] = exam_service
#             logger.info("ExamService initialized.")
#         else:
#             raise RuntimeError("Cannot initialize ExamService due to missing EvaluationProcessor.")

#         # Initialize other services
#         app_state["question_fetcher"] = QuestionFetcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
#         app_state["text_extractor"] = GeminiTextExtractor()

#         app_state["startup_error"] = None # Clear error if successful
#         logger.info("All core services initialized successfully.")

#     except (ValueError, ConnectionError, RuntimeError, Exception) as e:
#         logger.critical(f"FATAL: Startup initialization failed: {e}", exc_info=True)
#         app_state["startup_error"] = str(e)

#     yield # Application runs

#     # Shutdown
#     logger.info("App shutdown: Cleaning up resources...")
#     if manager and isinstance(manager, Neo4jManager): manager.close()
#     # Add cleanup for other resources if needed
#     app_state.clear()
#     logger.info("Shutdown complete.")




# async def get_evaluation_processor() -> EvaluationProcessor:
#     startup_error=app_state.get("startup_error"); processor=app_state.get("evaluation_processor")
#     if startup_error: raise HTTPException(503, f"Init Fail:{startup_error}")
#     if processor is None: raise HTTPException(503,"Evaluation processor unavailable")
#     return processor


# # Initialize FastAPI app with the lifespan manager
# app = FastAPI(
#     title="AI Answer Evaluation System", 
#     description="API for student exam submission and evaluation",
#     lifespan=lifespan  # Connect the lifespan function to the app
# )

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize router
# router = APIRouter()

# # Data models
# class QuestionUploadRequest(BaseModel):
#     question_id: str = Field(..., description="Unique identifier for the question (e.g., 'PHYSICS_KINEMATICS_Q1')")
#     subject_name: str = Field(..., description="The subject this question belongs to (must exist)")
#     question_text: str = Field(..., description="The full text of the question")
#     correct_answer_text: str = Field(..., description="The correct or model answer text")
#     max_marks: float = Field(..., gt=0, description="Maximum marks for the question (must be > 0)")
#     concepts: Optional[List[str]] = Field(None, description="Optional list of related concepts/context keywords")

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

# class TeacherRegister(BaseModel):
#     name: str
#     password: str
#     department: str

# class TeacherLogin(BaseModel):
#     name: str
#     password: str

# # --- Add near other Pydantic models ---

# class EvaluationResultDetail(BaseModel):
#     # Match the keys returned by EvaluationProcessor.get_student_evaluations's processing loop
#     student_roll_no: str
#     student_name: Optional[str] = None
#     subject_name: str
#     question_id: str
#     question_text: Optional[str] = None
#     submitted_answer: Optional[str] = None
#     max_marks_possible: Any # Float or N/A
#     numeric_score: Optional[int] = None # 0-5 score from LLM
#     score_str: Optional[str] = None # e.g., "4/5" or "N/A"
#     marks_obtained: Optional[float] = None
#     percentage: Optional[float] = None
#     letter_grade: Optional[str] = None
#     status: Optional[str] = None # "Pass", "Fail", "N/A"
#     feedback: Optional[str] = None
#     evaluated_time: Optional[str] = None
#     evaluation_error: Optional[str] = None

# class StudentEvaluationsResponse(BaseModel):
#     student_roll_no: str
#     evaluations: List[EvaluationResultDetail]
#     total_evaluated: int

# class ExamResultSummary(BaseModel):
#     submission_id: str
#     subject: str
#     total_marks_obtained: float
#     total_max_marks: float
#     overall_percentage: float
#     overall_grade: str
#     overall_status: str
#     details: List[EvaluationResultDetail] # Include per-question details

# # --- Dependencies ---
# async def get_neo4j_manager():
#     """Dependency function to get the initialized Neo4jManager instance."""
#     # Check startup error first
#     startup_error = app_state.get("startup_error")
#     if startup_error:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail=f"Service not available due to initialization failure: {startup_error}"
#         )

#     manager = app_state.get("neo4j_manager")
#     if manager is None:
#         logger.error("Neo4jManager instance not found in app state after startup.")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Database connection service is not properly initialized. Contact admin."
#         )
#     return manager

# async def get_question_uploader() -> QuestionUploader:
#     """Dependency function to get the initialized QuestionUploader instance."""
#     # Check startup error first
#     startup_error = app_state.get("startup_error")
#     if startup_error:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail=f"Service not available due to initialization failure: {startup_error}"
#         )

#     uploader = app_state.get("question_uploader")
#     if uploader is None:
#         logger.error("QuestionUploader instance not found in app state after startup.")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Question Uploader service is not properly initialized. Contact admin."
#         )
#     return uploader

# async def get_exam_service() -> ExamService:
#     """Dependency function to get the initialized ExamService instance."""
#     startup_error = app_state.get("startup_error")
#     if startup_error:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail=f"Service unavailable due to startup error: {startup_error}"
#         )
#     service = app_state.get("exam_service")
#     if service is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Exam Service not initialized."
#         )
#     return service

# # Helper function to check Neo4j connection
# async def check_neo4j(manager=Depends(get_neo4j_manager)):
#     """Check if Neo4j is connected before allowing requests."""
#     try:
#         driver = manager.get_connection()
#         if driver is None:
#             raise HTTPException(status_code=500, detail="Database connection error. Try again later.")
#         return driver
#     except Exception as e:
#         logger.error(f"Neo4j connection error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Neo4j connection error: {str(e)}")

# # --- API Routes ---
# @app.get("/neo4j_status")
# async def neo4j_status(manager=Depends(get_neo4j_manager)):
#     """API to check if Neo4j is connected."""
#     try:
#         driver = manager.get_connection()
#         if driver is None:
#             return {"status": "error", "message": "Neo4j connection not available"}
#         return {"status": "success", "message": "Neo4j connection is active"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.post("/register_teacher")
# async def register_teacher(data: TeacherRegister, driver=Depends(check_neo4j)):
#     """Register a new teacher in the system."""
#     try:
#         # Check if teacher already exists
#         with driver.session() as session:
#             result = session.run(
#                 "MATCH (t:Teacher {name: $name}) RETURN t", 
#                 {"name": data.name}
#             )
#             if result.single():
#                 raise HTTPException(status_code=400, detail="Teacher already registered")

#         # Hash password
#         hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
        
#         # Create teacher node and relationship with department
#         query = """
#         MERGE (d:Department {name: $department})
#         CREATE (t:Teacher {
#             name: $name, 
#             password: $password,
#             department: $department,
#             created_at: timestamp()
#         })
#         CREATE (t)-[:TEACHES_IN]->(d)
#         RETURN t.name as name
#         """
        
#         with driver.session() as session:
#             result = session.run(query, {
#                 "name": data.name,
#                 "password": hashed_password,
#                 "department": data.department
#             })
            
#             if not result.single():
#                 raise HTTPException(status_code=500, detail="Failed to create teacher record")
                
#         logger.info(f"Teacher registered successfully: {data.name}")
#         return {"status": "success", "message": "Teacher registered successfully!"}
    
#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error registering teacher: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error registering teacher: {str(e)}")

# @app.post("/login_teacher")
# async def login_teacher(data: TeacherLogin, driver=Depends(check_neo4j)):
#     """Login teacher by verifying credentials."""
#     try:
#         with driver.session() as session:
#             result = session.run(
#                 "MATCH (t:Teacher {name: $name}) RETURN t.password AS stored_password", 
#                 {"name": data.name}
#             )
#             record = result.single()
            
#             if not record:
#                 raise HTTPException(status_code=400, detail="Teacher not found")

#             stored_password = record["stored_password"]
#             if not bcrypt.checkpw(data.password.encode(), stored_password.encode()):
#                 raise HTTPException(status_code=401, detail="Invalid credentials")

#         logger.info(f"Teacher login successful: {data.name}")
#         return {
#             "status": "success", 
#             "message": "Login successful!",
#             "teacher_name": data.name
#         }
    
#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error during teacher login: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error during teacher login: {str(e)}")

# @app.post("/register_student")
# async def register_student(data: StudentRegister, driver=Depends(check_neo4j)):
#     """Register a new student in the system."""
#     try:
#         # Check if student already exists
#         with driver.session() as session:
#             result = session.run(
#                 "MATCH (s:Student {roll_no: $roll_no}) RETURN s", 
#                 {"roll_no": data.roll_no}
#             )
#             if result.single():
#                 raise HTTPException(status_code=400, detail="Roll number already registered")

#         # Hash password
#         hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
        
#         # Create student node
#         query = """
#         MERGE (d:Department {name: $department})
#         CREATE (s:Student {
#             name: $name, 
#             roll_no: $roll_no, 
#             password: $password, 
#             department: $department,
#             semester: $semester,
#             created_at: timestamp()
#         })
#         CREATE (s)-[:BELONGS_TO]->(d)
#         RETURN s.roll_no as roll_no
#         """
        
#         with driver.session() as session:
#             result = session.run(query, {
#                 "name": data.name,
#                 "roll_no": data.roll_no,
#                 "password": hashed_password,
#                 "department": data.department,
#                 "semester": data.semester
#             })
            
#             if not result.single():
#                 raise HTTPException(status_code=500, detail="Failed to create student record")
                
#         logger.info(f"Student registered successfully: {data.roll_no}")
#         return {"status": "success", "message": "Student registered successfully!"}
    
#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error registering student: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error registering student: {str(e)}")

# @app.post("/login_student")
# async def login_student(data: StudentLogin, driver=Depends(check_neo4j)):
#     """Login student by verifying credentials."""
#     try:
#         with driver.session() as session:
#             result = session.run(
#                 "MATCH (s:Student {roll_no: $roll_no}) RETURN s.password AS stored_password, s.name AS name", 
#                 {"roll_no": data.roll_no}
#             )
#             record = result.single()
            
#             if not record:
#                 raise HTTPException(status_code=400, detail="User not found")

#             stored_password = record["stored_password"]
#             if not bcrypt.checkpw(data.password.encode(), stored_password.encode()):
#                 raise HTTPException(status_code=401, detail="Invalid credentials")

#         logger.info(f"Student login successful: {data.roll_no}")
#         return {
#             "status": "success", 
#             "message": "Login successful!",
#             "student_name": record["name"]
#         }
    
#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error during login: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

# @app.get("/subjects")
# async def get_subjects(driver=Depends(check_neo4j)):
#     """Fetch available subjects"""
#     try:
#         with driver.session() as session:
#             result = session.run("MATCH (s:Subject) RETURN s.name AS name, s.department AS department")
#             subjects = [{"name": r["name"], "department": r["department"]} for r in result]
            
#         logger.info(f"Retrieved {len(subjects)} subjects")
#         return subjects
    
#     except Exception as e:
#         logger.error(f"Error fetching subjects: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error fetching subjects: {str(e)}")

# @app.get("/questions")
# async def get_questions(subject: str, limit: int = 50, driver=Depends(check_neo4j)):
#     """
#     Fetch exam questions for a given subject.
    
#     Parameters:
#     - subject: The subject name for which to fetch questions
#     - limit: Number of questions to fetch (default: 50)
#     """
#     if not subject:
#         raise HTTPException(status_code=400, detail="Subject parameter is required")

#     try:
#         # Initialize fetcher with the current driver
#         fetcher = QuestionFetcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
#         questions = fetcher.fetch_questions(subject, limit)
        
#         if not questions:
#             logger.warning(f"No questions found for subject: {subject}")
#             return {"questions": []}
            
#         logger.info(f"Retrieved {len(questions)} questions for subject: {subject}")
#         return {"questions": questions}
    
#     except Exception as e:
#         logger.error(f"Error fetching questions for {subject}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error fetching questions: {str(e)}")

# @app.post("/upload-answer")
# async def upload_answer(file: UploadFile = File(...)):
#     """
#     Upload an answer sheet (image/PDF), extract text, and return it.
#     """
#     try:
#         # Check file type
#         allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
#         if file.content_type not in allowed_types:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Unsupported file type. Supported types are: {', '.join(allowed_types)}"
#             )
            
#         content = await file.read()
#         if not content:
#             raise HTTPException(status_code=400, detail="Empty file uploaded")
            
#         # Initialize extractor
#         extractor = GeminiTextExtractor()
#         extracted_text = extractor.extract_text(content, file.content_type)

#         if not extracted_text:
#             raise HTTPException(status_code=400, detail="Text extraction failed")

#         logger.info(f"Successfully extracted text from uploaded file: {file.filename}")
#         return {
#             "status": "success", 
#             "extracted_text": extracted_text,
#             "file_name": file.filename
#         }

#     except HTTPException as e:
#         # Re-throw HTTP exceptions
#         raise e
#     except Exception as e:
#         logger.error(f"Error processing file {file.filename}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# @app.post(
#     "/upload_question",
#     tags=["Questions"],
#     summary="Upload or Update a Question",
#     description="Adds a new question or updates details for an existing Question ID. Requires Subject to exist. Prevents duplicate question text within the same subject.",
#     status_code=status.HTTP_200_OK,
#     response_description="Success message or error detail"
# )
# async def upload_question_endpoint(
#     request: QuestionUploadRequest,
#     uploader: QuestionUploader = Depends(get_question_uploader)
# ):
#     """
#     Handles POST requests to upload/update question data.
#     """
#     logger.info(f"Received request for /upload_question, QID: {request.question_id}")
#     try:
#         success, message = uploader.upload_question(
#             question_id=request.question_id,
#             subject_name=request.subject_name,
#             question_text=request.question_text,
#             correct_answer_text=request.correct_answer_text,
#             max_marks=request.max_marks,
#             concepts=request.concepts or []
#         )

#         if success:
#             logger.info(f"Success processing QID {request.question_id}. Message: {message}")
#             return {
#                 "status": "success",
#                 "message": message,
#                 "question_id": request.question_id
#             }
#         else:
#             logger.warning(f"Failed processing QID {request.question_id}. Reason: {message}")
#             # Determine appropriate HTTP status code based on error message
#             status_code = status.HTTP_400_BAD_REQUEST
#             if "duplicate check failed" in message.lower() or "already exists" in message.lower():
#                 status_code = status.HTTP_409_CONFLICT
#             elif "transaction failed" in message.lower() or "database" in message.lower():
#                  status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
#             elif "subject name cannot be empty" in message.lower() or "id cannot be empty" in message.lower():
#                  status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
#             raise HTTPException(status_code=status_code, detail=message)

#     except HTTPException as http_exc:
#         # Re-raise expected HTTP exceptions
#         raise http_exc
#     except Exception as e:
#         # Catch unexpected internal errors during the process
#         logger.exception(f"Unexpected error during /upload_question processing for QID '{request.question_id}'")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An unexpected internal server error occurred."
#         )

# # IMPORTANT: Moving the submit_exam endpoint to the main app instead of router
# @app.post("/submit_exam", tags=["Exams"])
# async def submit_exam_endpoint(
#     submission: ExamSubmission,
#     service: ExamService = Depends(get_exam_service)
# ):
#     """
#     Submits student's collected answers and triggers automatic evaluation.
#     """
#     try:
#         # The ExamService now handles submission AND triggering evaluation
#         result_dict = service.submit_exam(submission)
#         # Assuming result_dict is {'status': 'success', 'message': '...', 'submission_id': '...'}
#         return JSONResponse(status_code=status.HTTP_201_CREATED, content=result_dict)
#     except HTTPException as http_exc:
#         # Log FastAPI known errors specifically maybe?
#         logger.warning(f"HTTPException during exam submission: {http_exc.status_code} - {http_exc.detail}")
#         raise http_exc
#     except Exception as e:
#         # Log unexpected errors from the service layer
#         logger.exception(f"Unexpected error in /submit_exam for student {submission.roll_no}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during exam submission.")
# # == Teacher Route ==
# @router.get(
#     "/teacher/student_results/{roll_no}", # Clearer path for teacher access
#     tags=["Teacher Portal", "Results"],
#     summary="Fetch All Stored Evaluations for a Student",
#     response_model=StudentEvaluationsResponse
# )
# async def teacher_get_student_results(
#     roll_no: str = Path(..., description="Roll number of the student to fetch results for"),
#     subject: Optional[str] = None, # Optional query parameter to filter by subject
#     processor: EvaluationProcessor = Depends(get_evaluation_processor)
# ):
#     """
#     (For Teachers) Retrieves all stored evaluation results for a given student,
#     optionally filtered by subject.
#     """
#     logger.info(f"Teacher request: Fetching results for student '{roll_no}', subject '{subject or 'ALL'}'")
#     try:
#         results = processor.get_student_evaluations(roll_no=roll_no, subject=subject)
#         # get_student_evaluations should return empty list if student not found or no results
#         if not results and isinstance(results, list):
#              # Optional: check if student exists for better error message
#              manager = await get_neo4j_manager() # Need manager to check existence
#              student_exists = manager.execute_read("MATCH (s:Student {roll_no:$r}) RETURN count(s)>0 as exists",{'r':roll_no})
#              if not student_exists or not student_exists[0]['exists']:
#                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Student '{roll_no}' not found.")

#         # Return validated response using Pydantic model
#         return StudentEvaluationsResponse(
#             student_roll_no=roll_no,
#             evaluations=results, # Already processed list of dicts/models
#             total_evaluated=len(results)
#         )
#     except HTTPException as e: raise e # Re-raise FastAPI errors
#     except Exception as e:
#         logger.exception(f"Error fetching evaluations for teacher view, student {roll_no}")
#         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error fetching student results.")


# # == Student Route ==


# @router.get(
#     "/student/exam_result/{submission_id}",
#     # ...(other decorator arguments)...
#     response_model=ExamResultSummary
# )
# async def student_get_exam_result(
#     submission_id: str = Path(..., description="The unique ID of the exam submission"),
#     manager: Neo4jManager = Depends(get_neo4j_manager) # Assuming dependency function exists
# ):
#     """ Retrieves the evaluated results for a completed exam submission. """
#     # ...(query and params definition remain the same)...
#     query = """ MATCH (s:Student)-[:SUBMITTED]->(e:ExamSubmission {id: $p_submission_id}) MATCH (e)-[:HAS_ANSWER]->(a:Answer)-[:ANSWERS_QUESTION]->(q:Question) MATCH (e)-[:FOR_SUBJECT]->(sub:Subject) WHERE a.evaluated_at IS NOT NULL OR a.evaluation_status IS NOT NULL RETURN s.roll_no AS student_roll_no, s.name AS student_name, e.id AS submission_id, sub.name as subject_name, q.id as question_id, q.text as question_text, a.text as submitted_answer, COALESCE(q.max_marks, 5.0) as max_marks_possible, a.evaluation_numeric_score AS numeric_score, a.evaluation_score_str AS score_str, a.evaluation_marks_obtained AS marks_obtained, a.evaluation_percentage AS percentage, a.evaluation_letter_grade AS letter_grade, a.evaluation_status AS status, a.evaluation_feedback AS feedback, a.evaluated_at AS evaluated_time, 'N/A' AS evaluation_error ORDER BY q.id """
#     params = {"p_submission_id": submission_id}

#     try:
#         raw_eval_details: List[Dict] = manager.execute_read(query, params)
#         # ...(check if raw_eval_details is empty, raise 404 or 202)...
#         if not raw_eval_details:
#              exists=manager.execute_read("MATCH (e:ExamSubmission {id:$sid}) RETURN e", {"sid":submission_id})
#              if not exists: raise HTTPException(404, f"Submission '{submission_id}' not found.")
#              else: raise HTTPException(202, f"Eval pending for '{submission_id}'.")


#         total_obtained: float = 0.0
#         total_possible: float = 0.0
#         processed_details_list: List[EvaluationResultDetail] = []
#         subject_name = raw_eval_details[0].get("subject_name", "Unknown Subject")

#         # --- Process loop ---
#         for detail_row in raw_eval_details:
#              # ...(Data cleaning/validation for marks_raw, max_marks_raw remains the same)...
#              marks_obt_num = None; max_marks_num = None; valid_marks = False
#              marks_raw=detail_row.get("marks_obtained"); max_marks_raw=detail_row.get("max_marks_possible")
#              if isinstance(max_marks_raw, (int, float)) and max_marks_raw > 0: max_marks_num = float(max_marks_raw)
#              else: logger.warning(f"Q:{detail_row.get('qid','?')} invalid max_marks: {max_marks_raw}")
#              if max_marks_num is not None and isinstance(marks_raw, (int, float)): marks_obt_num = float(marks_raw); valid_marks=True
#              elif max_marks_num is not None: logger.warning(f"Q:{detail_row.get('qid','?')} invalid marks_obtained: {marks_raw}")
#              if valid_marks: total_obtained += marks_obt_num; total_possible += max_marks_num

#              # Prepare data for Pydantic, use validated marks or None
#              data_for_pydantic = {k: detail_row.get(k) for k in detail_row} # Start with all fetched data
#              data_for_pydantic['max_marks_possible'] = max_marks_num # Override with float or None
#              data_for_pydantic['marks_obtained'] = marks_obt_num     # Override with float or None
#              data_for_pydantic['evaluated_time'] = str(data_for_pydantic.get('evaluated_time')) if data_for_pydantic.get('evaluated_time') else None

#              # --- Validate and Append detail model ---
#              try:
#                  detail_model = EvaluationResultDetail(**data_for_pydantic)
#                  processed_details_list.append(detail_model)
#              except Exception as pydantic_error:
#                  logger.error(f"Pydantic validation fail Q:{detail_row.get('qid','?')}: {pydantic_error}", exc_info=False)
#                  # Continue processing other rows

#         # --- Calculate Overall Summary ---
#         overall_percentage = None
#         overall_grade_info = {"letter_grade": "N/A", "status": "N/A"} # Defaults

#         if total_possible > 0: # Check if any questions had valid max_marks
#             overall_percentage = round((total_obtained / total_possible) * 100, 2)
#             # --- CALL THE HELPER FUNCTION DEFINED ABOVE ---
#             overall_grade_info = calculate_grade_and_status(total_obtained, total_possible)
#         else:
#              logger.warning(f"Cannot calculate overall summary for {submission_id}: total_possible is 0.")


#         # --- Assemble and return final response ---
#         summary = ExamResultSummary(
#             submission_id=submission_id,
#             subject=subject_name,
#             total_marks_obtained=round(total_obtained, 2),
#             total_max_marks=round(total_possible, 2),
#             overall_percentage=overall_percentage,
#             overall_grade=overall_grade_info.get("letter_grade", "N/A"),
#             overall_status=overall_grade_info.get("status", "N/A"),
#             details=processed_details_list
#         )

#         logger.info(f"Successfully generated result summary for submission {submission_id}")
#         return summary

#     except HTTPException as e: raise e
#     except Exception as e:
#         logger.exception(f"Unexpected error generating summary for {submission_id}")
#         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error generating results.")


# @app.get("/")
# def serve_frontend():
#     """Serve frontend HTML page"""
#     try:
#         with open("templates/id.html", "r") as f:
#             content = f.read()
#             return HTMLResponse(content=content, status_code=200)
#     except FileNotFoundError:
#         return HTMLResponse(content="<html><body><h1>AI Answer Evaluation System API</h1><p>Frontend template not found.</p></body></html>", status_code=200)
#     except Exception as e:
#         logger.error(f"Error serving frontend: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error serving frontend")

# # Include router - though we've moved submit_exam to the main app, 
# # this is still here for any future router endpoints
# app.include_router(router)

# if __name__ == "__main__":
#     # Initialize database before starting the server
#     logger.info("Starting AI Answer Evaluation System API")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


### test code of api