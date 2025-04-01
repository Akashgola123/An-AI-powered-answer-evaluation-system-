# db/connection.py
from neo4j import GraphDatabase

class DatabaseConnection:
    """Handles Neo4j database connection and basic transaction execution."""
    
    def __init__(self, uri, username, password):
        """
        Initializes the Neo4j connection.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def execute_read_transaction(self, func, *args):
        """
        Execute a read transaction using the Neo4j driver.
        
        Args:
            func: Function to execute within transaction
            *args: Arguments to pass to the function
            
        Returns:
            Result of the transaction or None if error occurs
        """
        try:
            with self.driver.session() as session:
                return session.execute_read(func, *args)
        except Exception as e:
            print(f"Error executing read transaction: {e}")
            return None

    def execute_write_transaction(self, func, *args):
        """
        Execute a write transaction using the Neo4j driver.
        
        Args:
            func: Function to execute within transaction
            *args: Arguments to pass to the function
            
        Returns:
            Result of the transaction or None if error occurs
        """
        try:
            with self.driver.session() as session:
                return session.execute_write(func, *args)
        except Exception as e:
            print(f"Error executing write transaction: {e}")
            return None
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        def _clear_database(tx):
            query = "MATCH (n) DETACH DELETE n"
            tx.run(query)
            return True
            
        return self.execute_write_transaction(_clear_database)


# db/repositories/subject_repository.py
class SubjectRepository:
    """Repository for Subject-related database operations."""
    
    def __init__(self, db_connection):
        """
        Initialize with a database connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    def create_subject(self, subject_id, subject_name):
        """
        Creates a subject with the given ID and name.
        
        Args:
            subject_id: ID of the subject (e.g., 'MATH001')
            subject_name: Name of the subject (e.g., 'Mathematics')
            
        Returns:
            Subject ID
        """
        def _create_subject(tx, subject_id, subject_name):
            query = "CREATE (s:Subject {id: $subject_id, name: $subject_name}) RETURN s.id AS subject_id"
            result = tx.run(query, subject_id=subject_id, subject_name=subject_name)
            return result.single()["subject_id"]
            
        return self.db.execute_write_transaction(_create_subject, subject_id, subject_name)
    
    def get_all_subjects(self):
        """
        Gets all subjects.
        
        Returns:
            List of subject records
        """
        def _get_all_subjects(tx):
            query = "MATCH (s:Subject) RETURN s.id AS id, s.name AS name"
            result = tx.run(query)
            return [record for record in result]
            
        return self.db.execute_read_transaction(_get_all_subjects)


# db/repositories/question_repository.py
class QuestionRepository:
    """Repository for Question-related database operations."""
    
    def __init__(self, db_connection):
        """
        Initialize with a database connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    def create_question(self, question_id, qid, question_text, marks):
        """
        Creates a question with the given ID, qid, text, and marks.
        
        Args:
            question_id: Internal ID of the question (e.g., 'MATH001_Q1')
            qid: User-defined question ID (e.g., 'Q001')
            question_text: Text of the question
            marks: Number of marks for the question
            
        Returns:
            Question ID
        """
        def _create_question(tx, question_id, qid, question_text, marks):
            query = "CREATE (q:Question {id: $question_id, qid: $qid, text: $question_text, marks: $marks}) RETURN q.id AS question_id"
            result = tx.run(query, question_id=question_id, qid=qid, question_text=question_text, marks=marks)
            return result.single()["question_id"]
            
        return self.db.execute_write_transaction(_create_question, question_id, qid, question_text, marks)
    
    def get_all_questions(self):
        """
        Gets all questions.
        
        Returns:
            List of question records
        """
        def _get_all_questions(tx):
            query = "MATCH (q:Question) RETURN q.id AS id, q.qid AS qid, q.text AS text, q.marks AS marks"
            result = tx.run(query)
            return [record for record in result]
            
        return self.db.execute_read_transaction(_get_all_questions)
    
    def get_questions_with_answers(self):
        """
        Gets all questions with their answers.
        
        Returns:
            List of question-answer records
        """
        def _get_questions_with_answers(tx):
            query = """
            MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer) 
            RETURN q.qid AS QID, q.text AS Question, a.text AS Answer, q.marks AS Marks
            """
            result = tx.run(query)
            return [record for record in result]
            
        return self.db.execute_read_transaction(_get_questions_with_answers)
    
    def get_questions_with_context(self):
        """
        Gets all questions with their context.
        
        Returns:
            List of question-context records
        """
        def _get_questions_with_context(tx):
            query = """
            MATCH (q:Question)-[:HAS_CONTEXT]->(c:Context) 
            RETURN q.qid AS QID, q.text AS Question, c.text AS Context, q.marks AS Marks
            """
            result = tx.run(query)
            return [record for record in result]
            
        return self.db.execute_read_transaction(_get_questions_with_context)
    
    def get_questions_by_subject(self, subject_id):
        """
        Gets all questions for a subject.
        
        Args:
            subject_id: ID of the subject
            
        Returns:
            List of question records with associated answer and context
        """
        def _get_questions_by_subject(tx, subject_id):
            query = """
            MATCH (s:Subject {id: $subject_id})-[:HAS_QUESTION]->(q:Question)
            OPTIONAL MATCH (q)-[:HAS_ANSWER]->(a:Answer)
            OPTIONAL MATCH (q)-[:HAS_CONTEXT]->(c:Context)
            RETURN q.id AS id, q.qid AS qid, q.text AS question, q.marks AS marks,
                a.text AS answer, c.text AS context
            """
            result = tx.run(query, subject_id=subject_id)
            return [record for record in result]
            
        return self.db.execute_read_transaction(_get_questions_by_subject, subject_id)
    
    def get_question_by_qid(self, qid):
        """
        Gets a question by its qid.
        
        Args:
            qid: The question ID to search for
            
        Returns:
            Question details or None if not found
        """
        def _get_question_by_qid(tx, qid):
            query = """
            MATCH (q:Question {qid: $qid})
            OPTIONAL MATCH (q)-[:HAS_ANSWER]->(a:Answer)
            OPTIONAL MATCH (q)-[:HAS_CONTEXT]->(c:Context)
            OPTIONAL MATCH (s:Subject)-[:HAS_QUESTION]->(q)
            RETURN q.id AS id, q.qid AS qid, q.text AS question, q.marks AS marks,
                a.text AS answer, c.text AS context, s.id AS subject_id, s.name AS subject_name
            """
            result = tx.run(query, qid=qid)
            records = [record for record in result]
            return records[0] if records else None
            
        return self.db.execute_read_transaction(_get_question_by_qid, qid)
    
    def link_subject_to_question(self, subject_id, question_id):
        """
        Links a subject to a question.
        
        Args:
            subject_id: ID of the subject
            question_id: ID of the question
            
        Returns:
            Question ID
        """
        def _link_subject_to_question(tx, subject_id, question_id):
            query = """
            MATCH (s:Subject {id: $subject_id})
            MATCH (q:Question {id: $question_id})
            CREATE (s)-[:HAS_QUESTION]->(q)
            RETURN q.id AS question_id
            """
            result = tx.run(query, subject_id=subject_id, question_id=question_id)
            return result.single()["question_id"]
            
        return self.db.execute_write_transaction(_link_subject_to_question, subject_id, question_id)


# db/repositories/answer_repository.py
class AnswerRepository:
    """Repository for Answer-related database operations."""
    
    def __init__(self, db_connection):
        """
        Initialize with a database connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    def create_answer(self, answer_id, answer_text):
        """
        Creates an answer with the given ID and text.
        
        Args:
            answer_id: ID of the answer (e.g., 'MATH001_A1')
            answer_text: Text of the answer
            
        Returns:
            Answer ID
        """
        def _create_answer(tx, answer_id, answer_text):
            query = "CREATE (a:Answer {id: $answer_id, text: $answer_text}) RETURN a.id AS answer_id"
            result = tx.run(query, answer_id=answer_id, answer_text=answer_text)
            return result.single()["answer_id"]
            
        return self.db.execute_write_transaction(_create_answer, answer_id, answer_text)
    
    def link_question_to_answer(self, question_id, answer_id):
        """
        Links a question to an answer.
        
        Args:
            question_id: ID of the question
            answer_id: ID of the answer
            
        Returns:
            Answer ID
        """
        def _link_question_to_answer(tx, question_id, answer_id):
            query = """
            MATCH (q:Question {id: $question_id})
            MATCH (a:Answer {id: $answer_id})
            CREATE (q)-[:HAS_ANSWER]->(a)
            RETURN a.id AS answer_id
            """
            result = tx.run(query, question_id=question_id, answer_id=answer_id)
            return result.single()["answer_id"]
            
        return self.db.execute_write_transaction(_link_question_to_answer, question_id, answer_id)


# db/repositories/context_repository.py
class ContextRepository:
    """Repository for Context-related database operations."""
    
    def __init__(self, db_connection):
        """
        Initialize with a database connection.
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
    
    def create_context(self, context_id, context_text):
        """
        Creates a context with the given ID and text.
        
        Args:
            context_id: ID of the context (e.g., 'MATH001_C1')
            context_text: Text of the context
            
        Returns:
            Context ID
        """
        def _create_context(tx, context_id, context_text):
            query = "CREATE (c:Context {id: $context_id, text: $context_text}) RETURN c.id AS context_id"
            result = tx.run(query, context_id=context_id, context_text=context_text)
            return result.single()["context_id"]
            
        return self.db.execute_write_transaction(_create_context, context_id, context_text)
    
    def link_question_to_context(self, question_id, context_id):
        """
        Links a question to a context.
        
        Args:
            question_id: ID of the question
            context_id: ID of the context
            
        Returns:
            Context ID
        """
        def _link_question_to_context(tx, question_id, context_id):
            query = """
            MATCH (q:Question {id: $question_id})
            MATCH (c:Context {id: $context_id})
            CREATE (q)-[:HAS_CONTEXT]->(c)
            RETURN c.id AS context_id
            """
            result = tx.run(query, question_id=question_id, context_id=context_id)
            return result.single()["context_id"]
            
        return self.db.execute_write_transaction(_link_question_to_context, question_id, context_id)


# services/exam_service.py
import time
import uuid

class ExamService:
    """Service layer for exam operations, using the repositories."""
    
    def __init__(self, subject_repo, question_repo, answer_repo, context_repo):
        """
        Initialize with all needed repositories.
        
        Args:
            subject_repo: SubjectRepository instance
            question_repo: QuestionRepository instance
            answer_repo: AnswerRepository instance
            context_repo: ContextRepository instance
        """
        self.subject_repo = subject_repo
        self.question_repo = question_repo
        self.answer_repo = answer_repo
        self.context_repo = context_repo
    
    def create_new_subject(self, subject_id, subject_name):
        """
        Creates a new subject.
        
        Args:
            subject_id: ID of the subject (e.g., 'MATH002')
            subject_name: Name of the subject (e.g., 'Advanced Mathematics')
            
        Returns:
            Subject ID
        """
        return self.subject_repo.create_subject(subject_id, subject_name)
    
    def create_question_and_answer(self, subject_id, qid, question_text, marks, answer_text, context_text=None):
        """
        Creates a new question with its answer and context for a specific subject.
        
        Args:
            subject_id: ID of the subject (must already exist)
            qid: User-defined question ID (e.g., 'Q001')
            question_text: Text of the question
            marks: Number of marks for the question
            answer_text: Text of the answer
            context_text: Optional context for the question
            
        Returns:
            Dictionary with question, answer, and context IDs
        """
        try:
            # Generate unique IDs for internal use
            timestamp = int(time.time())
            random_str = str(uuid.uuid4())[:8]
            question_id = f"{subject_id}_Q{timestamp}_{random_str}"
            answer_id = f"{subject_id}_A{timestamp}_{random_str}"
            context_id = f"{subject_id}_C{timestamp}_{random_str}"
            
            # If no context provided, create a default one
            if context_text is None:
                context_text = "This question is part of the assessment for this subject."
            
            # Create the question, answer, and context
            self.question_repo.create_question(question_id, qid, question_text, marks)
            self.answer_repo.create_answer(answer_id, answer_text)
            self.context_repo.create_context(context_id, context_text)
            
            # Link them together
            self.question_repo.link_subject_to_question(subject_id, question_id)
            self.answer_repo.link_question_to_answer(question_id, answer_id)
            self.context_repo.link_question_to_context(question_id, context_id)
            
            return {
                "success": True,
                "question_id": question_id,
                "qid": qid,
                "answer_id": answer_id,
                "context_id": context_id
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def list_all_subjects(self):
        """
        List all subjects in the database.
        
        Returns:
            List of subject records
        """
        subjects = self.subject_repo.get_all_subjects()
        for subject in subjects:
            print(f"ID: {subject['id']} - Name: {subject['name']}")
        return subjects
    
    def list_questions_by_subject(self, subject_id):
        """
        List all questions for a specific subject.
        
        Args:
            subject_id: ID of the subject
            
        Returns:
            List of question records
        """
        questions = self.question_repo.get_questions_by_subject(subject_id)
        for i, q in enumerate(questions, 1):
            print(f"QID: {q['qid']} - Q{i}: {q['question']} ({q['marks']} marks)")
            print(f"A: {q['answer']}")
            print(f"Context: {q['context']}")
            print()
        return questions
    
    def get_question_by_qid(self, qid):
        """
        Get question details by its qid.
        
        Args:
            qid: User-defined question ID
            
        Returns:
            Question details or None if not found
        """
        result = self.question_repo.get_question_by_qid(qid)
        if result:
            print(f"QID: {result['qid']}")
            print(f"Question: {result['question']}")
            print(f"Marks: {result['marks']}")
            print(f"Answer: {result['answer']}")
            print(f"Context: {result['context']}")
            print(f"Subject: {result['subject_name']} ({result['subject_id']})")
        else:
            print(f"Question with QID '{qid}' not found.")
        return result


# services/data_initializer.py
class DataInitializer:
    """Service for initializing the database with sample data."""
    
    def __init__(self, connection, exam_service):
        """
        Initialize with database connection and exam service.
        
        Args:
            connection: DatabaseConnection instance
            exam_service: ExamService instance
        """
        self.connection = connection
        self.exam_service = exam_service
    
    def initialize_database(self):
        """Initialize database with sample data."""
        # Clear existing data
        self.connection.clear_database()
        
        # Create subjects
        subjects = [
            {"id": "MATH001", "name": "Mathematics"},
            {"id": "PHY001", "name": "Physics"},
            {"id": "CHEM001", "name": "Chemistry"},
            {"id": "AI001", "name": "Artificial Intelligence"},
            {"id": "HIN001", "name": "Hindi"}
        ]
        
        for subject in subjects:
            self.exam_service.create_new_subject(subject["id"], subject["name"])
        
        # Create Mathematics questions
        math_questions = [
            {
                "qid": "M001",
                "q_text": "Solve the quadratic equation: x² - 5x + 6 = 0", 
                "marks": 5,
                "a_text": "The roots are x = 2 and x = 3. To solve, factor the equation: (x - 2)(x - 3) = 0",
                "c_text": "This problem tests understanding of quadratic equations and factorization techniques."
            },
            {
                "qid": "M002",
                "q_text": "Calculate the derivative of f(x) = 3x² + 2x - 5", 
                "marks": 5,
                "a_text": "The derivative is f'(x) = 6x + 2 using the power rule of differentiation.",
                "c_text": "This question evaluates knowledge of calculus and differentiation rules."
            },
            {
                "qid": "M003",
                "q_text": "What is the Fibonacci sequence?", 
                "marks": 5,
                "a_text": "A sequence where each number is the sum of the two preceding numbers: 0, 1, 1, 2, 3, 5, 8...",
                "c_text": "Fibonacci numbers appear in nature, spirals, and algorithm design."
            }
        ]
        
        self._create_questions_batch("MATH001", math_questions)
        
        # Create Physics questions
        physics_questions = [
            {
                "qid": "P001",
                "q_text": "Explain Newton's First Law of Motion", 
                "marks": 5,
                "a_text": "An object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force.",
                "c_text": "This tests understanding of inertia and classical mechanics."
            },
            {
                "qid": "P002",
                "q_text": "Calculate the kinetic energy of a 5kg object moving at 10m/s", 
                "marks": 5,
                "a_text": "Kinetic Energy = 1/2 * 5 * 10² = 250 Joules",
                "c_text": "Application of kinetic energy formula in mechanics."
            }
        ]
        
        self._create_questions_batch("PHY001", physics_questions)
        
        # Create Chemistry questions
        chemistry_questions = [
            {
                "qid": "C001",
                "q_text": "Write the balanced equation for the reaction between sodium and water", 
                "marks": 5,
                "a_text": "2Na + 2H₂O → 2NaOH + H₂",
                "c_text": "This question tests understanding of chemical reactions and balancing equations."
            },
            {
                "qid": "C002",
                "q_text": "Explain the concept of pH and its significance in chemistry", 
                "marks": 5,
                "a_text": "pH is a measure of the hydrogen ion concentration in a solution. It ranges from 0 to 14, with 7 being neutral, below 7 acidic, and above 7 alkaline.",
                "c_text": "This question evaluates understanding of acid-base chemistry."
            }
        ]
        
        self._create_questions_batch("CHEM001", chemistry_questions)
        
        # Create AI questions
        ai_questions = [
            {
                "qid": "AI001",
                "q_text": "What is Artificial Intelligence?", 
                "marks": 5,
                "a_text": "AI is the simulation of human intelligence in machines, including learning, reasoning, and perception.",
                "c_text": "This question introduces the concept of AI, machine learning, and deep learning."
            },
            {
                "qid": "AI002",
                "q_text": "What is reinforcement learning?", 
                "marks": 5,
                "a_text": "Reinforcement learning is a type of ML where agents learn by interacting with their environment to maximize rewards.",
                "c_text": "This topic is key in robotics and AI-driven automation."
            }
        ]
        
        self._create_questions_batch("AI001", ai_questions)
        
        # Create Hindi questions
        hindi_questions = [
            {
                "qid": "H001",
                "q_text": "संस्कृत भाषा का महत्व क्या है?", 
                "marks": 5,
                "a_text": "संस्कृत भारत की प्राचीन भाषा है और सभी भाषाओं की जननी मानी जाती है।",
                "c_text": "संस्कृत साहित्य में वेद, उपनिषद् और महाकाव्य शामिल हैं।"
            },
            {
                "qid": "H002",
                "q_text": "मुंशी प्रेमचंद कौन थे?", 
                "marks": 5,
                "a_text": "मुंशी प्रेमचंद हिंदी और उर्दू के महान उपन्यासकार थे।",
                "c_text": "उनकी प्रसिद्ध कृतियाँ गोदान, गबन और कर्मभूमि हैं।"
            }
        ]
        
        self._create_questions_batch("HIN001", hindi_questions)
    
    def _create_questions_batch(self, subject_id, questions):
        """
        Helper method to create a batch of questions for a subject.
        
        Args:
            subject_id: ID of the subject
            questions: List of question dictionaries
        """
        for q in questions:
            self.exam_service.create_question_and_answer(
                subject_id=subject_id,
                qid=q["qid"],
                question_text=q["q_text"],
                marks=q["marks"],
                answer_text=q["a_text"],
                context_text=q["c_text"]
            )


# utils/diagnostic.py
class DatabaseDiagnostic:
    """Utility for verifying database content."""
    
    def __init__(self, subject_repo, question_repo):
        """
        Initialize with repositories.
        
        Args:
            subject_repo: SubjectRepository instance
            question_repo: QuestionRepository instance
        """
        self.subject_repo = subject_repo
        self.question_repo = question_repo
    
    def verify_database(self):
        """Verify the database content."""
        print("\n=== SUBJECTS ===")
        subjects = self.subject_repo.get_all_subjects()
        for subject in subjects:
            print(f"ID: {subject['id']} - Name: {subject['name']}")
        
        print("\n=== QUESTIONS ===")
        questions = self.question_repo.get_all_questions()
        for question in questions:
            print(f"ID: {question['id']} - QID: {question['qid']} - Text: {question['text']} - Marks: {question['marks']}")
        
        print("\n=== QUESTIONS WITH ANSWERS ===")
        qa_pairs = self.question_repo.get_questions_with_answers()
        for qa in qa_pairs:
            print(f"QID: {qa['QID']}")
            print(f"Q: {qa['Question']}")
            print(f"A: {qa['Answer']}")
            print(f"Marks: {qa['Marks']}")
            print()
        
        print("\n=== QUESTIONS WITH CONTEXT ===")
        qc_pairs = self.question_repo.get_questions_with_context()
        for qc in qc_pairs:
            print(f"QID: {qc['QID']}")
            print(f"Q: {qc['Question']}")
            print(f"Context: {qc['Context']}")
            print(f"Marks: {qc['Marks']}")
            print()


# config/config.py
class DatabaseConfig:
    """Configuration for database connection."""
    
    def __init__(self, uri="neo4j://localhost:7687", username="neo4j", password="password"):
        """
        Initialize with database connection parameters.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password

