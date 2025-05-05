

import os
import logging
from neo4j import GraphDatabase, Driver, Session, Transaction, Result 
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError, ConstraintError 
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any 
import threading 
import time 
load_dotenv()


log = logging.getLogger(__name__) 
neo4j_driver_log = logging.getLogger("neo4j") 
neo4j_driver_log.setLevel(logging.WARNING) 
class Neo4jManager:
    """
    Manages Neo4j Driver instance (singleton) with lazy initialization,
    thread safety, and provides methods for reliable transaction execution.
    """
    _instances: Dict[tuple, 'Neo4jManager'] = {}
    _lock = threading.Lock() 

    def __init__(self, uri: str, user: str, password: str):
        """
        Private initializer. Use Neo4jManager.get_instance() to get an instance.
        Stores connection details but defers driver creation.
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Optional[Driver] = None 
        log.info(f"Neo4jManager instance configured for URI: {self._uri}")

    @classmethod
    def get_instance(cls, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> 'Neo4jManager':
        """
        Factory method for singleton instance (thread-safe).
        Fetches connection details from environment variables if not provided.
        Verifies password existence. Creates instance if needed.
        """
        uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD") 

        if not password:
            log.error("NEO4J_PASSWORD environment variable not set or empty.")
            raise ValueError("Neo4j password not provided or found in environment.")

        key = (uri, user)
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    log.info(f"Creating new Neo4jManager instance for {key}")
                    cls._instances[key] = cls(uri, user, password)
        return cls._instances[key]

    def get_connection(self) -> Driver:
        """
        Lazily initializes and returns the Neo4j Driver instance (thread-safe).
        Verifies connectivity on first successful creation.
        Raises ConnectionError if connection/verification fails.
        """

        if self._driver is not None:

            return self._driver

    
        with self._lock:
           
            if self._driver is None:
                log.info(f"Attempting Neo4j Driver connection to {self._uri}...")
                try:
                    driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
                    
                    driver.verify_connectivity()
                    self._driver = driver 
                    log.info("✅ Neo4j Driver connected and verified.")
                except AuthError as auth_err:
                     log.exception("❌ Neo4j authentication failed.")
                     raise ConnectionError(f"Neo4j authentication failed: {auth_err}") from auth_err
                except ServiceUnavailable as su_err:
                    log.exception(f"❌ Neo4j service unavailable: {self._uri}")
                    raise ConnectionError(f"Neo4j service unavailable: {su_err}") from su_err
                except Exception as e: # Catch other driver init errors
                    log.exception(f"❌ Failed to initialize Neo4j Driver: {e}")
                    raise ConnectionError(f"Neo4j Driver initialization failed: {e}") from e

        if self._driver is None:
            raise ConnectionError("Neo4j Driver failed to initialize.")
        return self._driver


    def close(self):
        """
        Closes the Neo4j Driver connection if it has been initialized.
        Should be called explicitly during application shutdown.
        """
        with self._lock: 
            if self._driver is not None:
                instance_key = (self._uri, self._user)
                log.info(f"Closing Neo4j Driver for {instance_key}...")
                try:
                    self._driver.close()
                    log.info("🔒 Neo4j Driver closed.")
                except Exception as e:
                    log.exception(f"Error during Neo4j Driver close: {e}")
                finally:
                    self._driver = None 
                    if instance_key in Neo4jManager._instances:
                        try:
                            del Neo4jManager._instances[instance_key]
                            log.info(f"Removed instance {instance_key} from registry.")
                        except KeyError:
                            
                             log.warning(f"Instance {instance_key} was already removed?")



    def execute_read(self, query: str, parameters: Optional[Dict] = None, database: Optional[str] = None) -> List[Dict]:
        """
        Executes a Cypher query within a managed **read** transaction. Recommended for reads.
        """
        records: List[Dict] = []
        try:
            driver = self.get_connection() 
            with driver.session(database=database) as session:
                records = session.execute_read(self._run_query_and_fetch, query, parameters or {})
            log.debug(f"Read tx ok: {query[:80]}...")
            return records
        except Neo4jError as db_err:
             log.error(f"Read tx failed (DB Error): {db_err} | Query: {query[:80]}...")
             return []
        except ConnectionError: raise 
        except Exception as e:
            log.exception(f"Read tx failed (Unexpected): Query: {query[:80]}...")
            return []

    def execute_write(self, query: str, parameters: Optional[Dict] = None, database: Optional[str] = None) -> Optional[Any]:
        """
        Executes a Cypher query within a managed **write** transaction. Recommended for writes.
        Returns summary counters object on success, None on failure.
        """
        summary_counters = None
        try:
            driver = self.get_connection() 
            with driver.session(database=database) as session:
                summary_counters = session.execute_write(self._run_write_query_get_counters, query, parameters or {})
            log.debug(f"Write tx ok: {query[:80]}... Summary: {summary_counters}")
            return summary_counters
        except ConstraintError as ce:
             log.error(f"Write tx failed (Constraint): {ce} | Query: {query[:80]}...")
             return None
        except Neo4jError as db_err:
             log.error(f"Write tx failed (DB Error): {db_err} | Query: {query[:80]}...")
             return None
        except ConnectionError: raise 
        except Exception as e:
            log.exception(f"Write tx failed (Unexpected): Query: {query[:80]}...")
            return None

    @staticmethod
    def _run_query_and_fetch(tx: Transaction, query: str, parameters: Dict) -> List[Dict]:
        """Static helper: Runs read query, returns list of data dicts."""
        log.debug(f"Tx execute: {query[:80]}...")
        result: Result = tx.run(query, parameters)
        data = [record.data() for record in result]
        log.debug(f"Tx query returned {len(data)} records.")
        return data

    @staticmethod
    def _run_write_query_get_counters(tx: Transaction, query: str, parameters: Dict) -> Optional[Any]:
        """Static helper: Runs write query, returns summary counters."""
        log.debug(f"Tx execute write: {query[:80]}...")
        result: Result = tx.run(query, parameters)
        try:
            summary = result.consume()
            log.debug(f"Tx write summary: {summary.counters}")
            return summary.counters
        except Exception as e:
             log.warning(f"Could not consume result summary: {e}. Query: {query[:80]}")
             return None

   
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Optional[List[Result]]:
        """
        DEPRECATED in favor of execute_read/execute_write.
        Executes a Neo4j query using basic session.run (auto-commit).
        """
        log.warning("DEPRECATION WARNING: execute_query is used. Consider execute_read/execute_write for managed transactions.")
        try:
            driver = self.get_connection() # Ensure connection is available
            with driver.session() as session:
                result = session.run(query, params or {})
                
                return list(result)
        except ConnectionError:
            # Allow connection error to propagate up
            raise
        except Exception as e:
            log.error(f"❌ Query execution error in basic execute_query: {e}")
            
            raise RuntimeError(f"Failed to execute basic query. Check logs for details. Error: {e}")




    def fetch_evaluation_results_for_student(self, roll_no: str, subject_filter: Optional[str] = None) -> List[Dict]:
        """
        Fetches evaluation data previously stored on Answer nodes for a specific student,
        optionally filtered by subject.

        :param roll_no: The student's roll number.
        :param subject_filter: Optional subject name to filter results.
        :return: List of dictionaries containing question info and evaluation properties.
        """
 
        query = """
        MATCH (s:Student {roll_no: $p_roll_no})-[:SUBMITTED]->(e:ExamSubmission)
              -[:HAS_ANSWER]->(a:Answer)-[:ANSWERS_QUESTION]->(q:Question)
        MATCH (e)-[:FOR_SUBJECT]->(sub:Subject)
        // Ensure evaluation has been done - check for a specific property like evaluated_at or evaluation_status
        WHERE a.evaluated_at IS NOT NULL OR a.evaluation_status IS NOT NULL
        """
        parameters = {"p_roll_no": roll_no}

    
        if subject_filter:
            query += " AND sub.name = $p_subject "
            parameters["p_subject"] = subject_filter

        query += """
        RETURN
            s.roll_no as student_roll_no,
            s.name as student_name,
            sub.name as subject_name,
            q.id as question_id,
            q.text as question_text,
            a.text as submitted_answer,
            COALESCE(q.max_marks, 5.0) as max_marks_possible, // Max marks from Question node
            a.evaluation_numeric_score AS numeric_score,    // Evaluation results from Answer node
            a.evaluation_score_str AS score_str,           // Evaluation results from Answer node
            a.evaluation_marks_obtained AS marks_obtained,   // Evaluation results from Answer node
            a.evaluation_percentage AS percentage,        // Evaluation results from Answer node
            a.evaluation_letter_grade AS letter_grade,       // Evaluation results from Answer node
            a.evaluation_status AS status,                // Evaluation results from Answer node
            a.evaluation_feedback AS feedback,              // Evaluation results from Answer node
            a.evaluation_error AS eval_error,             // Potential errors during eval
            a.evaluated_at as evaluated_time            // Timestamp of evaluation
        ORDER BY e.submitted_at DESC, q.id
        """
        log.info(f"Fetching EVALUATED answers for student '{roll_no}', subject '{subject_filter or 'ALL'}'")
        try:
            results = self.execute_read(query, parameters)
            log.info(f"Found {len(results)} evaluated answers for student {roll_no}.")
            return results
        except Exception as e:
            log.exception(f"Failed to fetch evaluation results for student {roll_no}.")
            return [] 