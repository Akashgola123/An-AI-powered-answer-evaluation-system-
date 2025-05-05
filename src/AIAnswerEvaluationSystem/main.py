# # --- db_uploader.py ---
# import logging
# import os
# from neo4j import GraphDatabase
# from typing import List, Dict, Optional, Any
# from datetime import datetime

# log = logging.getLogger(__name__) # Use specific logger

# class Neo4jConnection:
#     """Handles connections and managed transactions to Neo4j database."""
#     # --- __init__, close methods remain the same ---
#     def __init__(self, uri: str, username: str, password: str):
#         try:
#             neo4j_logger = logging.getLogger("neo4j")
#             neo4j_logger.setLevel(logging.WARNING) # Reduce driver verbosity
#             self._driver = GraphDatabase.driver(uri, auth=(username, password))
#             self._driver.verify_connectivity()
#             log.info(f"Neo4j connection initialized and verified for URI: {uri}")
#         except Exception as e:
#             log.exception(f"FATAL: Failed to initialize/verify Neo4j connection to {uri}")
#             self._driver = None
#             raise

#     def close(self):
#         if self._driver is not None:
#             self._driver.close()
#             log.info("Neo4j connection closed.")

#     # --- Add execute_read_transaction if not already present ---
#     def execute_read_transaction(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
#         """Executes a query within a managed read transaction and returns results."""
#         if self._driver is None:
#              log.error("Neo4j driver not initialized. Cannot execute query.")
#              return []
#         if parameters is None: parameters = {}

#         try:
#             with self._driver.session() as session:
#                 records = session.execute_read(self._run_query_and_fetch, query, parameters)
#                 log.debug(f"Read transaction successful for query: {query[:100]}...")
#                 return records
#         except Exception as e:
#             log.error(f"Neo4j read transaction failed: {e}")
#             log.error(f"Query: {query}")
#             log.error(f"Parameters: {parameters}")
#             return []

#     def execute_write_transaction(self, query: str, parameters: Optional[Dict] = None) -> Optional[Any]:
#         # ... (keep execute_write_transaction method as is) ...
#         if self._driver is None:
#              log.error("Neo4j driver not initialized. Cannot execute query.")
#              return None
#         if parameters is None: parameters = {}
#         summary = None
#         try:
#             with self._driver.session() as session:
#                 result = session.execute_write(self._run_write_query, query, parameters)
#                 log.debug(f"Write transaction successful for query: {query[:100]}...")
#                 return result # Return counters or other info if needed
#         except Exception as e:
#             log.error(f"Neo4j write transaction failed: {e}")
#             log.error(f"Query: {query}")
#             log.error(f"Parameters: {parameters}")
#             return None


#     @staticmethod
#     def _run_query_and_fetch(tx, query, parameters) -> List[Dict]:
#          """Helper function to run a query and return list of data dictionaries."""
#          result = tx.run(query, parameters)
#          return [record.data() for record in result] # Fetch results

#     @staticmethod
#     def _run_write_query(tx, query, parameters) -> Any:
#          """Helper function for execute_write, returning summary counters."""
#          result = tx.run(query, parameters)
#          try:
#              return result.consume().counters
#          except Exception:
#              log.warning("Could not get transaction counters, returning basic success status.")
#              return {"success": result.success()}

#     # --- NEW Method to check for duplicate question text within a subject ---
#     def check_question_exists(self, subject_name: str, question_text: str, current_question_id: Optional[str] = None) -> bool:
#         """
#         Checks if a question with the exact text already exists for the given subject,
#         optionally excluding the provided question_id (for update checks).

#         :param subject_name: The name of the subject.
#         :param question_text: The exact text of the question to check.
#         :param current_question_id: If provided, exclude this ID from the check (allows updating text of existing question).
#         :return: True if a duplicate question text exists, False otherwise.
#         """
#         if not subject_name or not question_text:
#              return False # Cannot check without both

#         # Use MATCH and WHERE for efficient filtering
#         query = """
#         MATCH (sub:Subject {name: $subject_name})<-[:HAS_SUBJECT]-(q:Question)
#         WHERE q.text = $question_text
#         """
#         # Add clause to exclude the current question ID if provided (during updates)
#         parameters = {
#              "subject_name": subject_name,
#              "question_text": question_text
#         }
#         if current_question_id:
#              query += " AND q.id <> $current_question_id"
#              parameters["current_question_id"] = current_question_id

#         query += " RETURN count(q) > 0 AS question_exists" # Return boolean directly

#         log.debug(f"Checking existence for Q text in Subject '{subject_name}' (excluding ID: {current_question_id})")
#         results = self.execute_read_transaction(query, parameters)

#         if results: # Check if query returned anything
#              return results[0].get('question_exists', False) # Extract the boolean result
#         return False # Default to false if query failed or returned no result


# class QuestionUploader:
#     # --- __init__ method remains the same ---
#     def __init__(self, neo4j_connection: Neo4jConnection):
#         if not isinstance(neo4j_connection, Neo4jConnection):
#                  raise ValueError("Invalid Neo4jConnection object provided.")
#         self.db = neo4j_connection
#         log.info("QuestionUploader initialized.")

#     # --- Updated upload_question method ---
#     def upload_question(self,
#                         question_id: str,
#                         subject_name: str,
#                         question_text: str,
#                         correct_answer_text: str,
#                         max_marks: float,
#                         concepts: Optional[List[str]] = None) -> tuple[bool, str]: # Return success status and message
#         """
#         Uploads or updates a single question. Checks for duplicate question text within the subject
#         before proceeding.

#         :return: Tuple (bool: success status, str: user-friendly message)
#         """
#         # --- Initial Validation (same as before) ---
#         if not all([question_id, subject_name, question_text, correct_answer_text]):
#             msg = "Missing required text parameters (id, subject, question, answer)."
#             log.error(msg)
#             return False, msg
#         if not isinstance(max_marks, (int, float)) or max_marks <= 0:
#              msg = f"Invalid max_marks: {max_marks}. Must be a positive number."
#              log.error(msg)
#              return False, msg
#         concepts = [c.strip() for c in concepts if c and c.strip()] if concepts else []
#         subject_name = subject_name.strip()
#         question_id = question_id.strip()
#         question_text = question_text.strip() # Important to trim for accurate checks

#         if not subject_name: msg = "Subject name cannot be empty."; log.error(msg); return False, msg
#         if not question_id: msg = "Question ID cannot be empty."; log.error(msg); return False, msg
#         if not question_text: msg = "Question text cannot be empty."; log.error(msg); return False, msg

#         # --- *** NEW: Duplicate Question Text Check *** ---
#         # We exclude the current question_id during the check, allowing updates
#         # to the text of an existing question without triggering the duplicate check on itself.
#         try:
#             is_duplicate_text = self.db.check_question_exists(
#                 subject_name=subject_name,
#                 question_text=question_text,
#                 current_question_id=question_id # Allows updating the text of THIS question
#             )

#             if is_duplicate_text:
#                 msg = f"Error: Another question with the same text already exists for the subject '{subject_name}'."
#                 log.warning(f"Duplicate check failed for QID '{question_id}': {msg}")
#                 return False, msg

#         except Exception as check_err:
#             msg = "Error during duplicate question check."
#             log.exception(f"{msg} QID: {question_id}") # Log full exception
#             return False, msg


#         # --- Proceed with Upload/Update Transaction (Query remains the same robust one) ---
#         query = """
#         MERGE (q:Question {id: $question_id})
#         ON CREATE SET q.text = $question_text, q.max_marks = $max_marks, q.created_at = timestamp(), q.last_modified = timestamp()
#         ON MATCH SET q.text = $question_text, q.max_marks = $max_marks, q.last_modified = timestamp()

#         MERGE (q)-[rel_ans:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer)
#         ON CREATE SET ca.text = $correct_answer_text, ca.created_at = timestamp()
#         ON MATCH SET ca.text = $correct_answer_text

#         WITH q, ca
#         OPTIONAL MATCH (q)-[old_rel_sub:HAS_SUBJECT]->(:Subject) DELETE old_rel_sub
#         MERGE (sub:Subject {name: $subject_name})
#         MERGE (q)-[:HAS_SUBJECT]->(sub)

#         WITH q
#         OPTIONAL MATCH (q)-[old_rel_con:RELATED_TO]->(:Concept) DELETE old_rel_con
#         WITH q
#         UNWIND $concepts AS concept_name
#         WITH q, concept_name
#         WHERE concept_name IS NOT NULL AND concept_name <> ''
#             MERGE (c:Concept {name: concept_name})
#             MERGE (q)-[:RELATED_TO]->(c)
#         """
#         parameters = {
#             "question_id": question_id, "subject_name": subject_name, "question_text": question_text,
#             "correct_answer_text": correct_answer_text, "max_marks": float(max_marks), "concepts": concepts,
#         }

#         log.info(f"Attempting transaction to upload/update question ID: {question_id} / Subject: {subject_name}")
#         result = self.db.execute_write_transaction(query, parameters)

#         if result is not None:
#              msg = f"Successfully uploaded/updated question ID: '{question_id}'!"
#              log.info(f"{msg} DB changes: {result}")
#              return True, msg
#         else:
#              msg = f"Database transaction failed for question ID: '{question_id}'."
#              log.error(msg)
#              return False, msg