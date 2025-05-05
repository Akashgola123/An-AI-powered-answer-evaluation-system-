# # # --- db_uploader.py ---
# # import logging
# # import os
# # from neo4j import GraphDatabase
# # from typing import List, Dict, Optional, Any
# # from datetime import datetime
# # from collections import defaultdict # Need this import

# # log = logging.getLogger(__name__)

# # class Neo4jConnection:
# #     # ... (keep __init__, close, execute_write_transaction, _run_write_query as before) ...
# #     def __init__(self, uri: str, username: str, password: str):
# #         try:
# #             neo4j_logger = logging.getLogger("neo4j")
# #             neo4j_logger.setLevel(logging.WARNING) # Reduce driver verbosity
# #             self._driver = GraphDatabase.driver(uri, auth=(username, password))
# #             self._driver.verify_connectivity()
# #             log.info(f"Neo4j connection initialized and verified for URI: {uri}")
# #         except Exception as e:
# #             log.exception(f"FATAL: Failed to initialize/verify Neo4j connection to {uri}")
# #             self._driver = None
# #             raise

# #     def close(self):
# #         if self._driver is not None:
# #             self._driver.close()
# #             log.info("Neo4j connection closed.")

# #     def execute_read_transaction(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
# #         """Executes a query within a managed read transaction and returns results."""
# #         if self._driver is None:
# #              log.error("Neo4j driver not initialized. Cannot execute query.")
# #              return []
# #         if parameters is None: parameters = {}
# #         try:
# #             # Ensure session is properly managed
# #             with self._driver.session(database=self._driver.database or "neo4j") as session: # Specify default db if needed
# #                 records = session.execute_read(self._run_query_and_fetch, query, parameters)
# #                 log.debug(f"Read transaction successful for query: {query[:100]}...")
# #                 return records # Returns list of dicts
# #         except Exception as e:
# #             log.error(f"Neo4j read transaction failed: {e}")
# #             log.error(f"Query: {query}")
# #             log.error(f"Parameters: {parameters}")
# #             return []

# #     def execute_write_transaction(self, query: str, parameters: Optional[Dict] = None) -> Optional[Any]:
# #         # ... (write transaction method) ...
# #         if self._driver is None:
# #              log.error("Neo4j driver not initialized. Cannot execute query.")
# #              return None
# #         if parameters is None: parameters = {}
# #         summary = None
# #         try:
# #             with self._driver.session(database=self._driver.database or "neo4j") as session: # Specify default db if needed
# #                 result = session.execute_write(self._run_write_query, query, parameters)
# #                 log.debug(f"Write transaction successful for query: {query[:100]}...")
# #                 return result # Return counters or other info if needed
# #         except Exception as e:
# #             log.error(f"Neo4j write transaction failed: {e}")
# #             log.error(f"Query: {query}")
# #             log.error(f"Parameters: {parameters}")
# #             return None


# #     @staticmethod
# #     def _run_query_and_fetch(tx, query, parameters) -> List[Dict]:
# #          """Helper function to run a query and return list of data dictionaries."""
# #          result = tx.run(query, parameters)
# #          # Make sure to fully consume and convert within the transaction lambda/function
# #          return [record.data() for record in result]

# #     @staticmethod
# #     def _run_write_query(tx, query, parameters) -> Any:
# #          """Helper function for execute_write, returning summary counters."""
# #          result = tx.run(query, parameters)
# #          try: return result.consume().counters
# #          except Exception:
# #              log.warning("Could not get transaction counters, returning basic success status.")
# #              return {"success": result.success()}


# #     def check_question_exists(self, subject_name: str, question_text: str, current_question_id: Optional[str] = None) -> bool:
# #         # ... (keep check_question_exists method as before) ...
# #         if not subject_name or not question_text: return False
# #         query = """
# #         MATCH (sub:Subject {name: $subject_name})<-[:HAS_SUBJECT]-(q:Question)
# #         WHERE q.text = $question_text
# #         """
# #         parameters = {"subject_name": subject_name, "question_text": question_text}
# #         if current_question_id:
# #              query += " AND q.id <> $current_question_id"
# #              parameters["current_question_id"] = current_question_id
# #         query += " RETURN count(q) > 0 AS question_exists"
# #         log.debug(f"Checking existence for Q text in Subject '{subject_name}' (excluding ID: {current_question_id})")
# #         results = self.execute_read_transaction(query, parameters)
# #         return results[0].get('question_exists', False) if results else False


# #     # --- Method to get all distinct subject names ---
# #     def get_all_subjects(self) -> List[str]:
# #         """Fetches a list of all distinct subject names."""
# #         query = "MATCH (s:Subject) RETURN s.name AS subject_name ORDER BY subject_name"
# #         results = self.execute_read_transaction(query)
# #         return [row['subject_name'] for row in results]

# #     # --- Keep fetch_subjects_with_questions method ---
# #     def fetch_subjects_with_questions(self, subject_name: Optional[str] = None) -> Dict[str, List[Dict]]:
# #         """Fetches subjects and their related questions and correct answers."""
# #         base_query = """
# #         MATCH (sub:Subject)<-[:HAS_SUBJECT]-(q:Question)
# #         // Use OPTIONAL MATCH for robustness if CorrectAnswer might be missing for some questions
# #         OPTIONAL MATCH (q)-[:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer)
# #         """
# #         filter_clause = ""
# #         parameters = {}
# #         if subject_name:
# #              filter_clause = "WHERE sub.name = $subject_name"
# #              parameters["subject_name"] = subject_name

# #         return_clause = """
# #         RETURN sub.name AS subject_name,
# #                q.id AS question_id,
# #                q.text AS question_text,
# #                COALESCE(q.max_marks, 'N/A') AS max_marks, // Handle potential missing marks
# #                COALESCE(ca.text, '[Correct Answer Missing]') AS correct_answer_text // Handle missing answer
# #         ORDER BY sub.name, q.id
# #         """
# #         query = f"{base_query} {filter_clause} {return_clause}"
# #         log.info(f"Fetching subjects and questions... (Filter: Subject='{subject_name or 'ALL'}')")
# #         query_results = self.execute_read_transaction(query, parameters)

# #         subjects_data = defaultdict(list)
# #         if not query_results:
# #              log.warning(f"No questions found matching the criteria (Subject='{subject_name or 'ALL'}').")
# #              return {}

# #         for row in query_results:
# #             subject = row['subject_name']
# #             question_details = {
# #                 "id": row['question_id'],
# #                 "text": row['question_text'],
# #                 "max_marks": row['max_marks'], # Value might be 'N/A' now
# #                 "correct_answer": row['correct_answer_text'] # Value might be placeholder
# #             }
# #             subjects_data[subject].append(question_details)

# #         log.info(f"Successfully fetched data for {len(subjects_data)} subjects.")
# #         return dict(subjects_data)

# # # --- QuestionUploader Class (Keep as before) ---
# # class QuestionUploader:
# #     # ... (keep __init__ and upload_question as before) ...
# #     def __init__(self, neo4j_connection: Neo4jConnection):
# #         if not isinstance(neo4j_connection, Neo4jConnection):
# #                  raise ValueError("Invalid Neo4jConnection object provided.")
# #         self.db = neo4j_connection
# #         log.info("QuestionUploader initialized.")

# #     def upload_question(self,
# #                         question_id: str,
# #                         subject_name: str,
# #                         question_text: str,
# #                         correct_answer_text: str,
# #                         max_marks: float,
# #                         concepts: Optional[List[str]] = None) -> tuple[bool, str]:
# #         # ... (keep robust upload_question logic here) ...
# #         # --- Initial Validation ---
# #         if not all([question_id, subject_name, question_text, correct_answer_text]):
# #             msg = "Missing required text parameters (id, subject, question, answer)."
# #             log.error(msg); return False, msg
# #         if not isinstance(max_marks, (int, float)) or max_marks <= 0:
# #              msg = f"Invalid max_marks: {max_marks}. Must be a positive number."; log.error(msg); return False, msg
# #         concepts = [c.strip() for c in concepts if c and c.strip()] if concepts else []
# #         subject_name = subject_name.strip(); question_id = question_id.strip(); question_text = question_text.strip()
# #         if not subject_name: msg = "Subject name cannot be empty."; log.error(msg); return False, msg
# #         if not question_id: msg = "Question ID cannot be empty."; log.error(msg); return False, msg
# #         if not question_text: msg = "Question text cannot be empty."; log.error(msg); return False, msg

# #         # --- Duplicate Question Text Check ---
# #         try:
# #             is_duplicate_text = self.db.check_question_exists(subject_name, question_text, question_id)
# #             if is_duplicate_text:
# #                 msg = f"Error: Question text already exists for subject '{subject_name}'."; log.warning(f"Duplicate check fail: {msg}"); return False, msg
# #         except Exception as check_err:
# #             msg = "Error during duplicate check."; log.exception(f"{msg}"); return False, msg

# #         # --- Upload Transaction ---
# #         query = """
# #             MERGE (q:Question {id: $question_id})
# #             ON CREATE SET q.text = $question_text, q.max_marks = $max_marks, q.created_at = timestamp(), q.last_modified = timestamp()
# #             ON MATCH SET q.text = $question_text, q.max_marks = $max_marks, q.last_modified = timestamp()
# #             MERGE (q)-[rel_ans:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer)
# #             ON CREATE SET ca.text = $correct_answer_text, ca.created_at = timestamp() ON MATCH SET ca.text = $correct_answer_text
# #             WITH q, ca OPTIONAL MATCH (q)-[old_rel_sub:HAS_SUBJECT]->(:Subject) DELETE old_rel_sub MERGE (sub:Subject {name: $subject_name}) MERGE (q)-[:HAS_SUBJECT]->(sub)
# #             WITH q OPTIONAL MATCH (q)-[old_rel_con:RELATED_TO]->(:Concept) DELETE old_rel_con
# #             WITH q UNWIND $concepts AS concept_name WITH q, concept_name WHERE concept_name IS NOT NULL AND concept_name <> '' MERGE (c:Concept {name: concept_name}) MERGE (q)-[:RELATED_TO]->(c)
# #             """
# #         parameters = {"question_id": question_id, "subject_name": subject_name, "question_text": question_text, "correct_answer_text": correct_answer_text, "max_marks": float(max_marks), "concepts": concepts}
# #         log.info(f"Attempting transaction for QID: {question_id}")
# #         result = self.db.execute_write_transaction(query, parameters)
# #         if result is not None:
# #              msg = f"Successfully processed QID: '{question_id}'!"; log.info(f"{msg} DB changes: {result}"); return True, msg
# #         else:
# #              msg = f"DB transaction failed for QID: '{question_id}'."; log.error(msg); return False, msg
        








# # --- db_uploader.py ---
# import logging
# import os
# from neo4j import GraphDatabase
# from typing import List, Dict, Optional, Any
# from datetime import datetime
# from collections import defaultdict # Need this import

# log = logging.getLogger(__name__)

# # --- (Neo4jConnection class including __init__, close, execute_read_transaction etc.) ---
# class Neo4jConnection:
#     # ... (Keep other methods as they are, including the fixed read/write transactions) ...
#     def __init__(self, uri: str, username: str, password: str):
#         # ... (existing init) ...
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
#         # ... (existing close) ...
#         if self._driver is not None: self._driver.close(); log.info("Neo4j connection closed.")

#     def execute_read_transaction(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
#         # ... (existing execute_read_transaction, ensure _run_query_and_fetch is called) ...
#         if self._driver is None: log.error("Neo4j driver not initialized."); return []
#         if parameters is None: parameters = {}
#         try:
#             with self._driver.session() as session:
#                 records = session.execute_read(self._run_query_and_fetch, query, parameters)
#                 log.debug(f"Read Tx OK for query: {query[:100]}...")
#                 return records
#         except Exception as e:
#             log.exception(f"Neo4j read transaction failed. Query: {query}, Params: {parameters}")
#             return []

#     @staticmethod
#     def _run_query_and_fetch(tx, query, parameters) -> List[Dict]:
#          # This is where data comes from the driver
#          result = tx.run(query, parameters)
#          data_list = [record.data() for record in result]
#          log.debug(f"[_run_query_and_fetch] Returning data: {data_list}") # <<< ADD DEBUG
#          return data_list

#     # --- (Include get_all_subjects, check_question_exists, etc. if needed) ---
#     def get_all_subjects(self) -> List[str]:
#         """Fetches a list of all distinct subject names."""
#         # This seems to work based on previous debugging, but keep implementation.
#         query = "MATCH (s:Subject) RETURN s.name AS subject_name ORDER BY subject_name"
#         results = self.execute_read_transaction(query)
#         return [row['subject_name'] for row in results] if results else []


#     def fetch_subjects_with_questions(self, subject_name: Optional[str] = None) -> Dict[str, List[Dict]]:
#         """Fetches subjects and their related questions and correct answers."""
#         # (Query building part remains the same)
#         base_query = """ MATCH (sub:Subject)<-[:HAS_SUBJECT]-(q:Question) OPTIONAL MATCH (q)-[:HAS_CORRECT_ANSWER]->(ca:CorrectAnswer) """
#         filter_clause = ""; parameters = {}
#         if subject_name: filter_clause = "WHERE sub.name = $subject_name"; parameters["subject_name"] = subject_name
#         return_clause = """ RETURN sub.name AS subject_name, q.id AS question_id, q.text AS question_text, COALESCE(q.max_marks, 'N/A') AS max_marks, COALESCE(ca.text, '[Correct Answer Missing]') AS correct_answer_text ORDER BY sub.name, q.id """
#         query = f"{base_query} {filter_clause} {return_clause}"

#         log.info(f"Fetching subjects and questions... (Filter: Subject='{subject_name or 'ALL'}')")
#         # This call triggers _run_query_and_fetch
#         query_results = self.execute_read_transaction(query, parameters) # List[Dict]

#         # --- *** INTENSIVE DEBUGGING HERE *** ---
#         log.info(f"Data received from DB query execution (len={len(query_results)}): {query_results}")

#         subjects_data = defaultdict(list)
#         if not query_results:
#              log.warning(f"Neo4j query returned no results for criteria (Subject='{subject_name or 'ALL'}').")
#              return {}

#         for idx, row in enumerate(query_results):
#             log.debug(f"Processing row {idx}: {row}") # Log each row received
#             try:
#                 # Extract using .get() for safety, though keys should exist based on Cypher aliases
#                 subject = row.get('subject_name')
#                 q_id = row.get('question_id')
#                 q_text = row.get('question_text')
#                 marks = row.get('max_marks') # Might be float or 'N/A'
#                 ans_text = row.get('correct_answer_text') # Might be string or placeholder

#                 if subject is None or q_id is None:
#                      log.warning(f"Skipping row {idx} due to missing subject_name or question_id: {row}")
#                      continue # Skip incomplete rows essential for grouping

#                 question_details = {
#                     "id": q_id, # Use extracted value
#                     "text": q_text, # Use extracted value
#                     "max_marks": marks, # Use extracted value
#                     "correct_answer": ans_text # Use extracted value
#                 }
#                 log.debug(f"  -> Appending to subject '{subject}': {question_details}")
#                 subjects_data[subject].append(question_details)

#             except Exception as e:
#                  # Catch any unexpected error during row processing
#                  log.exception(f"Error processing row {idx}: {row}")
#                  # Continue to next row if possible

#         log.info(f"Finished processing rows. Aggregated data for {len(subjects_data)} subjects.")
#         log.debug(f"Final aggregated data structure: {dict(subjects_data)}") # Log final dict
#         return dict(subjects_data)

