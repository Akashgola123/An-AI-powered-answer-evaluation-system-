


# from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
# from AIAnswerEvaluationSystem.logger import logging


# class QuestionFetcher:
# #     """Handles fetching random questions from Neo4j for a given subject."""

#     def __init__(self, uri, user, password):
#         """
#         Initializes the Neo4j connection.
        
#         :param uri: Neo4j database URI
#         :param user: Neo4j username
#         :param password: Neo4j password
#         """
#         self.neo4j_manager = Neo4jManager.get_instance(uri, user, password)

#     # def fetch_questions(self, subject: str, limit: int = 50):
#     #     """
#     #     Fetches random questions for a given subject.

#     #     :param subject: The subject name for which to fetch questions.
#     #     :param limit: Number of questions to fetch (default: 50).
#     #     :return: A list of dictionaries containing question ID, text, and marks.
#     #     """
#     #     query = """
#     #     MATCH (sub:Subject {name: $subject})-[:HAS_QUESTION]->(q:Question)
#     #     RETURN q.id AS id, q.text AS question, q.marks AS marks
#     #     ORDER BY rand()
#     #     LIMIT $limit
#     #     """

# #         try:
# #             with self.neo4j_manager.get_connection().session() as session:
# #                 logging.info(f"📚 Fetching questions for subject: {subject} (Limit: {limit})")

# #                 results = session.execute_read(
# #                     lambda tx: list(tx.run(query, {"subject": subject, "limit": limit}))
# #                 )

# #                 if not results:
# #                     logging.warning(f"⚠️ No questions found for subject: {subject}")
# #                     return []

# #                 questions = [{"id": r["id"], "question": r["question"], "marks": r["marks"]} for r in results]
# #                 logging.info(f"✅ Successfully fetched {len(questions)} questions for '{subject}'")
# #                 return questions

# #         except Exception as e:
# #             logging.error(f"❌ Database error while fetching questions for '{subject}': {str(e)}")
# #             return []


 
#     def fetch_questions(self, subject: str, limit: int = 50):
#         """
#         Fetches random questions for a given subject.
        
#         :param subject: The subject name for which to fetch questions.
#         :param limit: Number of questions to fetch (default: 50).
#         :return: A list of dictionaries containing question ID, text, and marks.
#         """
        # query = """
        # MATCH (sub:Subject {name: $subject})-[:CONTAINS]->(q:Question)
        # RETURN q.id AS id, q.text AS question, 
        #     CASE WHEN q.marks IS NOT NULL THEN q.marks ELSE 1 END AS marks
        # ORDER BY rand()
        # LIMIT $limit
        # """


        
        
#         try:
#             with self.neo4j_manager.get_connection().session() as session:
#                 logging.info(f"📚 Fetching questions for subject: {subject} (Limit: {limit})")
                
#                 results = session.execute_read(
#                     lambda tx: list(tx.run(query, {"subject": subject, "limit": limit}))
#                 )
                
#                 if not results:
#                     logging.warning(f"⚠️ No questions found for subject: {subject}")
#                     return []
                
#                 questions = [{"id": r["id"], "question": r["question"], "marks": r["marks"]} for r in results]
#                 logging.info(f"✅ Successfully fetched {len(questions)} questions for '{subject}'")
#                 return questions
        
#         except Exception as e:
#             logging.error(f"❌ Database error while fetching questions for '{subject}': {str(e)}")
#             return []


from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.logger import logging
from typing import List, Dict # Ensure List and Dict are imported

class QuestionFetcher:
    """Handles fetching questions from Neo4j for a given subject."""

    def __init__(self, uri, user, password):
        """Initializes the Neo4j connection."""
        # Assuming Neo4jManager handles singleton connection correctly
        self.neo4j_manager = Neo4jManager.get_instance(uri, user, password)

    def fetch_questions(self, subject: str, limit: int = 50):
        """
        Fetches questions (ID, text, marks) for a given subject.

        :param subject: The subject name for which to fetch questions.
        :param limit: Number of questions to fetch.
        :return: A list of dictionaries containing question ID, text, and marks. Returns empty list on error or if none found.
        """

        # --- QUERY WITHOUT CORRECT ANSWER ---
        # Select question details linked to the given subject
        query = """
        MATCH (sub:Subject {name: $subject})<-[:HAS_SUBJECT]-(q:Question)
        RETURN
            q.id AS question_id,
            q.text AS question_text,
            COALESCE(q.max_marks, 'N/A') AS max_marks // Use COALESCE for safety
        ORDER BY rand() // Re-added randomness
        LIMIT $limit
        """
        parameters = {
            "subject": subject,
            "limit": limit # Parameter for LIMIT clause
        }
        # ------------------------------------

        questions = [] # Initialize empty list
        try:
            # Assuming self.neo4j_manager.get_connection() returns a valid driver object
            with self.neo4j_manager.get_connection().session() as session:
                logging.info(f"📚 Fetching questions for subject: '{subject}' (Limit: {limit})...")

                # --- Define the transaction function ---
                def _execute_query(tx, cypher, params):
                    result = tx.run(cypher, params)
                    return [record.data() for record in result]

                # --- Execute the read transaction ---
                results = session.execute_read(_execute_query, query, parameters)

                if not results:
                    logging.warning(f"⚠️ No questions found for subject: '{subject}'")
                    return [] # Return empty list

                # --- Process results using correct dictionary keys ---
                for record in results:
                    questions.append({
                        "id": record.get("question_id"), # Use alias
                        "question": record.get("question_text"), # Use alias
                        "marks": record.get("max_marks") # Use alias
                    })

                logging.info(f"✅ Successfully fetched {len(questions)} questions for '{subject}'")
                return questions

        except AttributeError as ae:
             logging.error(f"❌ Attribute error (maybe Neo4jManager setup issue?): {str(ae)}")
             return []
        except Exception as e:
            # Catch specific driver/session errors if possible
            logging.error(f"❌ Database error while fetching questions for '{subject}': {str(e)}")
            logging.debug(f"Failed query: {query} with params: {parameters}")
            return [] # Return empty list on error