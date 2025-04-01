


from AIAnswerEvaluationSystem.neo4j_Manager import Neo4jManager
from AIAnswerEvaluationSystem.logger import logging


class QuestionFetcher:
#     """Handles fetching random questions from Neo4j for a given subject."""

    def __init__(self, uri, user, password):
        """
        Initializes the Neo4j connection.
        
        :param uri: Neo4j database URI
        :param user: Neo4j username
        :param password: Neo4j password
        """
        self.neo4j_manager = Neo4jManager.get_instance(uri, user, password)

    # def fetch_questions(self, subject: str, limit: int = 50):
    #     """
    #     Fetches random questions for a given subject.

    #     :param subject: The subject name for which to fetch questions.
    #     :param limit: Number of questions to fetch (default: 50).
    #     :return: A list of dictionaries containing question ID, text, and marks.
    #     """
    #     query = """
    #     MATCH (sub:Subject {name: $subject})-[:HAS_QUESTION]->(q:Question)
    #     RETURN q.id AS id, q.text AS question, q.marks AS marks
    #     ORDER BY rand()
    #     LIMIT $limit
    #     """

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


 
    def fetch_questions(self, subject: str, limit: int = 50):
        """
        Fetches random questions for a given subject.
        
        :param subject: The subject name for which to fetch questions.
        :param limit: Number of questions to fetch (default: 50).
        :return: A list of dictionaries containing question ID, text, and marks.
        """
        query = """
        MATCH (sub:Subject {name: $subject})-[:CONTAINS]->(q:Question)
        RETURN q.id AS id, q.text AS question, 
            CASE WHEN q.marks IS NOT NULL THEN q.marks ELSE 1 END AS marks
        ORDER BY rand()
        LIMIT $limit
        """
        
        try:
            with self.neo4j_manager.get_connection().session() as session:
                logging.info(f"📚 Fetching questions for subject: {subject} (Limit: {limit})")
                
                results = session.execute_read(
                    lambda tx: list(tx.run(query, {"subject": subject, "limit": limit}))
                )
                
                if not results:
                    logging.warning(f"⚠️ No questions found for subject: {subject}")
                    return []
                
                questions = [{"id": r["id"], "question": r["question"], "marks": r["marks"]} for r in results]
                logging.info(f"✅ Successfully fetched {len(questions)} questions for '{subject}'")
                return questions
        
        except Exception as e:
            logging.error(f"❌ Database error while fetching questions for '{subject}': {str(e)}")
            return []