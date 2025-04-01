# import os
# from AIAnswerEvaluationSystem.logger import logging
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# load_dotenv()

# class Neo4jManager:
#     _instances = {}

#     def __init__(self, uri, user, password):
#         self.uri = uri
#         self.user = user
#         self.password = password
#         self._driver = None

#     @classmethod
#     def get_instance(cls, uri=None, user=None, password=None):
#         """Singleton instance with dynamic parameters."""
#         if uri is None:
#             uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
#         if user is None:
#             user = os.getenv("NEO4J_USER", "neo4j")
#         if password is None:
#             password = os.getenv("NEO4J_PASSWORD", "your_secure_password")

#         key = (uri, user)
#         if key not in cls._instances:
#             cls._instances[key] = cls(uri, user, password)
#         return cls._instances[key]

#     def get_connection(self):
#         """Lazy initialization of Neo4j connection with actual database check."""
#         if self._driver is None:
#             try:
#                 self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                
#                 # ✅ Test connection with a simple query
#                 with self._driver.session() as session:
#                     session.run("RETURN 1")
                
#                 logging.info("✅ Neo4j connection established successfully")

#             except Exception as e:
#                 logging.error(f"❌ Neo4j connection error: {e}")
#                 self._driver = None  # Ensure driver is not stored if connection fails
#                 raise RuntimeError("Neo4j connection failed. Check logs for details.")

#         return self._driver
    
#         @classmethod
#         def execute_query(cls, query, params=None):
#             with cls.get_connection().session() as session:
#                 try:
#                     result = session.run(query, params or {})
#                     return list(result)
#                 except Exception as e:
#                     logger.error(f"Query execution error: {e}")
#                     raise

#     def close(self):
#         """Closes Neo4j connection if open."""
#         if self._driver:
#             self._driver.close()
#             self._driver = None
#             logging.info("🔒 Neo4j connection closed")


import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class Neo4jManager:
    _instances = {}

    def __init__(self, uri, user, password):
        """Initialize Neo4j connection details."""
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    @classmethod
    def get_instance(cls, uri=None, user=None, password=None):
        """Singleton instance with dynamic parameters."""
        if uri is None:
            uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        if user is None:
            user = os.getenv("NEO4J_USER", "neo4j")
        if password is None:
            password = os.getenv("NEO4J_PASSWORD", "your_secure_password")

        key = (uri, user)
        if key not in cls._instances:
            cls._instances[key] = cls(uri, user, password)
        return cls._instances[key]

    def get_connection(self):
        """Lazy initialization of Neo4j connection with actual database check."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

                # ✅ Test connection with a simple query
                with self._driver.session() as session:
                    session.run("RETURN 1")
                
                logging.info("✅ Neo4j connection established successfully")

            except Exception as e:
                logging.error(f"❌ Neo4j connection error: {e}")
                self._driver = None  # Ensure driver is not stored if connection fails
                raise RuntimeError("Neo4j connection failed. Check logs for details.")

        return self._driver

    def execute_query(self, query, params=None):
        """Execute a Neo4j query with error handling."""
        try:
            with self.get_connection().session() as session:
                result = session.run(query, params or {})
                return list(result)
        except Exception as e:
            logging.error(f"❌ Query execution error: {e}")
            raise RuntimeError("Failed to execute query. Check logs for details.")

    def close(self):
        """Closes Neo4j connection if open."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logging.info("🔒 Neo4j connection closed")
