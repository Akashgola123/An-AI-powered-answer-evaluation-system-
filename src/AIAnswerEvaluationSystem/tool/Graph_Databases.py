from neo4j import GraphDatabase

class Neo4jManager:
    DEFAULT_URI = "bolt://localhost:7687" 
    DEFAULT_USER = "neo4j"
    DEFAULT_PASSWORD = "password"

    def __init__(self, uri=None, user=None, password=None):
        """
        Initializes a connection to the Neo4j database with default credentials if none are provided.
        """
        self.uri = uri or self.DEFAULT_URI
        self.user = user or self.DEFAULT_USER
        self.password = password or self.DEFAULT_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        """Closes the database connection."""
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Executes a given query and returns results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
