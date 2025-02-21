from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase

app = Flask(__name__)

# Connect to Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

    def add_question(self, question_id, q_type, text, answer, max_score):
        query = """
        CREATE (q:Question {id: $question_id, type: $q_type, text_English: $text, 
                            answer_English: $answer, max_score: $max_score})
        """
        self.run_query(query, {
            "question_id": question_id,
            "q_type": q_type,
            "text": text,
            "answer": answer,
            "max_score": max_score
        })

    def add_student_answer(self, question_id, student_name, student_answer):
        query = """
        MATCH (q:Question {id: $question_id})
        CREATE (s:StudentAnswer {name: $student_name, answer: $student_answer})
        MERGE (s)-[:ANSWERED]->(q)
        """
        self.run_query(query, {
            "question_id": question_id,
            "student_name": student_name,
            "student_answer": student_answer
        })

    def fetch_questions(self):
        query = "MATCH (q:Question) RETURN q.id AS id, q.text_English AS text"
        return self.run_query(query)

db = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add_question", methods=["POST"])
def add_question():
    data = request.json
    db.add_question(data["question_id"], data["type"], data["text_English"], data["answer_English"], data["max_score"])
    return jsonify({"message": "Question added successfully"}), 201

@app.route("/student_answers")
def student_answers_page():
    questions = db.fetch_questions()
    return render_template("student_answers.html", questions=questions)

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    db.add_student_answer(data["question_id"], data["student_name"], data["student_answer"])
    return jsonify({"message": "Student answer submitted successfully"}), 201

if __name__ == "__main__":
    app.run(debug=True)
