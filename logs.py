from flask import Flask
from logger import logging

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    logging.info("Home page accessed")
    return "Welcome to AI Assisted Assessment System"

if __name__ == "__main__":
    app.run(debug=True)