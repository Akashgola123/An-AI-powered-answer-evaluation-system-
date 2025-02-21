from flask import Flask
import os,sys
from logger import logging
from Exception import CustomException

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        raise Exception("This is a test exception")
    except Exception as e:
        ml = CustomException(e,sys)
        logging.info(ml.error_message)
        # logging.info(f"Error: {e}")
        # raise CustomException(e,sys)

        logging.info("Home page accessed")
        return "welcome to inexia Ai world"

if __name__ == "__main__":
    app.run(debug=True) #5000(localhost:5000)