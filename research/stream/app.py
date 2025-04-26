# --- app.py ---
import streamlit as st
import os
import logging
# Import classes from your db_uploader module
from AIAnswerEvaluationSystem.main import Neo4jConnection, QuestionUploader

# --- Configure Logging for Streamlit App ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - StreamlitApp - %(message)s')
log = logging.getLogger(__name__)

# --- Neo4j Connection Handling (keep @st.cache_resource function as before) ---
@st.cache_resource
def init_neo4j_connection():
    log.info("Attempting to initialize Neo4j connection (cached)...")
    try:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USER", "neo4j")
        password =  os.environ.get("NEO4J_PASSWORD", "password") 
        conn = Neo4jConnection(uri=uri, username=username, password=password)
        log.info("Neo4j connection successful (cached).")
        return conn
    except KeyError as e:
         st.error(f"Neo4j credential '{e}' not found in Streamlit secrets (.streamlit/secrets.toml)")
         log.error(f"Neo4j credential '{e}' not found in Streamlit secrets.")
         return None
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        log.exception("Failed to connect to Neo4j during initialization.")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="Question Uploader", layout="wide")
st.title("📚 Question Uploader for Neo4j Database")
st.markdown("Add or update questions. If the Question ID exists, its details will be updated. Existing subjects will be reused. Duplicate question *text* within the same subject is prevented.")

connection = init_neo4j_connection()

if connection:
    uploader = QuestionUploader(neo4j_connection=connection)

    with st.form(key="question_form", clear_on_submit=True):
        st.subheader("Enter Question Details")
        col1, col2 = st.columns(2)

        with col1:
            # Subject input - text input allows adding new or using existing implicitly
            subject_name = st.text_input("Subject Name*", key="subject", placeholder="e.g., Physics, Chemistry")
            question_id = st.text_input("Unique Question ID*", key="qid", placeholder="e.g., PHYSICS_NEWTON_Q1 (Update uses this ID)")
            max_marks = st.number_input("Maximum Marks*", min_value=0.1, value=5.0, step=0.5, key="marks", format="%.1f")

        with col2:
            question_text = st.text_area("Question Text*", height=150, key="q_text", placeholder="Enter the full question text...")
            correct_answer_text = st.text_area("Correct Answer Text*", height=150, key="ca_text", placeholder="Enter the model/correct answer...")

        concepts_str = st.text_area(
            "Related Concepts / Context (optional, comma-separated)",
            key="concepts", placeholder="e.g., Inertia, Classical Mechanics, Acids and Bases"
        )
        submitted = st.form_submit_button("⬆️ Upload / Update Question")

    if submitted:
        log.info(f"Form submitted. QID: {question_id}, Subject: {subject_name}")

        # Perform validation before calling the uploader
        if not subject_name.strip(): st.error("Subject Name cannot be empty.")
        elif not question_id.strip(): st.error("Question ID cannot be empty.")
        elif not question_text.strip(): st.error("Question Text cannot be empty.")
        elif not correct_answer_text.strip(): st.error("Correct Answer Text cannot be empty.")
        elif not max_marks or max_marks <= 0: st.error("Maximum Marks must be a positive number.")
        else:
            # Validation passed, proceed with upload
            concepts_list = [c.strip() for c in concepts_str.split(',') if c.strip()]
            with st.spinner(f"Processing question '{question_id}'..."):
                try:
                    # Call the uploader method, now returns (success, message)
                    success, message = uploader.upload_question(
                        question_id=question_id,
                        subject_name=subject_name,
                        question_text=question_text,
                        correct_answer_text=correct_answer_text,
                        max_marks=float(max_marks),
                        concepts=concepts_list
                    )
                    # Display feedback based on the result
                    if success:
                        st.success(message) # Display success message from uploader
                        log.info(message)
                    else:
                        st.error(message) # Display error/failure message from uploader
                        log.warning(f"Upload attempt for QID '{question_id}' returned failure: {message}")
                except Exception as e:
                    # Catch unexpected errors during the upload process itself
                    st.error(f"An unexpected error occurred: {e}")
                    log.exception(f"Unexpected error during upload_question call for QID '{question_id}'")
else:
    st.error("🔴 Could not establish connection to Neo4j. Check secrets and DB status.")
    log.critical("Streamlit app cannot function without Neo4j connection.")