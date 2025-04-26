# --- app.py ---
import streamlit as st
import os
import logging
import json
from AIAnswerEvaluationSystem.prompts import Neo4jConnection # Assuming correct import path

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - StreamlitApp - %(message)s')
log = logging.getLogger(__name__)

# --- Neo4j Connection Handling (@st.cache_resource) ---
@st.cache_resource
def init_neo4j_connection():
    """Initializes and caches the Neo4j connection using Streamlit secrets."""
    # ... (keep existing init_neo4j_connection function - VERIFIED AS WORKING) ...
    log.info("Attempting to initialize Neo4j connection (cached)...")
    try:
        uri = uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USER", "neo4j")
        password =  os.environ.get("NEO4J_PASSWORD", "password") 
        conn = Neo4jConnection(uri=uri, username=username, password=password)
        log.info("Neo4j connection successful (cached).")
        return conn
    except KeyError as e:
        st.error(f"Neo4j credential '{e}' missing from secrets.")
        log.error(f"Neo4j credential '{e}' missing.")
        return None
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        log.exception("Neo4j connection failed.")
        return None

# --- Function to get subject list (Cached Data) ---
@st.cache_data(ttl=600)
def get_subject_list() -> list[str]:
    """Fetches sorted list of subjects from DB, using the cached connection."""
    # This function works, as the dropdown is populated. No debug needed here usually.
    conn = init_neo4j_connection()
    if conn:
        try: subjects = conn.get_all_subjects(); return ["-- Select a Subject --"] + sorted(subjects)
        except Exception as e: log.error(f"Error fetching subjects: {e}"); return ["-- Select a Subject --"]
    else: return ["-- Select a Subject --"]


# --- Function to Fetch Questions (Cached Data per Subject - ADD MORE DEBUGGING) ---
@st.cache_data(ttl=300)
def get_questions_for_subject(subject_name_input: str) -> list[dict]:
    """Fetches question details for a specific subject using the cached connection."""
    conn = init_neo4j_connection()
    log.debug(f"get_questions_for_subject called with: '{subject_name_input}'") # Log input

    # Ensure we don't process the placeholder string
    if not subject_name_input or subject_name_input == "-- Select a Subject --":
        log.debug("Placeholder subject selected, returning empty list.")
        return []

    if conn:
        log.info(f"Fetching questions from DB for actual subject: '{subject_name_input}'")
        try:
            # This function in db_uploader returns {subject_name: [question_list]}
            subject_data_dict = conn.fetch_subjects_with_questions(subject_name=subject_name_input) # Call backend

            log.debug(f"Raw dictionary returned by fetch_subjects_with_questions: {subject_data_dict}") # *** IMPORTANT DEBUG ***

            # Extract the list for the specific subject key. Check for existence AND case sensitivity.
            # Explicitly check if the key exists before .get() to understand the issue better.
            if subject_name_input in subject_data_dict:
                 result_list = subject_data_dict[subject_name_input]
                 log.debug(f"Successfully extracted list for key '{subject_name_input}'. List length: {len(result_list)}")
                 return result_list
            else:
                 log.warning(f"Subject key '{subject_name_input}' NOT FOUND in the dictionary returned by fetch_subjects_with_questions. Available keys: {list(subject_data_dict.keys())}")
                 st.warning(f"Data structure error: Key '{subject_name_input}' not found in DB result.") # Show in UI too
                 return [] # Key not found, return empty

        except Exception as e:
            log.exception(f"Error during fetch_subjects_with_questions call for {subject_name_input}") # Log stack trace
            st.error(f"Database error while fetching questions for {subject_name_input}: {e}") # Show in UI
            return [] # Return empty on error
    else:
        log.warning("Cannot fetch questions, Neo4j connection unavailable.")
        st.error("Database connection lost or unavailable.") # Show in UI
        return []

# --- Streamlit App UI ---
st.set_page_config(page_title="Question Viewer", layout="centered")
st.title("🔎 View Questions by Subject")
st.markdown("Select a subject from the dropdown menu.")

# Attempt to initialize connection
connection = init_neo4j_connection()

if connection:
    subject_options = get_subject_list()

    if len(subject_options) <= 1:
        st.warning("No subjects found in the database or failed to fetch list.")
    else:
        selected_subject = st.selectbox(
            "Select Subject:",
            options=subject_options,
            key="view_subject_select",
            index=0
        )

        # Display questions ONLY if a valid subject (not the placeholder) is selected
        if selected_subject and selected_subject != "-- Select a Subject --":
            st.divider()
            st.subheader(f"Questions for: {selected_subject}")

            # Fetch the data using the (debugged) function
            # This call triggers the execution and logging within get_questions_for_subject
            questions_data = get_questions_for_subject(selected_subject)

            # Now, check if questions_data list is actually populated before displaying
            if isinstance(questions_data, list) and questions_data: # Check if it's a non-empty list
                st.success(f"Displaying {len(questions_data)} question(s)...") # Use success for positive feedback
                # Display each question
                for idx, q_data in enumerate(questions_data):
                    expander_title = f"**ID:** {q_data.get('id', f'Unknown ID {idx+1}')} | **Marks:** {q_data.get('max_marks', 'N/A')}"
                    with st.expander(expander_title):
                        st.markdown("##### Question Text")
                        st.text(q_data.get('text', '[No Text Provided]'))
                        st.markdown("---")
                        st.markdown("##### Correct Answer")
                        st.text(q_data.get('correct_answer', '[No Correct Answer Provided]'))
            elif isinstance(questions_data, list) and not questions_data:
                 # This case means the function ran but returned an empty list.
                 st.warning(f"No questions found associated with the subject: '{selected_subject}' in the database.")
                 log.warning(f"get_questions_for_subject returned empty list for '{selected_subject}'. Check DB content/relationships.")
            else:
                 # This case handles if get_questions_for_subject somehow didn't return a list
                 st.error("An unexpected error occurred while retrieving question data.")
                 log.error(f"get_questions_for_subject returned unexpected type for '{selected_subject}': {type(questions_data)}")

        elif selected_subject == "-- Select a Subject --":
            # Don't display anything, maybe a small prompt
            st.info("Please select a subject from the dropdown menu above.")
        # No 'else' needed here as the placeholder selection is handled


else:
    # Connection failed state
    st.error("🔴 Could not establish connection to Neo4j.")
    log.critical("Streamlit app cannot function without Neo4j connection.")

# --- Footer ---
st.divider()
st.caption("Question Viewer v1.0")