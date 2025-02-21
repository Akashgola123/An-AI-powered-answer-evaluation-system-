import os
import json
from AIAnswerEvaluationSystem.llms import LLMManager
from AIAnswerEvaluationSystem.prompt import BASE_PROMPTS
from AIAnswerEvaluationSystem.tool.Graph_Databases import Neo4jManager
from AIAnswerEvaluationSystem.tool.ocrs import GeminiOCR

class agents:
    """
    This class is used to create the agents for the AIAnswerEvaluationSystem.

    Attributes:
        _llm(object): the llm model is used for the understanding the question and answer and give the response as marks with feedback.
        _tools(list): the tools are used for the agents to use the tools and get the response.
        _prompt(str): th initial prompt to query the llm.
        _histroy(list): A histroy of the marks of the students and data fetch of question and answer.
        _output_praser(callable): A function or method to parse the llm's response.
        _validated_tools(list): stores validated tools after perparation.
        _formatted_prompt(str): the formatted prompt to query the llm.
        _formatted_history(str): the formatted history to query the llm.
        _final_prompt(str):stores the prepared prompt with validated tools and formatted history.

    """


    def __init__(self, llm, tools, prompt, history, output_parser, gemini_api_key, neo4j_config, subject="General Knowledge", language="English"):
        self._llm = llm
        self._prompt = prompt
        self._history = history
        self._output_praser =  output_parser
        self._llm_manager = LLMManager()
        self._subject = subject
        self._language = language
        self._tools = {
            "ocr":GeminiOCR(gemini_api_key),
            "neo4j":Neo4jManager(**neo4j_config),
        }
        self._validated_tools = self._prepare_tools()
        self._formatted_history = self._prepare_history()
        self._final_prompt = self._prepare_final_prompt(self._validated_tools)

    
    def _prepare_tools(self,required_tools=None):
        "prepare and validate tools before the execution"
        required_tools = set(required_tools or self._tools.keys())
        return{
            tool:self._tools[tool]
            for tool in required_tools if tool in self._tools
        }
    
    def fetch_question_answer(self,question_id,language="English"):
        "fetching the question and answer from the neo4j database"
        neo4j_tool = self._tools["neo4j"]

        if not neo4j_tool:
            neo4jeroor= "Neo4j tool not found and not initialized"
            raise ValueError(neo4jeroor)
        

        query = f"""
        MATCH (q:Question) WHERE q.id = '{question_id}'
        RETURN q.text_{self.language} AS text, q.answer_{self.language} AS answer, 
               q.type AS type, q.solution_steps_{self.language} AS solution_steps, 
               q.key_points_{self.language} AS key_points, q.max_score AS max_score
        """
        result = neo4j_tool.run_query(query)
        
        if not result or not result[0].get("text"):
            print(f"⚠️ No {self.language} version found. Falling back to English...")
            query = f"""
            MATCH (q:Question) WHERE q.id = '{question_id}'
            RETURN q.text_English AS text, q.answer_English AS answer, 
                   q.type AS type, q.solution_steps_English AS solution_steps, 
                   q.key_points_English AS key_points, q.max_score AS max_score
            """
            result = neo4j_tool.run_query(query)
        
        # query = f"MATCH (q:Question) WHERE q.id = '{question_id}' RETURN q.text, q.answer, q.type, q.solution_steps, q.key_points, q.max_score"
        # result = neo4j_tool.run_query(query)

        if not result:
            return None
        
        question_text = result[0]["q.text"]
        answer = result[0]["q.answer"]
        question_type = result[0]["q.type"]
        solution_steps = result[0]["q.solution_steps"]
        key_points = result[0]["q.key_points"]
        max_score = result[0]["q.max_score"]
        

    def extract_student_answer(self,image_path=None, pdf_path=None):
        "extract the student answer from the image or pdf file"
        ocr_tool = self._tools["ocr"]

        if not ocr_tool:
            ocr_error = "OCR tool not found and not initialized"
            raise ValueError(ocr_error)
        
        extracted_text = ocr_tool.extract_text(image_path or pdf_path)
        return extracted_text.strip()
    
    def prepare_prompt(self,question_id=None,student_answer=None):
        "prepare the final prompt based on subject and language"

        if not question_id or not student_answer:
            return "Invalid question or answer input"
        
        question_data = self.fetch_question_answer(question_id)

        if not question_data:
            return "Question not found in the database"
        
        student_answer = self.extract_student_answer(student_answer)
        if not question_data:
            return "answer is not extracted from the image or pdf file"
        
        subject = question_data["type"]
        language = self._language

        prompt_template = BASE_PROMPTS.get(subject, {}).get(language, BASE_PROMPTS["General Knowledge"]["English"])

        self._prompt = prompt_template.format(
            question_type = question_data["type"],
            question_text = question_data["text"],
            student_answer = student_answer,
            correct_answer = question_data["answer"],
            solution_steps="\n".join(f"{i+1}. {step}" for i, step in enumerate(question_data["solution_steps"])),
            key_points="\n".join(f"- {point}" for point in question_data["key_points"]),
            max_score = question_data["max_score"],
            history = self._formatted_history
        )

        return self._prompt
    
    def evaluate_answer(self,question_id=None,student_answer=None,target_language="English"):
        "evaluate the answer and return the response"

        prompt_text = self.prepare_prompt(question_id,student_answer)
        if "Invalid question or answer input" in prompt_text:
            return "error: Unable to evaluate answer due to invalid input"
        
        evaluation_response = self._llm_manager.generate_response(prompt_text)
        if not evaluation_response:
            return "error: Unable to generate evaluation response"
        
        if target_language != "English":
            evaluation_response = self._translate_response(evaluation_response,target_language)
        
        return evaluation_response
    


    def save_evaluation(self, question_id, image_path, evaluation, target_language, output_dir="evaluations"):
        """
        Saves the evaluation result as a JSON file.
        """
       
        os.makedirs(output_dir, exist_ok=True)

        
        output_file = os.path.join(output_dir, f"evaluation_{question_id}.json")

        
        output_data = {
            "question_id": question_id,
            "student_answer_image": image_path,
            "evaluation": evaluation,
            "language": target_language
        }

        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, indent=4, ensure_ascii=False)

        return output_file

    
    
        
        
        

                
                










