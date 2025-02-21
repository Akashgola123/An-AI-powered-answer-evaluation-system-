BASE_PROMPTS = {
    "Mathematics": {
        "English": """Evaluate this mathematics answer and provide detailed feedback.

Question Type: {question_type}
Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Expected Solution Steps:
{solution_steps}

Key Concepts to Check:
{key_points}

Please evaluate:
1. Numerical accuracy
2. Mathematical reasoning
3. Solution steps shown
4. Use of correct formulas/methods
5. Proper notation and units
6. Common misconceptions displayed
7. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """गणितीय उत्तर का मूल्यांकन करें और विस्तृत प्रतिक्रिया प्रदान करें।

प्रश्न प्रकार: {question_type}
प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

अपेक्षित समाधान चरण:
{solution_steps}

महत्वपूर्ण अवधारणाएँ:
{key_points}

कृपया मूल्यांकन करें:
1. संख्यात्मक सटीकता
2. गणितीय तर्क
3. समाधान के चरण
4. सही सूत्रों / विधियों का उपयोग
5. उचित संकेत और इकाइयाँ
6. आम गलतफहमियाँ
7. {max_score} में से ग्रेड

मूल्यांकन:
"""
    },
    "Physics": {
        "English": """Evaluate this physics answer and provide feedback.

Question Type: {question_type}
Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Expected Explanation:
{solution_steps}

Key Concepts to Check:
{key_points}

Please evaluate:
1. Understanding of physical laws
2. Correctness of calculations
3. Use of proper formulas
4. Conceptual clarity
5. Logical explanation
6. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """भौतिकी उत्तर का मूल्यांकन करें और प्रतिक्रिया दें।

प्रश्न प्रकार: {question_type}
प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

अपेक्षित व्याख्या:
{solution_steps}

महत्वपूर्ण अवधारणाएँ:
{key_points}

कृपया मूल्यांकन करें:
1. भौतिकी के नियमों की समझ
2. गणनाओं की शुद्धता
3. उचित सूत्रों का उपयोग
4. संकल्पना की स्पष्टता
5. तार्किक व्याख्या
6. {max_score} में से ग्रेड

मूल्यांकन:
"""
    },
    "Chemistry": {
        "English": """Evaluate this chemistry answer.

Question Type: {question_type}
Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Key Concepts to Check:
{key_points}

Please evaluate:
1. Correct chemical equations/reactions
2. Proper use of chemical symbols
3. Logical explanation of chemical processes
4. Application of concepts
5. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """रसायन विज्ञान उत्तर का मूल्यांकन करें।

प्रश्न प्रकार: {question_type}
प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

महत्वपूर्ण अवधारणाएँ:
{key_points}

कृपया मूल्यांकन करें:
1. सही रासायनिक समीकरण/प्रतिक्रियाएं
2. रासायनिक प्रतीकों का उचित उपयोग
3. रासायनिक प्रक्रियाओं की तार्किक व्याख्या
4. अवधारणाओं का अनुप्रयोग
5. {max_score} में से ग्रेड

मूल्यांकन:
"""
    },
    "Hindi": {
        "English": """Evaluate this Hindi language answer.

Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Please evaluate:
1. Grammar and sentence structure
2. Vocabulary usage
3. Expression and creativity
4. Content accuracy
5. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """हिंदी भाषा के उत्तर का मूल्यांकन करें।

प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

कृपया मूल्यांकन करें:
1. व्याकरण और वाक्य संरचना
2. शब्द भंडार का प्रयोग
3. अभिव्यक्ति और रचनात्मकता
4. विषय-वस्तु की सटीकता
5. {max_score} में से ग्रेड

मूल्यांकन:
"""
    },
    "Sanskrit": {
        "English": """Evaluate this Sanskrit language answer.

Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Please evaluate:
1. Grammar and sandhi rules
2. Vocabulary and usage
3. Translation accuracy
4. Understanding of Sanskrit literature
5. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """संस्कृत भाषा के उत्तर का मूल्यांकन करें।

प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

कृपया मूल्यांकन करें:
1. व्याकरण और संधि नियम
2. शब्द भंडार और प्रयोग
3. अनुवाद की सटीकता
4. संस्कृत साहित्य की समझ
5. {max_score} में से ग्रेड

मूल्यांकन:
"""
    },
    "Social Science": {
        "English": """Evaluate this social science answer.

Question: {question_text}

Student's Answer: {student_answer}

Correct Answer: {correct_answer}

Key Points to Check:
{key_points}

Please evaluate:
1. Historical accuracy / Political correctness
2. Logical structure of argument
3. Clarity and coherence
4. Factual correctness
5. Grade out of {max_score}

Evaluation:
""",
        "Hindi": """सामाजिक विज्ञान के उत्तर का मूल्यांकन करें।

प्रश्न: {question_text}

छात्र का उत्तर: {student_answer}

सही उत्तर: {correct_answer}

महत्वपूर्ण बिंदु:
{key_points}

कृपया मूल्यांकन करें:
1. ऐतिहासिक सटीकता / राजनीतिक सटीकता
2. तर्क की तार्किक संरचना
3. स्पष्टता और सुसंगतता
4. तथ्यात्मक सटीकता
5. {max_score} में से ग्रेड

मूल्यांकन:
"""
    }, "English Language": {
        "English": """Evaluate this English language answer.

    Question Type: {question_type}
    Question: {question_text}

    Student's Answer: {student_answer}

    Correct Answer: {correct_answer}

    Please evaluate:
    1. Grammar and syntax
    2. Vocabulary usage
    3. Sentence structure
    4. Spelling and punctuation
    5. Reading comprehension
    6. Writing clarity and style
    7. Grade out of {max_score}

    Evaluation:
    """,
            "Hindi": """अंग्रेजी भाषा के उत्तर का मूल्यांकन करें।

    प्रश्न प्रकार: {question_type}
    प्रश्न: {question_text}

    छात्र का उत्तर: {student_answer}

    सही उत्तर: {correct_answer}

    कृपया मूल्यांकन करें:
    1. व्याकरण और वाक्य विन्यास
    2. शब्द भंडार का प्रयोग
    3. वाक्य संरचना
    4. वर्तनी और विराम चिह्न
    5. पठन बोध
    6. लेखन की स्पष्टता और शैली
    7. {max_score} में से ग्रेड

    मूल्यांकन:
    """
        },

        "Programming": {
            "English": """Evaluate this programming answer.

    Question Type: {question_type}
    Question: {question_text}

    Student's Code:
    {student_answer}

    Expected Solution:
    {correct_answer}

    Language: {programming_language}

    Please evaluate:
    1. Code correctness
    2. Proper syntax and formatting
    3. Algorithm efficiency
    4. Best practices followed
    5. Error handling
    6. Code documentation/comments
    7. Variable/function naming
    8. Grade out of {max_score}

    Additional Requirements:
    {key_points}

    Evaluation:
    """,
            "Hindi": """प्रोग्रामिंग उत्तर का मूल्यांकन करें।

    प्रश्न प्रकार: {question_type}
    प्रश्न: {question_text}

    छात्र का कोड:
    {student_answer}

    अपेक्षित समाधान:
    {correct_answer}

    प्रोग्रामिंग भाषा: {programming_language}

    कृपया मूल्यांकन करें:
    1. कोड की सटीकता
    2. उचित सिंटैक्स और फॉर्मेटिंग
    3. एल्गोरिथ्म की कुशलता
    4. सर्वोत्तम प्रथाओं का पालन
    5. एरर हैंडलिंग
    6. कोड डॉक्यूमेंटेशन/टिप्पणियां
    7. वेरिएबल/फंक्शन नामकरण
    8. {max_score} में से ग्रेड

    अतिरिक्त आवश्यकताएं:
    {key_points}

    मूल्यांकन:
    """
        },

        "General Knowledge": {
            "English": """Evaluate this general knowledge answer.

    Question Type: {question_type}
    Question: {question_text}

    Student's Answer: {student_answer}

    Correct Answer: {correct_answer}

    Topic Area: {topic_area}

    Please evaluate:
    1. Factual accuracy
    2. Completeness of answer
    3. Understanding of the topic
    4. Current relevance (if applicable)
    5. Proper terminology use
    6. Grade out of {max_score}

    Key Points to Check:
    {key_points}

    Evaluation:
    """,
            "Hindi": """सामान्य ज्ञान के उत्तर का मूल्यांकन करें।

    प्रश्न प्रकार: {question_type}
    प्रश्न: {question_text}

    छात्र का उत्तर: {student_answer}

    सही उत्तर: {correct_answer}

    विषय क्षेत्र: {topic_area}

    कृपया मूल्यांकन करें:
    1. तथ्यात्मक सटीकता
    2. उत्तर की पूर्णता
    3. विषय की समझ
    4. वर्तमान प्रासंगिकता (यदि लागू हो)
    5. उचित शब्दावली का प्रयोग
    6. {max_score} में से ग्रेड

    महत्वपूर्ण बिंदु:
    {key_points}

    मूल्यांकन:
    """
    },
    "Database": {
        "English": """Evaluate this database answer.

    Question Type: {question_type}
    Question: {question_text}

    Student's Answer: {student_answer}

    Correct Answer: {correct_answer}

    Database Type: {database_type}

    Please evaluate:
    1. Query correctness and syntax
    2. Database design principles
    3. Normalization and optimization
    4. Index usage (if applicable)
    5. SQL best practices
    6. Performance considerations
    7. Security considerations
    8. Grade out of {max_score}

    Key Concepts to Check:
    {key_points}

    Expected Output/Result:
    {expected_output}

    Additional Requirements:
    {additional_requirements}

    Evaluation:
    """,
            "Hindi": """डेटाबेस उत्तर का मूल्यांकन करें।

    प्रश्न प्रकार: {question_type}
    प्रश्न: {question_text}

    छात्र का उत्तर: {student_answer}

    सही उत्तर: {correct_answer}

    डेटाबेस प्रकार: {database_type}

    कृपया मूल्यांकन करें:
    1. क्वेरी की सटीकता और सिंटैक्स
    2. डेटाबेस डिज़ाइन सिद्धांत
    3. नॉर्मलाइजेशन और ऑप्टिमाइजेशन
    4. इंडेक्स का उपयोग (यदि लागू हो)
    5. SQL की सर्वोत्तम प्रथाएं
    6. प्रदर्शन विचार
    7. सुरक्षा विचार
    8. {max_score} में से ग्रेड

    महत्वपूर्ण अवधारणाएं:
    {key_points}

    अपेक्षित आउटपुट/परिणाम:
    {expected_output}

    अतिरिक्त आवश्यकताएं:
    {additional_requirements}

    मूल्यांकन:
    """
        },

        "Database Theory": {
            "English": """Evaluate this database theory answer.

    Question Type: {question_type}
    Question: {question_text}

    Student's Answer: {student_answer}

    Correct Answer: {correct_answer}

    Topic Area: {topic_area}

    Please evaluate:
    1. Understanding of database concepts
    2. Explanation of theoretical principles
    3. Real-world application understanding
    4. Proper terminology usage
    5. Understanding of:
    - ACID properties
    - Transaction management
    - Concurrency control
    - Recovery mechanisms
    6. Grade out of {max_score}

    Key Concepts to Check:
    {key_points}

    Evaluation:
    """,
            "Hindi": """डेटाबेस सिद्धांत उत्तर का मूल्यांकन करें।

    प्रश्न प्रकार: {question_type}
    प्रश्न: {question_text}

    छात्र का उत्तर: {student_answer}

    सही उत्तर: {correct_answer}

    विषय क्षेत्र: {topic_area}

    कृपया मूल्यांकन करें:
    1. डेटाबेस अवधारणाओं की समझ
    2. सैद्धांतिक सिद्धांतों की व्याख्या
    3. वास्तविक दुनिया के अनुप्रयोग की समझ
    4. उचित शब्दावली का उपयोग
    5. निम्न की समझ:
    - ACID गुण
    - ट्रांजैक्शन प्रबंधन
    - समवर्तीता नियंत्रण
    - रिकवरी तंत्र
    6. {max_score} में से ग्रेड

    महत्वपूर्ण अवधारणाएं:
    {key_points}

    मूल्यांकन:
    """
    }

}