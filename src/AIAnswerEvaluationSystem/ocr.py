import os
from AIAnswerEvaluationSystem.logger import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

class GeminiTextExtractor:
    """Class to extract text from images using Google Gemini API with logging."""

    def __init__(self):
        """Initialize Gemini API client and logger."""
        load_dotenv()  # Load API key from .env
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp-image-generation"  # Use Flash model for fast response
        self.logger = logging.getLogger(__name__)

    def extract_text(self, image_data, mime_type="image/jpeg"):
        """Extract text from an image (binary data) using the Gemini API."""
        if not image_data:
            return "Error: No image data provided."

        contents = [
            types.Content(
                parts=[
                    types.Part(text="Extract the text from the image"),  # ✅ Correct Usage
                    types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data)),  # ✅ Correct
                ]
            )
        ]

        config = types.GenerateContentConfig(
            response_modalities=["text"],  # Expecting only text output
            response_mime_type="text/plain",
        )

        try:
            self.logger.info("Sending request to Gemini API...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            extracted_text = response.candidates[0].content.parts[0].text
            self.logger.info(f"Extracted Text: {extracted_text}")
            return extracted_text
        except Exception as e:
            self.logger.error(f"Error during text extraction: {e}")
            return "Error: Failed to extract text."

