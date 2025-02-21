import fitz 
from collections import Counter
from google.cloud import vision
import re 
from ....logger import logging
import os

# class GeminiOCR:
#     """
#     A class to perform Optical Character Recognition (OCR) using Google's Gemini API.
#     Supports both images and PDFs with logging.
#     """

#     def __init__(self, api_key):
#         """
#         Initializes the Gemini OCR with the API key.

#         Args:
#             api_key (str): Your Gemini API key.
#         """
#         self.api_key = api_key
#         genai.configure(api_key=self.api_key)
#         self.model = genai.GenerativeModel("gemini-pro-vision")

#         logging.info("GeminiOCR initialized successfully.")

#     def extract_text_from_image(self, image_path):
#         """
#         Extracts text from an image using Gemini's vision model.

#         Args:
#             image_path (str): Path to the image file.

#         Returns:
#             str: Extracted text from the image.
#         """
#         try:
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(f"File {image_path} not found.")

#             with open(image_path, "rb") as img_file:
#                 image_data = img_file.read()

#             image = Image.open(io.BytesIO(image_data))

#             logging.info(f"Processing image: {image_path}")

#             # Use Gemini API for OCR
#             response = self.model.generate_content([image])

#             extracted_text = response.text if response.text else "No text detected."
#             logging.info(f"Extracted text from image: {image_path}")

#             return extracted_text

#         except Exception as e:
#             logging.error(f"Error extracting text from image {image_path}: {str(e)}")
#             return f"Error: {str(e)}"

#     def extract_text_from_pdf(self, pdf_path):
#         """
#         Extracts text from a PDF file by converting each page to an image and applying OCR.

#         Args:
#             pdf_path (str): Path to the PDF file.

#         Returns:
#             str: Extracted text from the entire PDF.
#         """
#         try:
#             if not os.path.exists(pdf_path):
#                 raise FileNotFoundError(f"File {pdf_path} not found.")

#             doc = fitz.open(pdf_path)
#             extracted_text = ""

#             logging.info(f"Processing PDF: {pdf_path}")

#             for page_num in range(len(doc)):
#                 # Convert PDF page to an image
#                 pix = doc[page_num].get_pixmap()
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#                 # Use Gemini API for OCR
#                 response = self.model.generate_content([img])
#                 text = response.text if response.text else "No text detected."

#                 extracted_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
#                 logging.info(f"Processed Page {page_num + 1} of {pdf_path}")

#             return extracted_text

#         except Exception as e:
#             logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
#             return f"Error: {str(e)}"


class OCRProcessor:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect_text_from_image(self, image_path):
        """
        Extracts text from an image using Google Cloud Vision API.
        """
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.client.document_text_detection(image=image)  # Dense text OCR

            texts = response.text_annotations
            extracted_text = texts[0].description if texts else ""

            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")

            logging.info(f"✅ OCR Successful for {image_path}")
            return extracted_text

        except Exception as e:
            logging.error(f"OCR Failed for {image_path}: {str(e)}")
            return ""

    def detect_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF by converting each page to an image and applying OCR.
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap()
                img_path = f"temp_page_{page_num + 1}.png"
                pix.save(img_path)

                text = self.detect_text_from_image(img_path)
                full_text += f"\n\nPage {page_num + 1}:\n{text}"

            logging.info(f"✅ OCR Successful for {pdf_path}")
            return full_text

        except Exception as e:
            logging.error(f"OCR Failed for {pdf_path}: {str(e)}")
            return ""
