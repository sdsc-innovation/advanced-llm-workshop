import os
import re
import base64
import requests

"""
class PDFExtractor:
    def __init__(self):
        pass

    def extract_text(self, pdf_file_path: str) -> str:
        return pymupdf4llm.to_markdown(pdf_file_path)

    def extract_images(self, pdf_file_path: str) -> list:
        pdf_document = fitz.open(pdf_file_path)

        all_images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                name = f"image_{page_num+1}_{img_index+1}.{image_ext}"

                all_images.append(
                    {
                        "name": name,
                        "base_image": base_image,
                        "image_bytes": image_bytes,
                        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                        "image_ext": image_ext,
                    }
                )

        return all_images
"""
class PDFExtractorAPI:
    def __init__(self):
        pass

    def extract_text_and_images(self, pdf_file_path: str) -> tuple[str, list]:
        BASE_URL = "https://vlm2vec-api.runai-innovation-clement.inference.compute.datascience.ch/"
        # Send the request with the file directly
        with open(pdf_file_path, "rb") as pdf_file:
            files = {"pdf_file": (pdf_file_path, pdf_file, "application/pdf")}
            response = requests.post(f"{BASE_URL}/v1/pdf_to_markdown", files=files)

        # Process the response
        result = response.json()
        markdown_text = result["markdown"]
        images = result["images"]

        images_new_format = []
        for image in images:
            image_dict = {}
            image_dict["name"] = image["filename"]
            image_dict["image_base64"] = image["data"]
            image_dict["image_ext"] = image["format"]
            images_new_format.append(image_dict)

        return result, markdown_text, images_new_format
