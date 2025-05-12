import sys
import re
import requests
import subprocess
import base64

from src.constants_and_data_classes import Chunk, DataType, API_BASE_URL


class PDFExtractor:
    def __init__(self):
        self.extractor = None

    def extract_text_and_images(
        self, pdf_file_path: str
    ) -> tuple[str, str, list[dict]]:
        if self.extractor is None:
            try:
                self.extractor = PDFExtractorAPI()
                return self.extractor.extract_text_and_images(pdf_file_path)
            except (requests.RequestException, Exception) as e:
                print(
                    "API IS NOT REACHABLE!\n"
                    "Using fallback extractor instead: performances might be worse and "
                    "notebook comments might not be matching the outputs!\n"
                    f"{e}"
                )
                self.extractor = PDFExtractorFallback()
                return self.extractor.extract_text_and_images(pdf_file_path)

        return self.extractor.extract_text_and_images(pdf_file_path)


class PDFExtractorAPI:
    def __init__(self):
        pass

    def extract_text_and_images(
        self, pdf_file_path: str
    ) -> tuple[str, str, list[dict]]:
        # Send the request with the file directly
        with open(pdf_file_path, "rb") as pdf_file:
            files = {"pdf_file": (pdf_file_path, pdf_file, "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/v1/pdf_to_markdown", files=files)

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


class PDFExtractorFallback:
    # Will be used only if the API is not reachable

    def __init__(self):
        try:
            import pymupdf4llm
            import fitz
        except ImportError:
            print("Installing libraries for fallback PDFExtractor...")
            for library in ["pymupdf", "pymupdf4llm"]:
                subprocess.check_call([sys.executable, "-m", "pip", "install", library])

        pass

    def extract_text_and_images(
        self, pdf_file_path: str
    ) -> tuple[str, list, list[dict]]:
        import pymupdf4llm
        import fitz

        markdown_text = pymupdf4llm.to_markdown(pdf_file_path)

        images_new_format = []

        pdf_document = fitz.open(pdf_file_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                name = f"image_{page_num+1}_{img_index+1}.{image_ext}"

                images_new_format.append(
                    {
                        "name": name,
                        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                        "image_ext": image_ext,
                    }
                )

        return markdown_text, markdown_text, images_new_format

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


class SimpleChunker:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.current_chunk_id = 0

    def _split_text(self, text: str) -> list[str]:
        """Splits text hierarchically: first by markdown headers, then paragraphs, then sentences."""
        # Split by headers first (always)
        chunks = re.split(r"(?=^#{1,4})", text, flags=re.MULTILINE)
        fine_grained_chunks = []

        for chunk in chunks:
            sub_chunks = chunk.split("\n\n")
            for sub_chunk in sub_chunks:
                sentences = re.split(r"(?<=\.) ", sub_chunk)
                fine_grained_chunks.extend(sentences)

        return [c.strip() for c in fine_grained_chunks if c.strip()]

    def chunk_text(self, text: str, metadata: dict) -> list[Chunk]:
        """Chunks the text based on the defined splitting strategy."""
        raw_chunks = self._split_text(text)
        chunks = []
        current_chunk = []
        current_size = 0
        document_chunk_id = 0

        for raw_chunk in raw_chunks:
            # Split only if the current chunk exceeds max size
            if current_size + len(raw_chunk) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            " ".join(current_chunk), metadata, document_chunk_id
                        )
                    )
                    document_chunk_id += 1
                # Start a new chunk, adding the current raw_chunk
                current_chunk = [raw_chunk]
                current_size = len(raw_chunk)
            else:
                # Otherwise, just add the raw_chunk to the current chunk
                current_chunk.append(raw_chunk)
                current_size += len(raw_chunk)

        # Add the remaining chunk if any
        if current_chunk:
            chunks.append(
                self._create_chunk(" ".join(current_chunk), metadata, document_chunk_id)
            )

        # Replace all "\n" per " " as often just a new line, not new sentence.
        for chunk in chunks:
            chunk.content = chunk.content.replace("\n", " ")

        return chunks

    def chunk_images(self, images: list, metadata: dict) -> list[Chunk]:
        chunks = []
        document_chunk_id = 0

        for image in images:
            chunks.append(
                self._create_chunk(
                    image["image_base64"],
                    metadata,
                    document_chunk_id,
                    data_type=DataType.IMAGE,
                )
            )
            document_chunk_id += 1

        return chunks

    def _create_chunk(
        self,
        content: str,
        metadata: dict,
        document_chunk_id: int,
        data_type: DataType = DataType.TEXT,
    ) -> Chunk:
        chunk_id = self.current_chunk_id
        self.current_chunk_id += 1
        metadata = metadata.copy()
        metadata["document_chunk_id"] = document_chunk_id
        return Chunk(
            chunk_id=chunk_id, content=content, metadata=metadata, data_type=data_type
        )
