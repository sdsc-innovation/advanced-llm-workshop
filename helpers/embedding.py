import sys
import io
import base64
import requests
import subprocess

import numpy as np

import tiktoken
from openai import OpenAI

from PIL import Image

from .constants_and_data_classes import Chunk, DataType, API_BASE_URL, API_TOKEN


class OpenAITextEmbeddings:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.client = OpenAI()
        self.model_name = model_name

    def get_embedding(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = np.asarray(
            [response.data[i].embedding for i in range(len(response.data))]
        )

        return embeddings

class OpenAITextEmbeddingsAzure:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],
            api_version=os.environ["AZURE_API_VERSION"]
        )
        self.model_name = model_name
 
    def get_embedding(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
 
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = np.asarray(
            [response.data[i].embedding for i in range(len(response.data))]
        )
 
        return embeddings


def compute_openai_large_embedding_cost(chunks, verbose: bool = True) -> float:
    encoder = tiktoken.get_encoding(
        "cl100k_base"
    )  # The encoding used by text-embedding-3-large

    def count_tokens(text):
        return len(encoder.encode(text))

    total_tokens = sum(count_tokens(chunk.content) for chunk in chunks)

    # The current costs, potentially to be updated (but likely cheaper in the future)
    cost_per_1M_tokens = 0.13
    total_cost = (total_tokens / 1e6) * cost_per_1M_tokens

    if verbose:
        print(f"Total tokens: {total_tokens}")
        print(f"Estimated cost: ${total_cost:.4f}")

    return total_cost


class ImageEmbeddingsForText:

    def __init__(self):
        self.text_embedding = None

    def get_embedding(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        if self.text_embedding is None:
            try:
                self.text_embedding = VLM2VecTextEmbeddings()
                return self.text_embedding.get_embedding(chunks)
            except (requests.RequestException, Exception) as e:
                print(
                    "API IS NOT REACHABLE!\n"
                    "Using fallback text embedding for images embedding instead,"
                    "performances will be worse and "
                    "notebook comments might not be matching the outputs!\n"
                    f"{e}"
                )
                self.text_embedding = CLIPTextEmbeddings()
                return self.text_embedding.get_embedding(chunks)

        return self.text_embedding.get_embedding(chunks)


class ImageEmbeddings:

    def __init__(self):
        self.image_embedding = None

    def get_embedding(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        if self.image_embedding is None:
            try:
                self.image_embedding = VLM2VecImageEmbeddings()
                return self.image_embedding.get_embedding(chunks)
            except (requests.RequestException, Exception) as e:
                print(
                    "API IS NOT REACHABLE!\n"
                    "Using fallback image embedding instead, performances will be worse"
                    " and notebook comments might not be matching the outputs!\n"
                    f"{e}"
                )
                self.image_embedding = ClipImageEmbeddings()
                return self.image_embedding.get_embedding(chunks)

        return self.image_embedding.get_embedding(chunks)


class VLM2VecTextEmbeddings:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.token = API_TOKEN

    # Work only with 1 text (no batching)
    def get_embedding(self, text: str) -> np.ndarray:
        text_data = {"text": text}
        url = f"{self.base_url}/v1/text_embeddings"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.post(url, json=text_data, headers=headers)
        # List of list because notebook expects a list of embeddings
        return np.array([response.json()["embeddings"]])


class VLM2VecImageEmbeddings:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.token = API_TOKEN

    # Work only with 1 image in base64 (no batching)
    def get_embedding(self, imagebase64: str) -> np.ndarray:
        image_data = {"imagebase64": imagebase64}
        url = f"{self.base_url}/v1/image_embeddings"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.post(url, json=image_data, headers=headers)
        return np.array(response.json()["embeddings"])


class CLIPTextEmbeddings:
    # Will be used only if the API is not reachable

    def __init__(self):
        try:
            import torch
            from transformers import (
                CLIPProcessor,
                CLIPModel,
                CLIPTokenizer,
                CLIPTextModel,
            )
        except ImportError:
            print("Installing libraries for fallback ImagesEmbedding...")
            for library in ["torch", "transformers"]:
                subprocess.check_call([sys.executable, "-m", "pip", "install", library])

        from transformers import CLIPTokenizer, CLIPTextModel

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_embedding(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        import torch

        if isinstance(chunks, str):
            chunks = [Chunk(chunk_id=0, content=chunks, data_type=DataType.TEXT)]

        if isinstance(chunks, Chunk):
            chunks = [chunks]

        embeddings_l = []

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk.content, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                text_embeddings = self.text_model(**inputs).pooler_output

            embeddings_l.append(text_embeddings.squeeze().numpy())

        embeddings_array = np.array(embeddings_l)

        return embeddings_array


class ClipImageEmbeddings:
    # Will be used only if the API is not reachable

    def __init__(self):
        try:
            import torch
            from transformers import (
                CLIPProcessor,
                CLIPModel,
                CLIPTokenizer,
                CLIPTextModel,
            )
        except ImportError:
            print("Installing libraries for fallback ImagesEmbedding...")
            for library in ["torch", "transformers"]:
                subprocess.check_call([sys.executable, "-m", "pip", "install", library])

        from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_embedding(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        import torch

        if isinstance(chunks, str):
            chunks = [Chunk(chunk_id=0, content=chunks, data_type=DataType.IMAGE)]

        if isinstance(chunks, Chunk):
            chunks = [chunks]

        decoded_images = []
        for chunk in chunks:
            img_data = base64.b64decode(chunk.content)
            img = Image.open(io.BytesIO(img_data))
            decoded_images.append(img)

        image_inputs = self.processor(
            images=decoded_images, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**image_inputs)

        embeddings_array = image_embeddings.squeeze().numpy()

        return embeddings_array
