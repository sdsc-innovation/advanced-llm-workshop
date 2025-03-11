import io
import base64
import pickle
import requests

import numpy as np

import tiktoken
from openai import OpenAI

# import torch
# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

from PIL import Image

from src.data_classes import Chunk, DataType


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


class VLM2VecTextEmbeddings:
    def __init__(self):
        self.base_url = "https://compute.datascience.ch/vlm2vec-innovation"

    # Work only with 1 text (no batching)
    def get_embedding(self, text: str) -> np.ndarray:
        text_data = {"text": text}
        url = f"{self.base_url}/v1/text_embeddings"
        response = requests.post(url, json=text_data)
        # List of list because notebook expects a list of embeddings
        return np.array([response.json()["embeddings"]])


class VLM2VecImageEmbeddings:
    def __init__(self):
        self.base_url = "https://compute.datascience.ch/vlm2vec-innovation"

    # Work only with 1 image in base64 (no batching)
    def get_embedding(self, imagebase64: str) -> np.ndarray:
        image_data = {"imagebase64": imagebase64}
        url = f"{self.base_url}/v1/image_embeddings"
        response = requests.post(url, json=image_data)
        return np.array(response.json()["embeddings"])


"""
class PrecomputedImageEmbeddingModel:
    def __init__(self):
        print(
            "WARNING! Pre-computed Embedding Model, only working for predefined values."
        )

        with open("../data/query_visual_embeddings.pkl", "rb") as file:
            query_embeddings = pickle.load(file)

        self.query_to_embeddings = {}
        for query_dict in query_embeddings:
            self.query_to_embeddings[query_dict["query"]] = query_dict[
                "visual_embeddings"
            ]

    def get_embedding(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            if text in self.query_to_embeddings:
                embeddings.append(self.query_to_embeddings[text])
            else:
                raise ValueError(f"Embedding for '{text}' not pre-computed")

        return np.array(embeddings)"
"""


def compute_openai_large_embedding_cost(chunks, verbose: bool = True) -> float:
    encoder = tiktoken.get_encoding(
        "cl100k_base"
    )  # The encoding used by text-embedding-3-large

    def count_tokens(text):
        return len(encoder.encode(text))

    total_tokens = sum(count_tokens(chunk.content) for chunk in chunks)

    cost_per_1M_tokens = 0.13
    total_cost = (total_tokens / 1e6) * cost_per_1M_tokens

    if verbose:
        print(f"Total tokens: {total_tokens}")
        print(f"Estimated cost: ${total_cost:.4f}")

    return total_cost


"""
class CLIPEmbeddings:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_text_embeddings(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            text_embeddings = self.text_model(**inputs).pooler_output

        return text_embeddings.squeeze().numpy()

    def get_image_embeddings(self, images: list[str]) -> np.ndarray:
        if isinstance(images, str):
            images = [images]

        decoded_images = []
        for img_base64 in images:
            img_data = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_data))
            decoded_images.append(img)

        image_inputs = self.processor(
            images=decoded_images, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**image_inputs)

        return image_embeddings.squeeze().numpy()

    def get_embedding(self, chunks: str | Chunk | list[Chunk]) -> np.ndarray:
        if isinstance(chunks, str):
            chunks = [Chunk(chunk_id=0, content=chunks, data_type=DataType.TEXT)]

        if isinstance(chunks, Chunk):
            chunks = [chunks]

        embeddings_l = []

        for chunk in chunks:
            if chunk.data_type == DataType.TEXT:
                embeddings_l.append(self.get_text_embeddings(chunk.content))
            elif chunk.data_type == DataType.IMAGE:
                embeddings_l.append(self.get_image_embeddings(chunk.content))
            else:
                raise ValueError("Unknown Data Type")

        combined_embeddings = np.array(embeddings_l)

        return combined_embeddings
"""
