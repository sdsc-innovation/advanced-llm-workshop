import numpy as np

import chromadb

from .constants_and_data_classes import Chunk


class ChromaDBVectorStore:
    def __init__(
        self,
        collection_name: str,
        vector_store_path: str | None = None,
        chunks: list[Chunk] | None = None,
    ):
        if vector_store_path is None:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(collection_name)
            self.chunks = []
        else:
            self.client = chromadb.PersistentClient(vector_store_path)
            self.collection = self.client.get_or_create_collection(collection_name)
            if chunks is not None:
                self.chunks = chunks
            else:
                self.chunks = []

    def insert_chunk(self, chunk: Chunk, embedding: np.ndarray):
        embedding_list = embedding.tolist()

        self.collection.add(ids=[str(chunk.chunk_id)], embeddings=[embedding_list])
        self.chunks.append(chunk)

    def insert_chunks(self, chunks: list[Chunk], embeddings: list[np.ndarray]):
        embeddings_list = [e.tolist() for e in embeddings]

        self.collection.add(
            ids=[str(chunk.chunk_id) for chunk in chunks], embeddings=embeddings_list
        )
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int) -> list[Chunk]:
        query_embedding_list = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=query_embedding_list, n_results=k
        )

        # Find the actual chunks, not just the id and embedding
        relevant_ids = [int(id) for id in results["ids"][0]]
        relevant_chunks = [
            chunk for chunk in self.chunks if chunk.chunk_id in relevant_ids
        ]
        relevant_chunks_dict = {str(chunk.chunk_id): chunk for chunk in relevant_chunks}

        return [
            {
                "chunk_id": results["ids"][0][i],
                "score": results["distances"][0][i],
                "chunk": relevant_chunks_dict[results["ids"][0][i]],
            }
            for i in range(len(results["ids"][0]))
        ]


class VectorStoreRetriever:
    def __init__(
        self,
        text_embedding_model,
        text_vector_store,
        image_embedding_model=None,
        image_vector_store=None,
    ):
        self.text_embedding_model = text_embedding_model
        self.text_vector_store = text_vector_store
        self.image_embedding_model = image_embedding_model
        self.image_vector_store = image_vector_store

    def retrieve(
        self, queries: str | list[str], top_k_text: int, top_k_image: int | None = None
    ) -> list[list[Chunk]]:
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            embedding = self.text_embedding_model.get_embedding(query)[0]
            search_results = self.text_vector_store.search(embedding, top_k_text)
            results.append(search_results)

        if top_k_image is not None:
            for query in queries:
                embedding = self.image_embedding_model.get_embedding(query)[0]
                search_results = self.image_vector_store.search(embedding, top_k_image)
                results.append(search_results)

        return results
