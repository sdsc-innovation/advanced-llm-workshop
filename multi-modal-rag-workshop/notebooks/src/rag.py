import io
import json
import base64

import matplotlib.pyplot as plt
from PIL import Image

from src.vectorstore import VectorStoreRetriever
from src.constants_and_data_classes import Chunk, LLMMessage, DataType, Roles


class Generator:
    def __init__(
        self,
        llm,
        developer_prompt: str,
        rag_template: str,
    ):
        self.llm = llm
        self.developer_prompt = developer_prompt
        self.rag_template = rag_template

    def _transform_chunks(self, chunks: list[Chunk]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"\n\nChunk {i+1}: \n"
            if chunk.data_type == DataType.TEXT:
                context += chunk.content
            elif chunk.data_type == DataType.IMAGE:
                context += f"Image from chunk {i+1} attached in next messages."
            else:
                raise ValueError("Unsupported data type")
        return context

    def _get_developer_message(self) -> dict[str, str]:
        developer_prompt = self.developer_prompt
        return {
            "role": Roles.DEVELOPER,
            "content": [{"type": "text", "text": developer_prompt}],
        }

    def _apply_rag_template(self, query: str, context: str) -> dict[str, str]:
        template_rag = self.rag_template
        return {
            "role": Roles.USER,
            "content": [
                {
                    "type": "text",
                    "text": template_rag.format(context=context, query=query),
                }
            ],
        }

    def process_generator_answer(self, generator_answer: str) -> dict[str, str]:
        answer = generator_answer[
            generator_answer.find("{") : generator_answer.rfind("}") + 1
        ]
        answer = json.loads(answer)

        if "step_by_step_thinking" in answer:
            step_by_step_thinking = answer["step_by_step_thinking"]
            chunk_used = answer["chunk_used"]
            answer = (
                json.dumps(answer["answer"], ensure_ascii=False)
                if isinstance(answer["answer"], dict)
                else answer["answer"]
            )

            return {
                "step_by_step_thinking": step_by_step_thinking,
                "chunk_used": chunk_used,
                "answer": answer,
            }
        else:
            raise ValueError("The output format is not correct.")

    def generate(
        self,
        history: list[dict[str, str]],
        query: str,
        chunks: list[Chunk],
        verbose: bool = False,
    ) -> tuple[LLMMessage, float]:
        conversation = [
            self._get_developer_message(),
            *history,
            self._apply_rag_template(query, self._transform_chunks(chunks)),
        ]

        for i, chunk in enumerate(chunks):
            if chunk.data_type == DataType.IMAGE:
                conversation.append(
                    {
                        "role": Roles.USER,  # Only allowed to send images as user...
                        "content": [
                            {
                                "type": "text",
                                "text": f"Annex: Image from chunk {i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{chunk.content}"
                                },
                            },
                        ],
                    }
                )

        if verbose:
            for msg in conversation:
                for msg_section in msg["content"]:
                    if msg_section["type"] == "text":
                        print(msg_section["text"])
                    elif msg_section["type"] == "image_url":
                        data = msg_section["image_url"]["url"]
                        data = data[len("data:image/jpeg;base64,") :]
                        img = Image.open(io.BytesIO(base64.b64decode(data)))
                        plt.imshow(img)
                        plt.axis("off")
                        plt.show()
                    else:
                        raise ValueError("Unknown content type.")

        return self.llm.generate(conversation, verbose)


def query_expansion(
    query: str,
    llm,
    developer_message: str,
    template_query_expansion: str,
    expansion_number: int = 2,
    verbose: bool = False,
) -> tuple[list[str], float]:
    query_expansion_message = {
        "role": "user",
        "content": template_query_expansion.format(
            query=query, expansion_number=expansion_number
        ),
    }
    conversation = [developer_message, query_expansion_message]

    generated_queries, cost = llm.generate(conversation)
    generated_queries = generated_queries.content.strip().split("\n")

    if verbose:
        print(f"Query expansion cost: {cost:.4f}")

    return generated_queries, cost


def reciprocal_rank_fusion(
    search_results_list: list[list[dict[str, any]]], k: int = 60
) -> list[dict[str, any]]:
    fused_scores = {}
    doc_chunks = {}
    score_sums = {}
    score_counts = {}

    for query_results in search_results_list:
        for result in query_results:
            doc_id = result["chunk_id"]
            score = result["score"]
            chunk = result["chunk"]

            # Initialize the dictionaries for new chunk IDs
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_chunks[doc_id] = chunk
                score_sums[doc_id] = 0
                score_counts[doc_id] = 0

            # Accumulate the total score and count the occurrences for each chunk ID
            score_sums[doc_id] += score
            score_counts[doc_id] += 1

            # Calculate the rank for reciprocal rank fusion
            rank = len(fused_scores)  # The current position in the rank
            fused_scores[doc_id] += 1 / (rank + 1 + k)

    avg_scores = {
        doc_id: score_sums[doc_id] / score_counts[doc_id] for doc_id in score_sums
    }

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    reranked_results = [
        {
            "chunk_id": doc_id,
            "chunk": doc_chunks[doc_id],
            "fused_score": score,
            "average_original_score": avg_scores[doc_id],
        }
        for doc_id, score in sorted_docs
    ]

    return reranked_results


class DefaultRAG:
    def __init__(
        self,
        llm,
        text_embedding_model,
        text_vector_store,
        generator,
        query_expansion_developer_message: str | None = None,
        query_expansion_template_query: str | None = None,
        params: dict[str, str | float | int] = {},
        image_text_embedding_model=None,
        image_vector_store=None,
    ):
        self.llm = llm
        if image_text_embedding_model is None or image_vector_store is None:
            self.retriever = VectorStoreRetriever(
                text_embedding_model, text_vector_store
            )
        else:
            self.retriever = VectorStoreRetriever(
                text_embedding_model,
                text_vector_store,
                image_text_embedding_model,
                image_vector_store,
            )

        self.generator = generator
        self.query_expansion_developer_message = query_expansion_developer_message
        self.query_expansion_template_query = query_expansion_template_query

        if "top_k_text" in params:
            self.top_k_text = params["top_k_text"]
        else:
            self.top_k_text = 5

        if "top_k_image" in params:
            self.top_k_image = params["top_k_image"]
        else:
            self.top_k_image = None

        if "number_query_expansion" in params:
            self.number_query_expansion = params["number_query_expansion"]
        else:
            self.number_query_expansion = 0

    def execute(
        self, query: str, history: list[dict[str, str]], verbose: bool = False
    ) -> tuple[dict, list[dict], float]:
        original_query = str(query)

        if self.number_query_expansion > 0:
            queries, expansion_cost = query_expansion(
                query,
                self.llm,
                self.query_expansion_developer_message,
                self.query_expansion_template_query,
                self.number_query_expansion,
                verbose,
            )
            if verbose:
                print("Expanded queries:")
                for q in queries:
                    print(q)
                print()
        else:
            expansion_cost = 0.0
            queries = [query]

        if self.top_k_image is None:
            search_results = self.retriever.retrieve(queries, self.top_k_text)
        else:
            search_results = self.retriever.retrieve(
                queries, self.top_k_text, top_k_image=self.top_k_image
            )

        rag_fusion_results = reciprocal_rank_fusion(search_results)

        chunks = [result["chunk"] for result in rag_fusion_results]

        rag_answer, call_cost = self.generator.generate(
            history, original_query, chunks, verbose
        )

        total_cost = expansion_cost + call_cost

        return (
            self.generator.process_generator_answer(rag_answer.content),
            rag_fusion_results,
            total_cost,
        )
