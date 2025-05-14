from typing import Optional

from openai import OpenAI

from .constants_and_data_classes import LLMMessage


class OpenAILLM:
    def __init__(
        self,
        model_name: Optional[str] = "gpt-4o-mini",
        temperature=0.5,
        seed=42,
    ):
        super().__init__()

        self.client = OpenAI()
        self.model = model_name
        self.temperature = temperature
        self.seed = seed

        print(
            f"OpenAI LLM loaded: {model_name}; temperature: {temperature}; seed: {seed}"
        )

    def generate(
        self, conversation: list[dict[str, str]], verbose=True
    ) -> tuple[LLMMessage, float]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            seed=self.seed,
        )

        cost = compute_chatgpt_4o_cost(completion, verbose=verbose)

        return (
            LLMMessage(
                content=completion.choices[0].message.content,
                role=completion.choices[0].message.role,
                tool_calls=completion.choices[0].message.tool_calls,
            ),
            cost,
        )


def compute_chatgpt_4o_cost(completion, verbose: bool = False) -> float:
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens

    cost_per_1M_input_tokens = 0.15
    cost_per_1M_output_tokens = 0.60

    total_cost = (input_tokens / 1e6) * cost_per_1M_input_tokens
    total_cost += (output_tokens / 1e6) * cost_per_1M_output_tokens

    if verbose:
        print(f"Total input tokens: {input_tokens}")
        print(f"Total output tokens: {output_tokens}")
        print(f"Total tokens: {input_tokens+output_tokens}")
        print(f"Estimated cost: ${total_cost:.4f}")

    return total_cost
