import os
import time
import logging
import numpy as np

import backoff
from openai import (
    OpenAI,
    OpenAIError,
    APIConnectionError,
    APIError,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
)

from typing import Union, List, Tuple

logger = logging.getLogger(__name__)
supported_chat_models = {"gpt-3.5-turbo": [0.5, 1.5], "gpt-4-turbo": [10, 30]}


def get_perplexity(logprobs: List[float]) -> float:
    assert len(logprobs) > 0, logprobs
    return np.exp(-sum(logprobs) / len(logprobs))


class OpenAIEngine:
    def __init__(
        self, api_key: Union[str, List[str], None] = None, rate_limit: int = -1
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
        """
        assert (
            os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")

        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0


class OpenAIChatEngine(OpenAIEngine):
    def __init__(
        self,
        model: str,
        api_key: Union[str, List[str], None] = None,
        stop: List[str] = ["\n\n"],
        rate_limit: int = -1,
    ) -> None:
        """Init an OpenAI Chat engine

        Args:
            model (str): Model family.
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            temperature (int, optional): Defaults to 0.
        """
        self.stop = stop
        self.model = model
        assert (
            model in supported_chat_models
        ), f"model must be one of {supported_chat_models}"

        self.cost_per_million_input_tokens = supported_chat_models[model][0]
        self.cost_per_million_output_tokens = supported_chat_models[model][1]

        super().__init__(api_key, rate_limit)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:

        # Calculate the costs
        input_cost = prompt_tokens * self.cost_per_million_input_tokens / 1000000
        output_cost = completion_tokens * self.cost_per_million_output_tokens / 1000000

        # Total cost
        total_cost = input_cost + output_cost
        return total_cost

    @backoff.on_exception(
        backoff.expo,
        (
            APIError,
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAIError,
        ),
    )
    def generate(
        self,
        prompt: Union[str, list[dict]],
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.1,
        json_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[str], float]:
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
            self.request_interval > 0
            and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        client = OpenAI(api_key=self.api_keys[self.current_key_idx])
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt
        if json_mode == True:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},
                **kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                max(start_time, self.next_avil_time[self.current_key_idx])
                + self.request_interval
            )

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        return [choice.message.content for choice in response.choices], cost