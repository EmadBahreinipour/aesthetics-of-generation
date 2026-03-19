import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 800
    top_p: float = 1.0


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, model: str, config: GenerationConfig) -> str: ...


class OpenAIProvider(BaseLLMProvider):

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, model: str, config: GenerationConfig) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        return response.choices[0].message.content


class OllamaProvider(BaseLLMProvider):

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def generate(self, prompt: str, model: str, config: GenerationConfig) -> str:
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            },
        )
        return response["message"]["content"]
