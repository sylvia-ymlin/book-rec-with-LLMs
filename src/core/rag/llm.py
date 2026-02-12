from typing import Optional, Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from pydantic import SecretStr

from src.utils import setup_logger


logger = setup_logger(__name__)


class LLMFactory:
    """
    Factory to create LLM instances based on provider and API key.
    Supports 'Bring Your Own Key' (BYOK) architecture.
    """

    @staticmethod
    def create(
        provider: Literal["openai", "ollama", "mock", "groq"] = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ) -> BaseChatModel:
        """
        Create and return a configured LangChain Chat Model.
        """
        logger.info(
            "Creating LLM instance: provider=%s, model=%s", provider, model_name
        )

        if provider == "mock":
            from langchain_community.chat_models import FakeListChatModel

            return FakeListChatModel(
                responses=[
                    "This is a MOCKED response from the RAG Agent.",
                    "I found the book 'Aurora Leigh' to be quite fascinating based on the description!",
                    "It fits your persona of liking Victorian literature.",
                ]
            )

        if provider == "openai":
            if not model_name:
                model_name = "gpt-3.5-turbo"

            if not api_key:
                raise ValueError(
                    "OpenAI API Key is required for 'openai' provider."
                )

            return ChatOpenAI(
                api_key=SecretStr(api_key),
                model_name=model_name,
                temperature=temperature,
                streaming=True,  # Support streaming by default
            )

        if provider == "ollama":
            # Ollama usually runs locally on default port 11434
            if not model_name:
                model_name = "llama3"  # Default for Ollama

            return ChatOllama(
                model=model_name,
                temperature=temperature,
            )

        if provider == "groq":
            if not model_name:
                model_name = "llama3-70b-8192"  # Stable default for Groq

            if not api_key:
                raise ValueError(
                    "Groq API Key is required for 'groq' provider."
                )

            # Use ChatOpenAI client but point to Groq's API
            return ChatOpenAI(
                api_key=SecretStr(api_key),
                base_url="https://api.groq.com/openai/v1",
                model_name=model_name,
                temperature=temperature,
                streaming=True,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_model(
    provider: str = "openai",
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Helper for dependency injection or simple usage."""
    try:
        return LLMFactory.create(provider=provider, api_key=api_key)
    except Exception as e:
        logger.error("Failed to create LLM: %s", e)
        raise


__all__ = ["LLMFactory", "get_llm_model"]

