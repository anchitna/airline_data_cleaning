from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class LLM:
    """
    A utility class for initializing Language Learning Models (LLMs).

    This class provides a static method to obtain instances of different LLMs based on the specified
    model name. It supports initializing both OpenAI's GPT models and Groq's LLaMA models.
    """

    @staticmethod
    def get_llm(llm_name: str = "") -> Optional[object]:
        """
        Initialize and return an instance of the specified Language Learning Model (LLM).

        This method supports initializing:
            - Groq's LLaMA model when `llm_name` is set to "groq".
            - OpenAI's GPT models by default or when a different `llm_name` is provided.

        Args:
            llm_name (str, optional): The name of the LLM to initialize.
                                      - "groq" for Groq's LLaMA model.
                                      - Any other value or empty string for OpenAI's GPT models.
                                      Defaults to "".

        Returns:
            Optional[object]: An instance of the specified LLM.
                              Returns `None` if the `llm_name` provided does not match any supported models.

        Raises:
            ValueError: If an unsupported `llm_name` is provided.
        """
        if llm_name:
            logger.info(f"Initializing Language Learning Model with name: '{llm_name}'")
        else:
            logger.info(f"Initializing Language Learning Model with name: gpt-4o")
        try:
            if llm_name.lower() == "groq":
                logger.debug("Selected LLM: Groq's LLaMA Model")
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.0
                )
                logger.info("Groq's LLaMA Model initialized successfully.")
                return llm
            elif llm_name.lower() in ["openai", "gpt", ""]:
                logger.debug("Selected LLM: OpenAI's GPT Model")
                llm = ChatOpenAI(
                    model_name='gpt-4o',
                    temperature=0.0
                )
                logger.info("OpenAI's GPT Model initialized successfully.")
                return llm
            else:
                logger.error(f"Unsupported LLM name provided: '{llm_name}'")
                raise ValueError(f"Unsupported LLM name provided: '{llm_name}'")
        except Exception as e:
            logger.exception(f"Failed to initialize LLM '{llm_name}': {e}")
            raise RuntimeError(f"Failed to initialize LLM '{llm_name}': {e}")