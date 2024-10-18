"""
Author: Laszlo Popovics
Version: 1.0
Program: SambaNova Cloud access with rate limits
"""

# Import the required libraries
import logging
import time
import traceback
from dataclasses import dataclass
from collections import deque

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define constants
SAMBANOVA_API_ENDPOINT = "https://api.sambanova.ai/v1"
# Initialize logging
# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# String parser
parser = StrOutputParser()
# Maximum retry
max_iteration = 5


@dataclass(frozen=True)
class Models:
    """Enumeration of available SambaNova models."""
    sambanova_llama31_8b = "Meta-Llama-3.1-8B-Instruct"
    sambanove_llama31_70b = "Meta-Llama-3.1-70B-Instruct"
    sambanova_llama31_405b = "Meta-Llama-3.1-405B-Instruct"
    sambanova_llama32_1b = "Meta-Llama-3.2-1B-Instruct"
    sambanova_llama32_3b = "Meta-Llama-3.2-3B-Instruct"


@dataclass
class ChatModel:
    """Data class representing a chat model configuration."""
    name: str = None
    endpoint: str = None
    api_key: str = None
    rate_limit: int = None
    context_lengths: list = None


class RateLimitedChat:
    """Class to handle rate-limited chat interactions with a language model."""

    def __init__(self, chat: ChatModel):
        """
        Initialize the RateLimitedChat instance.

        Args: chat (ChatModel): The chat model configuration.
        """
        self.chat = chat
        self.queue = deque(maxlen=self.chat.rate_limit)
        self.llm = ChatOpenAI(
            api_key=chat.api_key,
            model=chat.name,
            base_url=chat.endpoint,
            temperature=0.0,
            streaming=True
        )

    def get_llm(self, temperature: float):
        """
        Retrieve the language model, adhering to the rate limit.

        Args: temperature (float): The temperature setting for the model.

        Returns: ChatOpenAI: The language model instance.
        """
        self.llm.temperature = temperature
        self._enforce_rate_limit()
        return self.llm

    def _enforce_rate_limit(self):
        """Enforce the rate limit by managing the request timestamps."""
        current_time = time.time()
        if len(self.queue) >= self.chat.rate_limit:
            oldest_time = self.queue[-1]
            elapsed_time = current_time - oldest_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        self.queue.appendleft(current_time)

    def _llm_call(self, system_prompt: str, variables: dict, user_prompt: str = None, messages=None,
                  return_messages: bool = False):
        """
        Internal method to make a call to the language model.

        Args:
            system_prompt (str): The system prompt.
            variables (dict): Variables for the prompt template.
            user_prompt (str, optional): The user prompt. Defaults to None.
            messages (list, optional): Existing messages. Defaults to None.
            return_messages (bool, optional): Whether to return messages. Defaults to False.

        Returns: str or tuple: The model's response, optionally with messages.
        """
        try:
            if messages is None:
                messages = []

            if not messages:
                if "{" in system_prompt:
                    messages.append(("system", system_prompt))
                else:
                    variables["system_prompt"] = system_prompt
                    messages.append(("system", "{system_prompt}"))

            if user_prompt:
                if "{" in user_prompt:
                    messages.append(("human", user_prompt))
                else:
                    variables["user_prompt"] = user_prompt
                    messages.append(("human", "{user_prompt}"))

            prompt_template = ChatPromptTemplate.from_messages(messages)
            chain = prompt_template | self.llm | parser
            result = None
            attempt = 0

            while result is None and attempt < max_iteration:
                try:
                    result = chain.invoke(input=variables)
                except openai.RateLimitError:
                    logger.warning("OpenAI RateLimitError encountered. Retrying in 10 seconds...")
                    time.sleep(10)
                    attempt += 1
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    traceback.print_exc()
                    break

            if result is None:
                logger.error("Max retries exceeded while attempting to call the LLM.")
                return None

            return (result, messages) if return_messages else result

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error while executing the LLM call: {e}")
            return None

    def call_the_llm(self, system_prompt: str, variables: dict, user_prompt: str = None, messages=None,
                     return_messages: bool = False):
        """
        Public method to call the language model with rate limiting.

        Args:
            system_prompt (str): The system prompt.
            variables (dict): Variables for the prompt templates.
            user_prompt (str, optional): The user prompt. Defaults to None.
            messages (list, optional): Existing messages. Defaults to None.
            return_messages (bool, optional): Whether to return messages. Defaults to False.

        Returns:
            str or tuple: The model's response, optionally with messages.
        """
        try:
            self._enforce_rate_limit()
            return self._llm_call(system_prompt, variables, user_prompt, messages, return_messages)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error while calling the LLM: {e}")
            return None


class LLMHelper:
    """Helper class to facilitate interactions with various language models."""

    def __init__(self, sambanova_api_key: str):
        """
        Initialize the LLMHelper with the given API key.

        Args: sambanova_api_key (str): The API key for SambaNova Cloud.
        """
        self._sambanova_llama31_8b = ChatModel(
            name=Models.sambanova_llama31_8b,
            endpoint=SAMBANOVA_API_ENDPOINT,
            api_key=sambanova_api_key,
            rate_limit=30,
            context_lengths=[4096, 8192, 16384]
        )
        self._sambanova_llama31_70b = ChatModel(
            name=Models.sambanove_llama31_70b,
            endpoint=SAMBANOVA_API_ENDPOINT,
            api_key=sambanova_api_key,
            rate_limit=20,
            context_lengths=[4096, 8192, 16384, 32768, 65536]
        )
        self._sambanova_llama31_405b = ChatModel(
            name=Models.sambanova_llama31_405b,
            endpoint=SAMBANOVA_API_ENDPOINT,
            api_key=sambanova_api_key,
            rate_limit=10,
            context_lengths=[4096, 8192]
        )
        self._sambanova_llama32_1b = ChatModel(
            name=Models.sambanova_llama32_1b,
            endpoint=SAMBANOVA_API_ENDPOINT,
            api_key=sambanova_api_key,
            rate_limit=30,
            context_lengths=[4096]
        )
        self._sambanova_llama32_3b = ChatModel(
            name=Models.sambanova_llama32_3b,
            endpoint=SAMBANOVA_API_ENDPOINT,
            api_key=sambanova_api_key,
            rate_limit=30,
            context_lengths=[4096]
        )
        self.rate_limited_llama31_8b = RateLimitedChat(chat=self._sambanova_llama31_8b)
        self.rate_limited_llama31_70b = RateLimitedChat(chat=self._sambanova_llama31_70b)
        self.rate_limited_llama31_405b = RateLimitedChat(chat=self._sambanova_llama31_405b)
        self.rate_limited_llama32_1b = RateLimitedChat(chat=self._sambanova_llama32_1b)
        self.rate_limited_llama32_3b = RateLimitedChat(chat=self._sambanova_llama32_3b)

    def call_llama31_8b(self, system_prompt: str, variables: dict, user_prompt: str = None):
        """Call the LLama 3.1 8B model."""
        return self.rate_limited_llama31_8b.call_the_llm(system_prompt, variables, user_prompt)

    def call_llama31_70b(self, system_prompt: str, variables: dict, user_prompt: str = None):
        """Call the LLama 3.1 70B model."""
        return self.rate_limited_llama31_70b.call_the_llm(system_prompt, variables, user_prompt)

    def call_llama31_405b(self, system_prompt: str, variables: dict, user_prompt: str = None):
        """Call the LLama 3.1 405B model."""
        return self.rate_limited_llama31_405b.call_the_llm(system_prompt, variables, user_prompt)

    def call_llama32_1b(self, system_prompt: str, variables: dict, user_prompt: str = None):
        """Call the LLama 3.2 1B model."""
        return self.rate_limited_llama32_1b.call_the_llm(system_prompt, variables, user_prompt)

    def call_llama32_3b(self, system_prompt: str, variables: dict, user_prompt: str = None):
        """Call the LLama 3.2 3B model."""
        return self.rate_limited_llama32_3b.call_the_llm(system_prompt, variables, user_prompt)
