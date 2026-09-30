"""
src/llm
=======
Module responsible for communicating with Large Language Models (LLMs).
Formats grounded educational prompts, manages context windows, and generates student-friendly explanations.
"""

from src.llm.client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
