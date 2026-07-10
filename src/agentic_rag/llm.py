"""Pluggable LLM backends.

* :class:`EchoLLM` - an offline, dependency-free client that produces a grounded
  extractive answer from the retrieved context. Perfect for demos and tests with
  no API key.
* :class:`OpenAILLM` / :class:`GeminiLLM` - hosted providers (installed with the
  ``openai`` / ``gemini`` extras, respectively).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from .config import Settings

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class LLMClient(Protocol):
    """Generates a completion for a prompt."""

    def generate(self, prompt: str, *, system: str | None = None) -> str: ...


class EchoLLM:
    """Offline client that stitches together the most relevant context.

    It does not "reason", but it selects the sentence from the retrieved context
    that best overlaps with the question, so the whole pipeline is runnable and
    testable — and produces sensible answers — without external services.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        context = _extract_context(prompt)
        question = _extract_question(prompt)
        if not context:
            return "I couldn't find any relevant information in the provided documents."

        question_tokens = set(_WORD_RE.findall(question.lower()))
        best_sentence: str | None = None
        best_marker = 1
        best_score = -1
        for marker, block in enumerate(context, start=1):
            for sentence in _split_sentences(block):
                overlap = len(question_tokens & set(_WORD_RE.findall(sentence.lower())))
                if overlap > best_score:
                    best_score, best_sentence, best_marker = overlap, sentence, marker

        if not best_sentence or best_score <= 0:
            best_sentence = _split_sentences(context[0])[0]
            best_marker = 1
        return f"Based on the retrieved context: {best_sentence.strip()} [{best_marker}]"


class OpenAILLM:
    """Chat completion via the OpenAI API."""

    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'openai' extra is required for OpenAILLM. "
                'Install it with: pip install -e ".[openai]"'
            ) from exc
        if not api_key:
            raise ValueError("RAG_OPENAI_API_KEY must be set to use the OpenAI provider.")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


class GeminiLLM:
    """Text generation via Google Gemini."""

    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'gemini' extra is required for GeminiLLM. "
                'Install it with: pip install -e ".[gemini]"'
            ) from exc
        if not api_key:
            raise ValueError("RAG_GEMINI_API_KEY must be set to use the Gemini provider.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        model = self._genai.GenerativeModel(
            self._model,
            system_instruction=system,
            generation_config={"temperature": self._temperature},
        )
        response = model.generate_content(prompt)
        return response.text or ""


def _extract_context(prompt: str) -> list[str]:
    """Pull the numbered context blocks out of the rendered prompt."""
    if "Context:" not in prompt:
        return []
    body = prompt.split("Context:", 1)[1]
    body = body.split("Question:", 1)[0]
    blocks: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            current.append(stripped.split("]", 1)[1].strip())
        elif stripped:
            current.append(stripped)
    if current:
        blocks.append(" ".join(current).strip())
    return [b for b in blocks if b]


def _extract_question(prompt: str) -> str:
    if "Question:" in prompt:
        return prompt.split("Question:", 1)[1].strip().splitlines()[0].strip()
    return prompt.strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def build_llm(settings: Settings) -> LLMClient:
    """Instantiate the LLM client selected in ``settings``."""
    if settings.llm_provider == "openai":
        return OpenAILLM(settings.llm_model, settings.openai_api_key, settings.llm_temperature)
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings.llm_model, settings.gemini_api_key, settings.llm_temperature)
    return EchoLLM()
