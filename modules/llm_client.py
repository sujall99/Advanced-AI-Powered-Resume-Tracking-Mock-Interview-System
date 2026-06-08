"""
LLM Client Abstraction
======================

This project can use a hosted LLM (Gemini) OR a fully local LLM (Ollama).

We keep a small abstraction so downstream modules (question generation,
report generation, etc.) do not depend on one vendor.

Local provider: Ollama
- Requires installing Ollama and pulling a model (e.g. llama3.1:8b)
- Default endpoint: http://localhost:11434

Sample:
>>> from modules.llm_client import OllamaClient
>>> c = OllamaClient()
>>> c.generate("Write 2 interview questions for a data engineer.")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class LlmGeneration:
    text: str
    raw: Dict[str, Any]


class OllamaClient:
    """
    Minimal Ollama HTTP client.

    Env vars (optional):
    - OLLAMA_HOST: e.g. http://localhost:11434
    - OLLAMA_MODEL: e.g. llama3.1:8b
    """

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout_s: int = 120,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.4,
        top_p: float = 0.9,
        num_predict: int = 512,
    ) -> str:
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response. Check model availability and prompt.")
        return text

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

