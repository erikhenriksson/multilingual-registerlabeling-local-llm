# llm_client.py
import json
import os
from typing import Type, TypeVar, overload

os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("HF_TOKEN", "")

import sglang as sgl
from pydantic import BaseModel
from transformers import AutoTokenizer

T = TypeVar("T", bound=BaseModel)


class LLM:
    """Thin wrapper around sgl.Engine. One per process."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3.6-35B-A3B",
        tp_size: int = 2,
        mem_fraction_static: float = 0.90,
        context_length: int = 4096,
        reasoning_parser: str | None = "qwen3",
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.engine = sgl.Engine(
            model_path=model,
            tp_size=tp_size,
            mem_fraction_static=mem_fraction_static,
            context_length=context_length,
            reasoning_parser=reasoning_parser,
            trust_remote_code=True,
        )

    def _format(self, user_text: str, system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @overload
    def batch(
        self,
        prompts: list[str],
        *,
        schema: Type[T],
        system: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[T]: ...
    @overload
    def batch(
        self,
        prompts: list[str],
        *,
        schema: None = None,
        system: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]: ...

    def batch(
        self,
        prompts,
        *,
        schema=None,
        system=None,
        max_new_tokens=512,
        temperature=0.0,
    ):
        formatted = [self._format(p, system) for p in prompts]
        sampling = {"temperature": temperature, "max_new_tokens": max_new_tokens}
        if schema is not None:
            sampling["json_schema"] = json.dumps(schema.model_json_schema())

        outs = self.engine.generate(formatted, sampling)
        texts = [o["text"] for o in outs]

        if schema is None:
            return texts
        return [schema.model_validate_json(t) for t in texts]

    def generate(self, prompt: str, **kwargs):
        """Convenience for single prompts. Returns one item, not a list."""
        return self.batch([prompt], **kwargs)[0]

    def close(self):
        self.engine.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
