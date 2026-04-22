# example_task.py
from typing import Literal

from pydantic import BaseModel

from llm_client import LLM


class RegisterClassification(BaseModel):
    category: Literal["casual", "formal", "legal", "academic", "narrative"]
    confidence: float


TEXTS = [
    "Hey dude, what's up?",
    "The defendant pleads not guilty.",
    "LOL that's so random omg",
    "Pursuant to section 4.2 of the agreement...",
]

SYSTEM = "You are a linguistic register classifier. Output JSON only."

prompts = [f"Classify the register of: {t!r}" for t in TEXTS]

with LLM() as llm:
    results = llm.batch(prompts, schema=RegisterClassification, system=SYSTEM)
    for text, r in zip(TEXTS, results):
        print(f"{r.category:10s} ({r.confidence:.2f})  {text}")
