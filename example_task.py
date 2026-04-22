# example_task.py
from typing import Literal

from pydantic import BaseModel, Field

from llm_client import LLM


class RegisterClassification(BaseModel):
    reasoning: str = Field(
        description="One sentence identifying the key register cues in the text."
    )
    category: Literal["casual", "formal", "legal", "academic", "narrative"]


SYSTEM = """You classify the linguistic register of short texts into exactly one of:

- casual: informal conversation, slang, contractions, internet speak (lol, omg), \
friends chatting
- formal: polished, professional but not domain-specific; business emails, \
formal announcements, public-facing prose
- legal: legal or contractual language; references to statutes, parties, \
clauses, pleadings, "pursuant to", "the defendant", "hereinafter"
- academic: scholarly prose, scientific claims, citations, textbook-style \
exposition of concepts
- narrative: storytelling or fiction; "once upon a time", scene-setting, \
characters, narrative past tense

First identify the key register cues in one sentence, then pick the single \
best-fit category. Output JSON only."""


TEXTS = [
    "Hey dude, what's up?",
    "The defendant pleads not guilty.",
    "LOL that's so random omg",
    "Pursuant to section 4.2 of the agreement...",
    "Dear Sir or Madam, I am writing to inquire...",
    "yo check this out its fire",
    "The mitochondrion is the powerhouse of the cell.",
    "Once upon a time in a faraway land...",
]


def main():
    prompts = [f"Classify the register of this text: {t!r}" for t in TEXTS]
    with LLM() as llm:
        results = llm.batch(
            prompts,
            schema=RegisterClassification,
            system=SYSTEM,
            max_new_tokens=200,
        )
        print()
        for text, r in zip(TEXTS, results):
            print(f"{r.category:10s}  {text}")
            print(f"           └─ {r.reasoning}")
            print()


if __name__ == "__main__":
    main()
