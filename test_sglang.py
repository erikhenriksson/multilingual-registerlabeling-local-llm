#!/usr/bin/env python3
"""Qwen3.6 generation via SGLang offline engine, thinking disabled."""

import os
import time

os.environ["HF_HOME"] = ".hf_cache"

import sglang as sgl
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"


def main():
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("Loading model...")
    llm = sgl.Engine(
        model_path=MODEL,
        tp_size=2,
        mem_fraction_static=0.90,
        context_length=4096,
        reasoning_parser="qwen3",
        trust_remote_code=True,
        # cuda graphs on (default), max_running_requests auto-sized (default)
    )

    def format_prompt(user_text: str) -> str:
        """Apply chat template with thinking disabled."""
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(user_text: str, max_new_tokens: int = 300) -> str:
        prompt = format_prompt(user_text)
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        }
        t0 = time.time()
        out = llm.generate(prompt, sampling_params)
        dt = time.time() - t0
        text = out["text"]
        n_tokens = out.get("meta_info", {}).get("completion_tokens", len(text.split()))
        print(f"  [{n_tokens} tokens in {dt:.2f}s -> {n_tokens / dt:.1f} tok/s]")
        return text

    def generate_batch(user_texts: list, max_new_tokens: int = 300):
        prompts = [format_prompt(t) for t in user_texts]
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        }
        t0 = time.time()
        outs = llm.generate(prompts, sampling_params)
        dt = time.time() - t0
        total_tokens = sum(
            o.get("meta_info", {}).get("completion_tokens", len(o["text"].split()))
            for o in outs
        )
        print(
            f"  [batch={len(prompts)}: {total_tokens} tokens in {dt:.2f}s "
            f"-> {total_tokens / dt:.1f} tok/s aggregate, "
            f"{total_tokens / dt / len(prompts):.1f} tok/s per-seq]"
        )
        return [o["text"] for o in outs]

    # Warm-up (first call triggers flashinfer JIT + cuda graph capture)
    print("\nWarm-up:")
    _ = generate("Hi", max_new_tokens=10)

    # Single-stream benchmarks
    print("\nGeneration 1 (single):")
    print(generate('Type "I love Qwen3.6" backwards'))

    print("\nGeneration 2 (single):")
    print(generate("Explain MoE models in one paragraph."))

    # Batched benchmark — this is where SGLang should shine
    print("\nGeneration 3 (batch of 8):")
    batch_prompts = [
        "Classify the register of this text: 'Hey dude, what's up?'",
        "Classify the register of this text: 'The defendant pleads not guilty.'",
        "Classify the register of this text: 'LOL that's so random omg'",
        "Classify the register of this text: 'Pursuant to section 4.2 of the agreement...'",
        "Classify the register of this text: 'Dear Sir or Madam, I am writing to inquire...'",
        "Classify the register of this text: 'yo check this out its fire'",
        "Classify the register of this text: 'The mitochondrion is the powerhouse of the cell.'",
        "Classify the register of this text: 'Once upon a time in a faraway land...'",
    ]
    results = generate_batch(batch_prompts, max_new_tokens=150)
    for i, r in enumerate(results):
        print(f"  [{i}] {r[:100]}...")

    llm.shutdown()


if __name__ == "__main__":
    main()
