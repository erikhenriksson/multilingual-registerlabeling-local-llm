#!/usr/bin/env python3
"""Fast Qwen3.6 generation via SGLang offline engine."""

import os
import time

os.environ["HF_HOME"] = os.environ.get("LOCAL_SCRATCH", ".") + "/hf_cache"

import sglang as sgl

MODEL = "Qwen/Qwen3.6-35B-A3B"


def main():
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print("Loading model...")

    llm = sgl.Engine(
        model_path=MODEL,
        tp_size=2,  # tensor parallel = num GPUs you allocated
        mem_fraction_static=0.85,
        context_length=8192,  # lower than model default to save KV memory
        reasoning_parser="qwen3",  # Qwen3.6 has thinking mode; this parses it
        trust_remote_code=True,
    )

    def generate(prompt: str, max_new_tokens: int = 200) -> str:
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        }
        t0 = time.time()
        out = llm.generate(prompt, sampling_params)
        dt = time.time() - t0

        text = out["text"]
        # token count isn't always in the output dict; fall back to rough estimate
        n_tokens = out.get("meta_info", {}).get("completion_tokens", len(text.split()))
        print(f"  [{n_tokens} tokens in {dt:.2f}s -> {n_tokens / dt:.1f} tok/s]")
        return text

    # Warm-up
    print("\nWarm-up:")
    _ = generate("Hi", max_new_tokens=10)

    print("\nGeneration 1:")
    print(generate('Type "I love Qwen3.6" backwards'))

    print("\nGeneration 2:")
    print(generate("Explain MoE models in one paragraph."))

    llm.shutdown()


if __name__ == "__main__":
    main()
