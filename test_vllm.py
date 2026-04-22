#!/usr/bin/env python3
"""Fast Qwen3.5 generation via vLLM, no thinking."""

import os
import time

os.environ["HF_HOME"] = ".hf_cache"

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.6-35B-A3B"


def generate(llm, prompt: str, max_tokens: int = 200) -> str:
    msgs = [{"role": "user", "content": prompt}]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    t0 = time.time()
    outputs = llm.chat(
        messages=msgs,
        sampling_params=sp,
        chat_template_kwargs={"enable_thinking": False},
    )
    dt = time.time() - t0

    out = outputs[0].outputs[0]
    n_new = len(out.token_ids)
    print(f"  [{n_new} tokens in {dt:.2f}s -> {n_new / dt:.1f} tok/s]")
    return out.text


def main():
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print("Loading model...")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
    )

    print("\nWarm-up:")
    _ = generate(llm, "Hi", max_tokens=10)

    print("\nGeneration 1:")
    print(generate(llm, 'Type "I love Qwen3.6" backwards'))

    print("\nGeneration 2:")
    print(generate(llm, "Explain MoE models in one paragraph."))


if __name__ == "__main__":
    main()
