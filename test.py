#!/usr/bin/env python3
"""Fast direct Qwen3.6 generation, no server, no thinking."""

import os

os.environ["HF_HOME"] = ".hf_cache"

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"

print(f"HF_HOME={os.environ['HF_HOME']}")
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()


def generate(prompt: str, max_new_tokens: int = 200) -> str:
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"  [{n_new} tokens in {dt:.2f}s → {n_new / dt:.1f} tok/s]")
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


# Warm-up (first call is slow — kernel compilation)
print("\nWarm-up:")
_ = generate("Hi", max_new_tokens=10)

# Real generations
print("\nGeneration 1:")
print(generate('Type "I love Qwen3.6" backwards'))

print("\nGeneration 2:")
print(generate("Explain MoE models in one paragraph."))
