from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_prompt


def load_model(base: str, adapter: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device})
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts: List[str], max_new_tokens: int = 128, sample: bool = False) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        temperature=0.7 if sample else None,
        top_p=0.9 if sample else None,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # Base models sometimes run on into a new "###" block; cut there
    return [t.split("\n###")[0].strip() for t in texts]


def run_task(model, tokenizer, task: str, texts: List[str], **kw) -> List[str]:
    return generate(model, tokenizer, [build_prompt(task, t, tokenizer) for t in texts], **kw)
