"""Score a model + adapter on data/test.jsonl.

    python eval.py                                   # current adapter
    python eval.py --model meta-llama/Llama-3.2-1B --adapter ./old-adapter --legacy   # old prompt format

Metrics per task:
  chrF          character n-gram F-score against the reference (higher = closer)
  slang_rate    en-to-slang outputs that use at least one vocabulary term
  leftover      slang-to-en outputs that still contain one of the slang terms (lower is better)
  term_hit      define answers that mention the term asked about
"""
import argparse
import json
import os
import random

from dotenv import load_dotenv
from sacrebleu.metrics import CHRF

from common import TermMatcher, load_vocab, read_jsonl
from inference import generate, load_model, run_task

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.getenv("BASE_MODEL", "gpt2-large"))
parser.add_argument("--adapter", default="./adapter")
parser.add_argument("--device", default=os.getenv("DEVICE", "cuda:0"))
parser.add_argument("--n", type=int, default=200, help="examples per task")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--legacy", action="store_true", help="v1 prompt format (no define task)")
parser.add_argument("--out", default=None, help="write predictions to this JSONL")
args = parser.parse_args()

LEGACY_INSTRUCTIONS = {
    "en-to-slang": "Rewrite the following english sentence to slang and identify the words replaced.",
    "slang-to-en": "Rewrite the following slang sentence to english and identify the words replaced.",
}

random.seed(0)
matcher = TermMatcher(load_vocab())
model, tokenizer = load_model(args.model, args.adapter, args.device)
rows = read_jsonl("./data/test.jsonl")
tasks = ["en-to-slang", "slang-to-en"] + ([] if args.legacy else ["define"])

report, dump = {}, []
for task in tasks:
    subset = [r for r in rows if r["task"] == task]
    subset = random.sample(subset, min(args.n, len(subset)))
    preds = []
    for i in range(0, len(subset), args.batch_size):
        batch = [r["input"] for r in subset[i:i + args.batch_size]]
        if args.legacy:
            prompts = [f"{LEGACY_INSTRUCTIONS[task]}\nInput: {t}\nOutput: " for t in batch]
            outs = generate(model, tokenizer, prompts)
            preds += [o.split("Words replaced")[0].strip() for o in outs]
        else:
            preds += run_task(model, tokenizer, task, batch)

    refs = [r["output"] for r in subset]
    scores = dict(n=len(subset), chrF=round(CHRF().corpus_score(preds, [refs]).score, 1))
    if task == "en-to-slang":
        scores["slang_rate"] = round(sum(bool(matcher.words(p)) for p in preds) / len(preds), 3)
    elif task == "slang-to-en":
        scores["leftover"] = round(sum(bool(set(matcher.words(p)) & set(r["terms"])) for p, r in zip(preds, subset)) / len(preds), 3)
    else:
        scores["term_hit"] = round(sum(r["terms"][0] in matcher.words(p) for p, r in zip(preds, subset)) / len(preds), 3)
    report[task] = scores
    dump += [dict(task=task, input=r["input"], reference=r["output"], prediction=p) for r, p in zip(subset, preds)]

    print(f"\n== {task}: {scores}")
    for r, p in list(zip(subset, preds))[:5]:
        print(f"  in:   {r['input']}\n  ref:  {r['output']}\n  pred: {p}\n")

print(json.dumps(report, indent=2))
if args.out:
    with open(args.out, "w") as f:
        for d in dump:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
