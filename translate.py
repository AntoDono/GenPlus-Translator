"""Try the fine-tuned model interactively.

    python translate.py --task en-to-slang
    python translate.py --task slang-to-en
    python translate.py --task define      # ask what a word means
"""
import argparse
import os

from dotenv import load_dotenv

from common import TASKS, adapter_dir
from inference import load_model, run_task

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=list(TASKS), default="en-to-slang")
parser.add_argument("--model", default=os.getenv("BASE_MODEL", "gpt2-large"))
parser.add_argument("--adapter", default=None, help="default: ./adapters/<model name>")
parser.add_argument("--device", default=os.getenv("DEVICE", "cuda:0"))
parser.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
args = parser.parse_args()
args.adapter = args.adapter or adapter_dir(args.model)

model, tokenizer = load_model(args.model, args.adapter, args.device)

while True:
    try:
        text = input(f"{args.task}> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if text:
        print(run_task(model, tokenizer, args.task, [text], sample=args.sample)[0])
