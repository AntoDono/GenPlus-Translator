"""LoRA fine-tune on data/{train,val}.jsonl (built by `data_generation.py build`).

Works for base models without a chat template (GPT-2) and instruct models with one.
Loss is computed only on the response tokens, never on the prompt or padding.

    python finetune.py                     # BASE_MODEL from .env
    python finetune.py --epochs 3 --lr 2e-4
"""
import argparse
import os
import random

from dotenv import load_dotenv

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.getenv("BASE_MODEL", "gpt2-large"))
parser.add_argument("--device", default=os.getenv("DEVICE", "cuda:0"))
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-length", type=int, default=256)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--eval-steps", type=int, default=200)
parser.add_argument("--output", default="./adapter")
args = parser.parse_args()

# Expose only the chosen GPU. Otherwise Trainer spreads over every visible GPU (DataParallel),
# including ones that are already full. Index matches nvidia-smi.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.device.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
    args.device = "cuda:0"

import torch  # noqa: E402  (after CUDA_VISIBLE_DEVICES)
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from common import build_prompt, read_jsonl  # noqa: E402

random.seed(12345)
torch.manual_seed(12345)
device = torch.device(args.device)
torch.cuda.set_device(device)
print(f"Fine-tuning {args.model} on {device} ({torch.cuda.get_device_name(device)})")

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": device})
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.rank,
    lora_alpha=args.rank * 2,
    lora_dropout=0.05,
    target_modules="all-linear",  # q/k/v/o + MLP (c_attn/c_proj/c_fc on GPT-2)
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


class SFTDataset(torch.utils.data.Dataset):

    def __init__(self, rows):
        self.examples = []
        skipped = 0
        for row in rows:
            prompt_ids = tokenizer(build_prompt(row["task"], row["input"], tokenizer), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row["output"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(answer_ids) > args.max_length:
                skipped += 1
                continue
            self.examples.append(dict(
                input_ids=prompt_ids + answer_ids,
                labels=[-100] * len(prompt_ids) + answer_ids,
            ))
        if skipped:
            print(f"Skipped {skipped} examples longer than {args.max_length} tokens")

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)


def collate(batch):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(labels), attention_mask=torch.tensor(mask))


train_dataset = SFTDataset(read_jsonl("./data/train.jsonl"))
val_dataset = SFTDataset(read_jsonl("./data/val.jsonl"))
print(f"train: {len(train_dataset)}  val: {len(val_dataset)}")

training_args = TrainingArguments(
    output_dir="./runs",
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,  # ratio of total steps
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.0,
    bf16=True,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=25,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate,
)
trainer.train()
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
print(f"Saved adapter to {args.output}")
