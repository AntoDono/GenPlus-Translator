"""Generate training data with the CHUNGUS API (OpenAI-compatible).

    python data_generation.py pairs --n 6000      # English <-> slang translation pairs
    python data_generation.py define --per-word 16 # Q&A teaching what each term means
    python data_generation.py build               # dedupe, split -> data/{train,val,test}.jsonl

Generation appends to data/*.raw.jsonl and resumes where it left off.
"""
import argparse
import asyncio
import json
import os
import random
import re

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from common import TermMatcher, load_vocab, read_json, read_jsonl, surface_forms, write_jsonl

load_dotenv(override=True)

GEN_MODEL = os.getenv("GEN_MODEL", "qwen3.8:27b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:567m")
PAIRS_PATH = "./data/pairs.raw.jsonl"
DEFINE_PATH = "./data/define.raw.jsonl"
ITEMS_PER_CALL = 6
CONCURRENCY = 4  # CHUNGUS is ~60 tok/s total; more parallelism only queues

STYLES = [
    "text message to a friend", "group chat message", "comment on a video", "tweet",
    "question to a friend", "complaint", "excited reaction", "story about what happened today",
    "advice to a friend", "description of a person", "review of a place or product",
    "reply to someone's post",
]
LENGTHS = ["a few words", "one sentence", "one sentence", "two sentences"]

client = AsyncOpenAI(
    base_url=os.getenv("CHUNGUS_BASE_URL", "https://chungus.antodono.com/v1"),
    api_key=os.getenv("CHUNGUS_API_KEY"),
    max_retries=8,  # honors Retry-After on 429
    timeout=120,
)


async def chat_json(prompt: str, temperature: float = 0.9) -> dict | None:
    res = await client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "You write training data for a Gen Z slang translator. Reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2000,
        response_format={"type": "json_object"},
        extra_body={"thinking": False},
    )
    raw = res.choices[0].message.content or ""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def term_line(term: dict) -> str:
    return f'"{term["word"]}" ({term["definition"]})'


# ---------------------------------------------------------------- translation pairs

def make_pair_specs(vocab, topics, n):
    """Specs with 1-3 required terms each, cycling the vocabulary so every term gets coverage."""
    deck = []
    specs = []
    for _ in range(n):
        k = random.choice([1, 1, 1, 2, 2, 3])
        terms = []
        while len(terms) < k:
            if not deck:
                deck = random.sample(vocab, len(vocab))
            t = deck.pop()
            if t not in terms:
                terms.append(t)
        specs.append(dict(topic=random.choice(topics), style=random.choice(STYLES),
                          length=random.choice(LENGTHS), terms=terms))
    return specs


def pairs_prompt(specs) -> str:
    lines = []
    for i, s in enumerate(specs, 1):
        terms = "; ".join(term_line(t) for t in s["terms"])
        lines.append(f'{i}. topic: {s["topic"]} | style: {s["style"]} | length: {s["length"]} | required terms: {terms}')
    return f"""Write {len(specs)} items. For each item, first write "slang", then "original":
- "slang": a message the way Gen Z / Gen Alpha actually text online, on the topic, in the style and length given. Use every required term naturally and with its correct meaning (inflections like -ing/-ed are fine). Casual spelling (lowercase, "u", "rn") is fine. If the terms don't fit the topic well, bend the topic, not the meaning of the terms.
- "original": the same message the way a normal person would say it casually in plain English, with NO slang. Same meaning, same facts, nothing added or dropped. Use everyday words, not dictionary definitions.
Do not explain anything.

{chr(10).join(lines)}

Return {{"items": [{{"id": 1, "slang": "...", "original": "..."}}, ...]}}"""


def validate_pair(item, spec, matcher):
    original = (item.get("original") or "").strip()
    slang = (item.get("slang") or "").strip()
    if not original or not slang or original.lower() == slang.lower():
        return None
    if len(original) > 400 or len(slang) > 400:
        return None
    found = set(matcher.words(slang))
    used = [t["word"] for t in spec["terms"] if t["word"] in found]
    if not used:
        return None
    return dict(original=original, slang=slang, terms=used, topic=spec["topic"], style=spec["style"])


async def generate_pairs(target: int):
    vocab, topics = load_vocab(), read_json("./topics.json")
    matcher = TermMatcher(vocab)
    have = len(read_jsonl(PAIRS_PATH))
    if have >= target:
        print(f"{PAIRS_PATH} already has {have} pairs")
        return
    pbar = tqdm(total=target, initial=have, desc="pairs")
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    state = dict(have=have, rejected=0)

    async def worker():
        while state["have"] < target:
            specs = make_pair_specs(vocab, topics, ITEMS_PER_CALL)
            async with sem:
                try:
                    data = await chat_json(pairs_prompt(specs))
                except Exception as e:
                    tqdm.write(f"request failed: {e}")
                    await asyncio.sleep(5)
                    continue
            items = (data or {}).get("items") or []
            rows = []
            for i, spec in enumerate(specs):
                item = next((x for x in items if x.get("id") == i + 1), items[i] if i < len(items) else None)
                row = validate_pair(item, spec, matcher) if isinstance(item, dict) else None
                if row:
                    rows.append(row)
                else:
                    state["rejected"] += 1
            async with lock:
                rows = rows[: max(0, target - state["have"])]
                with open(PAIRS_PATH, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["have"] += len(rows)
                pbar.update(len(rows))
                pbar.set_postfix(rejected=state["rejected"])

    await asyncio.gather(*[worker() for _ in range(CONCURRENCY)])
    pbar.close()


# ---------------------------------------------------------------- definitions (Q&A)

DEFINE_ANGLES = [
    'plain "what does X mean" questions, phrased many different ways',
    'questions quoting a message that uses the term, asking what the sender meant',
    'questions asking how to use the term, answered with an example sentence',
    'reverse lookups: describe the meaning and ask what the slang word for it is (answer names the term)',
    'questions on whether the term is positive, negative, rude, or ironic',
    'confused-parent style questions, answered simply',
]


def define_prompt(term: dict, k: int, angle: str) -> str:
    forms = ", ".join(f'"{f}"' for f in surface_forms(term))
    return f"""Slang term: {term_line(term)}. Forms: {forms}.

Write {k} varied question-and-answer pairs where someone asks about this term. Focus on: {angle}.
Answers: 1-3 sentences, plain English, accurate to the definition, and include a short example sentence when it helps. Never invent other meanings. Each question or answer must contain the term.

Return {{"items": [{{"question": "...", "answer": "..."}}, ...]}}"""


async def generate_define(per_word: int):
    vocab = load_vocab()
    matcher = TermMatcher(vocab)
    existing = read_jsonl(DEFINE_PATH)
    counts = {}
    for row in existing:
        counts[row["word"]] = counts.get(row["word"], 0) + 1
    jobs = []
    for term in vocab:
        need = per_word - counts.get(term["word"], 0)
        for angle in random.sample(DEFINE_ANGLES, len(DEFINE_ANGLES)):
            if need <= 0:
                break
            jobs.append((term, angle))
            need -= 4
    pbar = tqdm(total=len(jobs), desc="define calls")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run(term, angle):
        async with sem:
            try:
                data = await chat_json(define_prompt(term, 4, angle), temperature=0.8)
            except Exception as e:
                tqdm.write(f"request failed: {e}")
                data = None
        rows = []
        for item in (data or {}).get("items") or []:
            q, a = (item.get("question") or "").strip(), (item.get("answer") or "").strip()
            if not q or not a or len(a) > 600:
                continue
            if term["word"] not in matcher.words(q + "\n" + a):
                continue
            rows.append(dict(word=term["word"], question=q, answer=a))
        with open(DEFINE_PATH, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pbar.update(1)

    await asyncio.gather(*[run(t, a) for t, a in jobs])
    pbar.close()


# ---------------------------------------------------------------- build dataset

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


async def embed(texts, batch=64):
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="embedding"):
        res = await client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        out += [d.embedding for d in res.data]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def near_dupes(texts, threshold=0.95):
    """Indices to drop: any text too similar to an earlier kept one."""
    v = await embed(texts)
    drop = set()
    for i in range(len(v)):
        if i in drop:
            continue
        sims = v[i + 1:] @ v[i]
        for j in np.nonzero(sims > threshold)[0]:
            drop.add(i + 1 + int(j))
    return drop


async def build(use_embeddings: bool):
    vocab = load_vocab()
    pairs = read_jsonl(PAIRS_PATH)
    seen, uniq = set(), []
    for p in pairs:
        key = normalize(p["original"])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if use_embeddings:
        drop = await near_dupes([p["original"] for p in uniq])
        uniq = [p for i, p in enumerate(uniq) if i not in drop]
    print(f"pairs: {len(pairs)} raw -> {len(uniq)} after dedupe")

    defines = read_jsonl(DEFINE_PATH)
    seen, qa = set(), []
    for d in defines:
        key = normalize(d["question"])
        if key not in seen:
            seen.add(key)
            qa.append(d)
    print(f"define: {len(defines)} raw -> {len(qa)} after dedupe")

    random.shuffle(uniq)
    random.shuffle(qa)
    n_eval = max(1, int(len(uniq) * 0.05))
    pair_splits = dict(test=uniq[:n_eval], val=uniq[n_eval:2 * n_eval], train=uniq[2 * n_eval:])
    n_qa = max(1, int(len(qa) * 0.05))
    qa_splits = dict(test=qa[:n_qa], val=qa[n_qa:2 * n_qa], train=qa[2 * n_qa:])

    for split in ["train", "val", "test"]:
        rows = []
        for p in pair_splits[split]:
            rows.append(dict(task="en-to-slang", input=p["original"], output=p["slang"], terms=p["terms"]))
            rows.append(dict(task="slang-to-en", input=p["slang"], output=p["original"], terms=p["terms"]))
        for d in qa_splits[split]:
            rows.append(dict(task="define", input=d["question"], output=d["answer"], terms=[d["word"]]))
        if split == "train":
            # Anchor every term to its dictionary definition
            for t in vocab:
                rows.append(dict(task="define", input=f'What does "{t["word"]}" mean?',
                                 output=f'"{t["word"]}" means: {t["definition"]}', terms=[t["word"]]))
        random.shuffle(rows)
        write_jsonl(f"./data/{split}.jsonl", rows)
        counts = {}
        for r in rows:
            counts[r["task"]] = counts.get(r["task"], 0) + 1
        print(f"{split}: {len(rows)} rows {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--n", type=int, default=6000)
    d = sub.add_parser("define")
    d.add_argument("--per-word", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--no-embed", action="store_true", help="skip bge-m3 near-duplicate filtering")
    parser.add_argument("--seed", type=int, default=7777)
    args = parser.parse_args()

    os.makedirs("./data", exist_ok=True)
    if args.cmd == "pairs":
        asyncio.run(generate_pairs(args.n))
    elif args.cmd == "define":
        asyncio.run(generate_define(args.per_word))
    else:
        random.seed(args.seed)  # reproducible splits; generation stays unseeded so resumes don't repeat specs
        asyncio.run(build(use_embeddings=not args.no_embed))
