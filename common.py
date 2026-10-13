"""Shared helpers: vocabulary, term matching, prompt format, JSONL io.

The backend keeps a copy of the prompt + term-matching code; keep them in sync.
"""
import json
import re
from typing import Dict, List

TASKS = {
    "en-to-slang": "Rewrite this English text as Gen Z / Gen Alpha brainrot slang.",
    "slang-to-en": "Rewrite this brainrot slang as plain English.",
    "define": "Answer this question about Gen Z / Gen Alpha slang.",
}


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str) -> List[Dict]:
    try:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def adapter_dir(model: str) -> str:
    """Default adapter location for a base model, e.g. Qwen/Qwen3-0.6B -> ./adapters/qwen3-0.6b"""
    return f"./adapters/{model.split('/')[-1].lower()}"


def load_vocab(path: str = "./vocabulary.json") -> List[Dict]:
    return read_json(path)


def surface_forms(term: Dict) -> List[str]:
    return [term["word"], *term.get("aliases", [])]


class TermMatcher:
    """Finds vocabulary terms in text on word boundaries (so 'era' never matches 'therapy')."""

    def __init__(self, vocab: List[Dict]):
        self.by_form = {}
        for term in vocab:
            for form in surface_forms(term):
                self.by_form[form.lower()] = term
        # Longest first so 'no cap' wins over 'cap'
        forms = sorted(self.by_form, key=len, reverse=True)
        pattern = "|".join(re.escape(f) for f in forms)
        self.regex = re.compile(rf"(?<![\w'])(?:{pattern})(?![\w'])", re.IGNORECASE)

    def find(self, text: str) -> List[Dict]:
        """Every match as {word, definition, start, end}, in order of appearance."""
        res = []
        for m in self.regex.finditer(text):
            term = self.by_form[m.group(0).lower()]
            res.append(dict(word=term["word"], definition=term["definition"], start=m.start(), end=m.end()))
        return res

    def words(self, text: str) -> List[str]:
        """Unique canonical words found in text, in order of appearance."""
        return list(dict.fromkeys(m["word"] for m in self.find(text)))


def build_prompt(task: str, text: str, tokenizer=None) -> str:
    """Prompt up to where the model's answer starts.

    Models with a chat template (instruct models) get it; base models like GPT-2 get a plain format.
    """
    instruction = TASKS[task]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": f"{instruction}\n\n{text}"}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"
