import requests
import os 
from dotenv import load_dotenv
from typing import List
import random
import json

load_dotenv()
random.seed(7777)

API_ENDPOINT = os.getenv("API_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

def vocab_used(vocab: List, sentence: str) -> List:
    sentence = sentence.lower()
    res = []
    for term in vocab:
        word = term.get("word").lower()
        if word in sentence:
            res.append(word)
    return res
    

def read_vocab(path: str) -> List:
    try:
        with open(path, 'r') as r:
            data = json.load(r)
        return data
    except Exception as e:
        print(e)
        return []

def get_generate_prompt(vocabulary: List, topic: str = "anything", vocab_size: int = 1) -> str:

    vocab = random.sample(vocabulary, vocab_size)
    vocab_string = ""

    for term in vocab:
        vocab_string += f"\t - {term.get('word')}: {term.get('definition')}\n"

    return f"The following consists of terms and its definitions, they can be used as verbs, adjectives and nouns:\n{vocab_string}\
    \nOutput one sentence about {topic} that incorporates ALL of the terms."


def generate(prompt: str) -> str | None:
    res = requests.post(API_ENDPOINT, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    return res.json().get("response")

if __name__ == "__main__":
    vocabulary = read_vocab("./vocabulary.json")
    prompt = get_generate_prompt(vocabulary, "fencing", 10)
    print(f"Prompt: {prompt}")
    res = generate(prompt)
    words_used = vocab_used(vocabulary, res)
    print(f"Generated: {res}")
    print(f"Words used: {words_used}")
