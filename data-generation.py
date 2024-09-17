import requests
from tqdm import tqdm
import os 
from dotenv import load_dotenv
from typing import List
import random
from threading import Thread
import time
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
    

def read_json(path: str) -> any:
    try:
        with open(path, 'r') as r:
            data = json.load(r)
        return data
    except Exception as e:
        print(e)
        return []
    
def write_json(path: str, data: any) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(e)
        return []

def get_generate_prompt(vocabulary: List, topic: str = "anything", vocab_size: int = 1) -> str:

    vocab = random.sample(vocabulary, vocab_size)
    vocab_string = ""

    for term in vocab:
        vocab_string += f"\t - {term.get('word')}: {term.get('definition')}\n"

    return f"""
Provide one sentence about {topic}, then rewrite the sentence using only a few slang terms from {vocab_string}. 
Do not provide definitions or explanations. Keep the slang version no longer than one sentence.
Example:
Original: I value my family deeply.
Translated: Ain't nothing like my fam, they got my back.
"""


def generate(prompt: str) -> str | None:
    res = requests.post(API_ENDPOINT, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    return res.json().get("response")

def extract_from_generate(raw: str) -> List[str]:
    if "Original" not in raw or "Translated" not in raw:
        return []
    else:
        try:
            res = raw.split("Translated: ")
            res[0] = res[0].replace("Original: ", "")
            
            return [res[0].strip(), res[1].strip()]
        except Exception as e:
            print(e)
            return []

def generate_worker(max_vocab: int, vocabulary: List, turns: int, topics: List, return_list: List, pbar = None, en_to_slang = True) -> None:
    
    cnt = 0
    
    while(cnt < turns):
        
        prompt = get_generate_prompt(vocabulary, random.choice(topics), max_vocab)
        res = generate(prompt)
        extracted = extract_from_generate(res)

        if len(extracted) != 2:
            continue
        
        words_used = vocab_used(vocabulary, extracted[1])

        if len(words_used) == 0:
            continue

        cnt += 1
        
        if en_to_slang:
            return_list.append({
                "original": extracted[0],
                "translated": extracted[1],
                "terms": words_used
            })
        else:
            return_list.append({
                "original": extracted[1],
                "translated": extracted[0],
                "terms": words_used
            })
        
        
        write_json("./generated.json", generated_data)
    
        if pbar:
            pbar.update(1)

if __name__ == "__main__":

    total_datasets = 1
    generated_data = []
    thread_num = 1
    threads = []
    vocabulary = read_json("./vocabulary.json")
    topics = read_json("./topics.json")

    print(f"Generating a dataset of length {total_datasets}")
    pbar = tqdm(total=total_datasets)
    
    for i in range(thread_num):
        t = Thread(target=generate_worker, args=[
            len(vocabulary),
            vocabulary,
            int(total_datasets/thread_num),
            topics,
            generated_data,
            pbar,
            True
        ])
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("Completed generation.")
        
    
    