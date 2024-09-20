from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.getenv("FINETUNE_MODEL")
device = os.getenv("DEVICE")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(MODEL, max_length=256)
model = PeftModel.from_pretrained(model, "./adapter", max_length=256).to(device)

while True:
    
    en = input("Translate to slang-> ")
    encoded = tokenizer(
        f"Rewrite the following english sentence to slang and identify the words replaced.\nInput: {en}", 
        return_tensors="pt").to(device)
    output_ids = model.generate(
        **encoded
    )
    output = tokenizer.decode(output_ids[0])
    print(output)