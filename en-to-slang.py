from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv
import os
import torch

load_dotenv()

MODEL = os.getenv("FINETUNE_MODEL")
device = torch.device(os.getenv("DEVICE"))
torch.cuda.set_device(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(MODEL, max_length=256, device_map=device)
model = PeftModel.from_pretrained(model, "./adapter", max_length=256).to(device)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
        self.stop_id = stops
        StoppingCriteria.__init__(self)

    def __call__(self, input_ids, scores):
        last_id = input_ids.tolist()[-1] 
        for id in stop_words_ids:
            if id == last_id:
                return True
        return False

stop_words_ids = [tokenizer.eos_token_id]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_ids)])

while True:
    
    en = input("Translate to slang-> ")
    encoded = tokenizer(
        f"Rewrite the following english sentence to slang and identify the words replaced.\nInput: {en}\nOutput: ", 
        return_tensors="pt").to(device)
    output_ids = model.generate(
        **encoded,
        top_k=50,
        temperature=0.5,
        do_sample=True,
        stopping_criteria=stopping_criteria
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output)