from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPTQConfig, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from typing import List
import random
import torch
from data_generation import read_json
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MODEL = os.getenv("FINETUNE_MODEL")
FP16 = os.getenv("FP16").lower() == "true"
device = torch.device(os.getenv("DEVICE"))
force_quantization = os.getenv("FORCE_QUANTIZATION").lower() == "true"
torch.cuda.set_device(device)

print(f"Finetuning model {MODEL} on device: {device}")

TEST_SPLIT = 0.2
LR=3e-5
EPOCHS=4
BATCH_SIZE=4
random.seed(12345)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
    'q_proj',
    'k_proj',
    'v_proj'
    'o_proj'
    ]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if not tokenizer.pad_token:
    print("No pad token, assigned eos token as pad")
    tokenizer.pad_token = tokenizer.eos_token
    
if force_quantization:
    print("Forced quantization enabled, quantizing...")
    gptq_config = GPTQConfig(bits=4, dataset="c4-new", tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=device, quantization_config=gptq_config)
    tokenizer.save_pretrained(f"{MODEL}-4bit-GPTQ")
    model.save_pretrained(f"{MODEL}-4bit-GPTQ")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=device)
print(model)
model = get_peft_model(model=model, peft_config=peft_config)
model.print_trainable_parameters()

class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, data: List):

        self.input_ids = []
        self.labels = []
        self.masks = []

        half = int(len(data)/2)
        random.shuffle(data)

        en_to_slang = data[:half]
        slang_to_en = data[half:]
        
        encoded_en_to_slang = self.tokenize_translation(data=en_to_slang, en_to_slang=True)
        encoded_slang_to_en = self.tokenize_translation(data=slang_to_en, en_to_slang=False)
        
        self.input_ids += encoded_en_to_slang.get("input_ids") + encoded_slang_to_en.get("input_ids")
        self.labels += encoded_en_to_slang.get("labels") + encoded_slang_to_en.get("labels")
        self.masks += encoded_en_to_slang.get("attention_mask") + encoded_slang_to_en.get("attention_mask")
        
        # Zip the input_ids, labels, and masks together for consistent shuffling
        combined = list(zip(self.input_ids, self.labels, self.masks))

        random.shuffle(combined)
        
        self.input_ids, self.labels, self.masks = zip(*combined)

        self.input_ids = list(self.input_ids)
        self.labels = list(self.labels)
        self.masks = list(self.masks)

    def tokenize_translation(self, data: List, en_to_slang: bool):
        
        input_ids = []
        labels = []
        masks = []
        
        instruction = "Rewrite the following english sentence to slang and identify the words replaced."

        if not en_to_slang:
            instruction = "Rewrite the following slang sentence to english and identify the words replaced."

        for translation in data:

            concatenated = \
                f"{instruction}\nInput: {translation['original']} \
                \nOutput: {translation['translated']}\nWords replaced: {', '.join(translation['terms'])} {tokenizer.eos_token}"
            
            if not en_to_slang:
                concatenated = \
                    f"{instruction}\nInput: {translation['translated']} \
                    \nOutput: {translation['original']}\nWords replaced: {', '.join(translation['terms'])} {tokenizer.eos_token}"
            
            tokenized = tokenizer(concatenated, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
            input_ids.append(tokenized['input_ids'][0])
            labels.append(tokenized['input_ids'][0])
            masks.append(tokenized['attention_mask'][0])
            
        return dict(input_ids=input_ids, labels=labels, attention_mask=masks)

    def __getitem__(self, id):
        return dict(input_ids=self.input_ids[id], labels=self.labels[id], attention_mask=self.masks[id])
    
    def __len__(self):
        return len(self.input_ids)
            

raw_data = read_json("./data/generated-10000.json")
test_split_index = int(len(raw_data) * TEST_SPLIT)
train_dataset = TranslationDataset(raw_data[test_split_index:])
validate_dataset = TranslationDataset(raw_data[:test_split_index])

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.05,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=FP16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("./adapter")
