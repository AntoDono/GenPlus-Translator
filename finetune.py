from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPTQConfig, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from typing import List
import random
import torch
from data_generation import read_json

random.seed(12345)
MODEL = "gpt2-large"
TEST_SPLIT = 0.2
device = "cuda:0"

torch.cuda.set_device(device)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj"]
    target_modules=["c_proj", "c_attn"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=device)

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
                \nOutput: {translation['translated']}\nWords replaced: {', '.join(translation['terms'])}"
            
            if not en_to_slang:
                concatenated = \
                    f"{instruction}\nInput: {translation['translated']} \
                    \nOutput: {translation['original']}\nWords replaced: {', '.join(translation['terms'])}"
            
            tokenized = tokenizer(concatenated, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
            input_ids.append(tokenized['input_ids'][0])
            labels.append(tokenized['input_ids'][0])
            masks.append(tokenized['attention_mask'][0])
            
        return dict(input_ids=input_ids, labels=labels, attention_mask=masks)

    def __getitem__(self, id):
        return dict(input_ids=self.input_ids[id], labels=self.labels[id], attention_mask=self.masks[id])
    
    def __len__(self):
        return len(self.input_ids)
            

raw_data = read_json("./data/generated-6000.json")
test_split_index = int(len(raw_data) * TEST_SPLIT)
train_dataset = TranslationDataset(raw_data[test_split_index:])
validate_dataset = TranslationDataset(raw_data[:test_split_index])

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True
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
model.save_pretrained("./model")