import torch.utils
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from typing import List
import random
import torch
from data_generation import read_json

random.seed(12345)
MODEL = "google/gemma-2-2b"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = AutoModel.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
        
        self.tokenize_translation(data=en_to_slang, en_to_slang=True)
        self.tokenize_translation(data=slang_to_en, en_to_slang=False)

    def tokenize_translation(self, data: List, en_to_slang: bool):
        
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
            
            tokenized = tokenizer(concatenated, max_length=1024, padding=True, truncation=True)
            self.input_ids.append(tokenized['input_ids'])
            self.labels.append(tokenized['input_ids'])
            self.masks.append(tokenized['attention_mask'])

    def __get_item__(self, id):
        return dict(input_ids=self.input_ids[id], labels=self.labels[id], attention_mask=self.mask[id])
    
    def __len__(self):
        return len(self.input_ids)

raw_data = read_json("./data/generated-6000.json")
dataset = TranslationDataset(raw_data)

# training_args = TrainingArguments(
#     output_dir="model",
#     learning_rate=2e-3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=8,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()