# Gen+ Translator 🧠🚽
World's first bi-drectional brainrot translator.

<i>Youwei Zhen 2024</i>

### Data Collection
Gen+ Translator is what can be considered as a distilled model. The definition of all slang words was taken from [List of Generation Z slang - Wikipedia](https://en.wikipedia.org/wiki/List_of_Generation_Z_slang). Using a locally ran LLM [mistral-nemo](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407), precisely 6,000 datasets of translations were generated. These translations included the most common topics in day to day life (reference topics.json). Additionally, each translation included the slang words used.

These 6,000 datasets were then split into half, where half were english to slang and the other were slang to english.

### Training
Gen+ Translator is finetuned based on [gpt2-large](https://huggingface.co/openai-community/gpt2-large) and trained on 2x NVIDIA RTX 3090. The model was finetuned using PeFT and Lora to reduce computations.

### Setup
1. Create a python virtual environment (optional):
```
python -m venv venv
```
2. Install the required dependencies: 
```
pip install -r requirements.txt
```
3. Change parameters in the .env (yes, I know I committed the .env, because it is being used as a config):
```
API_ENDPOINT="http://localhost:11434/api/generate" <- Using ollama for running model locally
MODEL_NAME="mistral-nemo:latest"
FINETUNE_MODEL="gpt2-large"
DEVICE="cuda:0"
```
4. Generating data. Running data_generation.py will use the local LLM to generate 6,000 datasets with 10 threads. To change these parameters please modify the file.
```
python data_generation.py
```
5. Run finetune.py. If your CUDA runs out of memory, adjust the training parameters inside the file.
```
python finetune.py
```
The finetuned Peft will be saved inside ./adapter

