from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer
import torch
from transformers import AutoConfig

import bitsandbytes as bnb
print(bnb.__version__)

# === Config ===
model_name = "mistralai/Mistral-7B-v0.1"
dataset_path = "sigma_train_from_files.json"
output_dir = "mistral-sigma-lora"

# === Load tokenizer ===
#tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    low_cpu_mem_usage=True
    #max_memory={0: "6GiB"} 
)

# === Configure LoRA ===
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Specific to Mistral
    inference_mode=False
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === Load dataset ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

# === Format prompt/response into a single field ===
def formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

dataset = dataset.map(lambda x: {"text": formatting_func(x)})

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,  # Set to False if your GPU doesn't support bf16
    report_to="none"
)

# === Train ===
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()

# === Save ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

for name, param in model.named_parameters():
    if param.device.type == 'meta':
        print(f"{name} is on meta device")