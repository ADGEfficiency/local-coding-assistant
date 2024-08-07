# Original notebook - https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing
# Installs Unsloth, Xformers (Flash Attention) and all other packages!

# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
# !pip install wandb -U

# +
from unsloth import FastLanguageModel
import torch
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import wandb
import shutil

import pathlib


def format_prompt(examples: dict):
    model_inputs = tokenizer(
        examples["input"], max_length=512, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"], max_length=512, truncation=True, padding="max_length"
        )["input_ids"]
        model_inputs["labels"] = labels

    return model_inputs


def latest_checkpoint(output_dir: pathlib.Path):
    """Finds the latest checkpoints if any exist."""
    checkpoints = [
        cp for cp in output_dir.iterdir() if cp.is_dir() and "checkpoint-" in cp.name
    ]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda cp: int(cp.name.split("-")[-1]))
        return str(latest_checkpoint)
    return None


def get_output_dir():
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        output_dir = pathlib.Path("./drive/MyDrive/fine-tune-codellama/")
    except:
        output_dir = pathlib.Path("./experiments/fine-tune-codellama")

    output_dir.mkdir(exist_ok=True)
    return output_dir


output_dir = get_output_dir()

wandb.login()
run = wandb.init(project="energy-py-linear-codellama-v2", reinit=True)

evaluate_after_training = False
save_after_training = True

# -
# # Model and Tokenizer

# +
max_seq_length = 1024
dtype = None
load_in_4bit = True
model = "unsloth/codellama-7b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
# -
# # Load Dataset

# +
dataset = datasets.load_dataset("adgefficiency/energy-py-linear", split="train")
dataset = dataset.map(format_prompt)
dataset_te = datasets.load_dataset("adgefficiency/energy-py-linear", split="test")
dataset_te = dataset_te.map(format_prompt)

# -
# # Training

# +
from callbacks import *

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset_te,
    dataset_text_field="prompt-responses",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id
    ),
    args=TrainingArguments(
        bf16=is_bfloat16_supported(),
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        fp16=not is_bfloat16_supported(),
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=11,
        lr_scheduler_type="linear",
        num_train_epochs=5,
        optim="adamw_8bit",
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        seed=3407,
        warmup_steps=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        remove_unused_columns=True
    ),
)
cb = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=dataset_te,
    num_samples=10,
    freq=1,
)
trainer.add_callback(cb)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir))

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
# -
# # Inference

# +
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer([dataset["prompt"][0]], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for response in decoded_outputs:
    print(f"{response=}")
    print(f"{dataset['output'][0]=}")
# -
# # Evaluate

# +
if evaluate_after_training:
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
# -
# # GGUF / llama.cpp Conversion

# +
if save_after_training:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q5_k_m")
    source_path = "model/unsloth.Q5_K_M.gguf"
    destination_path = output_dir / source_path
    pathlib.Path(output_dir / "model").mkdir(exist_ok=True)
    shutil.copy(source_path, destination_path)
