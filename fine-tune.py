# Original notebook - https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing
# Installs Unsloth, Xformers (Flash Attention) and all other packages!

# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# # Model and Tokenizer

# +
from unsloth import FastLanguageModel
import torch
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048
dtype = None
load_in_4bit = True
model = "unsloth/codellama-7b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

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
def formatting_prompts_func(examples: dict):
    prompt_responses = []
    for example in examples["prompt-response"]:
        prompt_responses.append(
            example + tokenizer.eos_token,
        )
    return {"prompt-responses": prompt_responses}


dataset = datasets.load_dataset("adgefficiency/energy-py-linear", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# dataset_te = load_dataset("adgefficiency/energy-py-linear", split="test")
# dataset_te = dataset_te.map(formatting_prompts_func, batched=True)

# -
# # Training

# +
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # eval_dataset=dataset_te,
    eval_dataset=dataset,
    dataset_text_field="prompt-response",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Set num_train_epochs = 1 for full training runs
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

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
# Let's run the model! You can change the instruction and input - leave the output blank!

# +
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [dataset["prompt"][0]],
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for response in decoded_outputs:
    print(response)
    print(dataset["response"][0])
# -
# Evaluate
# +
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# -
# # GGUF / llama.cpp Conversion

# +
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q5_k_m")
