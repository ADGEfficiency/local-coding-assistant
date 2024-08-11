import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def create_conversation(sample: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Show the location of the source code."},
            {"role": "user", "content": sample["code"]},
            {"role": "assistant", "content": sample["file"]},
        ]
    }


def tokenize(conversation: dict, tokenizer: AutoTokenizer) -> dict:
    """
    adaped from https://huggingface.co/spaces/codellama/codellama-13b-chat/blob/main/model.py#L25-L36

    """
    messages = conversation["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    prompt = [
        f"{tokenizer.eos_token}<s>[INST] <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n",
        f"{messages[1]['content']} [/INST]",
    ]

    response = f"<s>{messages[2]['content'].strip()}</s>{tokenizer.eos_token}"

    sample = tokenizer(
        prompt, padding=True, max_length=64, truncation=True, return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            response, padding=True, max_length=16, truncation=True, return_tensors="pt"
        )
        sample["labels"] = labels["input_ids"]
    return sample


name = "codellama/CodeLlama-13b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
model = AutoModelForCausalLM.from_pretrained(
    name,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

dataset = {"code": ["def hello()"], "file": ["./hello.py"]}

train = datasets.Dataset.from_dict(dataset)
train = train.map(create_conversation, batched=False)
train = train.map(
    lambda batch: tokenize(batch, tokenizer),
    batched=False,
    remove_columns=["code", "file", "messages"],
)
train.set_format(
    "pt", columns=["input_ids", "labels", "attention_mask"], output_all_columns=True
)

model.eval()
for sample in train.shuffle().select(range(1)):
    print(sample)
    print(
        f"Input: {tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=True)}"
    )
    print(
        f"Labels: {tokenizer.batch_decode(sample['labels'], skip_special_tokens=True)}"
    )

    # evaluate the base model
    with torch.no_grad():
        response_tokens = model.generate(**sample, max_new_tokens=100)
        response = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
        print(f"Response: {tokenizer.batch_decode(response, skip_special_tokens=True)}")
