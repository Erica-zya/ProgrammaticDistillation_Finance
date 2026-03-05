import json
import os
from pathlib import Path
import modal
import time




# test N data
N = 32
# image: install packages needed for tokenizer
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "peft",
    "huggingface_hub",
)

app = modal.App("finance-train-lora", image=image)

finance_vol = modal.Volume.from_name("finance-data")
hf_cache_vol = modal.Volume.from_name("hf-cache")

PROMPT_TMPL = """Write a Python program that answers the QUESTION.

Sources of truth:
- TABLE and PARAGRAPHS are the ONLY sources of facts and numbers.
- Determine the unit/scale ONLY from explicit cues in TABLE, QUESTION, or PARAGRAPHS.

Strict output rules:
- Output MUST be raw Python code only. Do NOT use Markdown fences.
- Do NOT print anything except the final answer.
- End with EXACTLY ONE print(...) statement (the very last line).

Answer requirements:
1) Compute the final answer and store it in a variable named ans.
2) ans MUST be either:
   - a string, OR
   - a list of strings (for questions with multiple valid answers).
3) If the QUESTION can have multiple answers (e.g., "which years", "list all", filtering conditions), set ans to a list of strings containing ALL matching answers.
4) Do not include duplicates in list answers.

Scale requirements (must match exactly one of these 5):
- The JSON field "scale" MUST be exactly one of: "", "thousand", "million", "billion", "percent".

Final output format:
- Create a dictionary named out with exactly these keys:
  out = {{"ans": ..., "scale": ...}}
- Print EXACTLY one line of valid JSON.
- The LAST line of the program MUST be:
  print(json.dumps(out, ensure_ascii=False))

TABLE:
{table}

PARAGRAPHS:
{paras}

QUESTION:
{question}
"""


def build_prompt(example: dict) -> str:
    return PROMPT_TMPL.format(
        table=example.get("table", ""),
        paras=example.get("paragraphs", ""),
        question=example.get("question", ""),
    )

@app.function(
    gpu="L4",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=3600,
)
def train_lora_smoke():
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    t0 = time.time()

    train_path = Path("/root/finance-data/data/student_train_from_teacher.jsonl")
    out_dir = Path("/root/finance-data/outputs/lora_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    # read jsonl to data [sample1, sample2, sample 3,...]
    with train_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # test:take the first N sample
    data = data[:N]
    print("num smoke samples:", len(data))
    print("time: read data =", round(time.time() - t0, 2), "s")

    # load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    local_model_path = snapshot_download(
        repo_id=model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )
    print("local model path:", local_model_path)

    # tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("time: load model + lora =", round(time.time() - t0, 2), "s")

    # process
    max_seq_length = 2048

    def preprocess(example):
        prompt = build_prompt(example)
        target = example.get("generated_code", "")

        
        prompt_text = prompt + "\n"
        target_text = target + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        #label_pad_token_id (int, optional, defaults to -100) — The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        # 
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    tokenized_data = [preprocess(x) for x in data]

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    train_dataset = SimpleDataset(tokenized_data)

    def collate_fn(features):
        # find maximum length in the batch

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, labels, attention_mask = [], [], []
    
        for f in features:
            # calculate padding
            pad_len = max_len - len(f["input_ids"])

            input_ids.append(
                f["input_ids"] + [tokenizer.pad_token_id] * pad_len
            )
            labels.append(
                f["labels"] + [-100] * pad_len
            )
            attention_mask.append(
                f["attention_mask"] + [0] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # test whether we can train on the selected gpu
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=5,
        logging_steps=1,
        save_steps=5,
        save_strategy="steps",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    t_train0 = time.time()
    trainer.train()
    train_sec = time.time() - t_train0
    print("time: train only =", round(train_sec, 2), "s")
    print("time: total =", round(time.time() - t0, 2), "s")

    # save adapter
    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # save info
    summary = {
        "num_samples": len(data),
        "max_steps": 5,
        "output_dir": str(out_dir),
        "train_sec": round(train_sec, 2),
        "total_sec": round(time.time() - t0, 2),
    }
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    finance_vol.commit()
    hf_cache_vol.commit()


@app.local_entrypoint()
def main():
    train_lora_smoke.remote()