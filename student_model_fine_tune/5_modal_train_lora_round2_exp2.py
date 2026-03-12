import json
from pathlib import Path
import modal
import time

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "peft",
    "huggingface_hub",
)

app = modal.App("finance-train-lora-round2", image=image)

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
    gpu="RTX-PRO-6000",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=6 * 3600,
)
def train_lora_round2():
    import torch
    from huggingface_hub import snapshot_download
    #from peft import LoraConfig, get_peft_model
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    t0 = time.time()

    #train_path = Path("/root/finance-data/data/student_train_from_teacher.jsonl")
    train_path = Path("/root/finance-data/outputs/lora_round1_no_max_seq_mlp_exp2/outputs/student_train_correct_round2.jsonl")
    out_dir = Path("/root/finance-data/outputs/lora_round2_no_max_seq_mlp_exp2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("train_path exists:", train_path.exists(), str(train_path))
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    with train_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if len(data) == 0:
        raise ValueError("Training data is empty.")

    print("num round2 full-train samples:", len(data))
    print("time: read data =", round(time.time() - t0, 2), "s")


    # load round1 model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    local_model_path = snapshot_download(
        repo_id=model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )
    print("local model path:", local_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    round1_adapter_dir = Path("/root/finance-data/outputs/lora_round1_no_max_seq_mlp_exp2/adapter")
    print("round1_adapter_dir exists:", round1_adapter_dir.exists(), str(round1_adapter_dir))
    if not round1_adapter_dir.exists():
        raise FileNotFoundError(f"Round-1 adapter not found: {round1_adapter_dir}")
    
    model = PeftModel.from_pretrained(
        base_model,
        str(round1_adapter_dir),
        is_trainable=True,
    )

    #

    #lora_config = LoraConfig(
     #   r=8,
      #  lora_alpha=16,
       # lora_dropout=0.05,
        #bias="none",
        #task_type="CAUSAL_LM",
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],#["q_proj", "k_proj", "v_proj", "o_proj"],
    #)


    #model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("time: load base model + round1 adapter =", round(time.time() - t0, 2), "s")

    #max_seq_length = 4800

    def preprocess(example):
        prompt = build_prompt(example)
        target = example.get("generated_code", "")

        prompt_text = prompt + "\n"
        target_text = target + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        #input_ids = input_ids[:max_seq_length]
        #labels = labels[:max_seq_length]
        #attention_mask = attention_mask[:max_seq_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    tokenized_data = [preprocess(x) for x in data]

    lengths = [len(x["input_ids"]) for x in tokenized_data]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    print("min tokenized length:", lengths_sorted[0])
    print("max tokenized length:", lengths_sorted[-1])
    print("p95 tokenized length:", lengths_sorted[min(n - 1, int(0.95 * n))])
    print("p99 tokenized length:", lengths_sorted[min(n - 1, int(0.99 * n))])



    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    train_dataset = SimpleDataset(tokenized_data)

    def collate_fn(features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, labels, attention_mask = [], [], []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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
    train_result = trainer.train()
    train_sec = time.time() - t_train0
    total_sec = time.time() - t0

    print("time: train only =", round(train_sec, 2), "s")
    print("time: total =", round(total_sec, 2), "s")

    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # save trainer log history
    log_history_path = out_dir / "trainer_log_history.json"
    log_history_path.write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # save final metrics returned by trainer.train()
    final_metrics = dict(train_result.metrics) if train_result is not None else {}
    final_metrics["train_sec"] = round(train_sec, 2)
    final_metrics["total_sec"] = round(total_sec, 2)
    (out_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "round": 2,
        "num_samples": len(data),
        "num_train_epochs": 1,
        "output_dir": str(out_dir),
        "adapter_dir": str(out_dir / "adapter"),
        "log_history_path": str(log_history_path),
        "train_sec": round(train_sec, 2),
        "total_sec": round(total_sec, 2),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    finance_vol.commit()
    hf_cache_vol.commit()

    print("saved adapter to:", str(out_dir / "adapter"))
    print("saved log history to:", str(log_history_path))
    print("saved final metrics to:", str(out_dir / "final_metrics.json"))
    print("saved summary to:", str(out_dir / "train_summary.json"))


@app.local_entrypoint()
def main():
    train_lora_round2.remote()