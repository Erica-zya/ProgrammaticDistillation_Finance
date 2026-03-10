"""
LoRA hyperparameter sweep on 500 samples from student_train_from_teacher.jsonl.
Run with CLI args to try different settings; each run saves to a unique dir and logs config + final loss.

Suggested order to sweep:
  1) --lr
  2) --lora_r and --lora_alpha
  3) --num_epochs
  4) --per_device_batch_size and --gradient_accumulation_steps
  5) --lora_dropout

Example:
  modal run finetune_test/modal_train_lora_sweep.py
  modal run finetune_test/modal_train_lora_sweep.py --lr 0.0001
  modal run finetune_test/modal_train_lora_sweep.py --lora_r 16 --lora_alpha 32
  modal run finetune_test/modal_train_lora_sweep.py --num_epochs 2
  modal run finetune_test/modal_train_lora_sweep.py --per_device_batch_size 2 --gradient_accumulation_steps 2
  modal run finetune_test/modal_train_lora_sweep.py --lora_dropout 0.1

Each run writes to finance-data volume at outputs/sweep/<tag>/ with adapter. All loss PNGs go to outputs/sweep_plots/training_loss_<tag>.png and all result JSONs to outputs/sweep_results/sweep_result_<tag>.json (tag = params in filename). Download:  modal volume get finance-data outputs/sweep_plots finetune_test/sweep_plots  &&  modal volume get finance-data outputs/sweep_results finetune_test/sweep_results
"""
import json
import random
import time
from pathlib import Path

import modal

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.37.0",
    "sentencepiece",
    "accelerate",
    "peft",
    "huggingface_hub",
    "matplotlib",
)

app = modal.App("finance-train-lora-sweep", image=image)
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


def _make_tag(lr, lora_r, lora_alpha, num_epochs, batch_size, grad_accum, lora_dropout) -> str:
    return (
        f"lr{lr:.0e}_r{lora_r}_a{lora_alpha}_ep{num_epochs}_bs{batch_size}_ga{grad_accum}_drop{lora_dropout}"
    ).replace(".", "_")


@app.function(
    gpu="A100-80GB",
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=3600,
)
def train_lora_sweep(
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_dropout: float = 0.05,
    max_samples: int = 500,
    seed: int = 42,
):
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    t0 = time.time()
    tag = _make_tag(lr, lora_r, lora_alpha, num_epochs, per_device_batch_size, gradient_accumulation_steps, lora_dropout)
    out_dir = Path(f"/root/finance-data/outputs/sweep/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("/root/finance-data/data/student_train_from_teacher.jsonl")
    with train_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    if max_samples and len(data) > max_samples:
        data = random.sample(data, max_samples)
    print(f"Training on {len(data)} samples (seed={seed})")
    print(f"Config: lr={lr}, r={lora_r}, alpha={lora_alpha}, epochs={num_epochs}, bs={per_device_batch_size}, ga={gradient_accumulation_steps}, dropout={lora_dropout}")

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    local_model_path = snapshot_download(
        repo_id=model_name,
        cache_dir="/root/.cache/huggingface",
        local_files_only=True,
    )

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

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    max_seq_length = 2048

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
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

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
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = [f["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(f["input_ids"])) for f in features]
        labels = [f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features]
        attention_mask = [f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        max_steps=-1,
        logging_steps=1,
        save_steps=500,
        save_strategy="no",
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

    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # Final loss and plot from in-memory log (trainer_state.json may not exist when save_strategy="no")
    final_loss = None
    loss_entries = [x for x in getattr(trainer.state, "log_history", []) if "loss" in x]
    if loss_entries:
        final_loss = loss_entries[-1]["loss"]
    steps = [x["step"] for x in loss_entries]
    losses = [x["loss"] for x in loss_entries]
    if steps and losses:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots_dir = Path("/root/finance-data/outputs/sweep_plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, losses, marker="o", markersize=3)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training loss ({tag})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = plots_dir / f"training_loss_{tag}.png"
        plt.savefig(png_path, dpi=120)
        plt.close()
        print("Saved plot to", png_path)

    config = {
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_dropout": lora_dropout,
        "max_samples": len(data),
        "seed": seed,
    }
    result = {
        "tag": tag,
        "config": config,
        "final_loss": final_loss,
        "train_sec": round(train_sec, 2),
        "total_sec": round(time.time() - t0, 2),
        "output_dir": str(out_dir),
    }
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    (out_dir / "sweep_result.json").write_text(result_json, encoding="utf-8")
    # Also save to shared folder with param-based name (like PNGs)
    results_dir = Path("/root/finance-data/outputs/sweep_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"sweep_result_{tag}.json").write_text(result_json, encoding="utf-8")

    print("\n=== Sweep run result ===")
    print(json.dumps(result, indent=2))
    print("========================")

    finance_vol.commit()
    hf_cache_vol.commit()
    return result


@app.local_entrypoint()
def main(
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_dropout: float = 0.05,
    max_samples: int = 500,
    seed: int = 42,
):
    """CLI args are passed to the remote sweep. Use --lr 0.0001 for 1e-4."""
    train_lora_sweep.remote(
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_epochs=num_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_dropout=lora_dropout,
        max_samples=max_samples,
        seed=seed,
    )
