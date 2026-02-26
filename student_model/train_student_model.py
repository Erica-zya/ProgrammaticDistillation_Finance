import sys
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 获取项目根目录 (假设该脚本在 student_model/ 文件夹下)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

PROMPT_TMPL = """Write a Python program that answers the QUESTION using only TABLE and PARAGRAPHS.
Rules:
- Use ONLY the given TABLE and PARAGRAPHS.
- Infer and apply any scale/unit ONLY if explicitly stated.
- Output MUST be raw Python code only.
- End with a single print(...) of the final answer.

TABLE:\n{table}\nPARAGRAPHS:\n{paras}\nQUESTION:\n{question}\nGOLD_DERIVATION:\n{derivation}"""

def load_and_merge_data(raw_json_path, teacher_jsonl_path):
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    context_map = {}
    for doc in raw_data:
        table_text = "\n".join(" | ".join(map(str, r)) for r in doc.get("table", {}).get("table", []))
        paras_text = "\n".join(p.get("text", "") for p in doc.get("paragraphs", []))
        
        for q in doc.get("questions", []):
            qid = str(q.get("uid"))
            context_map[qid] = {
                "table": table_text,
                "paras": paras_text,
                "question": q.get("question", ""),
                "derivation": q.get("derivation", "")
            }

    formatted_data = {"text": []}
    with open(teacher_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            qid = row["qid"]
            if qid in context_map:
                ctx = context_map[qid]
                user_prompt = PROMPT_TMPL.format(**ctx)
                
                messages = [
                    {"role": "system", "content": "Return ONLY raw Python code. No Markdown fences."},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": row["generated_code"]}
                ]
                formatted_data["text"].append(messages)
                
    return Dataset.from_dict(formatted_data)

def main():
    # 1. 动态获取数据路径
    dataset_path = os.path.join(PROJECT_ROOT, "dataset_filtered", "tatqa_dataset_train_filtered.json")
    filtered_preds_path = os.path.join(PROJECT_ROOT, "distillation_data", "teacher_codegen_train_results_filtered.jsonl")

    print(f"Loading raw dataset from: {dataset_path}")
    print(f"Loading filtered predictions from: {filtered_preds_path}")
    
    dataset = load_and_merge_data(raw_json_path=dataset_path, teacher_jsonl_path=filtered_preds_path)
    print(f"Successfully loaded {len(dataset)} training samples.")

    # 2. 模型加载与 4-bit 量化
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def apply_template(example):
        example["text"] = tokenizer.apply_chat_template(example["text"], tokenize=False, add_generation_prompt=False)
        return example

    dataset = dataset.map(apply_template)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. 训练参数与路径设置
    student_outputs_dir = os.path.join(PROJECT_ROOT, "student_model", "outputs")
    os.makedirs(student_outputs_dir, exist_ok=True)
    
    output_model_dir = os.path.join(student_outputs_dir, "qwen2.5-7b-tatqa-lora")
    logging_dir = os.path.join(student_outputs_dir, "logs")

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
        peft_config=lora_config
    )

    # 5. 开始训练
    print("Starting training...")
    trainer.train()

    # 6. 导出日志与保存模型
    log_history = trainer.state.log_history
    log_file_path = os.path.join(student_outputs_dir, "training_logs.json")

    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=4)
        
    print(f"Training logs specifically saved to {log_file_path}")

    final_model_dir = os.path.join(student_outputs_dir, "qwen2.5-7b-tatqa-lora-final")
    trainer.model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Training complete! Final model saved to {final_model_dir}")

if __name__ == "__main__":
    main()