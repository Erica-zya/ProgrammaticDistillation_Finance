###
# 1. install model
# pip3 install modal
# modal setup
# 
# 2. create volumn
# modal volume create hf-cache
# modal volume create finance-data
# 
# 3. upload training file
# modal volume put finance-data "./student_train_from_teacher.jsonl" /data/
# This will create a "data" folder under "finance-data", and upload "/student_train_from_teacher.jsonl" under "data"
###

import json
from pathlib import Path
import modal



#  image: install packages needed for tokenizer
image = modal.Image.debian_slim().pip_install(
    "transformers>=4.37.0",
    "sentencepiece",
)

# attach image to app
app = modal.App("finance-tokenizer-smoke", image=image)

# existing volumes
finance_vol = modal.Volume.from_name("finance-data")
hf_cache_vol = modal.Volume.from_name("hf-cache")


# Prompt template
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

# decoration: Remote function: runs on Modal
@app.function(
    volumes={
        "/root/finance-data": finance_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=120,
)

def check_tokenizer():
    from transformers import AutoTokenizer

    sample_path = Path("/root/finance-data/data/student_train_from_teacher.jsonl")
    out_dir = Path("/root/finance-data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # read training data
    with sample_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print("num samples:", len(data))
    print("first keys:", list(data[0].keys()))

    first = data[0]
    prompt = PROMPT_TMPL.format(
        table=first.get("table", ""),
        paras=first.get("paragraphs", ""),
        question=first.get("question", ""),
    )
    target_code = first.get("generated_code", "")

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


    prompt_enc = tokenizer(prompt)
    target_enc = tokenizer(target_code)

    #prompt_enc = tokenizer(prompt, truncation=True, max_length=2048)
    #target_enc = tokenizer(target_code, truncation=True, max_length=1024)

    print("prompt token len:", len(prompt_enc["input_ids"]))
    print("target token len:", len(target_enc["input_ids"]))

    print("\n=== PROMPT PREVIEW ===")
    table_idx = prompt.find("TABLE:")
    if table_idx != -1:
        print(prompt[table_idx:table_idx + 800])
    else:
        print(prompt[:800])

    print("\n=== TARGET CODE PREVIEW ===")
    print(target_code[:800])

    (out_dir / "tokenizer_ok.txt").write_text("ok\n", encoding="utf-8")
    finance_vol.commit()
    hf_cache_vol.commit()




# similar to 
# if __name__ == "__main__":
#    main()
@app.local_entrypoint()
def main():
    check_tokenizer.remote()
