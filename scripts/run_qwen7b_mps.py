import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Load the model weights (first run will download them into HF cache).
# Use fp16 for better speed/memory on Apple MPS, and move the model to MPS (Apple GPU).
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map=None
).to("mps")

tokenizer = AutoTokenizer.from_pretrained(model_name)


def run_one(user_text, max_new_tokens=200):
    """
    Run a single chat-style generation:
    - Build system/user messages
    - Apply Qwen chat template to get the final prompt string
    - Tokenize and move input tensors to MPS
    - Generate output tokens
    - Measure tokens/sec
    - Decode only the newly generated part
    """
    messages = [
        {"role":"system","content":"Be as concise and informative as possible."},
        {"role":"user","content": user_text},
    ]
    # Convert messages into a single prompt string using the model's chat template.
    # add_generation_prompt=True appends the assistant "start" marker so the model knows to answer.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("mps")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
    dt = time.time() - t0

    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"generated {new_tokens} tokens in {dt:.2f}s -> {new_tokens/dt:.1f} tok/s")

    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

print("="*60)
print(run_one("2+2=?", max_new_tokens=120))
print("="*60)
print(run_one("Do you like dog or cat?", max_new_tokens=120))
print("="*60)
print(run_one("Who is Mark Twain?", max_new_tokens=120))
