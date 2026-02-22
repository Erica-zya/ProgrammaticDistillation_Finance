import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

response = query({
    "messages": [
        {"role": "system", "content": "Be concise."},
        {"role": "user","content": "What is the capital of France?"},
    ],
    "temperature": 0,
    "max_tokens": 100,
    "model": "Qwen/Qwen2.5-72B-Instruct:novita"
})

print(f"ans: {response["choices"][0]["message"]["content"]}")
print(f"total tokens: {response.get("usage")["total_tokens"]}")