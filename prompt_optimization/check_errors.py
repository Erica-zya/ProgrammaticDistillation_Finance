import dspy
from pathlib import Path

from optimize_prompt import FinancialCodegen, load_data, validate_code_metric, SAMPLE_PATH
from teacher_model.run_generated_code import exec_with_timeout

def run_debug_eval():
    lm = dspy.LM("huggingface/Qwen/Qwen2.5-72B-Instruct")
    dspy.configure(lm=lm)

    # 2. load in prompt
    compiled_program = dspy.Predict(FinancialCodegen)
    prompt_path = Path(__file__).resolve().parent / "best_financial_prompt.json"
    compiled_program.load(prompt_path)

    # 3. load in the 11 samples 
    dataset = load_data(SAMPLE_PATH)

    print(f"\n{'='*20} ERROR ANALYSIS {'='*20}\n")
    
    for i, example in enumerate(dataset):
        # Predict
        prediction = compiled_program(
            table=example.table,
            paragraphs=example.paragraphs,
            question=example.question,
            scale=example.scale
        )

        # Grade it
        is_correct = validate_code_metric(example, prediction)
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"Sample {i+1}: {example.question}")
        print(f"Status: {status}")
        
        if not is_correct:
            print("-" * 10 + " Debug Info " + "-" * 10)
            print(f"Expected Answer: {example.answer}")
            print(f"Generated Program:\n{prediction.program}")
            stdout, error, status = exec_with_timeout(prediction.program)
            print(f"Actual Output: {stdout.strip()}, Error: {error}, Status: {status}")
            print("-" * 32)
        print("\n")

if __name__ == "__main__":
    run_debug_eval()