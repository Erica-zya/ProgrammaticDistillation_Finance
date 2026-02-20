# Gold-Guided Programmatic Distillation for Verifiable Financial QA (TAT-QA)
Stanford CS224 Natural Language Processing with Deep Learning Final Project Course Website: https://web.stanford.edu/class/cs224n/index.html

## Overview
Financial disclosures are long, dense, and computation-heavy. While frontier LLMs can answer many finance questions, they are expensive to run and prone to arithmetic errors or unsupported statements. This project investigates whether a smaller language model can approach large-model performance on financial question answering over hybrid tables and text by learning executable reasoning.

We propose Gold-Guided Programmatic Distillation (GGPD), a PaD-style teacher–student framework that distills Program-of-Thought supervision rather than natural-language Chain-of-Thought. For arithmetic questions, a large teacher generates executable Python programs guided by gold symbolic derivations (from TAT-QA), and we retain only samples that pass execution-based verification. A smaller student is then fine-tuned to generate the verified programs conditioned on the question and context, enabling verifiable numeric reasoning via an external interpreter.

## Dataset
TAT-QA (Table-and-Text Question Answering)
Original Paper / Dataset: https://github.com/NExTplusplus/TAT-QA

### Answer Types
TAT-QA contains heterogeneous supervision:
- **Arithmetic / computation-heavy** (ratios, growth rates, sums, margins, etc.)
- **Non-arithmetic** (extractive spans / short grounded textual answers)

Our pipeline applies program synthesis + execution filtering primarily to arithmetic instances.

## Method Summary
### Teacher Phase (Gold-Guided Data Synthesis)
- Teacher: Qwen2.5-72B-Instruct (data generation only)
- Inputs: question and hybrid context (table + associated text) + gold symbolic derivation (when available)
- Output: a Python function that implements the derivation with explicit unit/scale normalization when required
- Verification: run the program in a sandboxed Python interpreter and compare against gold answer with tolerance (default: `1e-4`)
- Keep only verified samples

### Student Phase (Fine-Tuning)
- Student: Qwen2.5-7B-Instruct
- Training: SFT + LoRA
- Targets:
  - Arithmetic: verified Python programs (Program-of-Thought)
  - Non-arithmetic: grounded short textual answers (optional evidence span)
- Inference:
  - Arithmetic: generate program → execute → return numeric answer
  - Non-arithmetic: generate answer directly

## Repository Structure
