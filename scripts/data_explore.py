import json
import random
from pathlib import Path



# ============================================================
# Utilities for exploring the raw TAT-QA dataset (context-level JSON)
#
# Each JSON file is a list of "contexts". Each context contains:
#   - table: { "uid": str, "table": List[List[str]] }   # 2D table as list of rows
#   - paragraphs: List[ { "uid": str, "order": int, "text": str } ]
#   - questions: List[ question objects ... ]
#
# Goal:
#   1) Quickly inspect dataset structure (keys, types, example contents)
#   2) Count question answer types / sources
#   3) Print representative examples by (answer_type, answer_from, req_comparison)
# ============================================================

def preview_tatqa_json(
    json_path: str,
):
    # Load a context-level TAT-QA JSON file and print the first few contexts for inspection.
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    
    # Load data: a list of contexts
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"\nLoaded: {json_path}")
    print(f"Total contexts: {len(data)}")


    # Inspect a few contexts (table / paragraphs / questions)
    n=0
    # Only preview the first 3 contexts to avoid huge output
    check_n=3
    for ctx in data:
        n+=1
        print("="*80)
        print(f"sample No. {n}")
        print("="*30)
        print("Context:")
        print(ctx.keys())
        # Unpack the three top-level fields of each context
        table=ctx['table']
        para=ctx['paragraphs']
        questions=ctx['questions']
        print(f"Types: table={type(table)}, paragraphs={type(para)}, questions={type(questions)}")
         # ---- Table structure ----
        print("="*30)
        print("table keys:", table.keys())
        print("table[uid]",table["uid"])
        t = table["table"]
        print("t type:", type(t)) # 2D list
        n_row=len(t)
        n_col=len(t[0])
        print(f"number of rows:{n_row}, number of columns:{n_col}")
        print("="*30)
        # ---- Paragraph structure ----
        print("length of paragraphs:",len(para))
        for p in para:
            print(p.keys())
            print(p)
            print("\n")
        # ---- Question structure ----
        print("length of questions",len(questions))
        print("question keys:",questions[0].keys())
        for q in questions:
            print("="*10)
            print(q)
        if n==check_n:
            break



def check_answer_type(json_path: str):
    # Count question distributions by:
    #   - answer_type: span / multi-span / arithmetic / count
    #   - answer_from: table / text / table-text
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    # Load data: a list of contexts
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"\nLoaded: {json_path}")
    print(f"Total contexts: {len(data)}")


    ans_type={}
    ans_from_type={}

    # Iterate all questions across all contexts
    for ctx in data:
        table=ctx['table']
        para=ctx['paragraphs']
        questions=ctx['questions']
        for q in questions:
            # Count answer types
            ans_type_key=q['answer_type']
            ans_type[ans_type_key]=ans_type.get(ans_type_key,0)+1
            # Count answer sources
            ans_from_key=q['answer_from']
            ans_from_type[ans_from_key]=ans_from_type.get(ans_from_key,0)+1

    print(ans_type)
    print(ans_from_type)

def find_example(json_path: str, ans_type_str:str, ans_from_type_str:str,req_compare:bool):
    # Find and print the first example that matches:
    #   - answer_type == ans_type_str
    #   - answer_from == ans_from_type_str
    #   - req_comparison == req_compare
    #
    # This prints:
    #   - full question object
    #   - the associated table (2D list)
    #   - all paragraphs
    #   - question text, gold answer, derivation, relevant paragraphs, and scale
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    # Load data: a list of contexts
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Search contexts until we find a matching question

    for ctx in data:
        table=ctx['table']
        para=ctx['paragraphs']
        questions=ctx['questions']
        for q in questions:
            ans_type=q['answer_type']
            ans_from=q['answer_from']
            compare_flag=q['req_comparison']

            # Match target categories
            if ans_type==ans_type_str and ans_from_type_str==ans_from and compare_flag==req_compare:
                print(f"Answer type: {ans_type_str}, {ans_from_type_str}; Require comparison: {compare_flag}")
                print(q)
                print("="*80)
                # ---- Print table ----
                print("="*10+"Table"+"="*10)
                # print table
                t=table['table']
                for row in t:
                    print(row)
                # ---- Print paragraphs ----
                #print("="*10+"Paragraphs"+"="*10)
                #for p in para:
                 #   print(p['order'])
                  #  print(p['text'])
                # ---- Print selected question fields ----
                print("="*10+"Question"+"="*10)
                print(q['question'])
                print("="*10+"Gold answer"+"="*10)
                print(q['answer'])
                print("="*10+"Derivation"+"="*10)
                print(q['derivation'])
                print("="*10+"Relevant paragraphs"+"="*10)
                print(q['rel_paragraphs'])
                # Stop after printing the first matching example
                print("="*10+"Scale"+"="*10)
                print(q['scale'])
                
                
                
                
                
                return
        # (Optional) uncomment if you only want to search the first context
        #break



def count_nonempty_derivation(json_path: str):
    # Count how many question samples have a non-empty "derivation" field
    # in a context-level TAT-QA JSON file.
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    # Load data: a list of contexts
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"\nLoaded: {json_path}")
    print(f"Total contexts: {len(data)}")

    N_deri=0  # total number of questions whose 'derivation' is non-empty
    # Iterate over all contexts and all questions, and count non-empty derivations
    for ctx in data:
        questions=ctx['questions']
        for q in questions:
            d=q['derivation']
            # Treat an empty string as "no derivation"
            if d!="":
                N_deri+=1
    print(f"{N_deri} samples with nonempty derivation")
    return N_deri
       




    

        

def main():
    # Path to the raw TAT-QA JSON file (context-level)
    json_path = "../dataset_raw/tatqa_dataset_test_gold.json"
    # Uncomment to preview a few contexts
    #preview_tatqa_json(json_path=json_path)

    # Print dataset-level statistics (answer_type / answer_from counts)
    #check_answer_type(json_path=json_path)#

    # Enumerate combinations and print one representative example for each category
    ans_type=['multi-span','span','arithmetic','count']
    ans_from_type=['table','text','table-text']
    
    n=0
    for t in ans_type:
        for s in ans_from_type:
            for f in [True, False]:
                n+=1
                print("="*80)
                print(f"Example {n}")
                find_example(json_path=json_path, ans_type_str=t, ans_from_type_str=s,req_compare=f)


    N=count_nonempty_derivation(json_path)

if __name__ == "__main__":
    main()
