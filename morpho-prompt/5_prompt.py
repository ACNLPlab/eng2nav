import json
from pathlib import Path
from openai import OpenAI
import re

# Paths
json_path = "/home/common/ACNLP/eng2nav/morpho-prompt/chrf_withgloss.json"
output_dir = Path("/home/common/ACNLP/eng2nav/morpho-prompt/chrf_base_results")
gpt_with_gloss_results = output_dir / "gpt_chrf_withgloss_results.txt"
gpt_with_gloss_PROMPTS = output_dir / "gpt_chrf_withgloss_PROMPTS.txt"

# OpenAI client
client = OpenAI(api_key="")
model_name = "gpt-5"

# --- Normalization functions ---
def _normalize_item(item):
    item = item.strip()
    item = re.sub(r"\(nv\):\s*", "(nv), ", item)
    item = re.sub(r"\(en\):\s*", "(en), ", item)
    item = re.sub(r"\s{2,}", " ", item)
    item = re.sub(r",\s*,", ",", item)
    return item.strip()

def _process_with_gloss_item(item):
    return _normalize_item(item)

def format_with_gloss(morpho):
    s = morpho.strip()
    has_brackets = s.startswith("[") and s.endswith("]")
    if has_brackets:
        s = s[1:-1]
    parts = [p for p in s.split("|")]
    processed = [_process_with_gloss_item(p) for p in parts if p.strip()]
    out = " | ".join(processed)
    return f"[{out}]" if has_brackets else out

# --- Translation ---
def run_translation(prompt):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a machine translation system that translates English narrative sentences into Navajo. "
                    "Do not provide explanations, only output the translation. DO NOT RETURN ANY MORPHOSYNTACTIC DATA"
                )
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()

# --- Main script ---
def main():
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Original dictionaries: {len(data)}")
    print(f"Will translate {len(data)} sentences")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run GPT translations
    with open(gpt_with_gloss_results, "w", encoding="utf-8") as f_wg, \
         open(gpt_with_gloss_PROMPTS, "w", encoding="utf-8") as f_prompts:
        
        for idx, item in enumerate(data, start=0):
            # Build 5-shot examples from THIS item's eng1-5, nav1-5, morpho1-5
            with_gloss_examples = []
            
            for i in range(1, 6):
                eng = item[f"eng{i}"]
                nav = item[f"nav{i}"]
                #morpho = item[f"morpho{i}"]
                #mg = format_with_gloss(morpho)
                with_gloss_examples.append(f"English: {eng}\nNavajo: {nav}\n")
            
            # Build complete prompt
            with_gloss_prompt_base = "\n\n".join(with_gloss_examples) + "\n\n"
            reference = item['reference']
            prompt = with_gloss_prompt_base + f"English: {reference}\nNavajo: "
            
            # Save prompt to file
            f_prompts.write(f"=== EXAMPLE {idx} ===\n")
            f_prompts.write(prompt + "\n\n")
            f_prompts.write("="*80 + "\n\n")
            f_prompts.flush()
            
            # Run translation
            try:
                translation = run_translation(prompt)
            except Exception as e:
                translation = f"Error: {e}"
                print(f"\nERROR on example {idx}: {e}")
            
            # Print progress
            print(f"\n--- TRANSLATION {idx}/{len(data)} ---")
            print(f"English: {reference}")
            print(f"Navajo: {translation}\n")
            
            # Write result to file
            f_wg.write(f"English: {reference}\n")
            f_wg.write(f"Translation: {translation}\n\n")
            f_wg.flush()
    
    print(f"\nDone! Translated {len(data)} sentences")
    print(f"  Results: {gpt_with_gloss_results}")
    print(f"  Prompts: {gpt_with_gloss_PROMPTS}")

if __name__ == "__main__":
    main()
