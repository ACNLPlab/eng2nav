import json
from pathlib import Path
import re

# Paths
json_path = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_withgloss.json"
output_dir = Path("/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_final_results")
gpt_with_gloss_PROMPTS = output_dir / "gpt_bertscore_withgloss_PROMPTS_pt2.txt"

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

# --- Main script ---
def main():
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Original dictionaries: {len(data)}")
    print(f"Will generate {len(data)} prompts")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save prompts
    with open(gpt_with_gloss_PROMPTS, "w", encoding="utf-8") as f_prompts:
        for idx, item in enumerate(data, start=1):
            # Build 5-shot examples from THIS item's eng1-5, nav1-5, morpho1-5
            with_gloss_examples = []
            
            for i in range(1, 6):
                eng = item[f"eng{i}"]
                nav = item[f"nav{i}"]
                morpho = item[f"morpho{i}"]
                mg = format_with_gloss(morpho)
                with_gloss_examples.append(f"English: {eng}\nNavajo: {nav}\n{mg}")
            
            # Build complete prompt
            with_gloss_prompt_base = "\n\n".join(with_gloss_examples) + "\n\n"
            reference = item['reference']
            prompt = with_gloss_prompt_base + f"English: {reference}\nNavajo: "
            
            # Save to file
            f_prompts.write(f"=== EXAMPLE {idx} ===\n")
            f_prompts.write(prompt + "\n\n")
            f_prompts.write("="*80 + "\n\n")
            f_prompts.flush()
            
            # Print first 3 prompts to console for quick verification
            if idx <= 3:
                print(f"\n{'='*80}")
                print(f"PROMPT {idx}:")
                print(f"{'='*80}")
                print(prompt)
                print(f"\nReference to translate: {reference}")
                print(f"{'='*80}\n")
    
    print(f"\n✅ Done! Generated {len(data)} prompts")
    print(f"  Prompts saved to: {gpt_with_gloss_PROMPTS}")
    print(f"\n📝 Check the file to verify all prompts are correct")
    print(f"   First 3 prompts were printed above for quick review")

if __name__ == "__main__":
    main()
