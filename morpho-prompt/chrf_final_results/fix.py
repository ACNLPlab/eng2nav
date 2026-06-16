from pathlib import Path

input_file = Path("gpt_chrf_withgloss_results_deduped.txt")
output_file = Path("gpt_bertscore_english_deduped.txt")

seen_english = set()
kept_blocks = []

with input_file.open(encoding="utf-8") as f:
    content = f.read()

# Split into blocks separated by blank lines
blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

for block in blocks:
    lines = block.splitlines()
    if not lines or not lines[0].startswith("English:"):
        continue

    english = lines[0][len("English:"):].strip()

    if english in seen_english:
        continue  # drop duplicate + its translation

    seen_english.add(english)
    kept_blocks.append(block)

# Write cleaned output
with output_file.open("w", encoding="utf-8") as f:
    f.write("\n\n".join(kept_blocks) + "\n")

print(f"Kept {len(kept_blocks)} unique English–Translation pairs")
print(f"Output written to {output_file}")

