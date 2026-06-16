import evaluate
import os
import re

# Load CHRF metric (CHRf++ is word_order=2)
chrfpp = evaluate.load("chrf")

# Quick test
results = chrfpp.compute(
    predictions=["text"],
    references=[["gold"]],
    word_order=2
)
print(results["score"])

# Input and output directories
input_dir = "/home/common/ACNLP/eng2nav/umr-prompt/bertscore_results"
output_dir = "/home/common/ACNLP/eng2nav/umr-prompt/bertscore_results/chrf++_scores"

input_files = [
    "eng_engumr_nav_output.txt",
    "eng_nav_output.txt",
    "eng_engumr_nav_navumr_output.txt",
    "eng_nav_navumr_output.txt"
]

for input_file in input_files:
    input_path = os.path.join(input_dir, input_file)
    output_file = f"CHRFSCORES++_{input_file}"
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    scores = []
    output_lines = []
    entries = content.split("File: ")[1:]

    for entry in entries:
        lines = entry.strip().split('\n')
        gold = None
        translation = None

        for line in lines:
            if line.startswith("Gold: "):
                gold = line.replace("Gold: ", "").strip()
            elif line.startswith("Translation: "):
                translation = line.replace("Translation: ", "").strip()

        if gold and translation:
            result = chrfpp.compute(
                predictions=[translation],
                references=[[gold]],   # <--- list of lists
                word_order=2           # <--- activate CHRF++
            )
            score = result["score"]
            scores.append(score)

            output_lines.append(f"SCORE: {score:.3f}")
            output_lines.append(f"GOLD: {gold}")
            output_lines.append(f"PRED: {translation}")
            output_lines.append("")

    if scores:
        average = sum(scores) / len(scores)
        output_lines.append(f"AVERAGE_CHRF++: {average:.3f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Processed {input_file}: {len(scores)} pairs, Average CHRF++ = {average:.4f}")

print("All files processed!")

