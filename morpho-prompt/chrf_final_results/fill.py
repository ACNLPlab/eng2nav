gold_file = "gpt_chrf_withgloss_results_pt2_reformatted.txt"
morpho_file = "morpho-nav-sents.txt"
output_file = "gpt_chrf_withgloss_results_pt2_filled.txt"

# Read morpho sentences
with open(morpho_file, "r", encoding="utf-8") as f:
    morpho_sents = [line.strip() for line in f if line.strip()]

gold_index = 0
output_lines = []

with open(gold_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() == "Gold:":
            if gold_index >= len(morpho_sents):
                raise ValueError("Not enough morpho sentences to fill all Gold lines.")
            output_lines.append("Gold: ")
            output_lines.append(morpho_sents[gold_index] + "\n")
            gold_index += 1
        else:
            output_lines.append(line)

if gold_index < len(morpho_sents):
    raise ValueError("Unused morpho sentences remain after filling all Gold lines.")

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(output_lines)

