input_file = "gpt_chrf_withgloss_results.txt"
output_file = "gpt_chrf_nogloss_results_reformatted.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0

while i < len(lines):
    line = lines[i]

    if line.startswith("English:"):
        out.append(line)
        out.append("Gold:\n")
        i += 1
    else:
        out.append(line)
        i += 1

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(out)
