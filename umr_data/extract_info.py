import re

# File name template
FILE_TEMPLATE = "navajo_umr-{:04d}.umr"

# Output files
NAV_FILE = "nav_sents.txt"
ENG_FILE = "eng_sents.txt"
UMR_FILE = "nav_umrs.txt"

nav_sents = []
eng_sents = []
umr_graphs = []

for file_num in range(1, 6): 
    file_name = FILE_TEMPLATE.format(file_num)
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        continue

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- Navajo sentence ---
        if line.startswith("Words:"):
            navajo_sentence = re.sub(r"^Words:\s+", "", line)
            # remove all spaces in Navajo sentence
            navajo_sentence = navajo_sentence.replace(" ", "")
            nav_sents.append(navajo_sentence)

        # --- English translation ---
        elif line.startswith("Translation(English):"):
            english_sentence = re.sub(r"^Translation\(English\):\s+", "", line)
            eng_sents.append(english_sentence)

        # --- UMR graph ---
        elif line.startswith("# sentence level graph:"):
            umr_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("# alignment:"):
                umr_lines.append(lines[i].rstrip("\n"))
                i += 1
            umr_graphs.append("\n".join(umr_lines))
            continue  # skip extra i+=1 below

        i += 1

# Write all outputs at once
with open(NAV_FILE, "w", encoding="utf-8") as f:
    for sent in nav_sents:
        f.write(sent + "\n")

with open(ENG_FILE, "w", encoding="utf-8") as f:
    for sent in eng_sents:
        f.write(sent + "\n")

with open(UMR_FILE, "w", encoding="utf-8") as f:
    for graph in umr_graphs:
        f.write(graph + "\n\n")

