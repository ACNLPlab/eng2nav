import re

# -----------------------------
# Step 0: Load morpho-data.txt
# -----------------------------
with open("morpho-data.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

# Build morpho_dict: English sentence → {"nv": navajo_sent, "words": [...]}
morpho_dict = {}
current_nv = ""
current_eng = ""
current_words = []
current_word = {}

for line in lines:
    if line.startswith("Sentence (nv):"):
        current_nv = line.replace("Sentence (nv):", "").strip()
        current_words = []
    elif line.startswith("Sentence gloss (en):"):
        current_eng = line.replace("Sentence gloss (en):", "").strip()
    elif line.startswith("Word (nv):"):
        if current_word:
            current_words.append(current_word)
        current_word = {"nv": line.replace("Word (nv):","").strip(), "gloss": "", "morphs": []}
    elif line.startswith("Gloss (en):"):
        current_word["gloss"] = line.replace("Gloss (en):","").strip()
    elif line.startswith("Morph:"):
        current_word["morphs"].append(line.replace("Morph:","").strip())
    elif line == "============================================================":
        if current_word:
            current_words.append(current_word)
        if current_eng:
            morpho_dict[current_eng] = {"nv": current_nv, "words": current_words}
        current_word = {}
        current_words = []
        current_nv = ""
        current_eng = ""

# -----------------------------
# Step 1: Parse chrf_most_similar.txt
# -----------------------------
with open("bertscore_most_similar.txt", "r", encoding="utf-8") as f:
    chrf_lines = f.readlines()

# Store reference sentences and their top-5 matches
references = []
current_ref = ""
current_matches = []

for line in chrf_lines:
    line = line.strip()
    ref_match = re.match(r"REFERENCE \d+: (.+)", line)
    sim_match = re.match(r"\d+\. .*?] (.+)", line)

    if ref_match:
        if current_ref:
            references.append((current_ref, current_matches))
        current_ref = ref_match.group(1)
        current_matches = []
    elif sim_match:
        current_matches.append(sim_match.group(1))

# Append last reference
if current_ref:
    references.append((current_ref, current_matches))

# -----------------------------
# Step 2: Helper function to get morpho info
# -----------------------------
def get_morpho_info(eng_sentence):
    if eng_sentence not in morpho_dict:
        return "[Navajo missing]", "[]"
    data = morpho_dict[eng_sentence]
    nv_sent = data["nv"]
    words = data["words"]
    word_entries = []
    for w in words:
        morphs_str = ": ".join(w["morphs"]) if w["morphs"] else ""
        word_entries.append(f"{w['nv']} (nv): {w['gloss']} (en): {morphs_str}")
    morpho_line = "[{}]".format(" | ".join(word_entries))
    return nv_sent, morpho_line

# -----------------------------
# Step 3: Write all data to chrf_withgloss.txt
# -----------------------------
with open("bertscore_withgloss.txt", "w", encoding="utf-8") as out_file:
    for ref_idx, (ref_sentence, sim_sentences) in enumerate(references, start=1):
        out_file.write(f"(REFERENCE {ref_idx})\n")
        out_file.write(f"English: {ref_sentence}\n")
        # Extract the 5 most similar sentences in reverse order (5→1)
        for i, eng_sentence in enumerate(reversed(sim_sentences[:5]), start=1):
            nv_sent, morpho_line = get_morpho_info(eng_sentence)
            out_file.write(f"{i}\n")
            out_file.write(f"English: {eng_sentence}\n")
            out_file.write(f"Navajo: {nv_sent}\n")
            out_file.write(f"{morpho_line}\n\n")
        out_file.write(f"Reference: {ref_sentence}\n\n")

