import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
json_path = "/home/common/ACNLP/eng2nav/prompt-pipeline/bertscore_eng_nav_umr.json"
output_dir = Path("/home/common/ACNLP/eng2nav/prompt-pipeline/bertscore_results")
model_name = "gpt-5"

# Output files for the four prompt settings
output_files = {
    "eng_engumr_nav_navumr": output_dir / "eng_engumr_nav_navumr_output.txt",
    "eng_engumr_nav": output_dir / "eng_engumr_nav_output.txt",
    "eng_nav_navumr": output_dir / "eng_nav_navumr_output.txt",
    "eng_nav": output_dir / "eng_nav_output.txt",
}

# Debug file for prompt inspection
prompt_debug_file = output_dir / "prompts_debug.txt"

# Initialize client
client = OpenAI(
  api_key=""
)

# === HELPER FUNCTIONS ===

def build_prompt(five_shot_examples, new_sentence, mode):
    """
    mode options:
        1: eng + eng_umr + nav + nav_umr
        2: eng + eng_umr + nav
        3: eng + nav + nav_umr
        4: eng + nav
    """
    prompt = ""
    for ex in five_shot_examples:
        prompt += f"English: {ex['eng']}\n"
        if mode in (1, 2):
            prompt += f"English UMR: {ex['eng_umr']}\n"
        prompt += f"Navajo: {ex['nav']}\n"
        if mode in (1, 3):
            prompt += f"Navajo UMR: {ex['nav_umr']}\n"
        prompt += "\n"

    prompt += f"English: {new_sentence['eng']}\n"
    if mode in (1, 2):
        prompt += f"English UMR: {new_sentence['eng_umr']}\n"
    prompt += "Navajo: "
    return prompt


def run_translation(prompt):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a machine translation system that translates English narrative sentences into Navajo. "
                    "Do not provide explanations, only output the translation."
                )
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


# === MAIN EXECUTION ===

def main():
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare output files (clear any existing)
    for fpath in output_files.values():
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("")

    # Clear prompt debug file
    with open(prompt_debug_file, "w", encoding="utf-8") as f:
        f.write("=== DEBUG PROMPTS LOG ===\n\n")

    # Iterate through dataset
    for ref_name, content in tqdm(data.items(), desc="Processing entries"):
        new_sentence = content["new_sentence"]
        five_shot_examples = content["five_shot_examples"]
        gold_nav = new_sentence["nav"]

        for mode, label in zip(
            (1, 2, 3, 4),
            ("eng_engumr_nav_navumr", "eng_engumr_nav", "eng_nav_navumr", "eng_nav")
        ):
            try:
                prompt = build_prompt(five_shot_examples, new_sentence, mode)

                # Write prompt to debug file
                with open(prompt_debug_file, "a", encoding="utf-8") as dbg:
                    dbg.write(f"=== FILE: {ref_name} ===\n")
                    dbg.write(f"--- MODE: {label} ---\n")
                    dbg.write(prompt + "\n")
                    dbg.write("--- END PROMPT ---\n\n")

                # Get translation
                translation = run_translation(prompt)
                outpath = output_files[label]

                with open(outpath, "a", encoding="utf-8") as out:
                    out.write(f"File: {ref_name}\n")
                    out.write(f"Gold: {gold_nav}\n")
                    out.write(f"Translation: {translation}\n\n")

            except Exception as e:
                print(f"Error processing {ref_name} [{label}]: {e}")

    print("✅ All translations and prompts logged successfully.")


if __name__ == "__main__":
    main()

