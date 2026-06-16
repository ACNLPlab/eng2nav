import random

INPUT_FILE = "eng_nav_navumr_output.txt"
NUM_SAMPLES = 15


def load_sentence_pairs(filepath):
    pairs = []
    gold = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Gold:"):
                gold = line.replace("Gold:", "").strip()

            elif line.startswith("Translation:") and gold is not None:
                translation = line.replace("Translation:", "").strip()
                pairs.append((gold, translation))
                gold = None  # reset for next pair

    return pairs


def main():
    pairs = load_sentence_pairs(INPUT_FILE)
    print(f"Loaded {len(pairs)} sentence pairs.\n")

    if len(pairs) < NUM_SAMPLES:
        raise ValueError(
            f"Only {len(pairs)} sentence pairs found, cannot sample {NUM_SAMPLES}."
        )

    for i, (gold, translation) in enumerate(random.sample(pairs, NUM_SAMPLES), 1):
        print(f"{i}.")
        print(f"Gold: {gold}")
        print(f"Translation: {translation}")
        print("-" * 60)


if __name__ == "__main__":
    main()
