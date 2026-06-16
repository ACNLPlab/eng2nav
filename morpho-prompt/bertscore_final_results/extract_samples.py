import random

INPUT_FILE = "FINAL_RESULTS.txt"
NUM_SAMPLES = 15


def load_sentence_triples(filepath):
    triples = []

    english = None
    gold = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("English:"):
                english = line.replace("English:", "").strip()

            elif line.startswith("Gold:"):
                gold = line.replace("Gold:", "").strip()

            elif (
                line.startswith("Translation:")
                and english is not None
                and gold is not None
            ):
                translation = line.replace("Translation:", "").strip()
                triples.append((english, gold, translation))

                # reset for next chunk
                english = None
                gold = None

    return triples


def main():
    triples = load_sentence_triples(INPUT_FILE)
    print(f"Loaded {len(triples)} sentence triples.\n")

    if len(triples) < NUM_SAMPLES:
        raise ValueError(
            f"Only {len(triples)} sentence triples found, cannot sample {NUM_SAMPLES}."
        )

    for i, (english, gold, translation) in enumerate(
        random.sample(triples, NUM_SAMPLES), 1
    ):
        print(f"{i}.")
        print(f"English: {english}")
        print(f"Gold: {gold}")
        print(f"Translation: {translation}")
        print("-" * 60)


if __name__ == "__main__":
    main()

