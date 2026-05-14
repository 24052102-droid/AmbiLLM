import json
import os
import random
from datetime import datetime

from generator.llama_generate import generate_sentences
from ner.entity_extractor import extract_entities
from validator.wikidata_checker import check_entity_exists
from scorer.authenticity_score import calculate_authenticity_score
from utils.text_cleaner import clean_sentence

DATASET_PATH = "datasets/ambiguity_data.json"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_to_json(data: list) -> None:
    """Append *data* to the existing JSON dataset file (creates it if absent)."""
    os.makedirs("datasets", exist_ok=True)

    existing_data = []

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except (json.JSONDecodeError, ValueError):
                existing_data = []

    existing_data.extend(data)

    with open(DATASET_PATH, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_sentence(sentence: str) -> dict:
    entities = extract_entities(sentence)

    validated_entities = []
    hallucinated_entities = []
    unknown_entities = []
    real_entities_count = 0

    for entity in entities:
        print(f"  Checking entity: {entity['entity']} [{entity['label']}]")

        validation = check_entity_exists(entity["entity"])
        exists = validation["exists"]

        if exists is True:
            real_entities_count += 1
        elif exists is False:
            hallucinated_entities.append(entity["entity"])
        else:
            unknown_entities.append(entity["entity"])

        validated_entities.append({
            "entity": entity["entity"],
            "label": entity["label"],
            "exists": exists,
            "matched_title": validation["matched_title"],
            "summary": validation["summary"]
        })

    authenticity = calculate_authenticity_score(
        real_entities=real_entities_count,
        total_entities=len(entities),
        unknown_entities=len(unknown_entities),
    )

    return {
        "timestamp": str(datetime.now()),
        "sentence": sentence,
        "entities": validated_entities,
        "hallucinated_entities": hallucinated_entities,
        "unknown_entities": unknown_entities,
        "entity_authenticity_score": authenticity,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:

    topics = [
        "space exploration", "technology", "politics", "sports",
        "history", "medicine", "music", "movies", "biology",
        "gaming", "business", "artificial intelligence",
        "quantum physics", "cybersecurity", "astronomy"
    ]

    print("\nGenerating sentences...\n")

    all_results = []
    seen_sentences: set[str] = set()

    for idx in range(15):
        
        topic = random.choice(topics)

        prompt = f"""
        Generate ONLY ONE sentence about {topic}.

        The sentence must contain:
        - a person
        - an organization
        - a location

        Some entities may be real.
        Some entities may be fictional.

        But don't specify which ones are real or fictional.

        IMPORTANT:
        - Use completely different entities every time.
        - Avoid repeating organizations.
        - Avoid repeating sentence structures.
        - Do NOT start with "The United Nations".
        - Do NOT explain anything.
        - Do NOT add notes.
        - Do NOT add bullet points.
        - Return ONLY the sentence.
        """

        sentences = generate_sentences(prompt=prompt, count=1)

        if not sentences:
            continue

        sentence = clean_sentence(sentences[0])

        if not sentence or sentence in seen_sentences:
            continue

        seen_sentences.add(sentence)

        print("\n" + "=" * 60)
        print(f"Sentence {idx + 1} [{topic}]")
        print("=" * 60)
        print(sentence)
        print()

        result = process_sentence(sentence)

        score_info = result["entity_authenticity_score"]
        print(
            f"\nAuthenticity Score : {score_info['score']} "
            f"(coverage: {score_info['coverage']})"
        )
        print(f"Real               : {score_info['raw_counts']['real']}")
        print(f"Hallucinated       : {score_info['raw_counts']['hallucinated']}")
        print(f"Unknown (API error): {score_info['raw_counts']['unknown']}")

        print("\nHallucinated Entities:")
        if result["hallucinated_entities"]:
            for h in result["hallucinated_entities"]:
                print(f"  - {h}")
        else:
            print("  None")

        if result["unknown_entities"]:
            print("\nUnverified Entities (API error — re-run to retry):")
            for u in result["unknown_entities"]:
                print(f"  ? {u}")

        all_results.append(result)

    # FIX 4: Save ONCE after all 15 sentences are processed
    save_to_json(all_results)
    print("\n\nALL RESULTS SAVED SUCCESSFULLY →", DATASET_PATH)


if __name__ == "__main__":
    main()